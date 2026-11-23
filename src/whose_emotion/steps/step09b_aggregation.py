# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 9b — AGGREGATION AND PRE-REGISTERED HYPOTHESIS TESTS (multiseed aggregation cell)
# Reads the metrics of every evaluated run of every stage from the Hub and computes, where the data exist:
#   tuning selection -> shared/tuned_recipes.json | seed tables (mean +- sample SD, n) | item-paired and seed-paired
#   prompt contrasts (exact McNemar, Wald CI, MDES, equivalence margin, t / sign / permutation, random-effects)
#   H1 explicit share | H2 explicit vs implicit transfer (GEE) | H3 CRI vs transfer (OLS, MixedLM, cluster bootstrap,
#   ATC comparison) | H4 prompt effect by stratum | H5 cue masking vs random masking | convergence (Fisher).
# Every analysis is skipped with a log message when its inputs are not yet available.
# =====================================================================================
import warnings
from scipy import stats as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf

ANALYSIS_DIR = LOCAL_ROOT / ANALYSIS_PREFIX
(ANALYSIS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(ANALYSIS_DIR / "figures").mkdir(parents=True, exist_ok=True)
ANALYSIS = {}

def load_all_metrics() -> pd.DataFrame:
    rows = []
    for f in sorted(HUB_FILES):
        if not (f.endswith("/metrics/metrics.json") and "/runs/" in f):
            continue
        lp = LOCAL_ROOT / f
        path = lp if lp.exists() else hub_download(f)
        with open(path) as fh:
            m = json.load(fh)
        meta_path = f.replace("/metrics/metrics.json", "/metadata/run_metadata.json")
        lm = LOCAL_ROOT / meta_path
        mp = lm if lm.exists() else hub_download(meta_path)
        meta = json.load(open(mp)) if mp is not None else {}
        extra = {k: meta.get(k) for k in ("train_seconds", "test_items_per_second", "trainable_parameters", "total_parameters",
                                          "epochs_run", "steps_run", "precision_used", "prefix", "gpu")}
        rows.append({**m["spec"], **extra, **{k: v for k, v in m.items() if k != "spec"}})
    df = pd.DataFrame(rows)
    logger.info(f"Loaded metrics for {len(df)} evaluated runs across stages")
    return df

def load_test_correct(spec_row) -> np.ndarray:
    rp = f"{EXPERIMENT_NAME}/{spec_row['stage']}/runs/{spec_row['spec_id']}"
    lp = LOCAL_ROOT / rp / "outputs" / "logits.npz"
    path = lp if lp.exists() else hub_download(f"{rp}/outputs/logits.npz")
    with np.load(path) as z:
        return z["test_logits"].argmax(1) == z["test_y"]

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def paired_item_stats(ca: np.ndarray, cb: np.ndarray) -> dict:
    ca, cb = np.asarray(ca, bool), np.asarray(cb, bool)
    n = len(ca); b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    d = (b - c) / n; psi = (b + c) / n
    se = math.sqrt(max(psi - d * d, 0.0) / n)
    p = float(sps.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return {"n": n, "b": b, "c": c, "delta_pp": 100 * d, "ci95_pp": [100 * (d - 1.96 * se), 100 * (d + 1.96 * se)],
            "ci90_pp": [100 * (d - 1.645 * se), 100 * (d + 1.645 * se)], "p_exact": p,
            "mdes80_pp": 100 * (1.959964 + 0.841621) * math.sqrt(psi / n),
            "equivalence_margin_pp": 100 * max(abs(d - 1.645 * se), abs(d + 1.645 * se))}

def sign_flip_p(d: np.ndarray, seed: int = 0) -> float:
    d = np.asarray(d, float); obs = abs(d.mean())
    if len(d) <= N_PERMUTATIONS_MAX_EXACT:
        signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(100000, len(d)))
    return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-12))

def seed_paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    t = sps.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    nz = d[d != 0]
    return {"n": n, "mean": mean, "sd": sd, "ci95": [mean - t * sd / math.sqrt(n), mean + t * sd / math.sqrt(n)],
            "p_t": float(sps.ttest_1samp(d, 0).pvalue) if n > 1 and sd > 0 else float("nan"),
            "p_sign": float(sps.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else 1.0,
            "p_perm": sign_flip_p(d), "var_mean": sd * sd / n if n > 1 else float("nan")}

def random_effects(effects, variances) -> dict:
    y, v = np.asarray(effects, float), np.asarray(variances, float)
    w = 1 / v; fixed = np.sum(w * y) / np.sum(w); q = float(np.sum(w * (y - fixed) ** 2)); k = len(y)
    cc = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / cc) if cc > 0 else 0.0
    wr = 1 / (v + tau2); mu = float(np.sum(wr * y) / np.sum(wr)); se = float(math.sqrt(1 / np.sum(wr)))
    return {"k": k, "mu": mu, "se": se, "ci95": [mu - 1.96 * se, mu + 1.96 * se], "ci90": [mu - 1.645 * se, mu + 1.645 * se],
            "tau2": tau2, "Q": q, "I2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0}

def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p)); out[k] = running
    return out

RECIPE_KEYS = ["stage", "corpus", "encoder", "placement", "head", "text_transform", "transform_eval", "fewshot_k",
               "peft_method", "rebalance", "lr", "epochs"]

def tex_table(df: pd.DataFrame, cols: list, path: Path, fmt: dict) -> None:
    """Write booktabs body rows (no environment) so the paper can \\input them; values formatted per column."""
    lines = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c](v) if c in fmt and v is not None and not (isinstance(v, float) and np.isnan(v)) else
                         ("--" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).replace("_", "\\_")))
        lines.append(" & ".join(cells) + " \\\\")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

pct = lambda v: f"{100 * v:.2f}"
f4 = lambda v: f"{v:.4f}"
pm_pct = lambda m, s: f"{100 * m:.2f} $\\pm$ {100 * s:.2f}" if s == s else f"{100 * m:.2f}"

def nontrain_tables(A: dict, D_LLM: pd.DataFrame, D_AUDIT: pd.DataFrame, TEX: Path) -> None:
    if len(D_LLM):
        lt = D_LLM[["model", "shots", "acc", "macro_f1", "format_failure_rate", "acc_on_parsed_only", "models_reported"]].sort_values(["model", "shots"])
        tex_table(lt, ["model", "shots", "acc", "macro_f1", "format_failure_rate"], TEX / "llm.tex", {"acc": pct, "macro_f1": f4, "format_failure_rate": pct})
        A["table_llm"] = lt.to_dict("records")
    if len(D_AUDIT):
        A["audit"] = D_AUDIT.iloc[-1]["audit"]

def cev_analyses(df: pd.DataFrame) -> dict:
    """crowd-enVENT analyses (pre-registration amendment).
    Confirmatory: E1 author-masked minimal pairs with a target-trained control (difference in differences on the same items);
    E2 CRI versus the minimal-pair drop; E3 writer, reader, and model on identical texts.
    Exploratory: E4 appraisal dimensions of model errors; E5 same-configuration reruns (cev stages versus main/shortcut)."""
    out = {}
    src, tgt = df[df.stage.isin(CEV_STAGES)], df[df.stage == "cev_target"]
    if not len(src) and not len(tgt):
        return out
    y, ch, rp, mj = CEV["label"], CEV["changed"], CEV["reader_pos"], CEV["reader_majority"]
    ch_r, ok = ch[rp], mj >= 0
    yr = y[rp]
    def npz(r):
        p = f"{EXPERIMENT_NAME}/{r.stage}/runs/{r.spec_id}"
        lp = LOCAL_ROOT / p / "outputs" / "logits.npz"
        return np.load(lp if lp.exists() else hub_download(f"{p}/outputs/logits.npz"))
    S, T = [], []
    for _, r in src.iterrows():
        with npz(r) as z:
            S.append((r, cev_restrict(z["cev_orig_logits"]), cev_restrict(z["cev_masked_logits"])))
    for _, r in tgt.iterrows():
        with npz(r) as z:
            T.append((r, cev_restrict(z["test_logits"]), cev_restrict(z["cevtest_masked_logits"])))
    TEX = ANALYSIS_DIR / "tables"
    # ---- E1: minimal pairs (863 author-masked texts); control = target-trained models on the 163 masked reader-set items
    if S:
        d = pd.DataFrame([{"stage": r.stage, "encoder": r.encoder, "condition": r.condition, "seed": int(r.seed),
                           "text_transform": r.text_transform, "cri": r.get("cri"), "in_acc": float(r.test_acc),
                           "drop863": 100 * (np.mean(po[ch] == y[ch]) - np.mean(pm[ch] == y[ch])),
                           "drop163": 100 * (np.mean(po[rp][ch_r] == yr[ch_r]) - np.mean(pm[rp][ch_r] == yr[ch_r]))} for r, po, pm in S])
        base = d[d.stage == "cev_source"]
        e1, per = {"n_pairs": int(ch.sum())}, []
        for enc, g in base.groupby("encoder"):
            if len(g) >= 2:
                per.append((enc, float(g.drop863.mean()), float(g.drop863.var(ddof=1) / len(g)), len(g)))
        e1["source_drop_pp_by_encoder"] = {e: {"mean": m_, "n_runs": n} for e, m_, _, n in per}
        if len(per) >= 2:
            e1["source_drop_pp_pooled"] = random_effects([p[1] for p in per], [max(p[2], 1e-9) for p in per])
        if T:
            td = pd.DataFrame([{"encoder": r.encoder, "seed": int(r.seed), "acc500": float(np.mean(po == yr)),
                                "drop163": 100 * (np.mean(po[ch_r] == yr[ch_r]) - np.mean(pm[ch_r] == yr[ch_r]))} for r, po, pm in T])
            did = []
            for enc in sorted(set(base.encoder) & set(td.encoder)):
                a, b = base[base.encoder == enc].drop163.to_numpy(), td[td.encoder == enc].drop163.to_numpy()
                if len(a) >= 2 and len(b) >= 2:
                    did.append((enc, float(a.mean() - b.mean()), float(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))))
            e1["target_drop163_pp_by_encoder"] = td.groupby("encoder").drop163.mean().to_dict()
            e1["target_acc500_by_encoder"] = td.groupby("encoder").acc500.mean().to_dict()
            e1["did_source_minus_target_pp"] = {e: v for e, v, _ in did}
            if len(did) >= 2:
                e1["did_pooled"] = random_effects([x[1] for x in did], [max(x[2], 1e-9) for x in did])
        out["cev_E1_minimal_pairs"] = e1
        # ---- E2: does the label-free CRI (our cue vocabulary) predict the drop on the authors' independent masking?
        d2 = d.dropna(subset=["cri"])
        e2 = {"n_runs": int(len(d2)), "n_encoders": int(d2.encoder.nunique())}
        if len(d2) >= 10 and d2.encoder.nunique() >= 2 and d2.cri.std() > 0:
            zf = lambda s: (s - s.mean()) / (s.std() + 1e-12)
            d2 = d2.assign(z_cri=zf(d2.cri), z_in=zf(d2.in_acc))
            ols = smf.ols("drop863 ~ z_in + z_cri", data=d2).fit()
            b = float(ols.params["z_cri"])
            e2.update(spearman_cri_drop=float(sps.spearmanr(d2.cri, d2.drop863)[0]), beta_cri_ols_pp_per_sd=b,
                      p_ols_one_sided=float(ols.pvalues["z_cri"] / 2 if b > 0 else 1 - ols.pvalues["z_cri"] / 2))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mix = smf.mixedlm("drop863 ~ z_in + z_cri", data=d2, groups=d2["encoder"]).fit(reml=True)
                ci = mix.conf_int().loc["z_cri"].tolist()
                if np.all(np.isfinite(ci)):
                    e2.update(beta_cri_mixed=float(mix.params["z_cri"]), ci95_mixed=ci,
                              p_mixed_one_sided=float(sps.norm.sf(mix.tvalues["z_cri"])))
                else:
                    e2["mixed_model_note"] = "singular fit; not used for inference"
            except Exception as e:
                logger.error(f"E2 mixed model failed: {e}")
        out["cev_E2_cri_validation"] = e2
        # ---- E3: writer, reader, and model on the same 500 texts (source models without masked training)
        g_ = CEV["reader_guesses"]; inset = g_ >= 0
        e3 = {"n_items": int(len(rp)), "n_items_with_reader_majority": int(ok.sum()), "n_masked_items": int(ch_r.sum()),
              "human_individual_acc_13way": float(CEV["reader_item_acc"].mean()),
              "human_individual_acc_inset_guesses": float((g_ == yr[:, None])[inset].mean()),
              "human_majority_acc": float(np.mean(mj[ok] == yr[ok])),
              "human_individual_acc_masked_items": float(CEV["reader_item_acc"][ch_r].mean()),
              "human_individual_acc_unmasked_items": float(CEV["reader_item_acc"][~ch_r].mean()),
              "note": ("Readers chose among 13 emotions; models are restricted to the five shared classes. "
                       "Human and model accuracies are therefore not directly comparable; the in-set figure conditions "
                       "readers on choosing one of the five classes.")}
        rows, conf_m = [], np.zeros((6, 6))
        for r, po, pm in S:
            if r.stage != "cev_source":
                continue
            p = po[rp]
            rows.append({"encoder": r.encoder, "condition": r.condition, "seed": int(r.seed),
                         "acc_writer": float(np.mean(p == yr)), "acc_writer_masked_items": float(np.mean(p[ch_r] == yr[ch_r])),
                         "acc_writer_unmasked_items": float(np.mean(p[~ch_r] == yr[~ch_r])),
                         "agree_reader_majority": float(np.mean(p[ok] == mj[ok]))})
            for a_, b_ in zip(yr, p):
                conf_m[a_, b_] += 1
        if rows:
            mt = pd.DataFrame(rows).groupby(["encoder", "condition"]).mean(numeric_only=True).reset_index()
            e3["models"] = mt.drop(columns=["seed"]).to_dict("records")
            tex_table(mt, ["encoder", "condition", "acc_writer", "acc_writer_masked_items", "acc_writer_unmasked_items", "agree_reader_majority"],
                      TEX / "cev_perspective.tex", {c: pct for c in ["acc_writer", "acc_writer_masked_items", "acc_writer_unmasked_items", "agree_reader_majority"]})
            # do models fail where readers fail? item-level GEE clustered by item (source runs without masked training)
            parts = [pd.DataFrame({"correct": (po[rp] == yr).astype(int), "item": np.arange(len(rp)), "masked": ch_r.astype(int),
                                   "reader_acc": CEV["reader_item_acc"]}) for r, po, pm in S if r.stage == "cev_source"]
            dd = pd.concat(parts, ignore_index=True)
            dd["z_reader_acc"] = (dd.reader_acc - dd.reader_acc.mean()) / (dd.reader_acc.std() + 1e-12)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.gee("correct ~ z_reader_acc + masked", groups="item", data=dd, family=sm.families.Binomial(),
                                  cov_struct=sm.cov_struct.Exchangeable()).fit()
                e3["gee_model_correct_on_reader_accuracy"] = {k: {"log_or": float(fit.params[k]), "ci95": fit.conf_int().loc[k].tolist(),
                                                                  "p": float(fit.pvalues[k])} for k in ("z_reader_acc", "masked")}
            except Exception as e:
                logger.error(f"E3 GEE failed: {e}")
            # confusion similarity over the 20 off-diagonal cells of the five shared classes (writer label x predicted/guessed)
            cls = PRESENT_CLASSES["cev"]
            conf_h = np.zeros((6, 6))
            for a_, gl in zip(yr, g_):
                for gg in gl:
                    if gg >= 0:
                        conf_h[a_, gg] += 1
            norm = lambda M: (M / np.maximum(M.sum(1, keepdims=True), 1))[np.ix_(cls, cls)]
            off = ~np.eye(len(cls), dtype=bool)
            e3["confusion_similarity_spearman_offdiag"] = float(sps.spearmanr(norm(conf_m)[off], norm(conf_h)[off])[0])
        out["cev_E3_perspective"] = e3
        # ---- E4 (exploratory): which writer appraisals characterize model errors? item-clustered GEE with class fixed effects
        e4 = {}
        none_runs = [(r, po) for r, po, pm in S if r.stage == "cev_source" and r.condition == "none"]
        if none_runs:
            A_ = CEV["appraisal"]; Z = (A_ - A_.mean(0)) / (A_.std(0) + 1e-12)
            parts = []
            for r, po in none_runs:
                f_ = pd.DataFrame(Z, columns=CEV_APPRAISALS)
                f_["correct"] = (po == y).astype(int); f_["item"] = np.arange(len(y)); f_["label"] = y; f_["masked"] = ch.astype(int)
                parts.append(f_)
            dd = pd.concat(parts, ignore_index=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.gee("correct ~ " + " + ".join(CEV_APPRAISALS) + " + C(label) + masked", groups="item", data=dd,
                                  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
                ci = fit.conf_int()
                pv = {k: float(fit.pvalues[k]) for k in CEV_APPRAISALS}
                ph = holm(pv)
                e4 = {k: {"log_or_per_sd": float(fit.params[k]), "ci95": ci.loc[k].tolist(), "p": pv[k], "p_holm": ph[k]} for k in CEV_APPRAISALS}
                e4["_masked"] = {"log_or": float(fit.params["masked"]), "ci95": ci.loc["masked"].tolist(), "p": float(fit.pvalues["masked"])}
                e4["_note"] = "exploratory; adjusted for emotion class; Holm over the 21 appraisal dimensions"
            except Exception as e:
                logger.error(f"E4 GEE failed: {e}")
        out["cev_E4_appraisals_exploratory"] = e4
    # ---- E5 (exploratory): same configuration trained twice (cev stages vs main/shortcut), same-GPU pairs separately
    keys = ["corpus", "encoder", "condition", "seed", "lr", "epochs", "placement", "head", "text_transform", "transform_eval"]
    ref = df[df.stage.isin(["main", "shortcut"])]
    pairs = []
    for _, r in src.iterrows():
        mrow = ref
        for k in keys:
            mrow = mrow[mrow[k] == r[k]]
        if len(mrow) == 1:
            m0 = mrow.iloc[0]
            c1, c0 = load_test_correct(r), load_test_correct(m0)
            pairs.append({"encoder": r.encoder, "same_gpu": bool(r.get("gpu") == m0.get("gpu")),
                          "abs_diff_pp": float(100 * abs(r.test_acc - m0.test_acc)), "item_disagreement": float(np.mean(c1 != c0))})
    if pairs:
        pp = pd.DataFrame(pairs)
        out["cev_E5_rerun_variation_exploratory"] = {
            "n_pairs": int(len(pp)), "n_same_gpu": int(pp.same_gpu.sum()),
            "same_gpu": pp[pp.same_gpu][["abs_diff_pp", "item_disagreement"]].describe().to_dict() if pp.same_gpu.any() else None,
            "all": pp[["abs_diff_pp", "item_disagreement"]].describe().to_dict()}
    return out

def aggregate(df: pd.DataFrame) -> dict:
    A = {}
    if df.empty:
        logger.warning("No evaluated runs yet; nothing to aggregate.")
        return A
    df = df.copy()
    df["kind"] = df["kind"].fillna("train") if "kind" in df else "train"
    D_LLM, D_AUDIT = df[df.kind == "llm"], df[df.kind == "audit"]
    df = df[df.kind == "train"].copy()
    for k in ["text_transform", "fewshot_k", "peft_method", "rebalance"]:
        if k not in df:
            df[k] = None
        df[k] = df[k].astype(object).where(df[k].notna(), "none")
    TEX = ANALYSIS_DIR / "tables"
    if not len(df):                                              # no training runs yet (e.g. after the audit stage)
        nontrain_tables(A, D_LLM, D_AUDIT, TEX)
        A["H1_explicit_share"] = {k: {"share": float(v.mean()), "n": int(len(v)), "wilson95": wilson(int(v.sum()), len(v))}
                                  for k, v in EXPLICIT.items()}
        logger.info("No training runs evaluated yet; only audit/LLM tables and H1 computed.")
        return A
    # ---- recipe selection (tuning: main encoders; lr_sweep: DeBERTa/ModernBERT), mean over both conditions ------
    selected = dict(TUNED_RECIPES or {})
    for st, encs in RECIPE_SELECTION_STAGES.items():
        tun = df[df.stage == st]
        if not len(tun):
            continue
        exp_n = len(expand_stage(st, None))
        if len(tun) < exp_n:
            logger.info(f"{st} incomplete ({len(tun)}/{exp_n} runs evaluated); recipes not selected yet.")
            continue
        g = tun.groupby(["encoder", "lr", "epochs"]).val_acc.agg(["mean", "std", "count"]).reset_index()
        A[f"{st}_table"] = g.to_dict("records")
        for enc, sub in g.groupby("encoder"):
            r = sub.sort_values(["mean", "lr"], ascending=[False, True]).iloc[0]      # tie -> smaller learning rate
            rec = {"lr": float(r.lr), "epochs": int(r.epochs), "val_acc_mean": float(r["mean"]), "val_acc_sd": float(r["std"]),
                   "n_runs": int(r["count"]), "selected_in": st}
            if enc in selected and (selected[enc]["lr"] != rec["lr"] or selected[enc]["epochs"] != rec["epochs"]):
                raise RuntimeError(f"Tuned recipe for {enc} on the Hub differs from the recomputed selection. Investigate.")
            selected[enc] = rec
    if selected and selected != (TUNED_RECIPES or {}):
        lp = local_path(TUNED_RECIPES_PATH); write_json(lp, selected); hub_upload_file(lp, TUNED_RECIPES_PATH)
        logger.info(f"Tuned recipes written to the Hub: { {k: (v['lr'], v['epochs']) for k, v in selected.items()} }")
    A["tuned_recipes"] = selected
    # ---- seed tables -------------------------------------------------------------------------------------------
    keys = ["stage", "corpus", "encoder", "condition", "placement", "head", "text_transform", "transform_eval", "fewshot_k",
            "peft_method", "rebalance"]
    tab = df.groupby(keys).agg(n=("test_acc", "size"), acc_mean=("test_acc", "mean"), acc_sd=("test_acc", "std"),
                               mf1_mean=("test_macro_f1", "mean"), mf1_sd=("test_macro_f1", "std"),
                               converged=("converged", "sum")).reset_index()
    A["seed_table"] = tab.to_dict("records")
    tab.to_csv(ANALYSIS_DIR / "metrics" / "seed_table.csv", index=False)
    # ---- prompt contrasts (condition vs none, same recipe and seed) ------------------------------------------------
    item_rows, seed_rows = [], []
    base = df[df.condition == "none"].set_index(RECIPE_KEYS + ["seed"])
    # the cev and eni stages repeat main/shortcut configurations: they never enter the prompt-effect statistics
    for key, grp in df[(df.condition != "none") & ~df.stage.isin(CEV_STAGES + ["cev_target", "cev_third"] + ENI_EXPORT_STAGES)].groupby(RECIPE_KEYS + ["condition"]):
        rk = key[:-1]; cond = key[-1]
        pa, pb = [], []
        for _, r in grp.iterrows():
            k0 = tuple(rk) + (r.seed,)
            if k0 not in base.index:
                continue
            r0 = base.loc[k0]
            pa.append(r.test_acc); pb.append(r0.test_acc)
            if r.stage in ("main", "intra", "shortcut", "scaling", "target"):
                ca, cb = load_test_correct(r), load_test_correct(r0.to_dict() | {"stage": k0[0], "spec_id": r0.spec_id})
                row = {**dict(zip(RECIPE_KEYS, rk)), "condition": cond, "seed": int(r.seed), **paired_item_stats(ca, cb)}
                sk = STRATA_KEY.get("setfit" if str(r.corpus).startswith("setfit") else r.corpus)
                if sk and len(EXPLICIT[sk]) == len(ca) and not r.transform_eval:
                    ex = EXPLICIT[sk]
                    row["explicit"] = paired_item_stats(ca[ex], cb[ex]) if ex.any() else None
                    row["implicit"] = paired_item_stats(ca[~ex], cb[~ex]) if (~ex).any() else None
                item_rows.append(row)
        if len(pa) >= 3:
            seed_rows.append({**dict(zip(RECIPE_KEYS, rk)), "condition": cond,
                              **seed_paired(100 * np.array(pa), 100 * np.array(pb))})
    A["item_paired"] = item_rows; A["seed_paired"] = seed_rows
    pooled = {}
    sp = pd.DataFrame(seed_rows)
    if len(sp):
        for key, g in sp.groupby(["stage", "corpus", "condition", "placement", "head", "text_transform", "transform_eval", "fewshot_k",
                                  "peft_method", "rebalance"]):
            g = g[g.var_mean > 0]
            if len(g) >= 2:
                pooled["|".join(map(str, key))] = random_effects(g["mean"], g["var_mean"])
    A["random_effects"] = pooled
    # ---- H1: explicit share -------------------------------------------------------------------------------------
    A["H1_explicit_share"] = {k: {"share": float(v.mean()), "n": int(len(v)), "wilson95": wilson(int(v.sum()), len(v))}
                              for k, v in EXPLICIT.items()}
    # ---- H2: explicit vs implicit transfer accuracy (GEE over item correctness, clustered by run) ----------------
    src = df[df.stage.isin(["main"]) & df.condition.isin(["none", "A"])]
    h2, h2p = {}, {}
    for tgt in TRANSFER_TARGETS:
        col = f"transfer_{tgt}"
        if col not in src or src[col].isna().all():
            continue
        ys, xs, gs = [], [], []
        for _, r in src[src[col].notna()].iterrows():
            rp = f"{EXPERIMENT_NAME}/{r.stage}/runs/{r.spec_id}"
            lp = LOCAL_ROOT / rp / "outputs" / "logits.npz"
            with np.load(lp if lp.exists() else hub_download(f"{rp}/outputs/logits.npz")) as z:
                lg, y = z[f"transfer_{tgt}_logits"], z[f"transfer_{tgt}_y"]
            allowed = PRESENT_CLASSES[tgt]
            rl = np.full_like(lg, -np.inf); rl[:, allowed] = lg[:, allowed]
            ex = EXPLICIT[tgt if tgt != "ed" else "ed_test"]
            # cluster by ITEM: explicitness is an item-level property and every item recurs once per run, so clustering by
            # run would treat the repeated items as independent and make the interval far too narrow
            ys.append((rl.argmax(1) == y).astype(int)); xs.append(ex.astype(int)); gs.append(np.arange(len(y)))
        y_all, x_all, g_all = np.concatenate(ys), np.concatenate(xs), np.concatenate(gs)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.GEE(y_all, sm.add_constant(x_all), groups=g_all, family=sm.families.Binomial(),
                             cov_struct=sm.cov_struct.Exchangeable()).fit()
            est, lo, hi = float(fit.params[1]), float(fit.conf_int()[1][0]), float(fit.conf_int()[1][1])
            p_one = float(sps.norm.sf(fit.params[1] / fit.bse[1]))          # H2 predicts explicit > implicit
        except Exception as e:
            logger.error(f"H2 GEE failed for {tgt}: {e}"); continue
        h2[tgt] = {"n_runs": len(ys), "acc_explicit": float(y_all[x_all == 1].mean()), "acc_implicit": float(y_all[x_all == 0].mean()),
                   "log_or_explicit": est, "ci95": [lo, hi], "p_one_sided": p_one}
        h2p[tgt] = p_one
    for t, p in holm(h2p).items():
        h2[t]["p_holm"] = p
    A["H2_stratified_transfer"] = h2
    # ---- H3: CRI predicts transfer beyond in-domain accuracy ---------------------------------------------------------
    h3, h3p = {}, {}
    # in-domain accuracy must be measured on the original test set, so runs evaluated on masked test text are excluded
    cand = (df[df.stage.isin(["main", "shortcut"]) & df["cri"].notna() & (df.transform_eval != True)] if "cri" in df else df.iloc[0:0])
    for tgt in TRANSFER_TARGETS:
        col = f"transfer_{tgt}"
        if col not in cand or cand[col].isna().all():
            continue
        c = cand[cand[col].notna()].copy()
        c["transfer_acc"] = c[col].map(lambda d: d["restricted"]["acc"]); c["atc"] = c[col].map(lambda d: d["atc_predicted_acc"])
        if len(c) < 10 or c.encoder.nunique() < 2:
            logger.info(f"H3 {tgt}: {len(c)} runs from {c.encoder.nunique()} encoders; waiting for more runs")
            continue
        z = lambda s: (s - s.mean()) / (s.std() + 1e-12)
        c["z_in"], c["z_cri"] = z(c.test_acc), z(c.cri)
        ols = smf.ols("transfer_acc ~ z_in + z_cri", data=c).fit()
        r2_in = smf.ols("transfer_acc ~ z_in", data=c).fit().rsquared
        res = {"n_runs": int(len(c)), "n_encoders": int(c.encoder.nunique()), "beta_cri_ols": float(ols.params["z_cri"]),
               "beta_in_ols": float(ols.params["z_in"]), "r2_in_only": float(r2_in), "r2_in_plus_cri": float(ols.rsquared),
               "spearman_cri": float(sps.spearmanr(c.cri, c.transfer_acc)[0]),
               "spearman_in_domain": float(sps.spearmanr(c.test_acc, c.transfer_acc)[0]),
               "spearman_atc": float(sps.spearmanr(c.atc, c.transfer_acc)[0])}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mix = smf.mixedlm("transfer_acc ~ z_in + z_cri", data=c, groups=c["encoder"], re_formula="1",
                                  vc_formula={"seed": "0 + C(seed)"}).fit(reml=True)
            ci = mix.conf_int().loc["z_cri"].tolist()
            p1 = float(sps.norm.cdf(mix.tvalues["z_cri"]))                  # H3 predicts a negative CRI coefficient
            if np.all(np.isfinite(ci)) and np.isfinite(p1):
                res.update(beta_cri_mixed=float(mix.params["z_cri"]), ci95_mixed=ci, p_mixed_one_sided=p1)
                h3p[tgt] = p1
            else:
                # WARNING: singular random-effects fit (e.g. no between-encoder variance); the cluster bootstrap is reported instead
                logger.warning(f"H3 {tgt}: mixed model gave a non-finite interval; reporting OLS and cluster bootstrap only")
                res["mixed_model_note"] = "singular fit; not used for inference"
        except Exception as e:
            logger.error(f"H3 mixed model failed for {tgt}: {e}")
        rng = np.random.default_rng(0); encs = c.encoder.unique(); boots = []
        for _ in range(2000):
            pick = rng.choice(encs, len(encs))
            bs = pd.concat([c[c.encoder == e] for e in pick])
            if bs.cri.nunique() > 2 and bs.test_acc.nunique() > 2:
                bs = bs.assign(z_in=z(bs.test_acc), z_cri=z(bs.cri))
                boots.append(smf.ols("transfer_acc ~ z_in + z_cri", data=bs).fit().params["z_cri"])
        res["ci95_cluster_bootstrap"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None
        h3[tgt] = res
    for t, p in holm(h3p).items():
        h3[t]["p_holm"] = p
    A["H3_cri_vs_transfer"] = h3
    # ---- H4: prompt effect by stratum (A vs none, seed-paired, per encoder) ------------------------------------------
    h4 = {}
    ip = [r for r in item_rows if r["condition"] == "A" and r.get("explicit") and r.get("implicit")]
    for corpus in sorted({r["corpus"] for r in ip}):
        rr = [r for r in ip if r["corpus"] == corpus and r["stage"] in ("main", "intra")]
        if len(rr) >= 3:
            diff = np.array([abs(r["implicit"]["delta_pp"]) - abs(r["explicit"]["delta_pp"]) for r in rr])
            h4[corpus] = {"n_pairs": len(rr), "mean_abs_implicit_minus_abs_explicit_pp": float(diff.mean()),
                          "p_perm": sign_flip_p(diff), "n_implicit_items": rr[0]["implicit"]["n"], "n_explicit_items": rr[0]["explicit"]["n"]}
    A["H4_prompt_by_stratum"] = h4
    # ---- H5: cue masking vs random masking -----------------------------------------------------------------------------
    sc = df[df.stage == "shortcut"]
    h5 = {}
    if len(sc) and "cri" in sc:
        piv = sc.pivot_table(index=["encoder", "condition", "transform_eval", "seed"], columns="text_transform", values="cri")
        if {"mask_feel", "mask_random"} <= set(piv.columns):
            piv = piv.dropna()
            if len(piv) >= 3:
                h5["cri_mask_feel_minus_mask_random"] = seed_paired(piv["mask_feel"].values, piv["mask_random"].values)
        for tgt in TRANSFER_TARGETS:
            col = f"transfer_{tgt}"
            if col in sc and sc[col].notna().any():
                s2 = sc[sc[col].notna()].assign(acc=lambda d: d[col].map(lambda x: x["restricted"]["acc"]))
                pv = s2.pivot_table(index=["encoder", "condition", "transform_eval", "seed"], columns="text_transform", values="acc").dropna()
                if {"mask_feel", "mask_random"} <= set(pv.columns) and len(pv) >= 3:
                    h5[f"transfer_{tgt}_mask_feel_minus_mask_random"] = seed_paired(100 * pv["mask_feel"].values, 100 * pv["mask_random"].values)
    A["H5_shortcut"] = h5
    # ---- H2 control: the same explicit/implicit gap in target-trained models would indicate item difficulty -----
    ctrl = {}
    for tgt, tcorp in [("goemotions", "goemotions_target"), ("meld", "meld_target"), ("isear", "isear_target"), ("ed", "ed")]:
        tr_runs = df[(df.corpus == tcorp) & (df.condition == "none") & df.stage.isin(["target", "intra"])]
        if tgt not in h2 or not len(tr_runs):
            continue
        g_t = float((tr_runs.test_acc_explicit - tr_runs.test_acc_implicit).mean())
        g_s = h2[tgt]["acc_explicit"] - h2[tgt]["acc_implicit"]
        if tgt == "isear":                                         # recompute the source gap on the 650 target-test items
            pos = np.asarray(DATA["isear_ft"]["test"]["pos_in_full"]); ex = EXPLICIT["isear"][pos]; accs = []
            for _, r in src[src["transfer_isear"].notna()].iterrows():
                rp = f"{EXPERIMENT_NAME}/{r.stage}/runs/{r.spec_id}"; lp = LOCAL_ROOT / rp / "outputs" / "logits.npz"
                with np.load(lp if lp.exists() else hub_download(f"{rp}/outputs/logits.npz")) as z:
                    lg, y = z["transfer_isear_logits"][pos], z["transfer_isear_y"][pos]
                rl = np.full_like(lg, -np.inf); rl[:, PRESENT_CLASSES["isear"]] = lg[:, PRESENT_CLASSES["isear"]]
                c = rl.argmax(1) == y; accs.append((c[ex].mean(), c[~ex].mean()))
            g_s = float(np.mean([a - b for a, b in accs]))
        ctrl[tgt] = {"gap_source_trained": g_s, "gap_target_trained": g_t, "difference_in_differences": g_s - g_t,
                     "n_target_runs": int(len(tr_runs)), "note": "source and target models scored on the same test items"}
    A["H2_control_target_trained"] = ctrl
    # ---- encoder differences: MixedLM test_acc ~ encoder + (1 | seed), main stage, per condition ----------------
    enc_mm = {}
    mn = df[(df.stage == "main") & df.condition.isin(["none", "A"])]
    for cond, g in mn.groupby("condition"):
        if g.encoder.nunique() >= 2 and g.seed.nunique() >= 3:
            ref = "roberta" if "roberta" in set(g.encoder) else sorted(g.encoder)[0]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.mixedlm(f"test_acc ~ C(encoder, Treatment('{ref}'))", g.assign(test_acc=100 * g.test_acc), groups=g["seed"]).fit(reml=True)
                ci = fit.conf_int()
                enc_mm[cond] = {"reference": ref, "effects_pp": {k.split("T.")[-1].rstrip("]"): {"est": float(fit.params[k]), "ci95": ci.loc[k].tolist(),
                                "p": float(fit.pvalues[k])} for k in fit.params.index if "T." in k}}
            except Exception as e:
                logger.error(f"encoder mixed model failed ({cond}): {e}")
    A["encoder_differences_mixed"] = enc_mm
    # ---- item-level GEE for the prompt effect (A vs none), clustered by test item, encoder fixed effects -----------
    gee = {}
    for (stage, corpus), g in df[df.condition.isin(["none", "A"]) & df.stage.isin(["main", "intra"]) & (df.text_transform == "none")].groupby(["stage", "corpus"]):
        if g.condition.nunique() < 2:
            continue
        parts = []
        for _, r in g.iterrows():
            c = load_test_correct(r)
            parts.append(pd.DataFrame({"y": c.astype(int), "prompt": int(r.condition == "A"), "encoder": r.encoder, "item": np.arange(len(c))}))
        d = pd.concat(parts, ignore_index=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.gee("y ~ prompt + C(encoder)", groups="item", data=d, family=sm.families.Binomial(),
                              cov_struct=sm.cov_struct.Exchangeable()).fit()
            ci = fit.conf_int().loc["prompt"].tolist()
            gee[f"{stage}|{corpus}"] = {"log_or_prompt": float(fit.params["prompt"]), "ci95": ci, "p": float(fit.pvalues["prompt"]),
                                         "n_rows": int(len(d)), "n_runs": int(len(g))}
        except Exception as e:
            logger.error(f"item-level GEE failed for {corpus}: {e}")
    A["prompt_effect_item_gee"] = gee
    # ---- tables for the paper ----------------------------------------------------------------------------------------
    main_none = df[(df.stage == "main")]
    if len(main_none):
        t = main_none.groupby(["encoder", "condition"]).agg(acc=("test_acc", "mean"), acc_sd=("test_acc", "std"), mf1=("test_macro_f1", "mean"),
                                                           mp=("test_macro_precision", "mean"), mr=("test_macro_recall", "mean"), n=("seed", "nunique"),
                                                           epochs=("epochs_run", "mean")).reset_index()
        t["acc_str"] = [pm_pct(a, b) for a, b in zip(t.acc, t.acc_sd)]
        tex_table(t, ["encoder", "condition", "acc_str", "mf1", "mp", "mr", "n"], TEX / "main_by_condition.tex", {"mf1": f4, "mp": f4, "mr": f4})
        A["table_main_by_condition"] = t.drop(columns=["acc_str"]).to_dict("records")
        cal = main_none[main_none.condition.isin(["none", "A"])].copy()
        cal["ece_before"] = cal.calibration.map(lambda c: c["before"]["ece"]); cal["ece_after"] = cal.calibration.map(lambda c: c["after"]["ece"])
        cal["brier_before"] = cal.calibration.map(lambda c: c["before"]["brier"]); cal["brier_after"] = cal.calibration.map(lambda c: c["after"]["brier"])
        cal["T"] = cal.calibration.map(lambda c: c["temperature"]); cal["T_bound"] = cal.calibration.map(lambda c: c["temperature_at_bound"])
        ct = cal.groupby(["encoder", "condition"])[["ece_before", "ece_after", "brier_before", "brier_after", "T", "T_bound"]].mean().reset_index()
        tex_table(ct, ["encoder", "condition", "ece_before", "ece_after", "brier_before", "brier_after", "T"], TEX / "calibration.tex",
                  {"ece_before": pct, "ece_after": pct, "brier_before": f4, "brier_after": f4, "T": lambda v: f"{v:.2f}"})
        A["table_calibration"] = ct.to_dict("records")
        eff = main_none[main_none.condition == "A"].groupby("encoder").agg(params=("total_parameters", "first"), train_min=("train_seconds", lambda x: x.mean() / 60),
                                                                           items_per_s=("test_items_per_second", "mean"), acc=("test_acc", "mean")).reset_index()
        tex_table(eff, ["encoder", "params", "train_min", "items_per_s", "acc"], TEX / "efficiency.tex",
                  {"params": lambda v: f"{int(v):,}", "train_min": lambda v: f"{v:.1f}", "items_per_s": lambda v: f"{v:.0f}", "acc": pct})
        A["table_efficiency"] = eff.to_dict("records")
    rows_t = []
    for tgt in TRANSFER_TARGETS:
        col = f"transfer_{tgt}"
        if col not in df:
            continue
        s_ = df[df.stage.isin(["main"]) & df[col].notna() & df.condition.isin(["none", "A"])]
        for (enc, cond), g in s_.groupby(["encoder", "condition"]):
            row = {"target": tgt, "encoder": enc, "condition": cond, "n": len(g)}
            for rule in ("unrestricted", "restricted", "prior_em"):
                v = g[col].map(lambda d: d[rule]["acc"]); row[f"{rule}_acc"], row[f"{rule}_sd"] = v.mean(), v.std()
            row["atc_pred"] = g[col].map(lambda d: d["atc_predicted_acc"]).mean()
            row["degenerate_runs"] = int(g[col].map(lambda d: d["unrestricted"]["degenerate"]).sum())
            # v8: cev_third and eni_third repeat the GoEmotions configurations of the target stage; keep the upper bound on the
            # target-stage runs only (cev_third gave the same test accuracies, so the published values do not change)
            up = df[(df.corpus == (tgt + "_target" if tgt != "ed" else "ed")) & (df.encoder == enc) & (df.condition == cond)
                    & ~df.stage.isin(["cev_third"] + ENI_EXPORT_STAGES)]
            row["target_trained_acc"] = up.test_acc.mean() if len(up) else None
            if tgt == "isear":                   # source models scored on exactly the 650 items of the target-trained test split
                pos = np.asarray(DATA["isear_ft"]["test"]["pos_in_full"]); accs = []
                for _, r in g.iterrows():
                    rp = f"{EXPERIMENT_NAME}/{r.stage}/runs/{r.spec_id}"; lp = LOCAL_ROOT / rp / "outputs" / "logits.npz"
                    with np.load(lp if lp.exists() else hub_download(f"{rp}/outputs/logits.npz")) as z:
                        lg, y = z["transfer_isear_logits"][pos], z["transfer_isear_y"][pos]
                    rl = np.full_like(lg, -np.inf); rl[:, PRESENT_CLASSES["isear"]] = lg[:, PRESENT_CLASSES["isear"]]
                    accs.append(float(np.mean(rl.argmax(1) == y)))
                row["restricted_acc_same_items_as_target"] = float(np.mean(accs))
            else:
                row["restricted_acc_same_items_as_target"] = row["restricted_acc"]
            rows_t.append(row)
    if rows_t:
        tt = pd.DataFrame(rows_t)
        tt["unrestricted"] = [pm_pct(a, b) for a, b in zip(tt.unrestricted_acc, tt.unrestricted_sd)]
        tt["restricted"] = [pm_pct(a, b) for a, b in zip(tt.restricted_acc, tt.restricted_sd)]
        tt["prior_em"] = [pm_pct(a, b) for a, b in zip(tt.prior_em_acc, tt.prior_em_sd)]
        tex_table(tt, ["target", "encoder", "condition", "unrestricted", "restricted", "prior_em", "atc_pred",
                       "restricted_acc_same_items_as_target", "target_trained_acc"],
                  TEX / "transfer.tex", {"atc_pred": pct, "restricted_acc_same_items_as_target": pct, "target_trained_acc": pct})
        A["table_transfer"] = tt.drop(columns=["unrestricted", "restricted", "prior_em"]).to_dict("records")
    for stage_name, keycol in (("peft", "peft_method"), ("rebalance", "rebalance")):
        g = df[df.stage == stage_name]
        if len(g):
            per = g.groupby(["encoder", keycol]).agg(acc=("test_acc", "mean"), acc_sd=("test_acc", "std"), mf1=("test_macro_f1", "mean"),
                                                     wf1=("test_weighted_f1", "mean"), trainable=("trainable_parameters", "first"),
                                                     f1_love=("test_f1_per_class", lambda x: np.mean([v[2] for v in x])),
                                                     f1_surprise=("test_f1_per_class", lambda x: np.mean([v[5] for v in x])),
                                                     n=("seed", "nunique")).reset_index()
            per["acc_str"] = [pm_pct(a, b) for a, b in zip(per.acc, per.acc_sd)]
            tex_table(per, ["encoder", keycol, "trainable", "acc_str", "mf1", "wf1", "f1_surprise", "f1_love", "n"], TEX / f"{stage_name}.tex",
                      {"trainable": lambda v: f"{int(v):,}", "mf1": f4, "wf1": f4, "f1_surprise": f4, "f1_love": f4})
            A[f"table_{stage_name}"] = per.drop(columns=["acc_str"]).to_dict("records")
    nontrain_tables(A, D_LLM, D_AUDIT, TEX)
    # ---- same-seed rerun variation -----------------------------------------------------------------------------------
    rr = df[df.stage == "rerun"]
    rerun = {}
    if len(rr) and "repeat" in rr:
        for enc, g in rr.groupby("encoder"):
            if g["repeat"].nunique() == 2:
                a, b = (g[g["repeat"] == k].iloc[0] for k in sorted(g["repeat"].unique()))
                ca, cb = load_test_correct(a), load_test_correct(b)
                rerun[enc] = {"acc_1": float(a.test_acc), "acc_2": float(b.test_acc), "abs_diff_pp": float(100 * abs(a.test_acc - b.test_acc)),
                              "item_disagreement": float(np.mean(ca != cb)), **paired_item_stats(ca, cb)}
    A["rerun_variation"] = rerun
    A.update(cev_analyses(df))
    # ---- convergence ---------------------------------------------------------------------------------------------------
    conv = {}
    cv = df[df.stage == "convergence"]
    for enc, g in cv.groupby("encoder"):
        a, b = g[g.condition == "A"].converged, g[g.condition == "none"].converged
        if len(a) and len(b):
            conv[enc] = {"converged_A": [int(a.sum()), int(len(a))], "converged_none": [int(b.sum()), int(len(b))],
                         "p_fisher": float(sps.fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])[1])}
    A["convergence"] = conv
    return A

METRICS_DF = load_all_metrics()
ANALYSIS = aggregate(METRICS_DF)
for name, obj in ANALYSIS.items():
    write_json(ANALYSIS_DIR / "metrics" / f"{name}.json", obj)
if len(METRICS_DF):
    METRICS_DF.drop(columns=[c for c in METRICS_DF.columns if c.startswith("transfer_")]).to_csv(ANALYSIS_DIR / "metrics" / "all_runs.csv", index=False)
logger.info(f"Aggregation complete: {sorted(ANALYSIS)}")
