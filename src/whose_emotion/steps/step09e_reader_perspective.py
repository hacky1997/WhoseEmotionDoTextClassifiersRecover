# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 9e — READER PERSPECTIVE: crowd-enVENT AND enISEAR (v8, reader-perspective revision; analysis only, trains nothing)
# Quantities (paper, Section 3.2): among texts whose reader majority is unique and lies within the classes of the corpus,
# R = majority equals the writer's label and Rbar = majority differs from it. Per run, on the version the readers saw:
# accuracy against the writer's label on R and on Rbar, the share pi of its errors (over R and Rbar) that fall on Rbar,
# and on Rbar the shares of predictions equal to the reader majority and to the writer's label.
# P0 (exploratory) crowd-enVENT: SetFit-trained (cev_source), GoEmotions-trained (cev_third), crowd-enVENT-trained
#    (cev_target), GPT-4o (llm_transfer chunks), masked version, five shared classes. Complements X4 of Cell 9d.
# enISEAR inputs: eni_source and eni_third runs (eni_logits) and the GPT-4o task llm_transfer/enisear (Cell 8c).
# P1 (confirmatory, Amendment 2) enISEAR: classifier correctness increases with reader item accuracy
#    (GEE, binomial, exchangeable, clustered by item; one-sided p).
# P2 (confirmatory, Amendment 2) enISEAR: on Rbar, classifiers choose the reader majority more often than the writer's
#    label (difference of shares; joint bootstrap over Rbar items and runs; one-sided p).
# Outputs: analysis/metrics/perspective_crowd_envent.json, perspective_enisear.json, perspective_numbers.json (values for
# the paper's \NN keys) and analysis/tables/perspective.tex, supp_enisear.tex (complete tabulars). Every part is skipped
# with a log message when its inputs are not yet on the Hub.
# =====================================================================================
def p_bootstrap(P: np.ndarray, y: np.ndarray, maj: np.ndarray, rng) -> dict:
    """P: (runs, items) predictions. Joint bootstrap over runs and items, items resampled within R and within Rbar so
    that both set sizes stay fixed. With one run, only items are resampled."""
    ok = maj >= 0
    iR, iB = np.flatnonzero(ok & (maj == y)), np.flatnonzero(ok & (maj != y))
    if not len(iR) or not len(iB):
        return {}
    boots = []
    for _ in range(P_N_BOOT):
        rr = P[rng.integers(0, len(P), len(P))] if len(P) > 1 else P
        bR, bB = rng.choice(iR, len(iR)), rng.choice(iB, len(iB))
        eR, eB = rr[:, bR] != y[bR], rr[:, bB] != y[bB]
        boots.append([1 - eR.mean(), 1 - eB.mean(), eB.sum(1).mean() / max((eR.sum(1) + eB.sum(1)).mean(), 1e-12),
                      np.mean(rr[:, bB] == maj[bB]), np.mean(rr[:, bB] == y[bB])])
    b = np.asarray(boots)
    ci = lambda j: [float(np.percentile(b[:, j], 2.5)), float(np.percentile(b[:, j], 97.5))]
    diff = b[:, 3] - b[:, 4]                                       # P2 statistic: chose readers minus chose writer on Rbar
    return {"acc_R_ci95": ci(0), "acc_Rbar_ci95": ci(1), "pi_ci95": ci(2), "rbar_chose_readers_ci95": ci(3),
            "rbar_chose_writer_ci95": ci(4), "readers_minus_writer_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "p_readers_gt_writer_one_sided": float(max(1.0 / P_N_BOOT, np.mean(diff <= 0)))}

def p_family(rows: list, y: np.ndarray, maj: np.ndarray, rng) -> dict:
    """rows: list of (spec-like dict, predictions on the reader texts)."""
    if not rows:
        return {}
    P = np.stack([p for _, p in rows])
    per = pd.DataFrame([{"encoder": r.get("encoder"), "condition": r.get("condition"), "seed": r.get("seed"),
                         **{k: v for k, v in perspective_metrics(p, y, maj).items() if not isinstance(v, list)}} for r, p in rows])
    keys = ["acc_writer", "agree_reader_majority", "acc_R", "acc_Rbar", "pi_error_share_Rbar", "rbar_chose_readers", "rbar_chose_writer"]
    cell = per.groupby(["encoder", "condition"])[keys].mean()
    return {"n_runs": int(len(P)), "mean": per[keys].mean().to_dict(),
            "range_of_encoder_condition_means": {k: [float(cell[k].min()), float(cell[k].max())] for k in keys},
            "by_encoder_condition": cell.reset_index().to_dict("records"), **p_bootstrap(P, y, maj, rng)}

def p_gee(rows: list, y: np.ndarray, item_acc: np.ndarray, extra: dict = None) -> dict:
    """P1: correct ~ z(reader item accuracy) [+ extra item covariates], GEE clustered by item."""
    if not rows:
        return {}
    z_acc = (item_acc - item_acc.mean()) / (item_acc.std(ddof=1) + 1e-12)
    parts = [pd.DataFrame({"correct": (p == y).astype(int), "item": np.arange(len(y)), "z_reader_acc": z_acc,
                           **(extra or {})}) for _, p in rows]
    d = pd.concat(parts, ignore_index=True)
    formula = "correct ~ z_reader_acc" + "".join(f" + {k}" for k in (extra or {}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.gee(formula, groups="item", data=d, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    b, ci = float(fit.params["z_reader_acc"]), fit.conf_int().loc["z_reader_acc"].tolist()
    return {"log_or_per_sd": b, "ci95": [float(ci[0]), float(ci[1])], "p_two_sided": float(fit.pvalues["z_reader_acc"]),
            "p_one_sided": float(sps.norm.sf(fit.tvalues["z_reader_acc"])), "n_rows": int(len(d)), "n_runs": int(len(rows))}

def p_llm_rows(task: str):
    files = sorted(f for f in HUB_FILES if f.startswith(f"{LLMT_PREFIX}/{task}/") and f.endswith(".jsonl"))
    rows = []
    for f in files:
        lp = LOCAL_ROOT / f
        rows += [json.loads(l) for l in open(lp if lp.exists() else hub_download(f)) if l.strip()]
    rows.sort(key=lambda r: r["i"])
    return rows

def p_runs(df: pd.DataFrame, stage: str, key: str, select=None) -> list:
    out = []
    for _, r in df[df.stage == stage].iterrows():
        with robust_npz(r) as z:
            if key not in z.files:
                continue
            lg = z[key]
        out.append((r.to_dict(), lg if select is None else select(lg)))
    return out

def perspective_crowd_envent(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(P_SEED)
    rp, yr, mj = CEV["reader_pos"], CEV["label"][CEV["reader_pos"]], CEV["reader_majority"]
    cls = PRESENT_CLASSES["cev"]
    fam = {"setfit_trained": [(r, restrict_to(lg, cls)[rp]) for r, lg in p_runs(df, "cev_source", "cev_masked_logits")],
           "goemotions_trained": [(r, restrict_to(lg, cls)[rp]) for r, lg in p_runs(df, "cev_third", "cev_masked_logits")],
           "crowd_envent_trained": [(r, restrict_to(lg, cls)) for r, lg in p_runs(df, "cev_target", "cevtest_masked_logits")]}
    lo, lm = p_llm_rows("cev_orig"), p_llm_rows("cev_masked")
    if len(lo) == len(CEV["label"]) and len(lm) == int(CEV["changed"].sum()):
        assert np.array_equal([r["label"] for r in lo], CEV["label"]), "llm_transfer cev_orig rows misaligned"
        assert np.array_equal([r["label"] for r in lm], CEV["label"][CEV["changed"]]), "llm_transfer cev_masked rows misaligned"
        p = np.array([r["pred"] for r in lo]); p[np.flatnonzero(CEV["changed"])] = [r["pred"] for r in lm]
        fam["gpt4o"] = [({"encoder": "gpt-4o-2024-08-06", "condition": "zero-shot", "seed": 0}, p[rp])]
    else:
        logger.info("P0: GPT-4o crowd-enVENT chunks (llm_transfer) not found; GPT-4o row skipped")
    ok = mj >= 0
    out = {"n_reader_texts": int(len(rp)), "n_unique_majority": int(ok.sum()), "n_R": int((ok & (mj == yr)).sum()),
           "n_Rbar": int((ok & (mj != yr)).sum()), "share_Rbar": float((ok & (mj != yr)).sum() / ok.sum()),
           "reader_majority_recovers_writer": float((ok & (mj == yr)).sum() / ok.sum()),
           "families": {k: p_family(v, yr, mj, rng) for k, v in fam.items() if v},
           "_note": "exploratory (P0); masked version, the version the readers saw; five shared classes"}
    if fam["setfit_trained"]:                              # check against the analysis already in the paper (masked text, 77 texts)
        s = out["families"]["setfit_trained"]["mean"]
        logger.info(f"P0 check: SetFit-trained masked accuracy {100 * s['acc_writer']:.2f}% (paper 46.44%), on Rbar chose readers "
                    f"{100 * s['rbar_chose_readers']:.1f}% and writer {100 * s['rbar_chose_writer']:.1f}% (paper 48.2% and 22.7%)")
    return out

def perspective_enisear(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(P_SEED + 1)
    y, mj, ia = ENI["label"], ENI["reader_majority"], ENI["reader_item_acc"]
    cls = PRESENT_CLASSES["eni"]
    ok = mj >= 0
    out = {"n_texts": int(len(y)), "n_unicode_marker_normalized": int(ENI["n_unicode_marker"]),
           "n_tied_majority": int((mj == -1).sum()), "n_majority_outside_four": int((mj == -2).sum()),
           "n_unique_majority": int(ok.sum()), "n_R": int((ok & (mj == y)).sum()), "n_Rbar": int((ok & (mj != y)).sum()),
           "share_Rbar": float((ok & (mj != y)).sum() / ok.sum()),
           "reader_majority_recovers_writer": float((ok & (mj == y)).sum() / ok.sum()),
           "individual_reader_acc_7way": float(ia.mean()), "families": {}, "P1": {}, "P2": {}}
    # in-set reader accuracy: votes for the writer's label over all votes that fall within the four shared classes
    col = {e: j for j, e in enumerate(ENI_EMOTIONS)}
    writer_votes = ENI["reader_votes"][np.arange(len(y)), [col[SETFIT[c].capitalize()] for c in y]]
    out["individual_reader_acc_inset"] = float(writer_votes.sum() / ENI["reader_votes"][:, [col[SETFIT[c].capitalize()] for c in cls]].sum())
    fam = {"setfit_trained": [(r, restrict_to(lg, cls)) for r, lg in p_runs(df, "eni_source", "eni_logits")],
           "goemotions_trained": [(r, restrict_to(lg, cls)) for r, lg in p_runs(df, "eni_third", "eni_logits")]}
    rows = p_llm_rows("enisear")                       # GPT-4o, zero-shot, Cell 8c (task "enisear")
    if len(rows) == len(y) and [q["i"] for q in rows] == list(range(len(y))):
        assert np.array_equal([q["label"] for q in rows], y), "llm_transfer enisear rows misaligned with enISEAR labels"
        fam["gpt4o"] = [({"encoder": LLMT_MODEL, "condition": "zero-shot", "seed": 0}, np.array([q["pred"] for q in rows]))]
        out["gpt4o_format_failure_rate"] = float(np.mean(fam["gpt4o"][0][1] == -1))
        out["gpt4o_models_reported"] = sorted({q["model_reported"] for q in rows})
    fam = {k: v for k, v in fam.items() if v}
    if not fam:
        logger.info("enISEAR: no eni_* runs evaluated yet; descriptive reader statistics only")
        return out
    for k, rows in fam.items():
        out["families"][k] = p_family(rows, y, mj, rng)
        out["P1"][k] = p_gee(rows, y, ia)
        f = out["families"][k]
        out["P2"][k] = {"n_Rbar": out["n_Rbar"], "chose_readers": f["mean"]["rbar_chose_readers"], "chose_writer": f["mean"]["rbar_chose_writer"],
                        "difference_ci95": f["readers_minus_writer_ci95"], "p_one_sided": f["p_readers_gt_writer_one_sided"]}
        out["families"][k]["sadness_prediction_share"] = float(np.mean(np.stack([p for _, p in rows]) == 0))
    out["true_sadness_share"] = float(np.mean(y == 0))
    out["_note"] = ("P1 and P2 are confirmatory (Amendment 2; directions: P1 positive, P2 readers > writer). The release contains "
                    "only the versions the readers saw; '…' was normalized to '...'. GoEmotions-trained models are known to "
                    "treat the three-dot marker as a sadness cue on crowd-enVENT: compare sadness_prediction_share with true_sadness_share.")
    return out

def p_fmt(v, nd=1, pct=True):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{(100 * v if pct else v):.{nd}f}"

def perspective_tables(pc: dict, pe: dict, TEX: Path) -> None:
    labels = [("setfit_trained", "SetFit-trained"), ("goemotions_trained", "GoEmotions-trained"),
              ("crowd_envent_trained", "crowd-enVENT-trained"), ("gpt4o", "GPT-4o, zero-shot")]
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{crowd-enVENT} & \multicolumn{3}{c}{enISEAR} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Classifier & $\mathcal{R}$ & $\bar{\mathcal{R}}$ & $\pi$ & $\mathcal{R}$ & $\bar{\mathcal{R}}$ & $\pi$ \\", r"\midrule"]
    for key, name in labels:
        cells = []
        for res in (pc, pe):
            m = res.get("families", {}).get(key, {}).get("mean")
            cells += [p_fmt(m["acc_R"]), p_fmt(m["acc_Rbar"]), p_fmt(m["pi_error_share_Rbar"])] if m else ["--", "--", "--"]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule", r"Share of $\bar{\mathcal{R}}$ texts & & & " + p_fmt(pc.get("share_Rbar")) + " & & & " + p_fmt(pe.get("share_Rbar")) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    (TEX / "perspective.tex").write_text("\n".join(lines) + "\n")
    rows = [r"\begin{tabular}{lcc}", r"\toprule", r"Classifier or reader & Writer & Majority \\", r"\midrule"]
    st = pe.get("families", {}).get("setfit_trained")
    names = {"albert": "ALBERT", "distilbert": "DistilBERT", "distilroberta": "DistilRoBERTa", "electra": "ELECTRA", "roberta": "RoBERTa"}
    rng_ = lambda vals: f"{p_fmt(min(vals))}--{p_fmt(max(vals))}"
    if st:
        cell = pd.DataFrame(st["by_encoder_condition"])
        for enc, g in cell.groupby("encoder"):
            rows.append(f"{names.get(enc, enc)} & {rng_(g.acc_writer)} & {rng_(g.agree_reader_majority)}" + r" \\")
    gt = pe.get("families", {}).get("goemotions_trained")
    if gt:
        cell = pd.DataFrame(gt["by_encoder_condition"])
        rows.append(f"GoEmotions-trained & {rng_(cell.acc_writer)} & {rng_(cell.agree_reader_majority)}" + r" \\")
    g4 = pe.get("families", {}).get("gpt4o")
    if g4:
        rows.append(f"GPT-4o, zero-shot & {p_fmt(g4['mean']['acc_writer'])} & {p_fmt(g4['mean']['agree_reader_majority'])}" + r" \\")
    rows += [f"Individual reader, seven classes & {p_fmt(pe['individual_reader_acc_7way'])} & n/a" + r" \\",
             f"Individual reader, four classes$^{{a}}$ & {p_fmt(pe['individual_reader_acc_inset'])} & n/a" + r" \\",
             f"Reader majority & {p_fmt(pe['reader_majority_recovers_writer'])} & n/a" + r" \\", r"\bottomrule", r"\end{tabular}"]
    (TEX / "supp_enisear.tex").write_text("\n".join(rows) + "\n")

def perspective_numbers(pc: dict, pe: dict) -> dict:
    """Values for the \\NN{key}{value} placeholders of the paper; add them to the numbers script under the same keys."""
    N, ci = {}, lambda v, nd=1, pct=True: f"[{p_fmt(v[0], nd, pct)}, {p_fmt(v[1], nd, pct)}]"
    N["e3-majrec"], N["e3-share-rbar"] = p_fmt(pc["reader_majority_recovers_writer"]), p_fmt(pc["share_Rbar"])
    s = pc["families"].get("setfit_trained")
    if s:
        N.update({"p0-cev-src-r": p_fmt(s["mean"]["acc_R"]), "p0-cev-src-r-ci": ci(s["acc_R_ci95"]),
                  "p0-cev-src-rbar": p_fmt(s["mean"]["acc_Rbar"]), "p0-cev-src-rbar-ci": ci(s["acc_Rbar_ci95"]),
                  "p0-cev-src-pi": p_fmt(s["mean"]["pi_error_share_Rbar"]), "p0-cev-src-pi-ci": ci(s["pi_ci95"])})
    for k, tag in (("goemotions_trained", "goe"), ("crowd_envent_trained", "cev"), ("gpt4o", "llm")):
        f = pc["families"].get(k)
        if f:
            N[f"p0-cev-{tag}-pi"], N[f"p0-cev-{tag}-pi-ci"] = p_fmt(f["mean"]["pi_error_share_Rbar"]), ci(f["pi_ci95"])
    N.update({"size-enisear-all": "1{,}001", "size-enisear-per-class": "143", "size-enisear-four": str(pe["n_texts"]),
              "size-enisear-ties": str(pe["n_tied_majority"]), "size-enisear-outside": str(pe["n_majority_outside_four"]),
              "enisear-nmaj": str(pe["n_unique_majority"]), "enisear-majrec": p_fmt(pe["reader_majority_recovers_writer"]),
              "enisear-share-rbar": p_fmt(pe["share_Rbar"]), "p2-n": str(pe["n_Rbar"]),
              "enisear-hind": p_fmt(pe["individual_reader_acc_7way"]), "enisear-hinset": p_fmt(pe["individual_reader_acc_inset"]),
              "enisear-marker": str(pe["n_unicode_marker_normalized"])})
    gs = pe.get("families", {}).get("goemotions_trained")
    if gs:                                                  # three-dot marker check, as for crowd-enVENT (true share: 25%)
        N["x1-enisear-goe-sad"] = p_fmt(gs["sadness_prediction_share"])
    s = pe.get("families", {}).get("setfit_trained")
    if s:
        r = s["range_of_encoder_condition_means"]["acc_writer"]
        N.update({"enisear-src-min": p_fmt(r[0]), "enisear-src-max": p_fmt(r[1]), "p0-enisear-src-pi": p_fmt(s["mean"]["pi_error_share_Rbar"]),
                  "p1-gee": p_fmt(pe["P1"]["setfit_trained"]["log_or_per_sd"], 2, False), "p1-gee-ci": ci(pe["P1"]["setfit_trained"]["ci95"], 2, False),
                  "p1-gee-p": f"{pe['P1']['setfit_trained']['p_one_sided']:.2g}",
                  "p2-r": p_fmt(pe["P2"]["setfit_trained"]["chose_readers"]), "p2-w": p_fmt(pe["P2"]["setfit_trained"]["chose_writer"]),
                  "p2-p": f"{pe['P2']['setfit_trained']['p_one_sided']:.2g}"})
    g = pe.get("families", {}).get("gpt4o")
    if g:
        N.update({"enisear-gpt4o": p_fmt(g["mean"]["acc_writer"]), "p1-gee-gpt4o": p_fmt(pe["P1"]["gpt4o"]["log_or_per_sd"], 2, False),
                  "p1-gee-gpt4o-ci": ci(pe["P1"]["gpt4o"]["ci95"], 2, False),
                  "p2-gpt4o-r": p_fmt(pe["P2"]["gpt4o"]["chose_readers"]), "p2-gpt4o-w": p_fmt(pe["P2"]["gpt4o"]["chose_writer"])})
    return N

PERSPECTIVE = {}
_pdf = robust_train_df()
if len(_pdf):
    PERSPECTIVE["perspective_crowd_envent"] = perspective_crowd_envent(_pdf)
    PERSPECTIVE["perspective_enisear"] = perspective_enisear(_pdf)
    PERSPECTIVE["perspective_numbers"] = perspective_numbers(PERSPECTIVE["perspective_crowd_envent"], PERSPECTIVE["perspective_enisear"])
    (ANALYSIS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    perspective_tables(PERSPECTIVE["perspective_crowd_envent"], PERSPECTIVE["perspective_enisear"], ANALYSIS_DIR / "tables")
    for name, obj in PERSPECTIVE.items():
        write_json(ANALYSIS_DIR / "metrics" / f"{name}.json", obj)
    for k, f in PERSPECTIVE["perspective_crowd_envent"]["families"].items():
        logger.info(f"P0 crowd-enVENT {k}: acc R {100 * f['mean']['acc_R']:.1f}%, Rbar {100 * f['mean']['acc_Rbar']:.1f}%, "
                    f"pi {100 * f['mean']['pi_error_share_Rbar']:.1f}% {[round(100 * x, 1) for x in f['pi_ci95']]} "
                    f"(share of Rbar texts {100 * PERSPECTIVE['perspective_crowd_envent']['share_Rbar']:.1f}%)")
    for k, v in PERSPECTIVE["perspective_enisear"].get("P2", {}).items():
        p1 = PERSPECTIVE["perspective_enisear"]["P1"][k]
        logger.info(f"enISEAR {k}: P1 log OR {p1['log_or_per_sd']:+.2f} {[round(x, 2) for x in p1['ci95']]} (one-sided p {p1['p_one_sided']:.2g}); "
                    f"P2 on {v['n_Rbar']} texts readers {100 * v['chose_readers']:.1f}% vs writer {100 * v['chose_writer']:.1f}% "
                    f"(one-sided p {v['p_one_sided']:.2g})")
logger.info(f"Reader-perspective analyses complete: {sorted(PERSPECTIVE)}")
