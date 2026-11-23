# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 9d — NEW CONTROLS (analysis only). X1 third-corpus E1 | X2 second source | X3 zero-shot LLM | X4 E3 on reader-version texts
# =====================================================================================
X_N_BOOT = 2000
X_SEED = 20261008

def x_runs(df, stage, **eq):
    d = df[df.stage == stage]
    for k, v in eq.items():
        d = d[d[k] == v]
    return d

def x_cev_reader():
    rp = np.asarray(CEV["reader_pos"]); ch = np.asarray(CEV["changed"]); y = np.asarray(CEV["label"])
    return rp, ch, y, ch[rp], y[rp], np.asarray(CEV["reader_majority"])

def x_drops(rows, kind: str) -> pd.DataFrame:
    rp, ch, y, w, yr, _ = x_cev_reader()
    out = []
    for _, r in rows.iterrows():
        with robust_npz(r) as z:
            if kind == "target":
                po, pm = cev_restrict(z["test_logits"]), cev_restrict(z["cevtest_masked_logits"])
                assert len(po) == len(rp), "crowd-enVENT test set is not the reader set"
                d163, d863 = 100 * (np.mean(po[w] == yr[w]) - np.mean(pm[w] == yr[w])), np.nan
            else:
                po, pm = cev_restrict(z["cev_orig_logits"]), cev_restrict(z["cev_masked_logits"])
                d863 = 100 * (np.mean(po[ch] == y[ch]) - np.mean(pm[ch] == y[ch]))
                d163 = 100 * (np.mean(po[rp][w] == yr[w]) - np.mean(pm[rp][w] == yr[w]))
        out.append({"encoder": r.encoder, "condition": r.condition, "seed": int(r.seed), "drop863": d863, "drop163": d163})
    return pd.DataFrame(out)

def x_pooled_diff(a, b, col) -> dict:
    ys, vs, per = [], [], {}
    for e in sorted(set(a.encoder) & set(b.encoder)):
        u, v = a[a.encoder == e][col].dropna(), b[b.encoder == e][col].dropna()
        if len(u) < 2 or len(v) < 2:
            continue
        ys.append(u.mean() - v.mean()); vs.append(u.var(ddof=1) / len(u) + v.var(ddof=1) / len(v)); per[e] = float(ys[-1])
    return {**random_effects(ys, vs), "per_encoder": per} if len(ys) >= 2 else {}

def x_e1_third(df) -> dict:
    third = x_runs(df, "cev_third")
    if not len(third):
        return {}
    D3 = x_drops(third, "source")
    DS = x_drops(x_runs(df, "cev_source"), "source"); DSn = DS[DS.condition == "none"]
    DT = x_drops(x_runs(df, "cev_target"), "target")
    pool = lambda D, c: {**random_effects([D[D.encoder == e][c].mean() for e in sorted(D.encoder.unique())],
                                          [D[D.encoder == e][c].var(ddof=1) / max(1, (D.encoder == e).sum()) for e in sorted(D.encoder.unique())])}
    tgt = x_runs(df, "target", corpus="goemotions_target", condition="none")
    m = third.merge(tgt, on=["encoder", "seed"], suffixes=("_x", "_t"))
    return {"n_runs": {"goemotions_trained": len(D3), "setfit_trained_none": len(DSn), "cev_trained": len(DT)},
            "drop863_pooled": {"goemotions_trained": pool(D3, "drop863"), "setfit_trained_none": pool(DSn, "drop863")},
            "drop163_pooled": {"goemotions_trained": pool(D3, "drop163"), "setfit_trained_none": pool(DSn, "drop163"),
                               "cev_trained": pool(DT, "drop163")},
            "diff_setfit_minus_goemotions_863": x_pooled_diff(DSn, D3, "drop863"),
            "diff_setfit_minus_goemotions_163": x_pooled_diff(DSn, D3, "drop163"),
            "diff_goemotions_minus_cev_163": x_pooled_diff(D3, DT, "drop163"),
            "diff_setfit_minus_cev_163_none_only": x_pooled_diff(DSn, DT, "drop163"),
            "goemotions_test_acc_by_encoder": third.groupby("encoder").test_acc.mean().to_dict(),
            "rerun_check_vs_target_stage": {"pairs": int(len(m)), "equal_test_acc": int((m.test_acc_x == m.test_acc_t).sum())},
            "_note": "Shortcut account predicts SetFit drop > GoEmotions drop. GoEmotions ~ crowd-enVENT drop => E1 reflects the target's own cue. Condition none."}

def x_correct_matrix(rows, key, allowed, pos=None):
    C = []
    for _, r in rows.iterrows():
        with robust_npz(r) as z:
            lg, yy = z[key], z[key.replace("_logits", "_y")]
        if pos is not None:
            lg, yy = lg[pos], np.asarray(yy)[pos]
        C.append(robust_restricted_correct(lg, np.asarray(yy), allowed))
    return np.array(C)

def x_gap_did(CS, CT, ex, rng) -> dict:
    def gap(C, idx):
        Ci = C[:, idx]; e = ex[idx]
        return 100 * (Ci[:, e].mean() - Ci[:, ~e].mean())
    n = CS.shape[1]; all_i = np.arange(n)
    gS, gT = gap(CS, all_i), gap(CT, all_i)
    boots = []
    for _ in range(X_N_BOOT):
        i = rng.integers(0, n, n)
        if ex[i].all() or (~ex[i]).all():
            continue
        s = CS[rng.integers(0, len(CS), len(CS))]; t = CT[rng.integers(0, len(CT), len(CT))]
        boots.append(gap(s, i) - gap(t, i))
    boots = np.array(boots)
    p = float(min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))))
    return {"gap_source_pp": float(gS), "gap_target_pp": float(gT), "did_pp": float(gS - gT),
            "did_ci95_pp": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "did_ci90_pp": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))], "p_two_sided_bootstrap": p,
            "source_acc_pct": float(100 * CS.mean()), "target_acc_pct": float(100 * CT.mean())}

def x_second_source(df) -> dict:
    third = x_runs(df, "cev_third")
    if not len(third) or "transfer_meld" not in third or third["transfer_meld"].isna().all():
        return {}
    rng = np.random.default_rng(X_SEED)
    isear_pos = np.asarray(DATA["isear_ft"]["test"]["pos_in_full"])
    spec = [("meld", "meld_target", EXPLICIT["meld"], None), ("isear", "isear_target", EXPLICIT["isear_target_test"], isear_pos),
            ("ed", "ed", EXPLICIT["ed_test"], None)]
    out, pv = {}, {}
    for tgt, tcorp, ex, pos in spec:
        tr = df[(df.corpus == tcorp) & (df.condition == "none") & df.stage.isin(["target", "intra"])]
        if not len(tr):
            continue
        CS = x_correct_matrix(third, f"transfer_{tgt}_logits", PRESENT_CLASSES[tgt], pos)
        CT = x_correct_matrix(tr, "test_logits", PRESENT_CLASSES[tgt])
        assert CS.shape[1] == CT.shape[1] == len(ex), f"{tgt}: item sets differ"
        out[tgt] = x_gap_did(CS, CT, np.asarray(ex), rng); pv[tgt] = out[tgt]["p_two_sided_bootstrap"]
        out[tgt].update(n_items=int(len(ex)), n_source_runs=int(len(CS)), n_target_runs=int(len(CT)))
    for t, ph in holm(pv).items():
        out[t]["p_holm"] = ph
    out["_note"] = "Source = GoEmotions-trained (cev_third, none). Restricted rule. ISEAR on the 650-item test split."
    return out

def x_llm_load(task):
    rows, ci = [], 0
    while True:
        rel = f"{EXPERIMENT_NAME}/llm_transfer/{task}/chunk_{ci:03d}.jsonl"
        lp = LOCAL_ROOT / rel
        if not lp.exists():
            if rel not in HUB_FILES:
                break
            lp = hub_download(rel)
        rows += [json.loads(l) for l in open(lp) if l.strip()]
        ci += 1
    return pd.DataFrame(rows)

def x_boot_gap(correct, ex, rng):
    g = lambda c, e: 100 * (c[e].mean() - c[~e].mean())
    b = []
    for _ in range(X_N_BOOT):
        i = rng.integers(0, len(correct), len(correct))
        if ex[i].all() or (~ex[i]).all():
            continue
        b.append(g(correct[i], ex[i]))
    return {"gap_pp": float(g(correct, ex)), "ci95_pp": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

def x_e3_gee(correct_runs: list, extra: dict) -> dict:
    rp, ch, y, w, yr, mj = x_cev_reader()
    racc = np.asarray(CEV["reader_item_acc"]); cons = np.asarray(CEV["reader_consensus"])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    dd = pd.concat([pd.DataFrame({"correct": c.astype(int), "item": np.arange(len(rp)), "z_racc": z(racc), "z_cons": z(cons),
                                  "changed": w.astype(int)}) for c in correct_runs], ignore_index=True)
    res = {}
    for name, f in [("reader_acc", "correct ~ z_racc + changed"), ("reader_acc_and_consensus", "correct ~ z_racc + z_cons + changed")]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.gee(f, groups="item", data=dd, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        res[name] = {k: {"log_or": float(fit.params[k]), "ci95": fit.conf_int().loc[k].tolist(), "p": float(fit.pvalues[k])}
                     for k in fit.params.index if k != "Intercept"}
    return {**res, **extra}

def x_mismatch_agreement(preds_runs: list) -> dict:
    rp, ch, y, w, yr, mj = x_cev_reader()
    mm = (mj >= 0) & (mj != yr)
    if not mm.any():
        return {}
    aw = [float(np.mean(p[mm] == yr[mm])) for p in preds_runs]; am = [float(np.mean(p[mm] == mj[mm])) for p in preds_runs]
    return {"n_texts": int(mm.sum()), "agree_writer_mean": float(np.mean(aw)), "agree_reader_majority_mean": float(np.mean(am)),
            "agree_writer_range": [min(aw), max(aw)], "agree_reader_majority_range": [min(am), max(am)]}

def x_e3_reader_version(df) -> dict:
    src = x_runs(df, "cev_source")
    if not len(src) or "reader_consensus" not in CEV:
        return {}
    rp, ch, y, w, yr, mj = x_cev_reader()
    orig, masked = [], []
    for _, r in src.iterrows():
        with robust_npz(r) as z:
            orig.append(cev_restrict(z["cev_orig_logits"])[rp]); masked.append(cev_restrict(z["cev_masked_logits"])[rp])
    return {"n_runs": len(src),
            "original_text": x_e3_gee([p == yr for p in orig], {"acc_pct": float(100 * np.mean([np.mean(p == yr) for p in orig]))}),
            "masked_text_reader_version": x_e3_gee([p == yr for p in masked], {"acc_pct": float(100 * np.mean([np.mean(p == yr) for p in masked]))}),
            "perspective_mismatch_masked_text": x_mismatch_agreement(masked),
            "corr_reader_acc_consensus": float(np.corrcoef(CEV["reader_item_acc"], CEV["reader_consensus"])[0, 1]),
            "_note": "Reader acc = share of 5 readers choosing writer's label; consensus = share choosing modal label (13 classes). Exploratory."}

def x_llm_transfer(df) -> dict:
    if f"{EXPERIMENT_NAME}/llm_transfer/manifest.json" not in HUB_FILES and not (LOCAL_ROOT / f"{EXPERIMENT_NAME}/llm_transfer/manifest.json").exists():
        return {}
    rng = np.random.default_rng(X_SEED + 1)
    exkey = {"goemotions": "goemotions", "meld": "meld", "isear": "isear_target_test", "ed": "ed_test"}
    out = {}
    for tgt in ["goemotions", "meld", "isear", "ed"]:
        d = x_llm_load(tgt).sort_values("i")
        ex = np.asarray(EXPLICIT[exkey[tgt]]); assert len(d) == len(ex), f"llm_transfer {tgt}: {len(d)} rows, expected {len(ex)}"
        c = (d.pred.values == d.label.values)
        out[tgt] = {"n": int(len(d)), "acc_pct": float(100 * c.mean()), "format_failure_pct": float(100 * (d.pred.values < 0).mean()),
                    "implicit_gap": x_boot_gap(c, ex, rng), "models_reported": sorted(d.model_reported.unique().tolist())}
    o = x_llm_load("cev_orig").sort_values("i"); m = x_llm_load("cev_masked").sort_values("i")
    rp, ch, y, w, yr, mj = x_cev_reader()
    po = o.pred.values.copy(); pm = po.copy(); pm[np.flatnonzero(ch)] = m.pred.values
    assert len(po) == len(y) and len(m) == int(ch.sum())
    drop = lambda idx: 100 * (np.mean(po[idx] == y[idx]) - np.mean(pm[idx] == y[idx]))
    ci_idx = np.flatnonzero(ch); b = [drop(rng.choice(ci_idx, len(ci_idx))) for _ in range(X_N_BOOT)]
    out["crowd_envent"] = {"acc_orig_all_pct": float(100 * np.mean(po == y)), "drop863_pp": float(drop(ci_idx)),
                           "drop863_ci95_pp": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                           "drop163_pp": float(drop(rp[w])),
                           "reader_set_acc_orig_pct": float(100 * np.mean(po[rp] == yr)),
                           "reader_set_acc_masked_pct": float(100 * np.mean(pm[rp] == yr)),
                           "reader_set_agree_majority_masked_pct": float(100 * np.mean(pm[rp][mj >= 0] == mj[mj >= 0])),
                           "e3_masked_text": x_e3_gee([pm[rp] == yr], {}), "perspective_mismatch": x_mismatch_agreement([pm[rp]])}
    out["_note"] = "Zero-shot gpt-4o-2024-08-06, temp 0, target-restricted label list, strict parser (failures = errors). ISEAR 650 items."
    return out

X_RESULTS = {}
_xdf = robust_train_df()
for name, fn in [("x_e1_third", x_e1_third), ("x_second_source", x_second_source),
                 ("x_e3_reader_version", x_e3_reader_version), ("x_llm_transfer", x_llm_transfer)]:
    # v8: x5_setfit_vs_goemotions_gap, x6_reader_baselines, x7_prefix_transfer, x8_explicit_validation and x9_relative_drops
    # were listed here without definitions (NameError) and were never run; no number in the paper comes from them.
    try:
        res = fn(_xdf)
    except Exception as e:
        logger.error(f"{name} failed: {type(e).__name__}: {e}")
        raise
    if res:
        X_RESULTS[name] = res
        write_json(ANALYSIS_DIR / "metrics" / f"{name}.json", res)
        logger.info(f"{name}: written")
    else:
        logger.info(f"{name}: inputs not available yet, skipped")
if "x_e1_third" in X_RESULTS:
    r = X_RESULTS["x_e1_third"]
    for k in ["diff_setfit_minus_goemotions_163", "diff_goemotions_minus_cev_163"]:
        if r.get(k):
            logger.info(f"X1 {k}: {r[k]['mu']:+.2f} pp [{r[k]['ci95'][0]:+.2f}, {r[k]['ci95'][1]:+.2f}]")
    logger.info(f"X1 rerun check vs target stage: {r['rerun_check_vs_target_stage']}")
