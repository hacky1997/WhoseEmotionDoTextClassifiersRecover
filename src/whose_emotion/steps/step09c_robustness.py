# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 9c — ROBUSTNESS RECOMPUTATIONS (analysis only: reads stored logits and metrics, trains nothing)
# R1  H4 with a size-calibrated permutation null. The original H4 statistic, mean(|effect on implicit items| -
#     |effect on explicit items|), is biased upward when the implicit stratum is smaller (sampling noise inflates
#     absolute effects on fewer items). The null here permutes the explicit/implicit labels across items while keeping
#     both stratum sizes, applying the same permutation to every encoder x seed pair (items are shared across runs).
# R2  H2 control with uncertainty: difference in differences (source-trained gap minus target-trained gap) with a joint
#     bootstrap over test items and runs. Both model families are scored with the restricted rule on identical items.
# R3  H3 restricted to main-stage runs, i.e. without the masked-training runs of the shortcut stage, which inject
#     cue-reliance variation by design.
# Results: analysis/metrics/robustness_R1_h4.json, robustness_R2_h2_control.json, robustness_R3_h3_main_only.json
# (uploaded to the Hub by Cell 11 together with the rest of the analysis folder).
# =====================================================================================
ROBUST_N_PERM = 5000
ROBUST_N_BOOT = 2000
ROBUST_SEED = 20261004

def robust_train_df() -> pd.DataFrame:
    df = METRICS_DF.copy()
    df["kind"] = df["kind"].fillna("train") if "kind" in df else "train"
    df = df[df.kind == "train"].copy()
    for k in ["text_transform", "fewshot_k", "peft_method", "rebalance"]:
        if k not in df:
            df[k] = None
        df[k] = df[k].astype(object).where(df[k].notna(), "none")
    return df

def robust_npz(r):
    p = f"{EXPERIMENT_NAME}/{r['stage']}/runs/{r['spec_id']}"
    lp = LOCAL_ROOT / p / "outputs" / "logits.npz"
    return np.load(lp if lp.exists() else hub_download(f"{p}/outputs/logits.npz"))

def robust_restricted_correct(lg: np.ndarray, y: np.ndarray, allowed: list) -> np.ndarray:
    r = np.full_like(lg, -np.inf)
    r[:, allowed] = lg[:, allowed]
    return r.argmax(1) == y

def robust_r1_h4(df: pd.DataFrame) -> dict:
    out, pv = {}, {}
    rng = np.random.default_rng(ROBUST_SEED)
    keys = ["stage", "corpus", "encoder", "lr", "epochs", "placement", "head", "seed"]
    pool = df[df.stage.isin(["main", "intra"]) & (df.text_transform == "none") & df.condition.isin(["none", "A"])]
    for corpus, g in pool.groupby("corpus"):
        sk = STRATA_KEY.get("setfit" if str(corpus).startswith("setfit") else corpus)
        if sk is None:
            continue
        ex = np.asarray(EXPLICIT[sk], bool)
        a, b = g[g.condition == "A"].set_index(keys), g[g.condition == "none"].set_index(keys)
        common = a.index.intersection(b.index)
        if len(common) < 3:
            logger.info(f"R1 {corpus}: {len(common)} prompt/no-prompt pairs; at least 3 needed")
            continue
        D = np.stack([load_test_correct({"stage": kk[0], "spec_id": a.loc[kk, "spec_id"]}).astype(float)
                      - load_test_correct({"stage": kk[0], "spec_id": b.loc[kk, "spec_id"]}).astype(float) for kk in common])
        assert D.shape[1] == len(ex), f"R1 {corpus}: {D.shape[1]} test items but {len(ex)} explicitness flags"
        n, k = len(ex), int(ex.sum())
        assert 0 < k < n, f"R1 {corpus}: one stratum is empty"
        def stat(masks: np.ndarray) -> np.ndarray:            # masks: (P, n) explicit flags -> (P,) statistic
            M = masks.astype(float)
            imp, exp_ = D @ (1.0 - M).T / (n - k), D @ M.T / k
            return (np.abs(imp) - np.abs(exp_)).mean(0)
        obs = float(stat(ex[None, :])[0])
        null = np.concatenate([stat(np.stack([rng.permutation(ex) for _ in range(min(1000, ROBUST_N_PERM - s))]))
                               for s in range(0, ROBUST_N_PERM, 1000)])
        assert len(null) == ROBUST_N_PERM
        p = float((1 + np.sum(null >= obs - 1e-12)) / (ROBUST_N_PERM + 1))
        out[corpus] = {"n_pairs": int(len(common)), "n_implicit_items": int(n - k), "n_explicit_items": k,
                       "observed_pp": 100 * obs, "null_mean_pp": 100 * float(null.mean()),
                       "null_95pct_pp": [100 * float(np.percentile(null, 2.5)), 100 * float(np.percentile(null, 97.5))],
                       "excess_over_null_pp": 100 * (obs - float(null.mean())), "p_perm_size_calibrated": p}
        pv[corpus] = p
    for c, ph in holm(pv).items():
        out[c]["p_holm"] = ph
    out["_note"] = ("statistic = mean over encoder x seed pairs of |prompt effect on implicit items| - |on explicit items| (pp); "
                    "null = explicit/implicit labels permuted across items with stratum sizes fixed; one-sided")
    return out

def robust_r2_h2_control(df: pd.DataFrame) -> dict:
    out, pv = {}, {}
    rng = np.random.default_rng(ROBUST_SEED + 1)
    targets = [("goemotions", "goemotions_target", "goemotions", None),
               ("meld", "meld_target", "meld", None),
               ("isear", "isear_target", "isear_target_test", np.asarray(DATA["isear_ft"]["test"]["pos_in_full"])),
               ("ed", "ed", "ed_test", None)]
    src_all = df[(df.stage == "main") & df.condition.isin(["none", "A"])]
    for tgt, tcorp, exkey, pos in targets:
        col = f"transfer_{tgt}"
        tr_runs = df[(df.corpus == tcorp) & (df.condition == "none") & df.stage.isin(["target", "intra"])]
        if col not in src_all or src_all[col].isna().all() or not len(tr_runs):
            continue
        src = src_all[src_all[col].notna()]
        allowed = PRESENT_CLASSES[tgt]
        S, T, y_src, y_tgt = [], [], None, None
        for _, r in src.iterrows():
            with robust_npz(r) as z:
                lg, y = z[f"transfer_{tgt}_logits"], z[f"transfer_{tgt}_y"]
            if pos is not None:
                lg, y = lg[pos], y[pos]
            y_src = y if y_src is None else y_src
            assert np.array_equal(y, y_src), f"R2 {tgt}: source runs disagree on test labels"
            S.append(robust_restricted_correct(lg, y, allowed))
        for _, r in tr_runs.iterrows():
            with robust_npz(r) as z:
                lg, y = z["test_logits"], z["test_y"]
            y_tgt = y if y_tgt is None else y_tgt
            T.append(robust_restricted_correct(lg, y, allowed))
        # identical items and labels for source-trained and target-trained models are a precondition of this analysis
        assert np.array_equal(y_src, y_tgt), f"R2 {tgt}: source and target models were not scored on identical items"
        ex = np.asarray(EXPLICIT[exkey], bool)
        if pos is not None:
            assert np.array_equal(ex, np.asarray(EXPLICIT["isear"], bool)[pos]), "R2 isear: explicitness flags misaligned"
        S, T = np.asarray(S, float), np.asarray(T, float)
        assert S.shape[1] == T.shape[1] == len(ex)
        gap = lambda C, e: float(np.mean(C[:, e].mean(1) - C[:, ~e].mean(1)))
        g_s, g_t = gap(S, ex), gap(T, ex)
        n, rs, rt, boots = len(ex), len(S), len(T), []
        while len(boots) < ROBUST_N_BOOT:
            ii = rng.integers(0, n, n); e = ex[ii]
            if e.all() or not e.any():
                continue
            bs, bt = S[rng.integers(0, rs, rs)][:, ii], T[rng.integers(0, rt, rt)][:, ii]
            boots.append((gap(bs, e), gap(bt, e)))
        bo = np.asarray(boots); did_b = bo[:, 0] - bo[:, 1]
        # a bootstrap with B resamples cannot resolve p below 1/B; report that floor instead of 0
        p = float(max(1.0 / ROBUST_N_BOOT, min(1.0, 2 * min(np.mean(did_b <= 0), np.mean(did_b >= 0)))))
        ci = lambda v: [100 * float(np.percentile(v, 2.5)), 100 * float(np.percentile(v, 97.5))]
        out[tgt] = {"n_items": int(n), "n_explicit": int(ex.sum()), "n_source_runs": int(rs), "n_target_runs": int(rt),
                    "gap_source_pp": 100 * g_s, "gap_source_ci95_pp": ci(bo[:, 0]),
                    "gap_target_pp": 100 * g_t, "gap_target_ci95_pp": ci(bo[:, 1]),
                    "did_pp": 100 * (g_s - g_t), "did_ci95_pp": ci(did_b), "p_two_sided_bootstrap": p}
        pv[tgt] = p
    for t, ph in holm(pv).items():
        out[t]["p_holm"] = ph
    out["_note"] = ("gap = accuracy on explicit minus implicit items (restricted rule); source = main-stage runs with no prompt "
                    "or template A; target = models trained on the target corpus without a prompt; joint bootstrap over items and runs")
    return out

def robust_r3_h3_main_only(df: pd.DataFrame) -> dict:
    out, pv = {}, {}
    rng = np.random.default_rng(ROBUST_SEED + 2)
    cand = df[(df.stage == "main") & df["cri"].notna()] if "cri" in df else df.iloc[0:0]
    for tgt in TRANSFER_TARGETS:
        col = f"transfer_{tgt}"
        if col not in cand or cand[col].isna().all():
            continue
        c = cand[cand[col].notna()].copy()
        c["transfer_acc"] = c[col].map(lambda d: d["restricted"]["acc"])
        if len(c) < 10 or c.encoder.nunique() < 2:
            continue
        zf = lambda s: (s - s.mean()) / (s.std() + 1e-12)
        c["z_in"], c["z_cri"] = zf(c.test_acc), zf(c.cri)
        ols = smf.ols("transfer_acc ~ z_in + z_cri", data=c).fit()
        res = {"n_runs": int(len(c)), "n_encoders": int(c.encoder.nunique()), "beta_cri_ols": float(ols.params["z_cri"]),
               "spearman_cri": float(sps.spearmanr(c.cri, c.transfer_acc)[0]),
               "r2_in_only": float(smf.ols("transfer_acc ~ z_in", data=c).fit().rsquared), "r2_in_plus_cri": float(ols.rsquared)}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mix = smf.mixedlm("transfer_acc ~ z_in + z_cri", data=c, groups=c["encoder"], re_formula="1",
                                  vc_formula={"seed": "0 + C(seed)"}).fit(reml=True)
            ci_m = mix.conf_int().loc["z_cri"].tolist()
            p1 = float(sps.norm.cdf(mix.tvalues["z_cri"]))            # H3 predicts a negative coefficient
            if np.all(np.isfinite(ci_m)) and np.isfinite(p1):
                res.update(beta_cri_mixed=float(mix.params["z_cri"]), ci95_mixed=ci_m, p_mixed_one_sided=p1)
                pv[tgt] = p1
            else:
                res["mixed_model_note"] = "singular fit; not used for inference"
        except Exception as e:
            logger.error(f"R3 mixed model failed for {tgt}: {e}")
        encs, boots = c.encoder.unique(), []
        for _ in range(ROBUST_N_BOOT):
            bs = pd.concat([c[c.encoder == e] for e in rng.choice(encs, len(encs))])
            if bs.cri.nunique() > 2 and bs.test_acc.nunique() > 2:
                bs = bs.assign(z_in=zf(bs.test_acc), z_cri=zf(bs.cri))
                boots.append(smf.ols("transfer_acc ~ z_in + z_cri", data=bs).fit().params["z_cri"])
        res["ci95_cluster_bootstrap"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None
        res["n_bootstrap_valid"] = int(len(boots))     # resamples with enough distinct encoders to fit the model
        res["n_distinct_bootstrap_estimates"] = int(len(np.unique(np.round(boots, 12)))) if boots else 0
        out[tgt] = res
    for t, ph in holm(pv).items():
        out[t]["p_holm"] = ph
    out["_note"] = "H3 model refitted on main-stage runs only (no masked-training runs); coefficients are accuracy per SD of CRI"
    return out

def robust_r1b_placebo(df: pd.DataFrame) -> dict:
    """R1b (SetFit only, where R1 was significant). Question: is the larger prediction change on implicit items specific to
    the instruction's meaning, or do harder items change more under ANY prefix? Placebo = the random-token prefix of the
    same token length as template A. For every encoder x seed, template A and the random prefix are both compared with the
    SAME no-prompt run (same initialization and data order), so a correctness change occurs exactly where a prefix changed
    the prediction. Primary statistic: log concentration ratio, log(change rate on implicit items / change rate on explicit
    items), template A minus random prefix, paired bootstrap over encoder x seed. In this paired design the ratio does not
    depend on how strongly each prefix perturbs the model (validated by simulation: false-positive rate 0.07-0.09 at a
    nominal 0.05 with perturbation sizes differing up to 2x; slightly liberal). Also reported: signed effects per stratum."""
    rng = np.random.default_rng(ROBUST_SEED + 3)
    g = df[(df.stage == "main") & (df.corpus == "setfit") & (df.text_transform == "none")
           & df.condition.isin(["none", "A", "random"])]
    if not len(g):
        return {}
    assert not g.duplicated(["encoder", "condition", "seed"]).any(), "R1b: duplicate main runs for one encoder/condition/seed"
    ex = np.asarray(EXPLICIT["setfit_test"], bool)
    corr = {(r.encoder, r.condition, int(r.seed)): load_test_correct(r).astype(float) for _, r in g.iterrows()}
    assert all(len(v) == len(ex) for v in corr.values()), "R1b: test size does not match explicitness flags"
    keys = sorted((e, s) for (e, c, s) in corr if c == "none" and (e, "A", s) in corr and (e, "random", s) in corr)
    if len(keys) < 3:
        logger.info(f"R1b: {len(keys)} complete encoder x seed triples (none, A, random); at least 3 needed")
        return {}
    DA = np.stack([corr[(e, "A", s)] - corr[(e, "none", s)] for e, s in keys])
    DR = np.stack([corr[(e, "random", s)] - corr[(e, "none", s)] for e, s in keys])
    encs = np.array([e for e, _ in keys])
    def logratio(D):
        fi, fe = np.abs(D[:, ~ex]).mean(), np.abs(D[:, ex]).mean()
        return float(np.log(fi / fe)) if fi > 0 and fe > 0 else float("nan")
    est = logratio(DA) - logratio(DR)
    boots = []
    while len(boots) < ROBUST_N_BOOT:
        idx = rng.integers(0, len(keys), len(keys))                  # paired: A and random resampled together
        v = logratio(DA[idx]) - logratio(DR[idx])
        if np.isfinite(v):
            boots.append(v)
    boots = np.asarray(boots)
    p = float(max(1.0 / ROBUST_N_BOOT, min(1.0, 2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))))
    per_enc = {str(e): logratio(DA[encs == e]) - logratio(DR[encs == e]) for e in sorted(set(encs))}
    def rates(D):
        return {"change_rate_implicit_pct": 100 * float(np.abs(D[:, ~ex]).mean()),
                "change_rate_explicit_pct": 100 * float(np.abs(D[:, ex]).mean()),
                "concentration_ratio": float(np.exp(logratio(D)))}
    def signed(D, m):
        v = 100 * D[:, m].mean(1); t = sps.t.ppf(0.975, len(v) - 1); se = v.std(ddof=1) / np.sqrt(len(v))
        return {"mean_pp": float(v.mean()), "ci95_pp": [float(v.mean() - t * se), float(v.mean() + t * se)]}
    return {"corpus": "setfit", "n_pairs": int(len(keys)), "n_implicit_items": int((~ex).sum()), "n_explicit_items": int(ex.sum()),
            "template_A_vs_none": rates(DA), "random_prefix_vs_none": rates(DR),
            "log_concentration_ratio_A_minus_random": {"est": float(est), "ci95": [float(np.percentile(boots, 2.5)),
                                                       float(np.percentile(boots, 97.5))], "p_two_sided_bootstrap": p},
            "per_encoder_log_ratio_difference": per_enc,
            "signed_effect_A_implicit": signed(DA, ~ex), "signed_effect_A_explicit": signed(DA, ex),
            "signed_effect_random_implicit": signed(DR, ~ex), "signed_effect_random_explicit": signed(DR, ex),
            "_note": ("Positive log-ratio difference: template A's changes concentrate on implicit items more than a meaningless "
                      "prefix's changes do (instruction-specific). Near 0: implicit items change more under any prefix "
                      "(generic volatility of harder items). Signed effects show whether a prefix helps (positive) or only "
                      "moves predictions. The bootstrap treats encoder x seed pairs as independent; per-encoder differences "
                      "show sign consistency. Simulation calibration: slightly liberal (0.07-0.09 at nominal 0.05).")}

ROBUSTNESS = {}
_rdf = robust_train_df()
if len(_rdf):
    for name, fn in [("robustness_R1_h4", robust_r1_h4), ("robustness_R1b_h4_placebo", robust_r1b_placebo),
                     ("robustness_R2_h2_control", robust_r2_h2_control),
                     ("robustness_R3_h3_main_only", robust_r3_h3_main_only)]:
        try:
            ROBUSTNESS[name] = fn(_rdf)
            write_json(ANALYSIS_DIR / "metrics" / f"{name}.json", ROBUSTNESS[name])
        except Exception as e:
            logger.error(f"{name} failed: {type(e).__name__}: {e}")
            raise
for name, res in ROBUSTNESS.items():
    if name.endswith("R1b_h4_placebo"):
        if res:
            lr = res["log_concentration_ratio_A_minus_random"]
            logger.info(f"R1b setfit: concentration ratio template A {res['template_A_vs_none']['concentration_ratio']:.2f} vs random "
                        f"prefix {res['random_prefix_vs_none']['concentration_ratio']:.2f}; log difference {lr['est']:+.3f} "
                        f"[{lr['ci95'][0]:+.3f}, {lr['ci95'][1]:+.3f}], p = {lr['p_two_sided_bootstrap']:.4f}; signed effect on implicit "
                        f"items: A {res['signed_effect_A_implicit']['mean_pp']:+.2f} pp "
                        f"{[round(x, 2) for x in res['signed_effect_A_implicit']['ci95_pp']]}, random "
                        f"{res['signed_effect_random_implicit']['mean_pp']:+.2f} pp")
        continue
    for key, v in res.items():
        if key.startswith("_"):
            continue
        if name.endswith("R1_h4"):
            logger.info(f"R1 {key}: observed {v['observed_pp']:+.2f} pp vs size-matched null {v['null_mean_pp']:+.2f} pp "
                        f"-> excess {v['excess_over_null_pp']:+.2f} pp, p = {v['p_perm_size_calibrated']:.4f}")
        elif name.endswith("R2_h2_control"):
            logger.info(f"R2 {key}: DiD {v['did_pp']:+.2f} pp, 95% CI [{v['did_ci95_pp'][0]:+.2f}, {v['did_ci95_pp'][1]:+.2f}], "
                        f"p {'<' if v['p_two_sided_bootstrap'] <= 1.0 / ROBUST_N_BOOT else '='} {v['p_two_sided_bootstrap']:.4f}")
        else:
            logger.info(f"R3 {key}: beta_CRI mixed {v.get('beta_cri_mixed', float('nan')):+.4f}, bootstrap CI "
                        f"{v.get('ci95_cluster_bootstrap')} ({v.get('n_distinct_bootstrap_estimates', 0)} distinct estimates), n = {v['n_runs']}")
logger.info(f"Robustness recomputations complete: {sorted(ROBUSTNESS)}")
