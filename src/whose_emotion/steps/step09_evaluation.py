# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 9 — EVALUATION
# Held-out evaluation from the logits exported in Cell 8. Model selection never used test data.
# Per run: in-domain metrics, explicit/implicit test accuracy, cue-reliance index (CRI), convergence flag,
# and for source runs with transfer: unrestricted / restricted / EM prior-corrected rules, explicit/implicit
# accuracy per rule, ATC predicted accuracy, and a degenerate-output flag.
# =====================================================================================
import tempfile
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

SOURCE_PRIOR = np.bincount(DATA["setfit"]["train"]["label"], minlength=6) / len(DATA["setfit"]["train"]["label"])
STRATA_KEY = {"setfit": "setfit_test", "ed": "ed_test", "tweeteval": "tweeteval_test", "meld_target": "meld",
              "goemotions_target": "goemotions", "isear_target": "isear_target_test"}

def ece(probs: np.ndarray, y: np.ndarray, bins: int = CALIBRATION_BINS) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(np.mean(pred[m] == y[m]) - conf[m].mean())
    return float(total)

def classwise_ece(probs: np.ndarray, y: np.ndarray, c: int, bins: int = CALIBRATION_BINS) -> float:
    p, t = probs[:, c], (y == c).astype(float)
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(t[m].mean() - p[m].mean())
    return float(total)

def calibration(val_logits, val_y, test_logits, test_y) -> dict:
    def nll(logits, y, T):
        z = logits / T; z = z - z.max(1, keepdims=True)
        return float(-np.mean(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))))
    opt = minimize_scalar(lambda T: nll(val_logits, val_y, T), bounds=TEMPERATURE_BOUNDS, method="bounded")
    T = float(opt.x)
    at_bound = min(abs(T - TEMPERATURE_BOUNDS[0]), abs(T - TEMPERATURE_BOUNDS[1])) < 1e-3
    out = {"temperature": T, "temperature_at_bound": bool(at_bound)}
    k = test_logits.shape[1]; onehot = np.eye(k)[test_y]
    for tag, Tt in (("before", 1.0), ("after", T)):
        p = softmax(test_logits / Tt)
        out[tag] = {"ece": ece(p, test_y), "brier": float(np.mean(((p - onehot) ** 2).sum(1))), "nll": nll(test_logits, test_y, Tt),
                    "acc": float(np.mean(p.argmax(1) == test_y)),
                    "classwise_ece": [classwise_ece(p, test_y, c) for c in range(k)]}
        edges = np.linspace(0, 1, CALIBRATION_BINS + 1); conf, pred = p.max(1), p.argmax(1); rb = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            rb.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()),
                       "acc": float(np.mean(pred[m] == test_y[m])) if m.any() else None, "conf": float(conf[m].mean()) if m.any() else None})
        out[tag]["reliability_bins"] = rb
    return out

def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def strat_acc(y, pred, ex) -> dict:
    y, pred, ex = np.asarray(y), np.asarray(pred), np.asarray(ex, bool)
    assert len(y) == len(pred) == len(ex)
    return {"acc_explicit": float(np.mean(pred[ex] == y[ex])) if ex.any() else None,
            "acc_implicit": float(np.mean(pred[~ex] == y[~ex])) if (~ex).any() else None,
            "n_explicit": int(ex.sum()), "n_implicit": int((~ex).sum())}

def em_prior(probs: np.ndarray, allowed: list, iters: int = 200, tol: float = 1e-8):
    """Saerens et al. (2002) EM re-estimation of target priors over the allowed classes."""
    p = probs[:, allowed]; p = p / p.sum(1, keepdims=True)
    ps = SOURCE_PRIOR[allowed] / SOURCE_PRIOR[allowed].sum()
    pt = ps.copy()
    for _ in range(iters):
        w = p * (pt / ps); w = w / w.sum(1, keepdims=True)
        new = w.mean(0)
        done = np.abs(new - pt).max() < tol
        pt = new
        if done:
            break
    return np.asarray(allowed)[w.argmax(1)], pt

def cev_restrict(logits: np.ndarray) -> np.ndarray:
    """Predictions restricted to the five classes crowd-enVENT shares with SetFit."""
    r = np.full_like(logits, -np.inf)
    r[:, PRESENT_CLASSES["cev"]] = logits[:, PRESENT_CLASSES["cev"]]
    return r.argmax(1)

def restrict_to(logits: np.ndarray, allowed: list) -> np.ndarray:
    """Predictions restricted to the classes of a target (v8; the general form of cev_restrict)."""
    r = np.full_like(logits, -np.inf)
    r[:, allowed] = logits[:, allowed]
    return r.argmax(1)

def perspective_metrics(pred, y, maj) -> dict:
    """v8. pred: predictions (-1 = no valid answer, scored as an error); y: writer labels; maj: reader majority
    (-1 tie, -2 outside the shared classes). R: unique majority equal to the writer's label; Rbar: unique majority that
    differs from it. pi: share of the errors against the writer's label, over R and Rbar, that fall on Rbar."""
    pred, y, maj = np.asarray(pred), np.asarray(y), np.asarray(maj)
    ok = maj >= 0
    R, Rb, err = ok & (maj == y), ok & (maj != y), pred != y
    return {"acc_writer": float(np.mean(~err)), "agree_reader_majority": float(np.mean(pred[ok] == maj[ok])),
            "acc_R": float(np.mean(~err[R])), "acc_Rbar": float(np.mean(~err[Rb])) if Rb.any() else None,
            "pi_error_share_Rbar": float(err[Rb].sum() / max(int(err[ok].sum()), 1)),
            "share_Rbar_items": float(Rb.sum() / max(int(ok.sum()), 1)),
            "rbar_chose_readers": float(np.mean(pred[Rb] == maj[Rb])) if Rb.any() else None,
            "rbar_chose_writer": float(np.mean(pred[Rb] == y[Rb])) if Rb.any() else None,
            "pred_share": (np.bincount(pred[pred >= 0], minlength=6) / len(pred)).round(4).tolist(),
            "n": int(len(y)), "n_R": int(R.sum()), "n_Rbar": int(Rb.sum())}

def atc(val_logits, val_y, tgt_logits, allowed) -> float:
    pv = softmax(val_logits)
    acc_v = float(np.mean(pv.argmax(1) == val_y))
    t = float(np.quantile(pv.max(1), 1 - acc_v))
    # threshold and target confidences use the same full softmax; restricting the target softmax would inflate confidence
    return float(np.mean(softmax(tgt_logits).max(1) > t))

def target_metrics(y, pred, allowed, ex) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    share = np.bincount(pred, minlength=6) / len(pred)
    return {"acc": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, labels=allowed, average="macro", zero_division=0)),
            "pred_share": share.round(4).tolist(), "degenerate": bool(share.max() >= 0.9), **strat_acc(y, pred, ex)}

def evaluate(spec: dict, z) -> dict:
    vy, ty = z["val_y"], z["test_y"]
    vp, tp = z["val_logits"].argmax(1), z["test_logits"].argmax(1)
    m = {"val_acc": float(np.mean(vp == vy)), "test_acc": float(np.mean(tp == ty)),
         "test_macro_f1": float(f1_score(ty, tp, average="macro", zero_division=0)),
         "test_weighted_f1": float(f1_score(ty, tp, average="weighted", zero_division=0)),
         "test_f1_per_class": f1_score(ty, tp, average=None, zero_division=0).round(4).tolist(),
         "converged": bool(np.mean(vp == vy) >= CONVERGED_VAL_ACC),
         "test_macro_precision": float(precision_score(ty, tp, average="macro", zero_division=0)),
         "test_macro_recall": float(recall_score(ty, tp, average="macro", zero_division=0)),
         "test_confusion": confusion_matrix(ty, tp, labels=list(range(z["test_logits"].shape[1]))).tolist(),
         "calibration": calibration(z["val_logits"], vy, z["test_logits"], ty)}
    base = "setfit" if spec["corpus"].startswith("setfit") else spec["corpus"]
    # if spec["corpus"] in ("meld_target", "goemotions_target", "isear_target", "ed"):   # macro-F1 over the classes present
    if spec["corpus"] in ("meld_target", "goemotions_target", "isear_target", "ed", "cev_target"):   # macro-F1 over present classes
        allowed = PRESENT_CLASSES[spec["corpus"].replace("_target", "")]
        m["test_macro_f1"] = float(f1_score(ty, tp, labels=allowed, average="macro", zero_division=0))
    if base in STRATA_KEY:
        m.update({f"test_{k}": v for k, v in strat_acc(ty, tp, EXPLICIT[STRATA_KEY[base]]).items()})
    if "cri_orig_logits" in z.files:
        o, c, r = (z[f"cri_{k}_logits"].argmax(1) for k in ("orig", "cue", "random"))
        m["cri_flip_cue"], m["cri_flip_random"] = float(np.mean(c != o)), float(np.mean(r != o))
        m["cri"] = m["cri_flip_cue"] - m["cri_flip_random"]; m["cri_n"] = int(len(o))
    for tgt in TRANSFER_TARGETS:
        if f"transfer_{tgt}_logits" not in z.files:
            continue
        lg, y = z[f"transfer_{tgt}_logits"], z[f"transfer_{tgt}_y"]
        allowed, ex = PRESENT_CLASSES[tgt], EXPLICIT[tgt if tgt != "ed" else "ed_test"]
        restricted = np.full_like(lg, -np.inf); restricted[:, allowed] = lg[:, allowed]
        em_pred, pt = em_prior(softmax(lg), allowed)
        m[f"transfer_{tgt}"] = {"unrestricted": target_metrics(y, lg.argmax(1), allowed, ex),
                                "restricted": target_metrics(y, restricted.argmax(1), allowed, ex),
                                "prior_em": {**target_metrics(y, em_pred, allowed, ex), "estimated_prior": pt.round(4).tolist()},
                                "true_prior": [float(np.mean(y == c)) for c in allowed],
                                "atc_predicted_acc": atc(z["val_logits"], vy, lg, allowed)}
    if "eni_logits" in z.files:                        # v8: enISEAR (four classes), scored against the writer's label
        m["enisear"] = perspective_metrics(restrict_to(z["eni_logits"], PRESENT_CLASSES["eni"]), ENI["label"], ENI["reader_majority"])
    return m

def evaluate_llm(spec: dict, rp: str) -> dict:
    rows = []
    for ci in range(math.ceil(len(DATA["setfit"]["test"]["text"]) / LLM_CHUNK)):
        rel = f"{rp}/outputs/llm_chunk_{ci:03d}.jsonl"
        lp = LOCAL_ROOT / rel
        path = lp if lp.exists() else hub_download(rel)
        assert path is not None, f"missing LLM chunk {rel}"
        rows += [json.loads(l) for l in open(path) if l.strip()]
    assert len(rows) == len(DATA["setfit"]["test"]["text"]) and len({r["i"] for r in rows}) == len(rows), "LLM outputs incomplete"
    y = np.array([r["label"] for r in rows]); p = np.array([r["pred"] for r in rows])
    return {"n": int(len(rows)), "acc": float(np.mean(p == y)), "format_failure_rate": float(np.mean(p == -1)),
            "macro_f1": float(f1_score(y, p, labels=list(range(6)), average="macro", zero_division=0)),
            "acc_on_parsed_only": float(np.mean(p[p >= 0] == y[p >= 0])) if (p >= 0).any() else None,
            "pred_share": (np.bincount(p[p >= 0], minlength=6) / len(p)).round(4).tolist(),
            "models_reported": sorted({r["model_reported"] for r in rows}), **strat_acc(y, p, EXPLICIT["setfit_test"])}

def evaluate_stage(specs: list) -> int:
    todo = [s for s in specs if RUN_STATUS.get(s["spec_id"]) == "done" and f"{run_prefix(s)}/metrics/metrics.json" not in HUB_FILES]
    logger.info(f"Evaluating {len(todo)} runs without metrics")
    batch_root = Path(tempfile.mkdtemp(prefix="metrics_batch_"))
    n_done = 0
    for i, s in enumerate(todo, 1):
        rp = run_prefix(s)
        if s["kind"] == "llm":
            metrics = {"spec": s, **evaluate_llm(s, rp), "evaluated": datetime.now().isoformat()}
        elif s["kind"] == "audit":
            lp = LOCAL_ROOT / rp / "outputs" / "audit.json"
            path = lp if lp.exists() else hub_download(f"{rp}/outputs/audit.json")
            with open(path) as fh:
                metrics = {"spec": s, "audit": json.load(fh), "evaluated": datetime.now().isoformat()}
        else:
            lp = LOCAL_ROOT / rp / "outputs" / "logits.npz"
            path = lp if lp.exists() else hub_download(f"{rp}/outputs/logits.npz")
            assert path is not None and Path(path).exists(), f"logits missing for completed run {s['spec_id']}"
            with np.load(path) as z:
                metrics = {"spec": s, **evaluate(s, z), "evaluated": datetime.now().isoformat()}
        out = batch_root / s["spec_id"] / "metrics" / "metrics.json"; out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, metrics)
        local = LOCAL_ROOT / rp / "metrics" / "metrics.json"; local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, local)
        n_done += 1
        if n_done % 100 == 0 or i == len(todo):                            # one Hub commit per 100 runs
            hub_upload_folder(batch_root, f"{EXPERIMENT_NAME}/{STAGE}/runs", f"metrics for {n_done} runs ({STAGE})")
            shutil.rmtree(batch_root, ignore_errors=True); batch_root.mkdir(parents=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n_done

N_EVALUATED = evaluate_stage(SPECS) if STAGE != "analysis" else 0
logger.info(f"Evaluation complete: {N_EVALUATED} runs evaluated in stage '{STAGE}'")
