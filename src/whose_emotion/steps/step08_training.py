# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 8 — TRAINING LOOP
# For every pending run of the stage: seed -> data -> model -> train (checkpoints every CHECKPOINT_EVERY_N_STEPS,
# early stopping on validation loss) -> export logits for every evaluation input while the model is in memory
# (validation, test, cue-masked and random-masked validation for CRI, transfer targets) -> optional final model
# with reload gate -> one Hub commit for the run -> DONE marker. A failed run is logged and skipped; the sweep
# continues. Metrics are computed from the exported logits in Cell 9 (no model reload is ever needed there).
# =====================================================================================
import traceback
from contextlib import nullcontext

LARGE_POOLS = {}

def stratified_subset(labels, k: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    idx = []
    for c in sorted(set(labels.tolist())):
        pool = np.flatnonzero(labels == c)
        assert len(pool) >= k, f"class {c} has {len(pool)} items < k={k}"
        idx.extend(rng.choice(pool, size=k, replace=False).tolist())
    return sorted(idx)

def prepare_run_data(spec: dict, mask_token: str) -> dict:
    corpus = spec["corpus"]
    if corpus.startswith("setfit_large_"):
        n = int(corpus.rsplit("_", 1)[1])
        if n not in LARGE_POOLS:
            LARGE_POOLS[n] = load_setfit_large(DATA["setfit"], n, SCALING_POOL_SEED)
        src, num_labels, base = LARGE_POOLS[n], 6, "setfit"
    else:
        src, num_labels, base = DATA[corpus], NUM_LABELS[corpus], corpus
    tr_t, tr_y = list(src["train"]["text"]), list(src["train"]["label"])
    if spec["fewshot_k"]:
        keep = stratified_subset(tr_y, spec["fewshot_k"], spec["seed"])
        tr_t, tr_y = [tr_t[i] for i in keep], [tr_y[i] for i in keep]
    if spec.get("rebalance") == "oversample":                      # every class up to the majority count, with replacement
        rng = np.random.default_rng(spec["seed"])
        y_arr = np.asarray(tr_y); top = np.bincount(y_arr).max(); idx = []
        for c in sorted(set(tr_y)):
            pool = np.flatnonzero(y_arr == c)
            idx.extend(pool.tolist() + rng.choice(pool, size=top - len(pool), replace=True).tolist())
        tr_t, tr_y = [tr_t[i] for i in idx], [tr_y[i] for i in idx]
        assert len(tr_y) == top * len(set(tr_y)), "oversampling size mismatch"
    va_t, te_t = list(src["validation"]["text"]), list(src["test"]["text"])
    if spec["text_transform"]:
        fn = {"mask_feel": lambda t: mask_feel_phrase(t, mask_token),
              "mask_random": lambda t: mask_random_words(t, mask_token, 2, spec["seed"])}[spec["text_transform"]]
        tr_t = [fn(t) for t in tr_t]
        if spec["transform_eval"]:
            va_t, te_t = [fn(t) for t in va_t], [fn(t) for t in te_t]
    assert len(tr_t) == len(tr_y) > 0 and len(va_t) == len(src["validation"]["label"]) and len(te_t) == len(src["test"]["label"])
    return {"train": (tr_t, tr_y), "validation": (va_t, list(src["validation"]["label"])), "test": (te_t, list(src["test"]["label"])),
            "num_labels": num_labels, "base_corpus": base,
            "orig_validation_text": list(src["validation"]["text"])}

def batches(enc: dict, order, labels=None):
    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
    for i in range(0, len(order), BATCH_SIZE if labels is not None else EVAL_BATCH_SIZE):
        idx = order[i:i + (BATCH_SIZE if labels is not None else EVAL_BATCH_SIZE)]
        yield idx, [{k: enc[k][j] for k in keys} for j in idx]

def autocast_ctx(mode: str):
    if mode == "fp32" or DEVICE.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if mode == "bf16" else torch.float16)

def predict_logits(model, tok, enc: dict, mode: str) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for _, feats in batches(enc, np.arange(len(enc["input_ids"]))):
            b = tok.pad(feats, return_tensors="pt")
            b = {k: v.to(DEVICE) for k, v in b.items()}
            try:
                with autocast_ctx(mode):
                    out.append(model(**b).logits.float().cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM during inference. Allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                torch.cuda.empty_cache(); raise
    logits = np.concatenate(out)
    assert logits.shape[0] == len(enc["input_ids"]) and np.isfinite(logits).all(), "invalid logits"
    return logits

def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    shutil.move(str(tmp), str(path))

def train_run(spec: dict, run_dir: Path, rp: str) -> dict:
    set_seed(spec["seed"])
    mode = precision_mode(spec["precision"])
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    data = prepare_run_data(spec, tok.mask_token or tok.unk_token)
    tr_t, tr_y = data["train"]
    tok, model, optimizer, scheduler, prefix, total_steps = build_model(spec, len(tr_y), data["num_labels"])
    enc_tr = encode(tok, tr_t, prefix, spec["placement"], spec["head"])
    enc_va = encode(tok, data["validation"][0], prefix, spec["placement"], spec["head"])
    enc_te = encode(tok, data["test"][0], prefix, spec["placement"], spec["head"])
    y_tr = torch.tensor(tr_y)
    y_va = np.asarray(data["validation"][1])
    class_w = None
    if spec.get("rebalance") == "weighted":                        # ASSUMPTION: inverse-frequency weights n / (K * n_c)
        cnt = np.bincount(tr_y, minlength=data["num_labels"]).astype(float)
        present = cnt > 0
        w = np.zeros_like(cnt); w[present] = cnt.sum() / (present.sum() * cnt[present])
        class_w = torch.tensor(w, dtype=torch.float32, device=DEVICE)
        logger.info(f"class weights: {np.round(w, 3).tolist()}")
    if mode == "fp16":
        scaler = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()
    else:
        scaler = None
    upload_mid = UPLOAD_MIDRUN_CHECKPOINTS is True or (UPLOAD_MIDRUN_CHECKPOINTS == "auto" and total_steps > AUTO_UPLOAD_MIN_STEPS)
    few = bool(spec["fewshot_k"])
    st = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "best_val_loss": float("inf"), "bad_epochs": 0, "best_state": None,
          "history": [], "stopped_early": False}

    # ---- resume: local checkpoint first (same session), then the Hub ------------------------------------------
    latest_local = run_dir / "checkpoints" / "latest.json"
    latest_file = latest_local if latest_local.exists() else hub_download(f"{rp}/checkpoints/latest.json")
    if latest_file is not None:
        with open(latest_file) as f:
            latest = json.load(f)
        ck_rel = f"{rp}/{latest['latest_checkpoint']}ckpt.pt"
        ck = local_path(ck_rel) if local_path(ck_rel).exists() else hub_download(ck_rel)
        if ck is not None:
            saved = torch.load(ck, map_location="cpu", weights_only=False)
            assert saved["spec_id"] == spec["spec_id"], "checkpoint belongs to a different run specification"
            model.load_state_dict(saved["model_state_dict"]); optimizer.load_state_dict(saved["optimizer_state_dict"])
            scheduler.load_state_dict(saved["scheduler_state_dict"])
            if scaler is not None and saved.get("scaler_state_dict"):
                scaler.load_state_dict(saved["scaler_state_dict"])
            st = saved["train_state"]
            torch.set_rng_state(saved["rng_torch"]); np.random.set_state(saved["rng_numpy"]); random.setstate(saved["rng_python"])
            if torch.cuda.is_available() and saved.get("rng_cuda") is not None:
                torch.cuda.set_rng_state_all(saved["rng_cuda"])            # GPU dropout masks resume exactly
            logger.info(f"Resumed {spec['spec_id']} from step {st['step']}, epoch {st['epoch']}, batch {st['batch_in_epoch']}")
        else:
            logger.warning(f"latest.json found but checkpoint file missing for {spec['spec_id']}; starting fresh")
    else:
        logger.info(f"No checkpoint for {spec['spec_id']}. Starting fresh.")

    def checkpoint(tag: str):
        rel = f"checkpoints/{tag}/"
        path = run_dir / rel / "ckpt.pt"
        save_checkpoint(path, {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                               "scheduler_state_dict": scheduler.state_dict(),
                               "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                               "train_state": st, "spec_id": spec["spec_id"], "run_id": RUN_ID, "config": CONFIG_SNAPSHOT,
                               "timestamp": datetime.now().isoformat(), "rng_torch": torch.get_rng_state(),
                               "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
                               "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
        if upload_mid:
            hub_upload_file(path, f"{rp}/{rel}ckpt.pt")
        latest = {"latest_checkpoint": rel, "step": st["step"], "epoch": st["epoch"], "timestamp": datetime.now().isoformat()}
        write_json(run_dir / "checkpoints" / "latest.json", latest)       # only after the save (and upload) succeeded
        if upload_mid:
            hub_upload_file(run_dir / "checkpoints" / "latest.json", f"{rp}/checkpoints/latest.json")
        for old in (run_dir / "checkpoints").glob("step_*"):              # keep only the newest local checkpoint
            if old.name != tag:
                shutil.rmtree(old, ignore_errors=True)
        if upload_mid:                                                     # and only the newest one on the Hub
            stale = sorted({"/".join(f.split("/")[:-1]) for f in HUB_FILES
                            if f.startswith(f"{rp}/checkpoints/step_") and not f.startswith(f"{rp}/{rel}")})
            for d in stale:
                upload_with_retry(lambda d=d: API.delete_folder(path_in_repo=d, repo_id=HF_REPO_ID, repo_type="dataset",
                                                                 commit_message=f"remove superseded checkpoint {d.rsplit('/', 1)[-1]}"))
                HUB_FILES.difference_update({f for f in list(HUB_FILES) if f.startswith(d + "/")})
        logger.debug(f"checkpoint {tag} saved (uploaded={upload_mid})")

    model.train()
    t0 = time.time()
    n = len(tr_y)
    try:
        while True:
            if few and st["step"] >= total_steps:
                break
            if not few and (st["epoch"] >= spec["epochs"] or st["stopped_early"]):
                break
            order = np.random.default_rng(spec["seed"] * 1000 + st["epoch"]).permutation(n)   # deterministic per epoch
            model.train()
            for bi, (idx, feats) in enumerate(batches(enc_tr, order, labels=True)):
                if bi < st["batch_in_epoch"]:
                    continue                                                   # resume without repeating batches
                b = tok.pad(feats, return_tensors="pt")
                b = {k: v.to(DEVICE) for k, v in b.items()}
                y = y_tr[idx].to(DEVICE)
                with autocast_ctx(mode):
                    loss = torch.nn.functional.cross_entropy(model(**b).logits.float(), y, weight=class_w)
                assert torch.isfinite(loss), f"non-finite loss at step {st['step']}"
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM); optimizer.step()
                scheduler.step()
                st["step"] += 1; st["batch_in_epoch"] = bi + 1
                if st["step"] % LOG_EVERY_N_STEPS == 0:
                    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    logger.info(f"[{spec['spec_id']}] step {st['step']}/{total_steps} loss {loss.item():.4f} "
                                f"lr {scheduler.get_last_lr()[0]:.2e} mem {mem:.2f} GB")
                if st["step"] % CHECKPOINT_EVERY_N_STEPS == 0:
                    checkpoint(f"step_{st['step']}")
                if few and st["step"] >= total_steps:
                    break
            st["epoch"] += 1; st["batch_in_epoch"] = 0
            if not few:                                                        # model selection on validation loss only
                lg = predict_logits(model, tok, enc_va, mode)
                vloss = float(torch.nn.functional.cross_entropy(torch.tensor(lg), torch.tensor(y_va)))
                vacc = float(np.mean(lg.argmax(1) == y_va))
                st["history"].append({"epoch": st["epoch"], "step": st["step"], "val_loss": vloss, "val_acc": vacc})
                logger.info(f"[{spec['spec_id']}] epoch {st['epoch']} val_loss {vloss:.4f} val_acc {vacc:.4f}")
                if vloss < st["best_val_loss"] - 1e-6:
                    st["best_val_loss"], st["bad_epochs"] = vloss, 0
                    st["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    st["bad_epochs"] += 1
                    st["stopped_early"] = st["bad_epochs"] >= spec.get("patience", EARLY_STOP_PATIENCE)
            gc.collect()
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM at step {st['step']}. Allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB. Error: {e}")
        try:
            checkpoint("oom_recovery"); logger.info("OOM recovery checkpoint saved.")
        except Exception as save_err:
            logger.error(f"Recovery checkpoint failed: {save_err}")
        torch.cuda.empty_cache()
        raise
    train_seconds = time.time() - t0
    if not few:
        assert st["best_state"] is not None, "no validation epoch completed"
        model.load_state_dict(st["best_state"])

    # ---- export logits for every evaluation input while the model is in memory ---------------------------------
    outs = {"val_logits": predict_logits(model, tok, enc_va, mode), "val_y": y_va}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_inf = time.time()
    outs["test_logits"] = predict_logits(model, tok, enc_te, mode)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    test_items_per_second = len(enc_te["input_ids"]) / max(time.time() - t_inf, 1e-9)
    outs["test_y"] = np.asarray(data["test"][1])
    trunc = float(np.mean([len(x) >= MAX_SEQ_LEN for x in enc_te["input_ids"]]))
    if data["base_corpus"] == "setfit":                                         # CRI on original validation texts
        mask = tok.mask_token or tok.unk_token
        cue_t, rnd_t, orig_t = [], [], []
        for t in data["orig_validation_text"]:
            m, k = mask_cues(t, mask)
            if k:
                orig_t.append(t); cue_t.append(m); rnd_t.append(mask_random_noncue(t, mask, k, spec["seed"]))
        for key, texts in (("cri_orig", orig_t), ("cri_cue", cue_t), ("cri_random", rnd_t)):
            outs[f"{key}_logits"] = predict_logits(model, tok, encode(tok, texts, prefix, spec["placement"], spec["head"]), mode)

        if spec["transfer"]:
            for tgt in TRANSFER_TARGETS:
                d = DATA[tgt]["test"]
                outs[f"transfer_{tgt}_logits"] = predict_logits(model, tok, encode(tok, d["text"], prefix, spec["placement"], spec["head"]), mode)
                outs[f"transfer_{tgt}_y"] = np.asarray(d["label"])
    elif spec["stage"] == "cev_third" and spec["transfer"]:      # GoEmotions-trained: transfer to the other three targets
        for tgt in [t for t in TRANSFER_TARGETS if t != "goemotions"]:
            d = DATA[tgt]["test"]
            outs[f"transfer_{tgt}_logits"] = predict_logits(model, tok, encode(tok, d["text"], prefix, spec["placement"], spec["head"]), mode)
            outs[f"transfer_{tgt}_y"] = np.asarray(d["label"])

    if spec["stage"] in CEV_EXPORT_STAGES:             # crowd-enVENT: original and masked texts (2,750 each)

        for key, texts in (("cev_orig", CEV["text_orig"]), ("cev_masked", CEV["text_masked"])):
            outs[f"{key}_logits"] = predict_logits(model, tok, encode(tok, texts, prefix, spec["placement"], spec["head"]), mode)
    if spec["corpus"] == "cev_target":                 # the same 500 test texts with the writers' emotion words masked
        outs["cevtest_masked_logits"] = predict_logits(
            model, tok, encode(tok, [CEV["text_masked"][i] for i in CEV["reader_pos"]], prefix, spec["placement"], spec["head"]), mode)
    if spec["stage"] in ENI_EXPORT_STAGES:             # v8: enISEAR, the 572 four-class descriptions as the readers saw them
        outs["eni_logits"] = predict_logits(model, tok, encode(tok, ENI["text"], prefix, spec["placement"], spec["head"]), mode)
    np.savez_compressed(run_dir / "outputs" / "logits.npz", **outs)

    # ---- optional final model with reload gate --------------------------------------------------------------
    reload_agreement = None
    sel = SAVE_FINAL_MODEL_FOR
    if (spec["head"] == "cls" and spec["stage"] in sel["stages"] and spec["seed"] in sel["seeds"] and spec["condition"] in sel["conditions"]):
        final_dir = run_dir / "checkpoints" / "final"
        model.save_pretrained(final_dir, safe_serialization=True); tok.save_pretrained(final_dir)
        write_json(final_dir / "run_spec.json", {**spec, "prefix": prefix})
        # reloaded = AutoModelForSequenceClassification.from_pretrained(final_dir).to(DEVICE)
        reloaded = AutoModelForSequenceClassification.from_pretrained(final_dir).float().to(DEVICE)
        reload_agreement = float(np.mean(predict_logits(reloaded, tok, enc_va, mode).argmax(1) == outs["val_logits"].argmax(1)))
        del reloaded
        if reload_agreement < 0.999:
            raise RuntimeError(f"Reload gate failed: saved model agrees on {reload_agreement:.4f} of validation predictions")
    n_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    n_total_params = int(sum(p.numel() for p in model.parameters()))
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {"prefix": prefix, "train_seconds": train_seconds, "total_steps": total_steps, "steps_run": st["step"],
            "epochs_run": st["epoch"], "history": st["history"], "stopped_early": st["stopped_early"],
            "test_truncated_frac": trunc, "n_train": len(tr_y), "precision_used": mode, "reload_agreement": reload_agreement,
            "midrun_checkpoints_uploaded": upload_mid, "test_items_per_second": test_items_per_second,
            "trainable_parameters": n_trainable_params, "total_parameters": n_total_params}

def llm_demos(k: int) -> list:
    if k == 0:
        return []
    assert k == len(SETFIT), "demonstrations are one per class"
    rng = np.random.default_rng(LLM_DEMO_SEED)
    y = np.asarray(DATA["setfit"]["train"]["label"])
    demos = [(DATA["setfit"]["train"]["text"][int(rng.choice(np.flatnonzero(y == c)))], c) for c in range(len(SETFIT))]
    return [demos[i] for i in rng.permutation(len(demos))]

def llm_messages(text: str, demos: list) -> list:
    msgs = [{"role": "system", "content": LLM_SYSTEM}]
    for t, c in demos:
        msgs += [{"role": "user", "content": f"Text: {t}\nEmotion:"}, {"role": "assistant", "content": SETFIT[c]}]
    return msgs + [{"role": "user", "content": f"Text: {text}\nEmotion:"}]

def parse_label(raw: str) -> int:
    """Strict parser: exactly one of the six label names must occur as a word; otherwise -1 (format failure,
    scored as an error)."""
    found = [c for c, name in enumerate(SETFIT) if re.search(rf"\b{name}\b", raw.lower())]
    return found[0] if len(found) == 1 else -1

def run_llm(spec: dict, run_dir: Path, rp: str) -> dict:
    texts, labels = DATA["setfit"]["test"]["text"], DATA["setfit"]["test"]["label"]
    demos = llm_demos(spec["shots"])
    n_chunks = math.ceil(len(texts) / LLM_CHUNK)
    gen = None
    if spec["backend"] == "hf":
        from transformers import AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(spec["hf_id"], token=SECRETS["HF_TOKEN"])
        kw = {"token": SECRETS["HF_TOKEN"]}
        if torch.cuda.is_available() and LLM_LOAD_4BIT:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            kw["device_map"] = "auto"
        else:
            logger.warning("LLM loaded without 4-bit quantization (no GPU or LLM_LOAD_4BIT=False)")
        lm = AutoModelForCausalLM.from_pretrained(spec["hf_id"], **kw)
        if "device_map" not in kw:
            lm.to(DEVICE)
        lm.eval()
        def template(m):
            # return_dict=True gives the same type (BatchEncoding with input_ids and attention_mask) in Transformers 4 and 5;
            # without it, Transformers 5 returns a BatchEncoding where Transformers 4 returned a tensor
            return tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        def gen(msgs):
            try:
                enc = template(msgs)
            except Exception:
                # WARNING: chat template rejected the system role (or is missing); system text merged into the first user turn
                merged = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}] + msgs[2:]
                if tok.chat_template:
                    enc = template(merged)
                else:
                    enc = tok("\n".join(m["content"] for m in merged) + "\n", return_tensors="pt")
            ids = enc["input_ids"].to(lm.device)
            mask = enc["attention_mask"].to(lm.device) if "attention_mask" in enc else torch.ones_like(ids)
            with torch.no_grad():
                out = lm.generate(input_ids=ids, attention_mask=mask, max_new_tokens=LLM_MAX_NEW_TOKENS, do_sample=False,
                                  pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True), spec["hf_id"]
    else:
        from openai import OpenAI
        client = OpenAI(api_key=SECRETS["OPENAI_API_KEY"])
        def gen(msgs):
            for attempt in range(6):
                try:
                    r = client.chat.completions.create(model=spec["api_model"], messages=msgs, temperature=0,
                                                       max_tokens=LLM_MAX_NEW_TOKENS, seed=LLM_DEMO_SEED)
                    return (r.choices[0].message.content or ""), r.model
                except Exception as e:
                    wait = 5 * 2 ** attempt
                    logger.warning(f"OpenAI call failed ({type(e).__name__}); retry in {wait}s"); time.sleep(wait)
            raise RuntimeError("OpenAI API failed after 6 attempts")
    reported = set()
    for ci in range(n_chunks):
        rel = f"outputs/llm_chunk_{ci:03d}.jsonl"
        if (run_dir / rel).exists() or hub_download(f"{rp}/{rel}") is not None:
            logger.info(f"chunk {ci + 1}/{n_chunks} already on the Hub"); continue
        rows = []
        for i in range(ci * LLM_CHUNK, min((ci + 1) * LLM_CHUNK, len(texts))):
            raw, model_id = gen(llm_messages(texts[i], demos))
            reported.add(model_id)
            rows.append({"i": i, "label": int(labels[i]), "raw": raw, "pred": parse_label(raw), "model_reported": model_id})
        with open(run_dir / rel, "w") as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        hub_upload_file(run_dir / rel, f"{rp}/{rel}")                # each chunk persisted before the next one starts
        logger.info(f"[{spec['spec_id']}] chunk {ci + 1}/{n_chunks} done")
    if spec["backend"] == "hf":
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gc.collect()
    return {"n_items": len(texts), "n_chunks": n_chunks, "demos": [[t, SETFIT[c]] for t, c in demos], "models_reported": sorted(reported)}

def run_audit(spec: dict, run_dir: Path, rp: str) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, f1_score
    tr, te = DATA["setfit"]["train"], DATA["setfit"]["test"]
    res = {"feel_rate": {k: float(np.mean([bool(FEEL_RE.search(t)) for t in d["text"]])) for k, d in
                         [("setfit_train", tr), ("setfit_test", te), ("goemotions_test", DATA["goemotions"]["test"]),
                          ("meld_test", DATA["meld"]["test"]), ("isear", DATA["isear"]["test"]), ("ed_test", DATA["ed"]["test"]),
                          ("tweeteval_test", DATA["tweeteval"]["test"])]}}
    raw_re = re.compile(r"\b(?:feel|feeling|feels|felt)\s+([a-z]+)", re.I)
    for skip in (True, False):                                   # one-token probe: majority training label of the word after "feel"
        nw = (lambda t: next_word(t)) if skip else (lambda t: (raw_re.search(t.lower()).group(1) if raw_re.search(t.lower()) else "<none>"))
        by = defaultdict(Counter)
        for t, y in zip(tr["text"], tr["label"]):
            by[nw(t)][y] += 1
        lut = {w: c.most_common(1)[0][0] for w, c in by.items()}
        fb = Counter(tr["label"]).most_common(1)[0][0]
        p = [lut.get(nw(t), fb) for t in te["text"]]
        res[f"probe_skip_intensifiers_{skip}"] = {"acc": float(accuracy_score(te["label"], p)), "macro_f1": float(f1_score(te["label"], p, average="macro"))}
    lab_by_text = defaultdict(set)
    for t, y in zip(tr["text"], tr["label"]):
        lab_by_text[t].add(y)
    res["train_texts_with_conflicting_labels"] = int(sum(len(v) > 1 for v in lab_by_text.values()))
    res["test_texts_also_in_train"] = int(sum(t in lab_by_text for t in te["text"]))
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(tr["text"]), vec.transform(te["text"])
    for name, clf in [("tfidf_lr", LogisticRegression(C=1.0, max_iter=2000)), ("tfidf_svm", LinearSVC(C=1.0))]:
        t0 = time.time(); clf.fit(Xtr, tr["label"]); fit_s = time.time() - t0
        t0 = time.time(); p = clf.predict(Xte); ips = len(te["text"]) / max(time.time() - t0, 1e-9)
        res[name] = {"acc": float(accuracy_score(te["label"], p)), "macro_f1": float(f1_score(te["label"], p, average="macro")),
                     "fit_seconds": fit_s, "items_per_second_prevectorized": ips}
    res["cue_vocabulary_size"] = len(CUE_VOCAB)
    res["explicit_share"] = {k: float(v.mean()) for k, v in EXPLICIT.items()}
    write_json(run_dir / "outputs" / "audit.json", res)
    logger.info(f"Audit: TF-IDF LR {res['tfidf_lr']['acc']:.4f}, SVM {res['tfidf_svm']['acc']:.4f}; probe {res['probe_skip_intensifiers_True']['acc']:.4f}")
    return {"audit_file": "outputs/audit.json"}

def run_one(spec: dict) -> bool:
    rp = run_prefix(spec)
    run_dir = LOCAL_ROOT / rp
    for sub in SUBDIRS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True); (run_dir / sub / ".gitkeep").touch()
    meta = {"run_id": RUN_ID, "spec": spec, "experiment_name": EXPERIMENT_NAME, "model_name": spec.get("hf_id") or spec.get("api_model") or spec["kind"],
            "timestamp_start": datetime.now().isoformat(), "timestamp_end": None, "status": "running", "completed": False,
            "library_versions": LIB_VERSIONS, "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    write_json(run_dir / "metadata" / "run_metadata.json", meta)
    try:
        info = {"train": train_run, "llm": run_llm, "audit": run_audit}[spec["kind"]](spec, run_dir, rp)
    except Exception as e:
        logger.error(f"Run {spec['spec_id']} failed: {type(e).__name__}: {e}")
        meta.update(status="failed", error=f"{type(e).__name__}: {e}", timestamp_end=datetime.now().isoformat())
        write_json(run_dir / "metadata" / "run_metadata.json", meta)
        try:
            hub_upload_file(run_dir / "metadata" / "run_metadata.json", f"{rp}/metadata/run_metadata.json")
        except Exception as up_err:
            logger.error(f"Could not upload failure metadata: {up_err}")
        # the traceback keeps the failed run's frames (and its model) alive; clear them so the GPU memory is really freed
        traceback.clear_frames(e.__traceback__)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False
    meta.update(status="completed", completed=True, timestamp_end=datetime.now().isoformat(), **info)
    write_json(run_dir / "metadata" / "run_metadata.json", meta)
    for p in (run_dir / "checkpoints").glob("step_*"):                  # mid-run checkpoints are not final artifacts
        shutil.rmtree(p, ignore_errors=True)
    (run_dir / "checkpoints" / "latest.json").unlink(missing_ok=True)
    if spec["kind"] == "train":
        assert (run_dir / "outputs" / "logits.npz").stat().st_size > 0
    hub_upload_folder(run_dir, rp, f"run {spec['spec_id']} completed")
    (run_dir / "DONE").write_text(datetime.now().isoformat())
    hub_upload_file(run_dir / "DONE", f"{rp}/DONE")                      # DONE only after the artifacts are on the Hub
    if info.get("midrun_checkpoints_uploaded"):                             # remove mid-run checkpoints only (never final/)
        step_dirs = sorted({"/".join(f.split("/")[:-1]) for f in HUB_FILES if f.startswith(f"{rp}/checkpoints/step_")
                            or f.startswith(f"{rp}/checkpoints/oom_recovery")})
        for d in step_dirs:
            upload_with_retry(lambda d=d: API.delete_folder(path_in_repo=d, repo_id=HF_REPO_ID, repo_type="dataset",
                                                             commit_message=f"remove mid-run checkpoint {d.rsplit('/', 1)[-1]}"))
        if f"{rp}/checkpoints/latest.json" in HUB_FILES:
            upload_with_retry(lambda: API.delete_file(path_in_repo=f"{rp}/checkpoints/latest.json", repo_id=HF_REPO_ID,
                                                       repo_type="dataset", commit_message="remove latest.json of completed run"))
        HUB_FILES.difference_update({f for f in list(HUB_FILES) if any(f.startswith(d + "/") for d in step_dirs)
                                     or f == f"{rp}/checkpoints/latest.json"})
    shutil.rmtree(run_dir / "checkpoints" / "final", ignore_errors=True)  # already on the Hub; free local disk
    RUN_STATUS[spec["spec_id"]] = "done"
    logger.info(f"Run {spec['spec_id']} completed and persisted ({spec.get('encoder', spec.get('model', spec['kind']))}, "
                f"{spec.get('condition', spec.get('shots', ''))}, seed {spec.get('seed', '-')})")
    return True

pending = [s for s in SPECS if RUN_STATUS[s["spec_id"]] != "done"]
logger.info(f"{len(pending)} pending runs in stage '{STAGE}'")
FAILED_RUNS = []
for i, spec in enumerate(pending, 1):
    logger.info("=" * 60)
    logger.info(f"Run {i}/{len(pending)}: {spec['spec_id']} " + " ".join(f"{k}={v}" for k, v in spec.items() if k not in ("spec_id", "stage")))
    if not run_one(spec):
        FAILED_RUNS.append(spec["spec_id"])
logger.info(f"Training finished for stage '{STAGE}': {len(pending) - len(FAILED_RUNS)} completed, {len(FAILED_RUNS)} failed")
