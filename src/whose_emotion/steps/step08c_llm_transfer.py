# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 8c — ZERO-SHOT LLM TRANSFER CONTROL (runs only when STAGE == "llm_transfer"; trains nothing)
# =====================================================================================
LLMT_MODEL = "gpt-4o-2024-08-06"
LLMT_PREFIX = f"{EXPERIMENT_NAME}/llm_transfer"
LLMT_CHUNK = 250

def llmt_tasks() -> dict:
    ch = np.asarray(CEV["changed"])
    return {
        "goemotions": (list(DATA["goemotions"]["test"]["text"]), list(DATA["goemotions"]["test"]["label"]), PRESENT_CLASSES["goemotions"]),
        "meld":       (list(DATA["meld"]["test"]["text"]), list(DATA["meld"]["test"]["label"]), PRESENT_CLASSES["meld"]),
        "isear":      (list(DATA["isear_ft"]["test"]["text"]), list(DATA["isear_ft"]["test"]["label"]), PRESENT_CLASSES["isear"]),
        "ed":         (list(DATA["ed"]["test"]["text"]), list(DATA["ed"]["test"]["label"]), PRESENT_CLASSES["ed"]),
        "cev_orig":   (list(CEV["text_orig"]), [int(v) for v in CEV["label"]], PRESENT_CLASSES["cev"]),
        "cev_masked": ([CEV["text_masked"][i] for i in np.flatnonzero(ch)], [int(v) for v in np.asarray(CEV["label"])[ch]], PRESENT_CLASSES["cev"]),
        # v8: enISEAR, the 572 four-class descriptions as the readers saw them (system message lists sadness, joy, anger, fear)
        "enisear":    (list(ENI["text"]), [int(v) for v in ENI["label"]], PRESENT_CLASSES["eni"]),
    }

def llmt_system(allowed: list) -> str:
    return ("You classify the emotion of a text. Answer with exactly one word from this list: "
            + ", ".join(SETFIT[c] for c in allowed) + ".")

def llmt_parse(raw: str, allowed: list) -> int:
    found = [c for c in allowed if re.search(rf"\b{SETFIT[c]}\b", (raw or "").lower())]
    return found[0] if len(found) == 1 else -1

def run_llm_transfer() -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=SECRETS["OPENAI_API_KEY"])
    def gen(system: str, text: str):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": f"Text: {text}\nEmotion:"}]
        for attempt in range(6):
            try:
                r = client.chat.completions.create(model=LLMT_MODEL, messages=msgs, temperature=0,
                                                   max_tokens=LLM_MAX_NEW_TOKENS, seed=LLM_DEMO_SEED)
                return (r.choices[0].message.content or ""), r.model
            except Exception as e:
                wait = 5 * 2 ** attempt
                logger.warning(f"OpenAI call failed ({type(e).__name__}); retry in {wait}s"); time.sleep(wait)
        raise RuntimeError("OpenAI API failed after 6 attempts")
    summary = {"model_requested": LLMT_MODEL, "tasks": {}}
    for task, (texts, labels, allowed) in llmt_tasks().items():
        system = llmt_system(allowed)
        n_chunks = math.ceil(len(texts) / LLMT_CHUNK)
        reported = set()
        for ci in range(n_chunks):
            rel = f"{LLMT_PREFIX}/{task}/chunk_{ci:03d}.jsonl"
            lp = local_path(rel)
            if lp.exists() or rel in HUB_FILES:
                continue
            rows = []
            for i in range(ci * LLMT_CHUNK, min((ci + 1) * LLMT_CHUNK, len(texts))):
                raw, model_id = gen(system, texts[i])
                reported.add(model_id)
                rows.append({"i": i, "label": int(labels[i]), "raw": raw, "pred": llmt_parse(raw, allowed), "model_reported": model_id})
            with open(lp, "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")
            hub_upload_file(lp, rel)
            logger.info(f"llm_transfer {task}: chunk {ci + 1}/{n_chunks} done")
        summary["tasks"][task] = {"n_items": len(texts), "n_chunks": n_chunks, "allowed": [SETFIT[c] for c in allowed],
                                  "system": system, "models_reported_this_session": sorted(reported)}
    # v8: a rerun only sends the tasks whose chunks are missing; keep the earlier entries (models reported, timestamp) for
    # every task that sent nothing in this session, so the manifest still documents the earlier calls
    old = hub_download(f"{LLMT_PREFIX}/manifest.json") if f"{LLMT_PREFIX}/manifest.json" in HUB_FILES else None
    if old is not None:
        prev = json.load(open(old))
        for task, entry in summary["tasks"].items():
            if not entry["models_reported_this_session"] and task in prev.get("tasks", {}):
                summary["tasks"][task] = {**prev["tasks"][task],
                                          "written_in_session": prev["tasks"][task].get("written_in_session", prev.get("timestamp"))}
    mp = local_path(f"{LLMT_PREFIX}/manifest.json")
    write_json(mp, {**summary, "timestamp": datetime.now().isoformat(), "library_versions": LIB_VERSIONS})
    hub_upload_file(mp, f"{LLMT_PREFIX}/manifest.json")
    return summary

if STAGE == "llm_transfer":
    LLMT_SUMMARY = run_llm_transfer()
    logger.info(f"llm_transfer complete: { {k: v['n_items'] for k, v in LLMT_SUMMARY['tasks'].items()} }")
