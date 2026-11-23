# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 5 — CHECKPOINT RESTORATION
# Expands the STAGE grid into run specifications and determines, per run, whether it is complete (DONE on the
# Hub), resumable (latest.json on the Hub), or new. Checkpoint files are downloaded lazily in Cell 8.
# =====================================================================================
import hashlib, itertools

TUNED_RECIPES_PATH = f"{SHARED_PREFIX}/tuned_recipes.json"

def load_tuned_recipes():
    p = hub_download(TUNED_RECIPES_PATH)
    if p is None:
        return None
    with open(p) as f:
        return json.load(f)

def _with_id(spec: dict) -> dict:
    spec["spec_id"] = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    return spec

def expand_stage(stage: str, tuned) -> list:
    grid = STAGE_GRIDS[stage]
    if grid is None:
        return []
    if grid["kind"] == "audit":
        return [_with_id({"stage": stage, "kind": "audit"})]
    axes = {k: v for k, v in grid.items() if isinstance(v, list)}
    keys = list(axes)
    specs = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        s = dict(zip(keys, combo))
        if grid["kind"] == "llm":
            specs.append(_with_id({"stage": stage, "kind": "llm", "model": s["model"], "shots": int(s["shots"]), **LLM_MODELS[s["model"]]}))
            continue
        enc = ENCODERS[s["encoder"]]
        peft_method = s.get("peft_method")
        condition = PEFT_METHODS[peft_method] if peft_method else s["condition"]
        spec = {"stage": stage, "kind": "train", "corpus": s["corpus"], "encoder": s["encoder"], "hf_id": enc["hf_id"],
                "condition": condition, "placement": s.get("placement", "prefix"), "head": s.get("head", "cls"),
                "text_transform": s.get("text_transform"), "transform_eval": bool(s.get("transform_eval", False)),
                "fewshot_k": s.get("fewshot_k"), "peft_method": peft_method, "rebalance": s.get("rebalance"),
                "seed": int(s["seed"]), "transfer": bool(grid["transfer"]), "precision": enc.get("precision", PRECISION)}
        if "repeat" in s:
            spec["repeat"] = int(s["repeat"])            # identical configuration trained again (distinct spec_id)
        if grid["recipe"] == "grid":
            spec["lr"], spec["epochs"] = float(s["lr"]), int(s["epochs"])
        else:  # tuned
            if tuned is None or s["encoder"] not in tuned:
                src = next(st for st, encs in RECIPE_SELECTION_STAGES.items() if s["encoder"] in encs)
                raise RuntimeError(f"Stage '{stage}' needs the tuned recipe for '{s['encoder']}', but {TUNED_RECIPES_PATH} "
                                   f"does not contain it. Run STAGE='{src}' to completion first (Cells 1-11).")
            spec["lr"], spec["epochs"] = float(tuned[s["encoder"]]["lr"]), int(tuned[s["encoder"]]["epochs"])
        if peft_method and peft_method.startswith("lora"):
            spec["lr"] = LORA_LR
        elif peft_method == "softprompt_frozen":
            spec["lr"] = SOFT_PROMPT_LR_FROZEN
        elif peft_method == "softprompt_full":
            spec["lr"] = SOFT_PROMPT_LR_FULL
        if grid.get("epochs_override"):
            spec["epochs"] = int(grid["epochs_override"])
        ov = CORPUS_OVERRIDES.get(spec["corpus"], {})
        spec["epochs"] = int(ov.get("epochs", spec["epochs"]))
        spec["patience"] = int(ov.get("patience", EARLY_STOP_PATIENCE))
        if spec["condition"] == "none" and spec["placement"] != "prefix":
            continue                                    # placement is meaningless without a prompt
        specs.append(_with_id(spec))
    assert len({s["spec_id"] for s in specs}) == len(specs), "duplicate run specifications in the grid"
    return specs

def run_prefix(spec: dict) -> str:
    return f"{EXPERIMENT_NAME}/{spec['stage']}/runs/{spec['spec_id']}"

def restore_checkpoint_status(specs: list) -> dict:
    status = {}
    for s in specs:
        rp = run_prefix(s)
        if f"{rp}/DONE" in HUB_FILES:
            status[s["spec_id"]] = "done"
        elif f"{rp}/checkpoints/latest.json" in HUB_FILES:
            status[s["spec_id"]] = "resumable"
        else:
            status[s["spec_id"]] = "new"
    return status

TUNED_RECIPES = load_tuned_recipes()
SPECS = expand_stage(STAGE, TUNED_RECIPES) if STAGE != "analysis" else []
RUN_STATUS = restore_checkpoint_status(SPECS)
counts = {k: sum(v == k for v in RUN_STATUS.values()) for k in ("done", "resumable", "new")}
logger.info(f"Stage '{STAGE}': {len(SPECS)} runs | done {counts['done']} | resumable {counts['resumable']} | new {counts['new']}")
if TUNED_RECIPES:
    logger.info(f"Tuned recipes: {TUNED_RECIPES}")
