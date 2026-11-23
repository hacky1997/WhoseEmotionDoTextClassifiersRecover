# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 1 — CONFIGURATION
# Every tunable parameter of the study lives here. No other cell defines parameters.
# Workflow: set STAGE, run all cells top to bottom; rerun the notebook (same STAGE) after a
# disconnect, and it resumes (completed runs are skipped, interrupted runs resume from latest.json).
# Stage order: audit -> tuning -> lr_sweep -> main -> intra -> target -> shortcut -> llm -> peft -> rebalance ->
# placement -> convergence -> scaling -> prompt_map; "analysis" only re-aggregates what is already on the Hub.
# v8 (reader-perspective revision): eni_source -> eni_third -> llm_transfer (adds enISEAR); deposit Amendment 2 first.
# =====================================================================================
import logging, os, random, sys, uuid, json
from datetime import datetime

# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# logger = logging.getLogger("emotion_cue")
logger = logging.getLogger("emotion_cue")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(_handler)
logger.propagate = False

EXPERIMENT_NAME = "emotion_explicit_cue"
# STAGE = "audit"                       # see STAGE_GRIDS below and the notebook introduction for the order
STAGE = os.environ.get("WHOSE_EMOTION_STAGE", "llm_transfer")   # set by run.py --stage
MODEL_NAME = "multi_encoder"          # run-directory component; the stage grid defines the actual encoders
RUN_ID = uuid.uuid4().hex[:6]         # identifies this notebook session
SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---- reproducibility ---------------------------------------------------------------
SEED = 42                             # session-level seed (data subsampling, random prefix); each run seeds itself
SEED_LIST = [42, 2024, 7, 100, 999, 1, 2, 3, 4, 5]   # evaluation seeds (main grid uses all ten)
TUNING_SEEDS = [11, 12, 13]           # disjoint from evaluation seeds
CONVERGENCE_SEEDS = list(range(101, 121))
MULTISEED_MODE = True

# ---- training ----------------------------------------------------------------------
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5                  # default only; tuned per encoder in the tuning stage
NUM_EPOCHS = 5                        # default only; tuned per encoder
MAX_SEQ_LEN = 128
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 2               # epochs without validation-loss improvement
PRECISION = "auto"                    # auto -> bf16 on compute capability >= 8.0, fp16 otherwise; "fp32" forces fp32
FEWSHOT_MIN_STEPS = 300               # few-shot runs never train for fewer optimizer steps than this
CHECKPOINT_EVERY_N_STEPS = 500
UPLOAD_MIDRUN_CHECKPOINTS = "auto"    # "auto": upload mid-run checkpoints only for runs longer than the threshold below
AUTO_UPLOAD_MIN_STEPS = 5000
REQUIRE_GPU = True
LOG_EVERY_N_STEPS = 100

# ---- encoders ------------------------------------------------------------------------
ENCODERS = {
    "distilbert":    {"hf_id": "distilbert-base-uncased"},
    "distilroberta": {"hf_id": "distilroberta-base"},
    "roberta":       {"hf_id": "roberta-base"},
    "albert":        {"hf_id": "albert-base-v2"},
    "electra":       {"hf_id": "google/electra-base-discriminator"},
    "deberta":       {"hf_id": "microsoft/deberta-v3-base", "precision": "fp32", "lr": 1e-5, "epochs": 5},
    "modernbert":    {"hf_id": "answerdotai/ModernBERT-base", "precision": "fp32", "lr": 1e-5, "epochs": 5},
}
MAIN_ENCODERS = ["distilbert", "distilroberta", "roberta", "albert", "electra"]
TUNING_GRID = {"lr": [1e-5, 2e-5, 3e-5, 5e-5], "epochs": [5, 10]}
# RESOLUTION: tuning covers BOTH no prompt and template A and selects by their mean validation accuracy. Tuning on
# the no-prompt condition alone would favour that condition and bias the prompt comparison toward a null result.
TUNING_CONDITIONS = ["none", "A"]
LR_SWEEP_GRID = {"lr": [5e-6, 1e-5, 2e-5, 3e-5], "epochs": [5]}     # DeBERTa-v3 / ModernBERT (FP32)

# ---- parameter-efficient adaptation (v6 settings) ------------------------------------------
PEFT_METHODS = {   # method -> text-prompt condition used with it (as in the v6 experiments)
    "lora_r4": "A", "lora_r8": "A", "lora_r16": "A", "lora_r32": "A", "softprompt_frozen": "none", "softprompt_full": "none"}
LORA_LR = 2e-4
LORA_ALPHA = 16                       # v6 used alpha = 16 for every rank (cells 34 and 35)
LORA_DROPOUT = 0.1
LORA_TARGETS = {"roberta": ["query", "value"], "bert": ["query", "value"], "electra": ["query", "value"], "albert": ["query", "value"],
                "distilbert": ["q_lin", "v_lin"], "deberta-v2": ["query_proj", "value_proj"], "modernbert": ["Wqkv"]}
SOFT_PROMPT_TOKENS = 20
SOFT_PROMPT_LR_FROZEN = 1e-3          # v6: backbone frozen, prompt vectors and head trained
SOFT_PROMPT_LR_FULL = 2e-5            # v6: prompt vectors trained together with the backbone

# ---- class imbalance ---------------------------------------------------------------------------
REBALANCE_METHODS = ["weighted", "oversample"]   # weighted: inverse-frequency class weights; oversample: every class to the majority count

# ---- generative LLM baselines --------------------------------------------------------------------
LLM_MODELS = {"llama31_8b": {"backend": "hf", "hf_id": "meta-llama/Llama-3.1-8B-Instruct"},
              "ministral_8b": {"backend": "hf", "hf_id": "mistralai/Ministral-8B-Instruct-2410"},
              "gpt4o": {"backend": "openai", "api_model": "gpt-4o-2024-08-06"}}   # dated snapshot, recorded per response
LLM_SHOTS = [0, 6]                    # 6 = one demonstration per class from the training split
LLM_DEMO_SEED = 42
LLM_MAX_NEW_TOKENS = 12
LLM_LOAD_4BIT = True                  # NF4 on GPU; ignored on CPU
LLM_CHUNK = 250                       # items per resumable chunk
LLM_SYSTEM = ("You classify the emotion of a text. Answer with exactly one word from this list: "
              "sadness, joy, love, anger, fear, surprise.")

# ---- calibration ---------------------------------------------------------------------------------
CALIBRATION_BINS = 15
TEMPERATURE_BOUNDS = (0.05, 20.0)

# ---- prompts ---------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "none": "",
    "A": "The emotion of the following sentence is:",
    "B": "Classify the emotion expressed:",
    "C": "This text expresses the following emotion:",
    "D": "What emotion does this text express?",
    "misleading": "The emotion of the following sentence is joy:",
    "random": None,                   # random tokens, same token count as template A, built per tokenizer
}
RANDOM_PREFIX_SEED = 1234
VERBALIZER_CANDIDATES = {"sadness": ["sad", "sadness"], "joy": ["joy", "happy"], "love": ["love"],
                         "anger": ["anger", "angry"], "fear": ["fear", "afraid"], "surprise": ["surprise", "surprised"]}

# ---- explicit-cue analysis (frozen by docs/PREREGISTRATION.md) ---------------------------
CUE_MIN_COUNT = 5
CUE_MIN_PURITY = 0.6
EXPECTED_CUE_VOCAB_SIZE = 358         # value from the pre-registered pilot; Cell 6 aborts if the data changed
INTENSIFIERS = ("so very really a bit little kind of quite too more pretty extremely rather somewhat "
                "completely totally incredibly slightly just like").split()
CONVERGED_VAL_ACC = 0.90
N_BOOTSTRAP = 5000
N_PERMUTATIONS_MAX_EXACT = 16

# ---- data ------------------------------------------------------------------------------
TRANSFER_TARGETS = ["goemotions", "meld", "isear", "ed"]
TARGET_CORPORA = ["meld_target", "isear_target", "goemotions_target"]    # target-trained upper bounds (ED: intra stage)
ISEAR_SPLIT_SEED = 42
CORPUS_OVERRIDES = {"meld_target": {"epochs": 10, "patience": 3}}   # v6 trained the MELD upper bound this way (paper Table A-II)                 # stratified 70/15/15, reproduces the v6 split (3,030 / 649 / 650)
SCALING_SIZES = [32000, 64000, 128000]
SCALING_POOL_SEED = 0
# FEWSHOT_KS = [16, 64, 256, 1024]
FEWSHOT_KS = [16, 64, 256, 512]       # 512 = largest balanced size (surprise has 572 training texts)
ISEAR_HF_REPO = "gsri-18/ISEAR-dataset-complete"
# ISEAR integrity: the converted parquet file is pinned by its SHA-256 in shared/data_manifest.json (Cell 6).

# ---- crowd-enVENT (Troiano, Oberlaender & Klinger 2023, Computational Linguistics 49(1)) -----------------
CEV_URL = "https://www.romanklinger.de/data-sets/crowd-enVent2023.zip"
CEV_TO_SETFIT = {"sadness": 0, "joy": 1, "anger": 3, "fear": 4, "surprise": 5}   # the five classes shared with SetFit
CEV_APPRAISALS = ["suddenness", "familiarity", "predict_event", "pleasantness", "unpleasantness", "goal_relevance",
                  "chance_responsblt", "self_responsblt", "other_responsblt", "predict_conseq", "goal_support", "urgency",
                  "self_control", "other_control", "chance_control", "accept_conseq", "standards", "social_norms",
                  "attention", "not_consider", "effort"]
CEV_SPLIT_SEED = 42
CEV_VAL_FRACTION = 0.1
CEV_STAGES = ["cev_source", "cev_masked"]       # SetFit-trained stages that export crowd-enVENT logits
CEV_EXPORT_STAGES = CEV_STAGES + ["cev_third"]   # stages that export crowd-enVENT logits; cev_third is NOT a SetFit source

# ---- enISEAR (Troiano, Pado & Klinger 2019, ACL; ODC-By 1.0) -- v8, reader-perspective revision ------------------------
# Writers completed "I felt [emotion] when/because ..." for a prompted emotion; five other workers read the description
# with the emotion word omitted and chose, among seven emotions, the one the writer most likely felt. The release contains
# only the versions the readers saw. Evaluation only: stages eni_source, eni_third, task enisear of llm_transfer, Cell 9e.
ENI_URL = "https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/emotions/deISEARenISEAR.zip"
ENI_SHA256 = "38aa785caaf668652d9278f583990802d7eccf7569d0a206cabb49f9bbb89861"   # archive as downloaded on 2026-10-25
ENI_EMOTIONS = ["Anger", "Disgust", "Fear", "Guilt", "Joy", "Sadness", "Shame"]       # reader vote columns of enISEAR.tsv
ENI_TO_SETFIT = {"Sadness": 0, "Joy": 1, "Anger": 3, "Fear": 4}                       # the four classes shared with SetFit
# RESOLUTION: the omitted emotion word is marked "..." in 407 and "\u2026" (one character) in 166 of the 572 four-class
# descriptions (one has both), and the second form occurs only in Fear and Anger descriptions. A marker that differs by
# class would be an artifact, so every "\u2026" is replaced by "..." (the marker of crowd-enVENT) before any model
# sees the text.
ENI_MARKER = "..."
ENI_EXPORT_STAGES = ["eni_source", "eni_third"]          # trained stages that export enISEAR logits
P_N_BOOT = 2000                                          # bootstrap resamples, reader-perspective analyses (Cell 9e)
P_SEED = 20261025

# ---- storage -----------------------------------------------------------------------------
WORK_DIRS = {"colab": "/content/work", "kaggle": "/kaggle/working/work",
             "local": os.environ.get("WHOSE_EMOTION_WORK_DIR", "work")}
HF_REPO_PRIVATE = True
SAVE_FINAL_MODEL_FOR = {"stages": ["main", "shortcut"], "seeds": [42], "conditions": ["none", "A", "misleading"]}

# ---- stage grids ---------------------------------------------------------------------------
# Each stage expands to a list of run specifications in Cell 5. "transfer": evaluate on TRANSFER_TARGETS.
STAGE_GRIDS = {
    # core: needed for the main claims | supp: supplementary analyses (run only if compute allows)
    "audit":       {"kind": "audit", "tier": "core"},
    "tuning":      {"kind": "train", "tier": "core", "corpus": ["setfit"], "encoder": MAIN_ENCODERS, "condition": TUNING_CONDITIONS,
                    "seed": TUNING_SEEDS, "lr": TUNING_GRID["lr"], "epochs": TUNING_GRID["epochs"], "transfer": False, "recipe": "grid"},
    "lr_sweep":    {"kind": "train", "tier": "supp", "corpus": ["setfit"], "encoder": ["deberta", "modernbert"], "condition": TUNING_CONDITIONS,
                    "seed": TUNING_SEEDS, "lr": LR_SWEEP_GRID["lr"], "epochs": LR_SWEEP_GRID["epochs"], "transfer": False, "recipe": "grid"},
    "main":        {"kind": "train", "tier": "core", "corpus": ["setfit"], "encoder": MAIN_ENCODERS,
                    "condition": ["none", "A", "B", "C", "D", "random", "misleading"], "seed": SEED_LIST, "transfer": True, "recipe": "tuned"},
    "intra":       {"kind": "train", "tier": "core", "corpus": ["tweeteval", "ed"], "encoder": MAIN_ENCODERS, "condition": ["none", "A"],
                    "seed": SEED_LIST[:5], "transfer": False, "recipe": "tuned"},
    "target":      {"kind": "train", "tier": "core", "corpus": TARGET_CORPORA, "encoder": MAIN_ENCODERS, "condition": ["none", "A"],
                    "seed": SEED_LIST[:5], "transfer": False, "recipe": "tuned"},
    "shortcut":    {"kind": "train", "tier": "core", "corpus": ["setfit"], "encoder": ["roberta", "distilroberta"], "condition": ["none", "A", "misleading"],
                    "text_transform": ["mask_feel", "mask_random"], "transform_eval": [False, True], "seed": SEED_LIST[:5],
                    "transfer": True, "recipe": "tuned"},
    "llm":         {"kind": "llm", "tier": "supp", "model": list(LLM_MODELS), "shots": LLM_SHOTS},
    "peft":        {"kind": "train", "tier": "supp", "corpus": ["setfit"], "encoder": ["distilroberta"], "peft_method": list(PEFT_METHODS),
                    "seed": SEED_LIST[:5], "transfer": False, "recipe": "tuned"},
    "rebalance":   {"kind": "train", "tier": "supp", "corpus": ["setfit"], "encoder": ["distilroberta", "electra"], "condition": ["A"],
                    "rebalance": REBALANCE_METHODS, "seed": SEED_LIST[:5], "transfer": False, "recipe": "tuned"},
    "placement":   {"kind": "train", "tier": "supp", "corpus": ["setfit"], "encoder": ["roberta", "distilbert"], "condition": ["A"],
                    "placement": ["suffix", "pair"], "seed": SEED_LIST[:5], "transfer": True, "recipe": "tuned"},
    "convergence": {"kind": "train", "tier": "supp", "corpus": ["setfit"], "encoder": ["deberta", "modernbert"], "condition": ["none", "A"],
                    "seed": CONVERGENCE_SEEDS, "transfer": False, "recipe": "tuned"},
    "scaling":     {"kind": "train", "tier": "supp", "corpus": [f"setfit_large_{n}" for n in SCALING_SIZES], "encoder": ["distilroberta"],
                    "condition": ["none", "A", "misleading"], "seed": SEED_LIST[:3], "transfer": True, "recipe": "tuned", "epochs_override": 3},
    "prompt_map":  {"kind": "train", "tier": "supp", "corpus": ["setfit"], "encoder": ["roberta", "distilroberta"],
                    "condition": ["none", "A", "random", "misleading"], "head": ["cls", "verbalizer"], "fewshot_k": FEWSHOT_KS,
                    "seed": SEED_LIST[:5], "transfer": False, "recipe": "tuned"},
    # same configuration and seed trained twice: measures run-to-run variation from GPU nondeterminism (paper RQ3)
    "rerun":       {"kind": "train", "tier": "core", "corpus": ["setfit"], "encoder": ["distilroberta", "roberta", "albert", "electra"],
                    "condition": ["A"], "seed": [42], "repeat": [1, 2], "transfer": False, "recipe": "tuned"},
    # verbalizer head with the full training set (the "Full" row of the data-regime table)
    "prompt_map_full": {"kind": "train", "tier": "supp", "corpus": ["setfit"], "encoder": ["roberta", "distilroberta"],
                    "condition": ["none", "A", "random", "misleading"], "head": ["verbalizer"], "seed": SEED_LIST[:5],
                    "transfer": False, "recipe": "tuned"},
    "analysis":    None,
    # crowd-enVENT analyses: these repeat main/shortcut configurations because finished runs did not export crowd-enVENT
    # logits; used ONLY for the crowd-enVENT analyses and rerun variation, never in the prompt-effect statistics.
    "cev_source":  {"kind": "train", "tier": "core", "corpus": ["setfit"], "encoder": MAIN_ENCODERS, "condition": ["none", "A"],
                    "seed": SEED_LIST[:5], "transfer": False, "recipe": "tuned"},
    "cev_masked":  {"kind": "train", "tier": "core", "corpus": ["setfit"], "encoder": ["roberta", "distilroberta"], "condition": ["none"],
                    "text_transform": ["mask_feel", "mask_random"], "transform_eval": [False], "seed": SEED_LIST[:5],
                    "transfer": False, "recipe": "tuned"},
    "cev_target":  {"kind": "train", "tier": "core", "corpus": ["cev_target"], "encoder": MAIN_ENCODERS, "condition": ["none"],
                    "seed": SEED_LIST[:5], "transfer": False, "recipe": "tuned"},
    # third-corpus control for E1 and second source corpus for transfer (GoEmotions-trained; exports crowd-enVENT logits)
    "cev_third":   {"kind": "train", "tier": "core", "corpus": ["goemotions_target"], "encoder": MAIN_ENCODERS, "condition": ["none"],
                    "seed": SEED_LIST[:5], "transfer": True, "recipe": "tuned"},
    # zero-shot GPT-4o on the transfer targets, crowd-enVENT, and (v8) enISEAR (Cell 8c; no training)
    "llm_transfer": None,
    # v8: enISEAR. Only seed-42 models were saved (SAVE_FINAL_MODEL_FOR), so the configurations of cev_source and cev_third are
    # retrained with the same recipes and seeds and export enISEAR logits; they never enter the prompt-effect statistics.
    "eni_source":  {"kind": "train", "tier": "core", "corpus": ["setfit"], "encoder": MAIN_ENCODERS, "condition": ["none", "A"],
                    "seed": SEED_LIST[:5], "transfer": False, "recipe": "tuned"},
    "eni_third":   {"kind": "train", "tier": "core", "corpus": ["goemotions_target"], "encoder": MAIN_ENCODERS, "condition": ["none"],
                    "seed": SEED_LIST[:5], "transfer": False, "recipe": "tuned"},
}
RECIPE_SELECTION_STAGES = {"tuning": MAIN_ENCODERS, "lr_sweep": ["deberta", "modernbert"]}
VALID_STAGES = list(STAGE_GRIDS)
assert STAGE in VALID_STAGES, f"STAGE must be one of {VALID_STAGES}, got {STAGE!r}"

# ---- reproducibility helper (GodMode standard) --------------------------------------------
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
# CONFIG_SNAPSHOT = {k: v for k, v in dict(globals()).items()
#                    if k.isupper() and isinstance(v, (int, float, str, bool, list, dict, type(None)))}
import re
_NOT_CONFIG = {"SECRETS", "API", "HUB_FILES", "DATA", "DATA_FILES", "EXPLICIT", "CUE_VOCAB", "SPECS", "RUN_STATUS",
               "TUNED_RECIPES", "SESSION_METADATA", "ANALYSIS", "METRICS_DF", "FIGURES", "FAILED_RUNS", "LIB_VERSIONS",
               "CONFIG_SNAPSHOT", "HF_REPO_ID", "WORK_DIR", "DEVICE", "ENV", "CEV", "CEV_DATA", "ENI", "PERSPECTIVE", "X_RESULTS"}
_SECRET_RE = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{20,})")

def _redact(o):
    """Config snapshots never carry credentials, even if a secret ends up in a global by mistake."""
    if isinstance(o, str):
        return _SECRET_RE.sub("<REDACTED>", o)
    if isinstance(o, dict):
        return {k: _redact(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_redact(v) for v in o]
    return o

CONFIG_SNAPSHOT = _redact({k: v for k, v in dict(globals()).items()
                           if k.isupper() and k not in _NOT_CONFIG
                           and isinstance(v, (int, float, str, bool, list, dict, type(None)))})
assert not _SECRET_RE.search(json.dumps(CONFIG_SNAPSHOT, default=str)), "credential found in config snapshot"
logger.info(f"Config ready: experiment={EXPERIMENT_NAME} stage={STAGE} session={SESSION_TIMESTAMP}_{RUN_ID}")
logger.info(f"Seed set to {SEED}")
