# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 2 — ENVIRONMENT DETECTION AND SECRET LOADING
# Required secrets: HF_TOKEN (write access), HF_USERNAME; OPENAI_API_KEY only for STAGE='llm'.
# Colab: key icon; Kaggle: Add-ons -> Secrets. Llama-3.1 is gated: accept its licence on the Hub with the HF_TOKEN account.
# =====================================================================================
import importlib.util

def detect_environment() -> str:
    if os.environ.get("WHOSE_EMOTION_ENV") == "local":          # set by run.py --local
        return "local"
    elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return "kaggle"
    elif os.environ.get("COLAB_RELEASE_TAG") or importlib.util.find_spec("google.colab") is not None:
        return "colab"
    else:
        raise RuntimeError("Unsupported environment. This notebook must run in Google Colab or Kaggle. "
                           "Local and unknown environments are not supported.")

def load_secrets(env: str) -> dict:
    required_keys = ["HF_TOKEN", "HF_USERNAME"]
    if (STAGE == "llm" and any(m["backend"] == "openai" for m in LLM_MODELS.values())) or STAGE == "llm_transfer":
        required_keys.append("OPENAI_API_KEY")          # only the LLM stage calls the OpenAI API
    secrets = {}
    if env == "local":
        getter = os.environ.get
        where = "environment variables (for example, export HF_TOKEN=...)"
    elif env == "colab":
        from google.colab import userdata
        getter = userdata.get
        where = "Colab Secrets (key icon in the left panel)"
    else:
        from kaggle_secrets import UserSecretsClient
        getter = UserSecretsClient().get_secret
        where = "Kaggle -> Add-ons -> Secrets"
    for key in required_keys:
        try:
            val = getter(key)
        except Exception as e:                      # platform raises if the secret does not exist
            raise ValueError(f"Required secret '{key}' is missing or not granted to this notebook. Add it via {where}.") from e
        if not val:
            raise ValueError(f"Required secret '{key}' is empty. Add it via {where}.")
        secrets[key] = val
    logger.info(f"Secrets loaded: {list(secrets.keys())}")      # keys only, never values
    return secrets

def setup_environment() -> tuple:
    env = detect_environment()
    logger.info(f"Environment detected: {env}")
    assert sys.version_info >= (3, 10), f"Python >= 3.10 required, found {sys.version.split()[0]}"
    secrets = load_secrets(env)
    os.environ["HF_TOKEN"] = secrets["HF_TOKEN"]
    if REQUIRE_GPU and not torch.cuda.is_available():
        raise RuntimeError("GPU required but not available. Enable a GPU runtime (REQUIRE_GPU=True in Cell 1).")
    elif not torch.cuda.is_available():
        logger.warning("GPU not available. Running on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {props.total_memory / 1e9:.1f} GB | "
                    f"compute capability {props.major}.{props.minor}")
    work_dir = WORK_DIRS[env]
    os.makedirs(work_dir, exist_ok=True)
    return env, secrets, device, work_dir

ENV, SECRETS, DEVICE, WORK_DIR = setup_environment()
