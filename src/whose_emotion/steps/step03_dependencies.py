# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 3 — DEPENDENCY INSTALLATION
# Pinned minimum versions. If an already-imported package gets upgraded, the runtime must restart.
# =====================================================================================
import importlib.metadata, subprocess

REQUIRED = {
    "torch": "2.0.0", "transformers": "4.48.0", "datasets": "2.14.0", "huggingface_hub": "0.20.0",
    "matplotlib": "3.7.0", "numpy": "1.24.0", "scikit-learn": "1.3.0", "accelerate": "0.24.0", "pyyaml": "6.0",
    # study-specific additions
    "pandas": "2.0.0", "pyarrow": "12.0.0", "scipy": "1.10.0", "statsmodels": "0.14.0",
    "sentencepiece": "0.1.99", "protobuf": "3.20.0", "safetensors": "0.4.0", "packaging": "23.0", "peft": "0.10.0",
}
if STAGE == "llm":
    REQUIRED["openai"] = "1.40.0"
    if torch.cuda.is_available() and LLM_LOAD_4BIT:
        REQUIRED["bitsandbytes"] = "0.43.0"
if STAGE == "llm_transfer":
    REQUIRED["openai"] = "1.40.0"
# ASSUMPTION: transformers >= 4.48 is required because ModernBERT (convergence stage) was added in 4.48.
IMPORT_NAMES = {"scikit-learn": "sklearn", "pyyaml": "yaml", "huggingface_hub": "huggingface_hub", "protobuf": "google.protobuf"}

def installed_version(pkg: str):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None

def install_dependencies() -> None:
    from packaging import version as V
    before = {p: installed_version(p) for p in REQUIRED}
    need = [f"{p}>={v}" for p, v in REQUIRED.items() if before[p] is None or V.parse(before[p]) < V.parse(v)]
    if need:
        logger.info(f"Installing/upgrading: {need}")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pip install failed: {e.stderr[-1500:]}") from e
    else:
        logger.info("All required packages already satisfy the minimum versions.")

        # peft refuses LoRA while an old torchao (< 0.16) is installed; the notebook never uses torchao, so remove it
    _tao = installed_version("torchao")
    if _tao is not None and V.parse(_tao) < V.parse("0.16.0"):
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
        logger.info(f"Removed torchao {_tao}: peft needs torchao >= 0.16 or none; this notebook does not use torchao")
    after = {p: installed_version(p) for p in REQUIRED}
    for pkg, min_ver in REQUIRED.items():
        assert after[pkg] is not None, f"{pkg} is not installed after pip install. Aborting."
        assert V.parse(after[pkg]) >= V.parse(min_ver), f"{pkg} version {after[pkg]} is below required {min_ver}. Aborting."
    changed = [p for p in REQUIRED if before[p] is not None and before[p] != after[p]
               and IMPORT_NAMES.get(p, p).split(".")[0] in sys.modules]
    if changed:
        raise RuntimeError(f"Upgraded packages that were already imported: {changed}. "
                           "Restart the runtime (Runtime -> Restart session) and run the notebook again from Cell 1.")
    for pkg in REQUIRED:
        __import__(IMPORT_NAMES.get(pkg, pkg).replace("-", "_"))
    logger.info("Library versions: " + ", ".join(f"{p}={after[p]}" for p in REQUIRED))
    return after

LIB_VERSIONS = install_dependencies()
