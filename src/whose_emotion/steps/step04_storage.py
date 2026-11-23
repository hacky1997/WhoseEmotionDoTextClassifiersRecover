# Step file generated from the archived notebook by tools/export_notebook.py.
# Run it through run.py, which executes every step in order in one shared namespace,
# as "Run all" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md

# =====================================================================================
# CELL 4 — STORAGE AND REPOSITORY INITIALIZATION
# Hub layout (private dataset repo {HF_USERNAME}/{EXPERIMENT_NAME}):
#   {EXPERIMENT_NAME}/{STAGE}/runs/{spec_id}/{checkpoints,logs,metrics,figures,outputs,metadata}/  + DONE
#   {EXPERIMENT_NAME}/{STAGE}/sessions/{timestamp}_{RUN_ID}/...                                     session log/metadata
#   {EXPERIMENT_NAME}/shared/        cue vocabulary, data manifest, tuned recipes
#   {EXPERIMENT_NAME}/analysis/      aggregated metrics, tables, figures
# RESOLUTION: GodMode names a run directory {timestamp}_{run_id}. A timestamped directory cannot be found again
# by a new session, which breaks resumption across Colab disconnects. Runs are therefore stored under a
# deterministic spec_id (hash of the run specification); the timestamped directory is kept for the session.
# RESOLUTION: every run bundles its artifacts into one folder upload (one Hub commit) instead of one commit per
# file, to stay within Hub commit-rate limits over hundreds of runs.
# =====================================================================================
import json, shutil, time
from pathlib import Path
from huggingface_hub import HfApi
import numpy as np
import torch

SUBDIRS = ["checkpoints", "logs", "metrics", "figures", "outputs", "metadata"]
LOCAL_ROOT = Path(WORK_DIR) / "hub_mirror"
STAGE_PREFIX = f"{EXPERIMENT_NAME}/{STAGE}"
SHARED_PREFIX = f"{EXPERIMENT_NAME}/shared"
ANALYSIS_PREFIX = f"{EXPERIMENT_NAME}/analysis"
SESSION_PREFIX = f"{STAGE_PREFIX}/sessions/{SESSION_TIMESTAMP}_{RUN_ID}"

def upload_with_retry(upload_fn, max_retries: int = 3, base_delay: int = 5):
    for attempt in range(max_retries):
        try:
            return upload_fn()
        except Exception as e:
            wait = base_delay * (2 ** attempt)
            logger.warning(f"Upload attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait}s.")
            time.sleep(wait)
    logger.error("All upload attempts failed. Local artifact preserved. Manual upload required.")
    raise RuntimeError("Persistent upload failure. Aborting to preserve state.")

def local_path(path_in_repo: str) -> Path:
    p = LOCAL_ROOT / path_in_repo
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def hub_upload_file(local: Path, path_in_repo: str) -> None:
    assert Path(local).exists(), f"cannot upload missing file {local}"
    upload_with_retry(lambda: API.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                                              repo_id=HF_REPO_ID, repo_type="dataset"))
    HUB_FILES.add(path_in_repo)
    logger.debug(f"uploaded {path_in_repo}")

def hub_upload_folder(local_dir: Path, path_in_repo: str, message: str) -> None:
    assert Path(local_dir).is_dir(), f"cannot upload missing folder {local_dir}"
    upload_with_retry(lambda: API.upload_folder(folder_path=str(local_dir), path_in_repo=path_in_repo, repo_id=HF_REPO_ID,
                                                repo_type="dataset", commit_message=message))
    for f in Path(local_dir).rglob("*"):
        if f.is_file():
            HUB_FILES.add(f"{path_in_repo}/{f.relative_to(local_dir).as_posix()}")

def hub_download(path_in_repo: str):
    """Return a local path for a Hub file, or None if it does not exist on the Hub."""
    if path_in_repo not in HUB_FILES:
        return None
    try:
        p = API.hf_hub_download(repo_id=HF_REPO_ID, filename=path_in_repo, repo_type="dataset", local_dir=str(LOCAL_ROOT))
        return Path(p)
    except Exception as e:
        logger.error(f"Download failed for {path_in_repo}: {e}")
        raise

def write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if isinstance(o, (np.ndarray, torch.Tensor)) else (o.item() if hasattr(o, "item") else (list(o) if isinstance(o, (set, tuple)) else str(o))))
    shutil.move(str(tmp), str(path))            # atomic rename on POSIX

def refresh_hub_listing() -> set:
    try:
        return set(API.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"))
    except Exception as e:
        logger.error(f"Could not list Hub repository {HF_REPO_ID}: {e}")
        raise

def initialize_storage():
    api = HfApi(token=SECRETS["HF_TOKEN"])
    repo_id = f"{SECRETS['HF_USERNAME']}/{EXPERIMENT_NAME}"
    upload_with_retry(lambda: api.create_repo(repo_id=repo_id, repo_type="dataset", private=HF_REPO_PRIVATE, exist_ok=True))
    logger.info(f"Repository confirmed: {repo_id}")
    return api, repo_id

API, HF_REPO_ID = initialize_storage()
HUB_FILES = refresh_hub_listing()
logger.info(f"Hub repository holds {len(HUB_FILES)} files")

session_dir = LOCAL_ROOT / SESSION_PREFIX
for sub in SUBDIRS:
    (session_dir / sub).mkdir(parents=True, exist_ok=True)
    (session_dir / sub / ".gitkeep").touch()
SESSION_METADATA = {"run_id": RUN_ID, "experiment_name": EXPERIMENT_NAME, "model_name": MODEL_NAME, "stage": STAGE,
                    "timestamp_start": datetime.now().isoformat(), "timestamp_end": None, "status": "running",
                    "config": CONFIG_SNAPSHOT, "library_versions": LIB_VERSIONS, "environment": ENV,
                    "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "latest_checkpoint_step": None, "completed": False}
write_json(session_dir / "metadata" / "run_metadata.json", SESSION_METADATA)
hub_upload_folder(session_dir, SESSION_PREFIX, f"session {SESSION_TIMESTAMP}_{RUN_ID} start ({STAGE})")
logger.info(f"Session storage initialized at {HF_REPO_ID}/{SESSION_PREFIX}")
