"""Run the study pipeline for one stage.

The pipeline is the archived notebook split into step files. The steps share their
variables, as notebook cells do, so they are executed in order in one namespace.
This keeps the released code identical to the code that produced the paper.
"""

import os
import sys
from pathlib import Path

STEPS_DIR = Path(__file__).resolve().parent / "steps"

STEPS = [
    "step01_config",
    "step02_environment",
    "step03_dependencies",
    "step04_storage",
    "step05_checkpoints",
    "step06_data",
    "step07_models",
    "step08_training",
    "step08c_llm_transfer",
    "step09_evaluation",
    "step09b_aggregation",
    "step09c_robustness",
    "step09d_controls",
    "step09e_reader_perspective",
    "step10_visualization",
    "step11_finalize",
]


def run(stage: str, local: bool = False, work_dir: str | None = None, namespace: dict | None = None) -> dict:
    """Run every step for one stage and return the shared namespace.

    stage:     a stage name such as "audit", "main", or "analysis" (see README.md).
    local:     run outside Colab and Kaggle; secrets are read from environment variables.
    work_dir:  local working folder (local runs only; the default is ./work).
    namespace: where the steps run. The default is the __main__ module, which matches a
               notebook kernel. In Colab, call run() from a cell so that Colab secrets work.
    """
    os.environ["WHOSE_EMOTION_STAGE"] = stage
    if local:
        os.environ["WHOSE_EMOTION_ENV"] = "local"
        if work_dir:
            os.environ["WHOSE_EMOTION_WORK_DIR"] = str(work_dir)

    if namespace is None:
        namespace = sys.modules["__main__"].__dict__

    for name in STEPS:
        path = STEPS_DIR / f"{name}.py"
        code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
        exec(code, namespace)
    return namespace
