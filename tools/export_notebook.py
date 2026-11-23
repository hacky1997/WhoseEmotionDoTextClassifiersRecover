"""Export the archived notebook into the step files under src/whose_emotion/steps/.

The archived notebook in notebooks/archive/ is the exact code that produced the paper.
This script copies each code cell into its own step file and applies the small patches
listed in PATCHES, which let the pipeline run outside Colab and Kaggle.

Usage:
    python tools/export_notebook.py           # write the step files
    python tools/export_notebook.py --check   # fail if the step files differ from the notebook

The script uses only the Python standard library.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "archive" / "notebook_v8_reader_perspective.ipynb"
STEPS_DIR = ROOT / "src" / "whose_emotion" / "steps"

# Notebook cell index -> step file name. The order is the execution order.
STEPS = [
    (2, "step01_config"),
    (4, "step02_environment"),
    (6, "step03_dependencies"),
    (8, "step04_storage"),
    (10, "step05_checkpoints"),
    (12, "step06_data"),
    (14, "step07_models"),
    (16, "step08_training"),
    (17, "step08c_llm_transfer"),
    (19, "step09_evaluation"),
    (21, "step09b_aggregation"),
    (23, "step09c_robustness"),
    (24, "step09d_controls"),
    (26, "step09e_reader_perspective"),
    (28, "step10_visualization"),
    (30, "step11_finalize"),
]

# Each patch replaces one exact piece of notebook code. docs/CHANGES_FROM_NOTEBOOK.md
# explains why each one is needed. None of them changes what a run computes.
PATCHES = {
    "step01_config": [
        (
            'STAGE = "llm_transfer"\n',
            'STAGE = os.environ.get("WHOSE_EMOTION_STAGE", "llm_transfer")   # set by run.py --stage\n',
        ),
        (
            'WORK_DIRS = {"colab": "/content/work", "kaggle": "/kaggle/working/work"}',
            'WORK_DIRS = {"colab": "/content/work", "kaggle": "/kaggle/working/work",\n'
            '             "local": os.environ.get("WHOSE_EMOTION_WORK_DIR", "work")}',
        ),
    ],
    "step02_environment": [
        (
            '    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):\n',
            '    if os.environ.get("WHOSE_EMOTION_ENV") == "local":          # set by run.py --local\n'
            '        return "local"\n'
            '    elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):\n',
        ),
        (
            '    if env == "colab":\n        from google.colab import userdata\n',
            '    if env == "local":\n'
            '        getter = os.environ.get\n'
            '        where = "environment variables (for example, export HF_TOKEN=...)"\n'
            '    elif env == "colab":\n        from google.colab import userdata\n',
        ),
    ],
}

HEADER = (
    "# Step file generated from the archived notebook by tools/export_notebook.py.\n"
    "# Run it through run.py, which executes every step in order in one shared namespace,\n"
    "# as \"Run all\" does in the notebook. Changes from the notebook: docs/CHANGES_FROM_NOTEBOOK.md\n"
    "\n"
)


def build_steps() -> dict:
    """Return {file name: file text} for every step."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    files = {}
    for index, name in STEPS:
        cell = cells[index]
        if cell["cell_type"] != "code":
            raise SystemExit(f"Cell {index} is not a code cell; the notebook layout has changed.")
        source = "".join(cell["source"])
        for old, new in PATCHES.get(name, []):
            if source.count(old) != 1:
                raise SystemExit(f"Patch target not found exactly once in {name}: {old[:60]!r}")
            source = source.replace(old, new)
        files[f"{name}.py"] = HEADER + source.rstrip("\n") + "\n"
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args()

    files = build_steps()
    if args.check:
        stale = [name for name, text in files.items()
                 if not (STEPS_DIR / name).exists()
                 or (STEPS_DIR / name).read_text(encoding="utf-8") != text]
        extra = sorted(p.name for p in STEPS_DIR.glob("step*.py") if p.name not in files)
        if stale or extra:
            print("Step files do not match the notebook.")
            for name in stale:
                print(f"  differs or missing: {name}")
            for name in extra:
                print(f"  not in the notebook: {name}")
            return 1
        print(f"All {len(files)} step files match the archived notebook.")
        return 0

    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (STEPS_DIR / name).write_text(text, encoding="utf-8")
    print(f"Wrote {len(files)} step files to {STEPS_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
