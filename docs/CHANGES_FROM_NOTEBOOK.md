# Changes from the archived notebook

The notebook in `notebooks/archive/` is the exact code that produced the results in the paper. The step files in `src/whose_emotion/steps/` were exported from it by `tools/export_notebook.py`.

Each step file is the code of one notebook cell. Four small patches were applied so that the pipeline can run from the command line and outside Colab and Kaggle. None of them changes what a run computes.

## The four patches

**1. The stage can be set from the command line** (`step01_config.py`)

The notebook sets the stage with the line `STAGE = "llm_transfer"`. The step file reads it from the environment variable `WHOSE_EMOTION_STAGE`, which `run.py --stage` sets. If the variable is not set, the value is `"llm_transfer"`, as in the notebook.

**2. A working folder for local runs** (`step01_config.py`)

The notebook defines working folders only for Colab and Kaggle. The step file adds a third entry, `"local"`, which uses the folder given by `--work-dir`. The default is `./work`.

**3. A local environment** (`step02_environment.py`)

The notebook stops with an error outside Colab and Kaggle. The step file first checks the environment variable `WHOSE_EMOTION_ENV`. When `run.py --local` sets it to `"local"`, the pipeline runs on the current machine. Colab and Kaggle are detected as before.

**4. Secrets from environment variables** (`step02_environment.py`)

In a local run, the secrets `HF_TOKEN`, `HF_USERNAME`, and `OPENAI_API_KEY` are read from environment variables. In Colab and Kaggle they are read from the platform's secret store, as before.

## Checking that the code still matches the notebook

Run this command from the repository folder:

```
python tools/export_notebook.py --check
```

It exports the notebook again, applies the same four patches, and compares the result with the step files. It fails if any step file differs. The GitHub workflow in `.github/workflows/check.yml` runs the same check on every push.

## What was not changed

The training, evaluation, and analysis code is unchanged. Code comments are kept as they were in the notebook, including references to notebook cell numbers.
