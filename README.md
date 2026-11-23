# Whose Emotion Do Text Classifiers Recover?

This repository contains the code for the paper *Whose Emotion Do Text Classifiers Recover? Explicit Cues, Reader-Inferred Emotion, and Cross-Corpus Transfer* by Sayak Naskar. The paper is under review at IEEE Transactions on Affective Computing.

The code runs every part of the study. It audits the corpora, selects the training recipes, trains all 1,740 models, queries the generative models, runs the statistical analyses, and writes the tables and figures of the paper and its supplement.

The outputs of every run are on the Hugging Face Hub at <https://huggingface.co/datasets/hacky1997/emotion_explicit_cue>.

## Repository layout

```
.
├── run.py                        Command-line entry point
├── src/whose_emotion/
│   ├── pipeline.py               Runs the steps in order for one stage
│   ├── cli.py                    Command-line options
│   └── steps/                    The pipeline, one file per step
├── notebooks/
│   ├── run_stage.ipynb           Runs one stage in Colab or Kaggle
│   └── archive/                  The original notebook that produced the paper
├── tools/export_notebook.py      Exports the notebook into steps/ and checks that they match
├── docs/CHANGES_FROM_NOTEBOOK.md The four small changes from the notebook
├── requirements.txt              Library versions of the final session
├── pyproject.toml                Package definition
└── .env.example                  Template for the secrets of a local run
```

## How the code is organized

The study was developed as one notebook. Each cell of that notebook is now a step file in `src/whose_emotion/steps/`. The steps share their variables, as notebook cells do, so `pipeline.py` runs them in order in one namespace. This keeps the released code identical to the code that produced the paper.

The steps are:

| Step | What it does |
|---|---|
| `step01_config.py` | Sets every parameter of the study and the grid of runs for each stage |
| `step02_environment.py` | Detects Colab, Kaggle, or a local machine, loads the secrets, and selects the device |
| `step03_dependencies.py` | Checks the installed libraries and installs missing ones |
| `step04_storage.py` | Connects to the Hugging Face Hub repository that stores the outputs |
| `step05_checkpoints.py` | Finds finished and interrupted runs so that a stage can resume |
| `step06_data.py` | Downloads each corpus, checks its checksum, and builds the splits |
| `step07_models.py` | Builds the encoders, heads, and optimizers |
| `step08_training.py` | Trains the runs of the current stage |
| `step08c_llm_transfer.py` | Queries GPT-4o zero-shot on the transfer targets and reader-annotated corpora |
| `step09_evaluation.py` | Computes the metrics of each run |
| `step09b_aggregation.py` | Aggregates all runs and runs the main statistical tests |
| `step09c_robustness.py` | Runs the robustness checks |
| `step09d_controls.py` | Runs the controls without SetFit/emotion training |
| `step09e_reader_perspective.py` | Runs the reader analyses on crowd-enVENT and enISEAR |
| `step10_visualization.py` | Draws the figures |
| `step11_finalize.py` | Uploads the final outputs and marks the session as complete |

## Running a stage

The study needs a GPU. All reported runs used an NVIDIA Tesla T4.

### On your own machine

1. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
2. Set the secrets. `HF_TOKEN` needs write access, because each run uploads its outputs to the Hub. Copy `.env.example` to `.env`, fill it in, and load it:
   ```
   set -a; source .env; set +a
   ```
3. Run a stage:
   ```
   python run.py --stage audit --local
   ```

Use `--work-dir` to choose the working folder. The default is `./work`.

You can also install the package with `pip install -e .` and then run `whose-emotion --stage audit --local`.

### In Colab or Kaggle

Open `notebooks/run_stage.ipynb`, add the secrets to the platform's secret store, set `STAGE`, and run all cells. The notebook clones this repository and runs the stage in the notebook's own namespace, so the platform's secrets work as usual.

The original notebook in `notebooks/archive/` also still runs on its own. Set `STAGE` in its first code cell and run all cells.

### Resuming

If a session stops, run the same stage again. Finished runs are skipped, and interrupted runs resume from their last saved state.

## Stages

Run the core stages in the order listed. The supplementary stages support the robustness checks in the supplement. The `analysis` stage only aggregates results that are already on the Hub.

| Stage | Tier | Runs | Used for |
|---|---|---|---|
| `audit` | core | 1 | Corpus audit: TF-IDF baselines, *feel* probe, duplicates, explicit shares |
| `tuning` | core | 240 | Recipe selection for each encoder |
| `main` | core | 350 | Main grid on SetFit/emotion, with transfer to four target corpora |
| `intra` | core | 100 | TweetEval and EmpatheticDialogues |
| `target` | core | 150 | Target-trained reference models on GoEmotions, MELD, and ISEAR |
| `shortcut` | core | 120 | Cue masking compared with random masking during training |
| `cev_source`, `cev_masked`, `cev_target`, `cev_third` | core | 50, 20, 25, 25 | crowd-enVENT reader comparison, minimal pairs, and the GoEmotions-trained control |
| `eni_source`, `eni_third` | core | 50, 25 | enISEAR reader comparison |
| `llm_transfer` | core | API | Zero-shot GPT-4o on the transfer targets, crowd-enVENT, and enISEAR |
| `lr_sweep`, `convergence` | supp | 48, 80 | Convergence check for DeBERTa-v3-base and ModernBERT-base |
| `llm` | supp | API and 4-bit | GPT-4o, Llama-3.1-8B-Instruct, and Ministral-8B-Instruct-2410 on the source test set |
| `peft`, `rebalance`, `placement`, `scaling` | supp | 30, 20, 20, 27 | Parameter-efficient adaptation, class rebalancing, prefix placement, and larger training sets |
| `prompt_map`, `prompt_map_full` | supp | 320, 40 | Few-shot data regimes with classification and verbalizer heads |
| `analysis` | none | 0 | Aggregation, statistical tests, tables, and figures |

The `llm` and `llm_transfer` stages need the secret `OPENAI_API_KEY`. The `llm` stage also needs accepted licences for Llama-3.1-8B-Instruct and Ministral-8B-Instruct-2410 on the account of `HF_TOKEN`.

Supplementary Table S30 of the paper lists the runs of each stage, and Table S31 lists the runs that were trained twice.

## Checking the code against the notebook

```
python tools/export_notebook.py --check
```

This command confirms that every step file matches the archived notebook, apart from the four changes described in `docs/CHANGES_FROM_NOTEBOOK.md`. The same check runs on GitHub after every push.

## Data

This repository does not include the corpora. The code downloads each one from its public source and checks it against the SHA-256 checksums in `shared/data_manifest.json` on the Hub. Supplementary Table S32 of the paper lists the same checksums.

- SetFit/emotion, a copy of the corpus by Saravia et al. (2018)
- GoEmotions by Demszky et al. (2020)
- MELD by Poria et al. (2019)
- ISEAR by Scherer and Wallbott (1994), from the Hugging Face copy `gsri-18/ISEAR-dataset-complete`
- EmpatheticDialogues by Rashkin et al. (2019)
- TweetEval emotion by Barbieri et al. (2020), based on Mohammad et al. (2018)
- crowd-enVENT by Troiano, Oberländer, and Klinger (2023)
- enISEAR by Troiano, Padó, and Klinger (2019)

Each corpus is subject to the terms set by its creators.

## Citation

Citation details will be added when the paper is published. Until then, please cite the paper by its title and author.
