# Whose Emotion Do Text Classifiers Recover?

[![check](https://github.com/hacky1997/WhoseEmotionDoTextClassifiersRecover/actions/workflows/check.yml/badge.svg)](https://github.com/hacky1997/WhoseEmotionDoTextClassifiersRecover/actions/workflows/check.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![Run outputs on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20run%20outputs-emotion__explicit__cue-yellow)](https://huggingface.co/datasets/hacky1997/emotion_explicit_cue)
![Paper status](https://img.shields.io/badge/paper-under%20review%20at%20IEEE%20TAC-lightgrey)

This repository contains the code for the paper *Whose Emotion Do Text Classifiers Recover? Explicit Cues, Reader-Inferred Emotion, and Cross-Corpus Transfer*, which is under review at IEEE Transactions on Affective Computing.

## About the study

Emotion corpora label their texts from different perspectives. Some labels come from hashtags, some from readers who infer an emotion, and some from writers who report what they felt. Cross-corpus evaluations usually treat these labels as interchangeable.

The study asks which emotion a text classifier recovers when it is applied outside its training corpus, and what limits its accuracy. Five transformer encoders are trained on hashtag-labelled tweets and evaluated on four target corpora. Their predictions are also compared with those of human readers on two corpora in which five readers inferred each writer's emotion. The comparison classifiers are encoders trained on reader or writer labels, zero-shot GPT-4o, and encoders trained on each target corpus.

The main findings are:

- When the readers' most frequent choice differed from the writer's emotion, 62.4% to 92.9% of classifier errors matched the readers' choice. Under shuffled reader choices, the share was 43.0% to 51.8%.
- Most encoder errors still occurred on texts where readers recovered the writer's emotion.
- All classifiers used explicit emotion words, and implicit items were harder on most targets regardless of the training corpus. Evidence of a shortcut specific to the source corpus appeared only on MELD.
- Within the narrow range observed, neither in-domain accuracy nor model confidence predicted cross-corpus accuracy.

These results suggest that writer-labelled benchmarks may partly penalize classifiers for agreeing with readers.

## Requirements

- Python 3.10 or later.
- An NVIDIA GPU. All reported runs used a Tesla T4, and the 1,740 training runs took about 157 GPU-hours in total.
- A Hugging Face account and an access token with write access. Every run uploads its outputs to a dataset repository in this account.
- An OpenAI API key for the two stages that query GPT-4o. The `llm_transfer` stage sends about 8,900 requests.

## Installation

```
git clone https://github.com/hacky1997/WhoseEmotionDoTextClassifiersRecover.git
cd WhoseEmotionDoTextClassifiersRecover
pip install -r requirements.txt
```

The file `requirements.txt` pins the library versions of the final experiment session. To install the code as a package with a `whose-emotion` command instead, run `pip install -e .`. For the stages that query generative models, run `pip install -e ".[llm]"`.

## Quick start

### On your own machine

1. Copy `.env.example` to `.env` and fill in your secrets.
2. Load the secrets into the shell:
   ```
   set -a; source .env; set +a
   ```
3. Run a stage:
   ```
   python run.py --stage audit --local
   ```

The `--local` option reads the secrets from environment variables. The `--work-dir` option sets the local working folder, which is `./work` by default.

### In Colab or Kaggle

1. Open `notebooks/run_stage.ipynb`.
2. Add `HF_TOKEN` and `HF_USERNAME` to the platform's secret store. Add `OPENAI_API_KEY` as well for the `llm` and `llm_transfer` stages.
3. Select a GPU runtime. In Kaggle, also turn on internet access.
4. Set `STAGE` and run all cells.

The notebook clones this repository and runs the stage inside the notebook, so the platform's secrets work as usual.

### Resuming an interrupted stage

Run the same stage again. Finished runs are skipped, and interrupted runs resume from their last saved state.

## Reproducing the paper

Run the core stages in the order listed in [Stages](#stages), and then run the `analysis` stage. The table below shows which stages each part of the paper depends on.

| Part of the paper | Stages |
|---|---|
| Readers and classifier errors (Sections 6.1 and 6.2, Tables 2 and 3) | `cev_source`, `cev_target`, `cev_third`, `eni_source`, `eni_third`, `llm_transfer` |
| Cross-corpus transfer (Section 6.3, Table 4) | `main`, `target`, `intra`, `cev_third`, `llm_transfer` |
| Explicit cues and shortcut tests (Section 6.4, Table 5, Fig. 2) | `audit`, `shortcut`, `cev_masked`, `target`, `cev_third`, `llm_transfer` |
| Prefix wording (Section 6.5) | `main`, `intra`, `prompt_map`, `prompt_map_full`, `placement`, `scaling` |
| Training setup (Supplementary Section S2) | `tuning`, `lr_sweep`, `convergence` |
| Other methods and generative models (Supplementary Section S4) | `llm`, `peft`, `rebalance` |

The `analysis` stage writes the tables as LaTeX files and the figures as PNG files to the `analysis/` folder of your Hugging Face dataset repository.

### Recomputing the analyses from the released outputs

The outputs of all reported runs are public at [hacky1997/emotion_explicit_cue](https://huggingface.co/datasets/hacky1997/emotion_explicit_cue). The code reads its outputs from the dataset repository `<HF_USERNAME>/emotion_explicit_cue`, so you can recompute every table and figure without training any model:

1. Copy the released dataset into a dataset repository named `emotion_explicit_cue` in your own Hugging Face account. One way is to clone it with `git`, change the remote to your repository, and push.
2. Run `python run.py --stage analysis --local` with your own `HF_TOKEN` and `HF_USERNAME`.

## Stages

Core stages produce the results in the main text. Supplementary stages produce the robustness checks in the supplement.

| Stage | Tier | Training runs | Purpose |
|---|---|---|---|
| `audit` | core | none | Corpus audit: TF-IDF baselines, the *feel* probe, duplicates, and explicit shares |
| `tuning` | core | 240 | Recipe selection for each encoder |
| `main` | core | 350 | Main grid on SetFit/emotion, with transfer to four target corpora |
| `intra` | core | 100 | TweetEval and EmpatheticDialogues |
| `target` | core | 150 | Target-trained reference models on GoEmotions, MELD, and ISEAR |
| `shortcut` | core | 120 | Cue masking compared with random masking during training |
| `cev_source`, `cev_masked`, `cev_target`, `cev_third` | core | 50, 20, 25, 25 | crowd-enVENT reader comparison, minimal pairs, and the GoEmotions-trained control |
| `eni_source`, `eni_third` | core | 50, 25 | enISEAR reader comparison |
| `llm_transfer` | core | none (API) | Zero-shot GPT-4o on the transfer targets, crowd-enVENT, and enISEAR |
| `lr_sweep`, `convergence` | supplementary | 48, 80 | Convergence check for DeBERTa-v3-base and ModernBERT-base |
| `llm` | supplementary | none | GPT-4o, Llama-3.1-8B-Instruct, and Ministral-8B-Instruct-2410 on the source test set |
| `peft`, `rebalance`, `placement`, `scaling` | supplementary | 30, 20, 20, 27 | Parameter-efficient adaptation, class rebalancing, prefix placement, and larger training sets |
| `prompt_map`, `prompt_map_full` | supplementary | 320, 40 | Few-shot data regimes with classification and verbalizer heads |
| `analysis` | | none | Aggregation, statistical tests, tables, and figures |

The `llm` stage also needs accepted licences for Llama-3.1-8B-Instruct and Ministral-8B-Instruct-2410 on the Hugging Face account of `HF_TOKEN`. Supplementary Table S30 of the paper lists the runs of each stage.

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
├── docs/CHANGES_FROM_NOTEBOOK.md The four changes made to the notebook code
├── requirements.txt              Library versions of the final experiment session
├── pyproject.toml                Package definition
└── .env.example                  Template for the secrets of a local run
```

## How the code is organized

The study was developed as a single notebook. Each code cell of that notebook is now a step file in `src/whose_emotion/steps/`. The steps share their variables, as notebook cells do, so `pipeline.py` runs them in order in one namespace. This design keeps the released code identical to the code that produced the paper.

| Step | What it does |
|---|---|
| `step01_config.py` | Defines every parameter of the study and the grid of runs for each stage |
| `step02_environment.py` | Detects Colab, Kaggle, or a local machine, loads the secrets, and selects the device |
| `step03_dependencies.py` | Checks the installed libraries and installs missing ones |
| `step04_storage.py` | Connects to the Hugging Face dataset repository that stores the outputs |
| `step05_checkpoints.py` | Finds finished and interrupted runs so that a stage can resume |
| `step06_data.py` | Downloads each corpus, verifies its checksum, and builds the splits |
| `step07_models.py` | Builds the encoders, classification heads, and optimizers |
| `step08_training.py` | Trains the runs of the current stage |
| `step08c_llm_transfer.py` | Queries GPT-4o zero-shot on the transfer targets and the reader-annotated corpora |
| `step09_evaluation.py` | Computes the metrics of each run |
| `step09b_aggregation.py` | Aggregates all runs and runs the main statistical tests |
| `step09c_robustness.py` | Runs the robustness checks |
| `step09d_controls.py` | Runs the controls without SetFit/emotion training |
| `step09e_reader_perspective.py` | Runs the reader analyses on crowd-enVENT and enISEAR |
| `step10_visualization.py` | Draws the figures |
| `step11_finalize.py` | Uploads the final outputs and marks the session as complete |

## Verifying the code against the original notebook

```
python tools/export_notebook.py --check
```

This command exports the archived notebook again and confirms that every step file matches it. The only differences are the four changes in [docs/CHANGES_FROM_NOTEBOOK.md](docs/CHANGES_FROM_NOTEBOOK.md), which let the code run from the command line and outside Colab and Kaggle. None of them changes what a run computes. The same check runs on GitHub after every push.

## Outputs

Every run writes its settings, test logits, and metrics to the dataset repository `<HF_USERNAME>/emotion_explicit_cue`. The code creates this repository as private if it does not exist. The dataset card of the [released outputs](https://huggingface.co/datasets/hacky1997/emotion_explicit_cue) describes the folder layout and the format of each file.

## Data

This repository does not include the corpora. The code downloads each corpus from its public source and verifies it against the SHA-256 checksums in `shared/data_manifest.json`. Supplementary Table S32 of the paper lists the same checksums.

| Corpus | Reference |
|---|---|
| SetFit/emotion | Saravia et al. (2018) |
| GoEmotions | Demszky et al. (2020) |
| MELD | Poria et al. (2019) |
| ISEAR | Scherer and Wallbott (1994), Hugging Face copy `gsri-18/ISEAR-dataset-complete` |
| EmpatheticDialogues | Rashkin et al. (2019) |
| TweetEval emotion | Barbieri et al. (2020), based on Mohammad et al. (2018) |
| crowd-enVENT | Troiano, Oberländer, and Klinger (2023) |
| enISEAR | Troiano, Padó, and Klinger (2019) |

Each corpus is subject to the terms set by its creators. Please read and follow those terms before you use the data.

## Citation
 
If you use this code or the released outputs, please cite the paper:
 
> S. Naskar, "Whose Emotion Do Text Classifiers Recover? Explicit Cues, Reader-Inferred Emotion, and Cross-Corpus Transfer," manuscript under review at *IEEE Transactions on Affective Computing*, 2026.
 
```bibtex
@unpublished{naskar2026whoseemotion,
  author = {Naskar, Sayak},
  title  = {Whose Emotion Do Text Classifiers Recover? Explicit Cues, Reader-Inferred Emotion, and Cross-Corpus Transfer},
  note   = {Manuscript under review at IEEE Transactions on Affective Computing},
  year   = {2026},
  url    = {https://github.com/hacky1997/WhoseEmotionDoTextClassifiersRecover}
}
```
 
This entry will be updated with the volume, pages, and DOI once the paper is published. GitHub's "Cite this repository" button gives the same reference in other formats.
