# Training and reproduction

## Setup

Python ≥ 3.11 is required. Reported runs used torch 2.6.0 (CUDA) on one H100 or H200 GPU.

```bash
git clone https://github.com/iamkucuk/tiny-resume-parser && cd tiny-resume-parser
pip install -e ".[train,dev]"
```

Reported runs use these package versions:
- **`transformers` 4.57.6** for ResuVL-S, ResuVL-M, and ResuVL-L v0.1, compared with version 5.5.4 for ResuVL-L.
- **Linear-attention kernels**: on GPU, ResuVL-L trains about 7× faster with them installed.
  - `flash-linear-attention` and `fla-core` 0.4.2 (0.5 needs torch 2.7).
  - `causal-conv1d` 1.7.0, built from source (`CAUSAL_CONV1D_FORCE_BUILD=TRUE`) when no prebuilt wheel matches the installed torch.

## Data

Label files are JSONL, with one page per line. Data and labels are not in the repository yet.

```json
{"cv_id": "CV-Anis-Mefteh", "page": 1, "image": "roboflow_source/train/....jpg", "record": {"basics": {...}, "work": [...]}}
```

| File (under `data/processed/`) | Pages | Use |
|---|---|---|
| `labels_train_v6.jsonl` | 1,472 | training (1,393) and validation (79, chosen by `data.is_validation`) |
| `labels_gold_v6_verified.jsonl` | 254 | reviewed gold: pages 1–50 development, 51–254 test |

Page images come from the [Cv Parsing Dataset](https://universe.roboflow.com/invoicemodelisation/cv-parsing-ksgbu) by invoiceModelisation (Roboflow Universe, CC BY 4.0). Labels are written by [`tiny_resume_parser.label`](../src/tiny_resume_parser/label.py). The [`archive-2026-10-15`](https://github.com/iamkucuk/tiny-resume-parser/tree/archive-2026-10-15/scripts) tag holds the one-off scripts that grouped pages into CVs and built the gold set.

The package looks for files in `data/` at the repository root, or in the folder named by `TINY_RESUME_PARSER_DATA`. Labelling new pages needs an OpenAI-compatible endpoint that serves a vision model. Set `LABEL_BASE_URL` and `LABEL_API_KEY` (and optionally `LABEL_MODEL`, default `glm-5.3-flash`) in the environment or in `.env`.

To try the pipeline first, [examples/](../examples) holds eight labelled pages in this format. Set `TINY_RESUME_PARSER_DATA=examples` and pass `--train-file examples/train.jsonl`.

## Train one student

A flag on the command line overrides the config.

```bash
tiny-resume-parser train --config configs/resuvl-l.toml --seed 42 --out runs/resuvl_l_s42
```

| Config | Model | Epochs | lr | Tiles | Time per seed (1 × H100, incl. scoring) |
|---|---|---|---|---|---|
| [`resuvl-l.toml`](../configs/resuvl-l.toml) | ResuVL-L | 6 | 1e-5 | 12 | ~1.3 h |
| [`resuvl-l-v0.1.toml`](../configs/resuvl-l-v0.1.toml) | ResuVL-L v0.1 | 6 | 1e-5 | 12 | ~1 h |
| [`resuvl-m.toml`](../configs/resuvl-m.toml) | ResuVL-M | 10 | 6e-5 | 30 | ~2.5 h |
| [`resuvl-s.toml`](../configs/resuvl-s.toml) | ResuVL-S | 10 | 1e-4 | 12 | ~1.5 h |

All four configurations share common training settings:
- **Full fine-tune** with the vision encoder and bf16 precision.
- Image noise p = 0.5.
- Effective batch 16.
- Cosine schedule with 3 % warm-up.
- Weight decay 0.01.

## Choose the epoch and average the seeds

Checkpoints are selected by validation score and averaged across seeds. Only the epoch with the highest validation lenient F1 is kept for each seed.

1. Score every saved epoch on the development sets (`evaluate` + `rescore`, see [evaluation.md](evaluation.md)).
2. Keep each seed's epoch with the highest validation lenient F1.
3. Average those models:

```bash
tiny-resume-parser soup --out runs/resuvl_l_avg/model runs/resuvl_l_s42/model runs/resuvl_l_s1/checkpoint-440 runs/resuvl_l_s2/model
```

ResuVL-L averages 3 runs (seeds 42, 1, 2), while ResuVL-M averages 3 runs. ResuVL-S averages 4 runs, and ResuVL-L v0.1 averages 10 runs.

## On a SLURM cluster

The [`slurm/`](../slurm) folder holds the jobs used for the reported runs:
- `exp.slurm` trains, scores every epoch from epoch 3 on, and keeps only the checkpoint chosen on the validation pages.
- `soup.slurm` averages and scores.
- `test.slurm` scores the test pages.

Cluster settings go on the `sbatch` command line and in `slurm/site.sh` ([slurm/README.md](../slurm/README.md)).

