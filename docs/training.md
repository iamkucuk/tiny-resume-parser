# Training and reproduction

## Setup

```bash
git clone https://github.com/iamkucuk/tiny-resume-parser && cd tiny-resume-parser
pip install -e ".[train,dev]"
```

Python ≥ 3.11. Reported runs used torch 2.6.0 (CUDA), transformers 4.57.6, peft 0.21.2 on one H100 or H200.

## Data

Label files are JSONL, one page per line:

```json
{"cv_id": "CV-Anis-Mefteh", "page": 1, "image": "roboflow_source/train/....jpg", "record": {"basics": {...}, "work": [...]}}
```

| File (under `data/processed/`) | Pages | Use |
|---|---|---|
| `labels_train_v6.jsonl` | 1,472 | training (1,393) and validation (79, chosen by `data.is_validation`) |
| `labels_gold_v6_verified.jsonl` | 254 | reviewed gold: pages 1–50 development, 51–254 test |

The page images come from the [Cv Parsing Dataset](https://universe.roboflow.com/invoicemodelisation/cv-parsing-ksgbu)
by invoiceModelisation (Roboflow Universe, CC BY 4.0); the labels are written by
[`tiny_resume_parser.label`](../src/tiny_resume_parser/label.py). Data preparation scripts are in
[`scripts/`](../scripts). Data and labels are not in the repository yet. The package looks for them in `data/` at the
repository root, or in the folder named by `TINY_RESUME_PARSER_DATA`.

Labelling new pages needs an OpenAI-compatible endpoint that serves a vision model. Set `LABEL_BASE_URL` and
`LABEL_API_KEY` (and optionally `LABEL_MODEL`, default `glm-5.3-flash`) in the environment or in `.env`.

## Train one student

```bash
tiny-resume-parser train --config configs/internvl3-1b.toml --seed 42 --out runs/internvl_s42
```

| Config | Base model | Epochs | lr | Tiles | Time per seed (1 × H100, incl. scoring) |
|---|---|---|---|---|---|
| [`internvl3-1b.toml`](../configs/internvl3-1b.toml) | OpenGVLab/InternVL3-1B-hf | 6 | 1e-5 | 12 | ~1 h |
| [`smolvlm-500m.toml`](../configs/smolvlm-500m.toml) | HuggingFaceTB/SmolVLM-500M-Instruct | 10 | 6e-5 | 30 | ~2.5 h |
| [`granite-docling-258m.toml`](../configs/granite-docling-258m.toml) | ibm-granite/granite-docling-258M | 10 | 1e-4 | 12 | ~1.5 h |

All three: full fine-tune with the vision encoder, image noise p = 0.5, effective batch 16, cosine schedule,
3 % warm-up, weight decay 0.01, bf16. A flag on the command line overrides the config.

## Choose the epoch and average the seeds

1. Score every saved epoch on the development sets (`evaluate` + `rescore`, see [evaluation.md](evaluation.md)).
2. Keep each seed's epoch with the highest validation lenient F1.
3. Average those models:

```bash
tiny-resume-parser soup --out runs/internvl_avg/model runs/internvl_s42/checkpoint-440 runs/internvl_s1/model ...
```

The InternVL model averages 10 runs, SmolVLM 3 and granite-docling 4.

## On a SLURM cluster

[`slurm/`](../slurm) holds the jobs used for the reported runs: `exp.slurm` trains, scores every epoch from epoch 3
on and keeps only the checkpoint chosen on the validation pages; `soup.slurm` averages and scores; `test.slurm`
scores the test pages. Cluster settings go on the `sbatch` command line and in `slurm/site.sh`
([slurm/README.md](../slurm/README.md)).

## Experiments not in the recipe

[`experiments/`](../experiments) keeps the code of methods that were tried and not adopted: preference training,
reinforcement learning and a first training stage on page transcripts. Their results are in [findings.md](findings.md).
