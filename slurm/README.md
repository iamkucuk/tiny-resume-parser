# Cluster jobs

SLURM scripts used for the reported runs on the TRUBA cluster (TÜBİTAK ULAKBİM). Paths (`/arf/scratch/...`), partition and account are
specific to that cluster; adapt them before use elsewhere. Each script puts `src/` on `PYTHONPATH` instead of
installing the package into the shared environment.

| Script | Job |
|---|---|
| `exp.slurm` | train, score every saved epoch from MIN_EPOCH on the development pages, keep only the validation-chosen checkpoint |
| `soup.slurm` | average several fine-tuned models, then score the average |
| `eval.slurm` | score one saved model on the development pages |
| `test.slurm` | score one model once on the 204 held-out test pages (milestones only) |
| `monitor.sh` | runs on the local machine: polls the cluster and prints one line per event that needs a decision |

```bash
sbatch -J NAME --time=03:00:00 --gres=gpu:1 --cpus-per-task=16 slurm/exp.slurm NAME "--config configs/granite-docling-258m.toml --seed 1" "--max-patches 12"
```
