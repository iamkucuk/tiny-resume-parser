# Cluster jobs

SLURM scripts for the reported runs. They contain no cluster-specific settings:

- Submit from the repository root. The scripts read code and data from that folder and write to `runs/NAME/`.
- Put the environment in `slurm/site.sh` (copy [`site.example.sh`](site.example.sh)): which Python environment to
  activate and where models are cached. Every script sources it if it exists.
- Pass cluster flags (partition, account, log paths) on the `sbatch` command line.

| Script | Job |
|---|---|
| `exp.slurm` | train, score every saved epoch from epoch 3 on the development pages, keep only the checkpoint chosen on the validation pages |
| `soup.slurm` | average several fine-tuned models, then score the average |
| `eval.slurm` | score one saved model on the development pages |
| `test.slurm` | score one model once on the 204 test pages (milestones only) |

```bash
sbatch -p gpu -A myaccount --gres=gpu:1 --cpus-per-task=16 --time=03:00:00 \
  slurm/exp.slurm resuvl_s_seed1 "--config configs/resuvl-s.toml --seed 1" "--max-patches 12"
```

A full fine-tune of ResuVL-L v0.1 takes about 1 hour on one H100, including scoring, and saves up to 6 checkpoints of
3.5 GB before all but one are deleted.
