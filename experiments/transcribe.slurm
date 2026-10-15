#!/bin/bash
# Transcribe training pages with granite-docling-258M (experiments/transcribe.py), one shard per job.
#   sbatch -J NAME experiments/transcribe.slurm OUT_FILE "transcribe args"     (paths relative to cv_parser/)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
set -euo pipefail
R=${SLURM_SUBMIT_DIR:-$PWD}
[ -f $R/slurm/site.sh ] && source $R/slurm/site.sh  # optional, untracked: environment, model cache (see slurm/README.md)
cd $R; export PYTHONPATH=$R/src TOKENIZERS_PARALLELISM=false
echo "== transcribe -> $1 | $2"
python -u experiments/transcribe.py --train-file data/processed/labels_train_v6_hk1k.jsonl --out $1 $2
echo "== done"
