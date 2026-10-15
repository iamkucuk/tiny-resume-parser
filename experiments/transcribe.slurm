#!/bin/bash
# Transcribe training pages with granite-docling-258M (experiments/transcribe.py), one shard per job.
#   sbatch -J NAME experiments/transcribe.slurm OUT_FILE "transcribe args"     (paths relative to cv_parser/)
#SBATCH --partition=kolyoz-cuda
#SBATCH --account=romer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz17,kolyoz19,kolyoz22,kolyoz31,kolyoz58,kolyoz59,kolyoz60
#SBATCH --output=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.out
#SBATCH --error=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.err
set -euo pipefail
D=/arf/scratch/fkucuk/cv_parser_test; R=$D/cv_parser
export HF_HOME=$D/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source $D/venv/bin/activate; cd $R; export PYTHONPATH=$R/src  # package without installing into the shared venv
echo "== transcribe -> $1 | $2"
python -u experiments/transcribe.py --train-file data/processed/labels_train_v6_hk1k.jsonl --out $1 $2
echo "== done"
