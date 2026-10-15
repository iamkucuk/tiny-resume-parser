#!/bin/bash
# Score one saved model on the dev pages (gold 1-50 + validation split) with extra scoring args; no training.
#   sbatch -J NAME slurm/eval.slurm NAME weights_dir "eval args"   (weights_dir relative to cv_parser/)
#SBATCH --partition=kolyoz-cuda
#SBATCH --account=romer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz17,kolyoz19,kolyoz22,kolyoz31,kolyoz58,kolyoz59,kolyoz60
#SBATCH --output=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.out
#SBATCH --error=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.err
set -euo pipefail
NAME=$1; W=$2; EVAL_ARGS=$3
D=/arf/scratch/fkucuk/cv_parser_test; R=$D/cv_parser; OUT=$R/runs/$NAME
export HF_HOME=$D/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source $D/venv/bin/activate; cd $R; export PYTHONPATH=$R/src; mkdir -p $OUT  # package without installing into the shared venv
echo "== $NAME: score $W | eval $EVAL_ARGS"
for SET in "gold --limit 50" "val"; do
  S=${SET%% *}; E=$OUT/eval_final_${S}
  [ -e $E/summary.json ] || python -u -m tiny_resume_parser.evaluate --model $W --set $SET --batch 8 --out $E $EVAL_ARGS > $E.log 2>&1 || echo "eval failed: $E"
  python -m tiny_resume_parser.rescore $E 2>/dev/null | grep -v Warn || true
done
echo "== done $NAME"
