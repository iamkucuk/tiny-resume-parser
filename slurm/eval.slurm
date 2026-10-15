#!/bin/bash
# Score one saved model on the dev pages (gold 1-50 + validation split) with extra scoring args; no training.
#   sbatch -J NAME slurm/eval.slurm NAME weights_dir "eval args"   (weights_dir relative to cv_parser/)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
set -euo pipefail
NAME=$1; W=$2; EVAL_ARGS=$3
R=${SLURM_SUBMIT_DIR:-$PWD}; OUT=$R/runs/$NAME
[ -f $R/slurm/site.sh ] && source $R/slurm/site.sh  # optional, untracked: environment, model cache (see slurm/README.md)
cd $R; export PYTHONPATH=$R/src TOKENIZERS_PARALLELISM=false; mkdir -p $OUT
echo "== $NAME: score $W | eval $EVAL_ARGS"
for SET in "gold --limit 50" "val"; do
  S=${SET%% *}; E=$OUT/eval_final_${S}
  [ -e $E/summary.json ] || python -u -m tiny_resume_parser.evaluate --model $W --set $SET --batch 8 --out $E $EVAL_ARGS > $E.log 2>&1 || echo "eval failed: $E"
  python -m tiny_resume_parser.rescore $E 2>/dev/null | grep -v Warn || true
done
echo "== done $NAME"
