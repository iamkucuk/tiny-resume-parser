#!/bin/bash
# Preference training (experiments/train_dpo.py) from an adapter, then score the result and its checkpoints on the dev pages.
#   sbatch -J NAME --gres=gpu:1 --cpus-per-task=16 experiments/dpo.slurm NAME ADAPTER PAGES "dpo args" "eval args"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
set -euo pipefail
NAME=$1; ADAPTER=$2; PAGES=$3; DPO_ARGS=$4; EVAL_ARGS=$5
R=${SLURM_SUBMIT_DIR:-$PWD}; OUT=$R/runs/$NAME
[ -f $R/slurm/site.sh ] && source $R/slurm/site.sh  # optional, untracked: environment, model cache (see slurm/README.md)
cd $R; export PYTHONPATH=$R/src TOKENIZERS_PARALLELISM=false
[ -e $OUT/adapter ] || python -u experiments/train_dpo.py --adapter $ADAPTER --pages $PAGES --out $OUT $DPO_ARGS
for CK in $(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n) $OUT/adapter; do
  T=$(basename $CK)
  for SET in "gold --limit 50" "val"; do
    S=${SET%% *}; E=$OUT/eval_${T}_${S}
    [ -e $E/summary.json ] || python -u -m tiny_resume_parser.evaluate --adapter $CK --set $SET --batch 8 --out $E $EVAL_ARGS > $E.log 2>&1 || echo "eval failed: $E"
    python -m tiny_resume_parser.rescore $E 2>/dev/null | grep -v Warn || true
  done
done
echo "== done $NAME"
