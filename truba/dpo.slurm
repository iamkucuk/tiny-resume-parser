#!/bin/bash
# Preference training (experiments/train_dpo.py) from an adapter, then score the result and its checkpoints on the dev pages.
#   sbatch -J NAME --gres=gpu:1 --cpus-per-task=16 truba/dpo.slurm NAME ADAPTER PAGES "dpo args" "eval args"
#SBATCH --partition=kolyoz-cuda
#SBATCH --account=romer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz17,kolyoz19,kolyoz22,kolyoz31,kolyoz58,kolyoz59,kolyoz60
#SBATCH --output=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.out
#SBATCH --error=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.err
set -euo pipefail
NAME=$1; ADAPTER=$2; PAGES=$3; DPO_ARGS=$4; EVAL_ARGS=$5
D=/arf/scratch/fkucuk/cv_parser_test; R=$D/cv_parser; OUT=$R/runs/$NAME
export HF_HOME=$D/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source $D/venv/bin/activate; cd $R; export PYTHONPATH=$R/src  # package without installing into the shared venv
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
