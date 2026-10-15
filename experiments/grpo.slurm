#!/bin/bash
# GRPO after supervised fine-tuning, then score the final model on the dev pages (gold 1-50 + validation split).
#   sbatch -J NAME experiments/grpo.slurm NAME "grpo args (--model CHECKPOINT ...)" "eval args"
# TRL may sit in its own folder (TRL_PATH, set in slurm/site.sh) so a shared environment stays unchanged. That folder
# must be under the environment prefix: dill pickles modules outside sys.prefix by value, which breaks datasets.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
set -euo pipefail
NAME=$1; GRPO_ARGS=$2; EVAL_ARGS=$3
R=${SLURM_SUBMIT_DIR:-$PWD}; OUT=$R/runs/$NAME
[ -f $R/slurm/site.sh ] && source $R/slurm/site.sh  # optional, untracked: environment, model cache (see slurm/README.md)
cd $R; export PYTHONPATH=$R/src TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name --format=csv,noheader
echo "== $NAME: grpo $GRPO_ARGS | eval $EVAL_ARGS"
[ -e $OUT/log_history.json ] || PYTHONPATH=$R/src${TRL_PATH:+:$TRL_PATH} python -u experiments/train_grpo.py --out $OUT $GRPO_ARGS
for SET in "gold --limit 50" "val"; do
  S=${SET%% *}; E=$OUT/eval_final_${S}
  [ -e $E/summary.json ] || python -u -m tiny_resume_parser.evaluate --model $OUT/model --set $SET --batch 8 --out $E $EVAL_ARGS > $E.log 2>&1 || echo "eval failed: $E"
  python -m tiny_resume_parser.rescore $E 2>/dev/null | grep -v Warn || true
done
rm -rf $OUT/checkpoint-*  # intermediate GRPO checkpoints, 3.5 GB each; model/ is kept
echo "== done $NAME"
