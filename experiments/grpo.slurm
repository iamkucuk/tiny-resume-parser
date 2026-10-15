#!/bin/bash
# GRPO after supervised fine-tuning, then score the final model on the dev pages (gold 1-50 + validation split).
#   sbatch -J NAME experiments/grpo.slurm NAME "grpo args (--model CHECKPOINT ...)" "eval args"
# TRL is in $D/venv/trl_pkgs: not on the shared venv path (PYTHONPATH only), pinned to its torch/transformers. It must
# sit under the venv prefix: dill pickles modules outside sys.prefix by value, which breaks datasets fingerprinting.
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
NAME=$1; GRPO_ARGS=$2; EVAL_ARGS=$3
D=/arf/scratch/fkucuk/cv_parser_test; R=$D/cv_parser; OUT=$R/runs/$NAME
export HF_HOME=$D/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source $D/venv/bin/activate; cd $R; export PYTHONPATH=$R/src  # package without installing into the shared venv
nvidia-smi --query-gpu=name --format=csv,noheader
echo "== $NAME: grpo $GRPO_ARGS | eval $EVAL_ARGS"
[ -e $OUT/log_history.json ] || PYTHONPATH=$R/src:$D/venv/trl_pkgs python -u experiments/train_grpo.py --out $OUT $GRPO_ARGS
for SET in "gold --limit 50" "val"; do
  S=${SET%% *}; E=$OUT/eval_final_${S}
  [ -e $E/summary.json ] || python -u -m tiny_resume_parser.evaluate --model $OUT/model --set $SET --batch 8 --out $E $EVAL_ARGS > $E.log 2>&1 || echo "eval failed: $E"
  python -m tiny_resume_parser.rescore $E 2>/dev/null | grep -v Warn || true
done
rm -rf $OUT/checkpoint-*  # intermediate GRPO checkpoints, 3.5 GB each; model/ is kept
echo "== done $NAME"
