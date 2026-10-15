#!/bin/bash
# Average the weights of several full fine-tunes (tiny_resume_parser.soup), then score the average on the dev pages.
#   sbatch -J NAME slurm/soup.slurm NAME "eval args" weights_dir1 weights_dir2 ...   (dirs relative to cv_parser/)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
set -euo pipefail
NAME=$1; EVAL_ARGS=$2; shift 2
R=${SLURM_SUBMIT_DIR:-$PWD}; OUT=$R/runs/$NAME
[ -f $R/slurm/site.sh ] && source $R/slurm/site.sh  # optional, untracked: environment, model cache (see slurm/README.md)
cd $R; export PYTHONPATH=$R/src TOKENIZERS_PARALLELISM=false
echo "== $NAME: average of $* | eval $EVAL_ARGS"
[ -e $OUT/model/model.safetensors ] || python -u -m tiny_resume_parser.soup --out $OUT/model "$@"
for SET in "gold --limit 50" "val"; do
  S=${SET%% *}; E=$OUT/eval_final_${S}
  [ -e $E/summary.json ] || python -u -m tiny_resume_parser.evaluate --model $OUT/model --set $SET --batch 8 --out $E $EVAL_ARGS > $E.log 2>&1 || echo "eval failed: $E"
  python -m tiny_resume_parser.rescore $E 2>/dev/null | grep -v Warn || true
done
echo "== done $NAME"
