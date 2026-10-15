#!/bin/bash
# Average the weights of several full fine-tunes (src/soup.py), then score the average on the dev pages.
#   sbatch -J NAME truba/soup.slurm NAME "eval args" weights_dir1 weights_dir2 ...   (dirs relative to cv_parser/)
#SBATCH --partition=kolyoz-cuda
#SBATCH --account=romer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz17,kolyoz19,kolyoz22,kolyoz31,kolyoz58,kolyoz59,kolyoz60
#SBATCH --output=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.out
#SBATCH --error=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.err
set -euo pipefail
NAME=$1; EVAL_ARGS=$2; shift 2
D=/arf/scratch/fkucuk/cv_parser_test; R=$D/cv_parser; OUT=$R/runs/$NAME
export HF_HOME=$D/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source $D/venv/bin/activate; cd $R
echo "== $NAME: average of $* | eval $EVAL_ARGS"
[ -e $OUT/model/model.safetensors ] || python -u src/soup.py --out $OUT/model "$@"
for SET in "gold --limit 50" "val"; do
  S=${SET%% *}; E=$OUT/eval_final_${S}
  [ -e $E/summary.json ] || python -u src/evaluate.py --model $OUT/model --set $SET --batch 8 --out $E $EVAL_ARGS > $E.log 2>&1 || echo "eval failed: $E"
  python src/rescore.py $E 2>/dev/null | grep -v Warn || true
done
echo "== done $NAME"
