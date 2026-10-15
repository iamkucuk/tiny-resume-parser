#!/bin/bash
# One experiment = train, then score every saved epoch from MIN_EPOCH on the dev pages (gold 1-50 + validation split).
#   sbatch --gres=gpu:N --cpus-per-task=16N slurm/exp.slurm NAME "train args" "eval args" [MIN_EPOCH]
# eval args must repeat what changes the input (--max-patches) and, for another base model, --model.
#SBATCH --partition=kolyoz-cuda
#SBATCH --account=romer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz17,kolyoz19,kolyoz22,kolyoz31,kolyoz58,kolyoz59,kolyoz60
#SBATCH --output=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.out
#SBATCH --error=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.err
set -euo pipefail
NAME=$1; TRAIN_ARGS=$2; EVAL_ARGS=$3; MIN_EPOCH=${4:-3}
D=/arf/scratch/fkucuk/cv_parser_test; R=$D/cv_parser; OUT=$R/runs/$NAME
export HF_HOME=$D/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source $D/venv/bin/activate; cd $R; export PYTHONPATH=$R/src  # package without installing into the shared venv
nvidia-smi --query-gpu=name --format=csv,noheader
NG=$(echo "$CUDA_VISIBLE_DEVICES" | tr , "\n" | grep -c .)  # nvidia-smi -L lists one GPU twice on some nodes (kolyoz31)
echo "== $NAME on $NG GPU(s): train $TRAIN_ARGS | eval $EVAL_ARGS"
if [ ! -e $OUT/log_history.json ]; then
  if [ "$NG" -gt 1 ]; then python -m torch.distributed.run --standalone --nproc_per_node=$NG -m tiny_resume_parser.train --out $OUT --check-pages 0 $TRAIN_ARGS
  else python -u -m tiny_resume_parser.train --out $OUT --check-pages 0 $TRAIN_ARGS; fi
fi
STEPS=$(python -c "import json;h=json.load(open('$OUT/log_history.json'));print(max(x.get('step',0) for x in h))")
EPOCHS=$(python -c "import json;h=json.load(open('$OUT/log_history.json'));print(round(max(x.get('epoch',0) for x in h)))")
for CK in $(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  EP=$(( ${CK##*-} * EPOCHS / STEPS ))
  [ $EP -ge $MIN_EPOCH ] || { rm -rf $CK; continue; }  # never scored; 3.5 GB each for a full fine-tune
  if [ -d $CK/../model ] || [ -e $CK/config.json ]; then W="--model $CK"; else W="--adapter $CK"; fi
  for SET in "gold --limit 50" "val"; do
    S=${SET%% *}; E=$OUT/eval_ep${EP}_${S}
    [ -e $E/summary.json ] || python -u -m tiny_resume_parser.evaluate $W --set $SET --batch 8 --out $E $EVAL_ARGS > $E.log 2>&1 || echo "eval failed: $E"
    python -m tiny_resume_parser.rescore $E 2>/dev/null | grep -v Warn || true
  done
done
# full fine-tune checkpoints are ~3.5 GB each: keep only the epoch chosen on validation lenient F1 (the protocol's
# choice); if that is the last epoch, model/ already holds the same weights, so its checkpoint is deleted too
if [ -d $OUT/model ]; then
  BEST=$(python -c "
import json,glob,re
s={int(re.search(r'ep(\d+)_val',f).group(1)):json.load(open(f))['lenient']['F1'] for f in glob.glob('$OUT/eval_ep*_val/rescore.json')}
print(max(s,key=s.get) if s else -1)")
  if [ $BEST -ge 0 ]; then
    for CK in $(ls -d $OUT/checkpoint-* 2>/dev/null); do
      EP=$(( ${CK##*-} * EPOCHS / STEPS )); [ $EP -eq $BEST ] && [ $EP -ne $EPOCHS ] || rm -rf $CK
    done
    [ $BEST -eq $EPOCHS ] || rm -rf $OUT/model
  fi
  echo "kept val-chosen epoch $BEST ($( [ $BEST -eq $EPOCHS ] && echo model/ || echo checkpoint))"
fi
echo "== done $NAME"
