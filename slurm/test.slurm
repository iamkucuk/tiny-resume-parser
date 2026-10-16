#!/bin/bash
# Score one model ONCE on the held-out test pages (reviewed gold pages 51-254), raw and repaired. Milestones only.
#   sbatch -J NAME slurm/test.slurm NAME weights_dir "eval args"   (weights_dir relative to cv_parser/)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
set -euo pipefail
NAME=$1; W=$2; EVAL_ARGS=$3
R=${SLURM_SUBMIT_DIR:-$PWD}; CODE=${CODE:-$R}; E=$R/runs/$NAME/eval_test
[ -f $R/slurm/site.sh ] && source $R/slurm/site.sh  # optional, untracked: environment, model cache (see slurm/README.md)
cd $CODE; export PYTHONPATH=$CODE/src${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH} TOKENIZERS_PARALLELISM=false  # EXTRA_PYTHONPATH: e.g. a newer transformers
echo "== $NAME: test pages 51-254 with $W | eval $EVAL_ARGS"
mkdir -p $R/runs/$NAME
[ -e $E/summary.json ] || python -u -m tiny_resume_parser.evaluate --model $R/$W --set gold --start 50 --batch 8 --out $E $EVAL_ARGS > $E.log 2>&1
python -m tiny_resume_parser.rescore $E
python -m tiny_resume_parser.rescore --repair $E
echo "== done $NAME"
