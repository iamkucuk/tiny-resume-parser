#!/bin/bash
# Score one model ONCE on the held-out test pages (reviewed gold pages 51-254), raw and repaired. Milestones only.
#   sbatch -J NAME slurm/test.slurm NAME weights_dir "eval args"   (weights_dir relative to cv_parser/)
#SBATCH --partition=kolyoz-cuda
#SBATCH --account=romer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --exclude=kolyoz10,kolyoz11,kolyoz13,kolyoz14,kolyoz17,kolyoz19,kolyoz22,kolyoz31,kolyoz58,kolyoz59,kolyoz60
#SBATCH --output=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.out
#SBATCH --error=/arf/scratch/fkucuk/cv_parser_test/logs/%x-%j.err
set -euo pipefail
NAME=$1; W=$2; EVAL_ARGS=$3
D=/arf/scratch/fkucuk/cv_parser_test; R=$D/cv_parser; CODE=${CODE:-$R}; E=$R/runs/$NAME/eval_test
export HF_HOME=$D/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source $D/venv/bin/activate; cd $CODE; export PYTHONPATH=$CODE/src
echo "== $NAME: test pages 51-254 with $W | eval $EVAL_ARGS"
[ -e $E/summary.json ] || python -u -m tiny_resume_parser.evaluate --model $R/$W --set gold --start 50 --batch 8 --out $E $EVAL_ARGS > $E.log 2>&1
python -m tiny_resume_parser.rescore $E
python -m tiny_resume_parser.rescore --repair $E
echo "== done $NAME"
