# Copy to slurm/site.sh (untracked) and edit for your cluster. Every job script sources it before running Python.
# Cluster flags (partition, account, log paths) go on the sbatch command line, e.g.
#   sbatch -p gpu -A myaccount -o logs/%x-%j.out --gres=gpu:1 --cpus-per-task=16 slurm/exp.slurm NAME "..." "..."
source "$HOME/venvs/tiny-resume-parser/bin/activate"   # an environment with `pip install -e ".[train]"`
export HF_HOME="$HOME/hf_cache"                         # where base models are downloaded
# export HF_HUB_OFFLINE=1                               # compute nodes without internet: download models first
# TRL_PATH=$HOME/trl_pkgs                               # only for experiments/grpo.slurm, if TRL lives in its own folder
