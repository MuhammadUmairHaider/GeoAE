#!/bin/bash
# One slice of experiments/paper/remote/run_control_extra.sh on Argo; choose the slice with --export=ALL,CTRL_ARMS=..,CTRL_DATASETS=..,CTRL_ATLAS_SEEDS=..
#   sbatch --gres=gpu:l40:1 --export=ALL,CTRL_ARMS=g4b_l33_d5120,CTRL_DATASETS=news20,CTRL_ATLAS_SEEDS= experiments/paper/argo/control_extra.sbatch
#   sbatch --gres=gpu:h100:1 --export=ALL,CTRL_ARMS=,CTRL_DATASETS=,CTRL_ATLAS_SEEDS=0 experiments/paper/argo/control_extra.sbatch
#SBATCH --job-name=ctrl-extra
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "${SLURM_SUBMIT_DIR:?}"
# wikimedia/wikipedia (perplexity texts) is streamed from the Hub, so offline mode must stay off
export CTRL_ROOT="$PWD" CTRL_LOG="results/delta/run_control_extra_argo_${SLURM_JOB_ID}.out" HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1
echo "[ctrl-extra] $(hostname) $(date) arms='${CTRL_ARMS-}' datasets='${CTRL_DATASETS-}' atlas='${CTRL_ATLAS_SEEDS-}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash experiments/paper/remote/run_control_extra.sh
