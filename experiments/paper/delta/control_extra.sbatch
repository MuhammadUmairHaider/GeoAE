#!/bin/bash
# One slice of experiments/paper/remote/run_control_extra.sh on Delta; choose the slice with --export=ALL,CTRL_ARMS=..,CTRL_DATASETS=..,CTRL_ATLAS_SEEDS=
#   sbatch --export=ALL,CTRL_ARMS=d6144,CTRL_DATASETS=news20,CTRL_ATLAS_SEEDS= experiments/paper/delta/control_extra.sbatch
#SBATCH --job-name=ctrl-extra
#SBATCH --account=bbyl-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "${SLURM_SUBMIT_DIR:?}"
export CTRL_ROOT="$PWD" CTRL_ENV=scripts/env.sh CTRL_LOG="results/delta/run_control_extra_delta_${SLURM_JOB_ID}.out" \
       HF_HUB_OFFLINE=0 PYTHONUNBUFFERED=1
echo "[ctrl-extra] $(hostname) $(date) arms='${CTRL_ARMS-}' datasets='${CTRL_DATASETS-}' atlas='${CTRL_ATLAS_SEEDS-}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash experiments/paper/remote/run_control_extra.sh
