#!/usr/bin/env bash
# Evaluation-only continuation after the original campaign releases its job.
# Use an explicit scheduler dependency when submitting this recovery job.
# No training, extraction or GPU profiling; stage at most two needed layers.
#SBATCH --job-name=gpt_run_recover
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/gpt_run/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/gpt2/argo_env.sh
exec "$GEOAE_PYTHON" -u scripts/gpt2/recover_evals.py "$@"
