#!/usr/bin/env bash
# One H100 for the complete GPT-2 all-layer campaign, after the Gemma jobs:
#   python experiments/gpt2/submit_argo.py --after-gemma <train_id>,<eval_id>
# The profile determines how many of the twelve layers can share this GPU.
# 369 GB of activation files are copied once to tmpfs because Argo GPFS does
# not serve random mmap reads from the Linux page cache.
#SBATCH --job-name=gpt_run
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=450G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/gpt_run/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source experiments/gpt2/argo_env.sh
# Read optional judge credentials without tracing or printing them.
set +x
set -a
[ ! -f "${GEOAE_ENV_FILE:-.env}" ] || source "${GEOAE_ENV_FILE:-.env}"
set +a
exec "$GEOAE_PYTHON" -u experiments/gpt2/run_gpt_run.py "$@"
