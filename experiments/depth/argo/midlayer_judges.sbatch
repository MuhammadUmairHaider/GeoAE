#!/usr/bin/env bash
# Serial API judgments and final reports after both H100 lanes succeed.
# Submitted by experiments/depth/campaign.py; this job requests no GPUs.
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export GEOAE_PYTHON="${GEOAE_PYTHON:-${UV_PROJECT_ENVIRONMENT}/bin/python}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
[ -x "$GEOAE_PYTHON" ] || { echo "Python not executable: $GEOAE_PYTHON" >&2; exit 1; }
set +x
set -a
[ ! -f "${GEOAE_ENV_FILE:-.env}" ] || source "${GEOAE_ENV_FILE:-.env}"
set +a
exec "$GEOAE_PYTHON" -u experiments/depth/campaign.py "$@"
