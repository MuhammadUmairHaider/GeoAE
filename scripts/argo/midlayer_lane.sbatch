#!/usr/bin/env bash
# Each job reserves one H100 and runs one model's three AE variants serially.
# Queue both jobs with scripts/midlayer/campaign.py --submit.
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=14-00:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export GEOAE_PYTHON="${GEOAE_PYTHON:-${UV_PROJECT_ENVIRONMENT}/bin/python}"
export OMP_NUM_THREADS="${MID_LAYER_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export GEOAE_TRAIN_WORKERS="${GEOAE_TRAIN_WORKERS:-2}"
export GEOAE_VAL_WORKERS="${GEOAE_VAL_WORKERS:-1}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
[ -x "$GEOAE_PYTHON" ] || { echo "Python not executable: $GEOAE_PYTHON" >&2; exit 1; }
# Credentials are loaded privately; shell tracing remains disabled.
set +x
set -a
[ ! -f "${GEOAE_ENV_FILE:-.env}" ] || source "${GEOAE_ENV_FILE:-.env}"
set +a
exec "$GEOAE_PYTHON" -u scripts/midlayer/campaign.py "$@"
