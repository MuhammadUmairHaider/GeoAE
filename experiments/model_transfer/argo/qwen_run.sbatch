#!/usr/bin/env bash
# Resources are explicit in experiments/model_transfer/qwen/submit.py: GPU phases use one H100,
# 12 CPUs and 160G; the final API phase uses 4 CPUs, 32G and no GPU.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
set +x
source configs/sites/argo.sh
export GEOAE_ENV_FILE="${GEOAE_ENV_FILE:-.env.midlast}"
set -a
[ ! -f "$GEOAE_ENV_FILE" ] || source "$GEOAE_ENV_FILE"
set +a
# Never propagate inherited sbatch resource/account flags to subprocesses.
for _qwen_option in ${!SBATCH_@}; do unset "$_qwen_option"; done
export GEOAE_PYTHON="${GEOAE_PYTHON:-${UV_PROJECT_ENVIRONMENT}/bin/python}"
export OMP_NUM_THREADS="${QWEN_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export GEOAE_TRAIN_WORKERS="${GEOAE_TRAIN_WORKERS:-2}"
export GEOAE_VAL_WORKERS="${GEOAE_VAL_WORKERS:-1}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
[ -x "$GEOAE_PYTHON" ] || { echo "Python not executable: $GEOAE_PYTHON" >&2; exit 1; }
exec "$GEOAE_PYTHON" -u experiments/model_transfer/qwen/submit.py "$@"
