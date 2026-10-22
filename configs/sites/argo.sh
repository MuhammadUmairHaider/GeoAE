# ACENET Argo allocation, storage and interpreter defaults. Source through scripts/env.sh.
_geoae_self="${BASH_SOURCE[0]:-$0}"
export GEOAE_ROOT="${GEOAE_ROOT:-$(cd "$(dirname "$_geoae_self")/../.." && pwd)}"
export GEOAE_REMOTE=1
export GEOAE_SITE=argo
export ARGO_ACCOUNT="${ARGO_ACCOUNT:-def-hsajjad}"
export ARGO_PARTITION="${ARGO_PARTITION:-c-hsajjad}"
export ARGO_SCRATCH="${ARGO_SCRATCH:-/scratch/${USER}}"
export DELTA_DATA="${DELTA_DATA:-${ARGO_SCRATCH}/geoae}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/dogar/venvs/geoae}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ARGO_SCRATCH}/uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${ARGO_SCRATCH}/uv_python}"
export UV_PYTHON="${UV_PYTHON:-3.14}"
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-managed}"
export HF_HOME="${HF_HOME:-${ARGO_SCRATCH}/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-${ARGO_SCRATCH}/wandb}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ARGO_SCRATCH}/triton_cache}"
mkdir -p "$DELTA_DATA" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$WANDB_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-$ARGO_ACCOUNT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export PATH="$HOME/.local/bin:$PATH"
cd "$GEOAE_ROOT"
