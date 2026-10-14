# Source this on ACENET Argo (login shell AND every job):  source scripts/argo/env.sh
#
# Argo facts this relies on (verified 2026-10-03 as user dogar):
#   /home/dogar                      50 GB quota   -> uv binary + (optionally) the venv
#   /scratch/dogar                   20 TB quota   -> data root, HF cache, uv cache, uv-managed python
#   /home/dogar/projects/def-hsajjad/dogar/GeoAE/repo   -> this checkout (code only)
#   GPU node argo73: 4x H100 NVL (driver 580 = CUDA 13.0 ok), 48 CPUs, ~1 TB RAM, NO node-local scratch
#   (TmpDisk=0), so dumps are read from /scratch directly and must live in page cache / RAM.
#   Slurm: account def-hsajjad, partition c-hsajjad (14-day limit). The group shares the node: ONE GPU per job.
#
# How the rest of the repo finds this site: GEOAE_REMOTE=1 makes scripts/delta/env.sh hand over to
# scripts/remote/env.sh, which keeps every variable set here (it only fills in unset ones), so the eval
# sheets and scripts/delta/py run unchanged. scripts/remote/env.sh also sources THIS file by itself when it
# runs on an *argo* host, so a sheet started without sourcing this first still lands on the right paths.

_geoae_self="${BASH_SOURCE[0]:-$0}"
export GEOAE_ROOT="${GEOAE_ROOT:-$(cd "$(dirname "$_geoae_self")/../.." && pwd)}"
export GEOAE_REMOTE=1
export GEOAE_SITE=argo

export ARGO_ACCOUNT="${ARGO_ACCOUNT:-def-hsajjad}"
export ARGO_PARTITION="${ARGO_PARTITION:-c-hsajjad}"
export ARGO_SCRATCH="${ARGO_SCRATCH:-/scratch/${USER}}"

# Bulk data root (same directory roles as scripts/delta/link_data.sh -> scripts/argo/link_data.sh).
export DELTA_DATA="${DELTA_DATA:-${ARGO_SCRATCH}/geoae}"

# The venv: OUTSIDE the repo, on /scratch. Cold `import torch` was the same on /scratch and /home (10-12 s vs 9-15 s, warm 1-4 s), so /scratch keeps the 50 GB /home quota free.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/dogar/venvs/geoae}"
# Caches that would otherwise land in $HOME (50 GB quota).
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ARGO_SCRATCH}/uv_cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${ARGO_SCRATCH}/uv_python}"
# uv would otherwise pick the system python 3.11 (satisfies requires-python >=3.10); Delta's lock was
# resolved and tested on a uv-managed 3.14.
export UV_PYTHON="${UV_PYTHON:-3.14}"
export UV_PYTHON_PREFERENCE="${UV_PYTHON_PREFERENCE:-only-managed}"
# /scratch/dogar/hf_cache is the HF cache already in use on this account (also exported by ~/.bashrc).
export HF_HOME="${HF_HOME:-${ARGO_SCRATCH}/hf_cache}"
export WANDB_DIR="${WANDB_DIR:-${ARGO_SCRATCH}/wandb}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ARGO_SCRATCH}/triton_cache}"
mkdir -p "$DELTA_DATA" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$WANDB_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true

# Charge `sbatch` to the group account without -A (an explicit -A still wins).
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-$ARGO_ACCOUNT}"

# Threads: inside a job follow the allocation; on a login node stay small (Alliance login nodes kill
# long CPU time, and torch spins up one thread per core otherwise).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

# No `module load` is needed: the torch 2.13+cu130 wheel carries its own CUDA 13 runtime/cuDNN/cuBLAS
# (nvidia-* wheels in the venv) and the node driver is 580.x. The Alliance StdEnv modules are left alone.

export PATH="$HOME/.local/bin:$PATH"   # where the uv installer put `uv`
cd "$GEOAE_ROOT"
