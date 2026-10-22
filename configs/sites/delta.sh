# NCSA Delta allocation, storage and interpreter defaults. Source through scripts/env.sh.
_geoae_self="${BASH_SOURCE[0]:-$0}"
export GEOAE_ROOT="${GEOAE_ROOT:-$(cd "$(dirname "$_geoae_self")/../.." && pwd)}"
export GEOAE_SITE=delta
: "${DELTA_PROJECT:=bbyl}"
export DELTA_PROJECT
: "${DELTA_ACCOUNT:=${DELTA_PROJECT}-delta-gpu}"
: "${DELTA_CPU_ACCOUNT:=bimc-delta-cpu}"
export DELTA_ACCOUNT DELTA_CPU_ACCOUNT
: "${DELTA_DATA:=/work/hdd/bimc/${USER}/GeoAE}"
export DELTA_DATA
export DELTA_WORK="/work/hdd/${DELTA_PROJECT}/${USER}"
export DELTA_FAST="/work/nvme/${DELTA_PROJECT}/${USER}"
export DELTA_DATA="${DELTA_DATA:-${DELTA_WORK}/GeoAE}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${DELTA_FAST}/venvs/geoae}"
if [ -d "${DELTA_DATA}/cache/huggingface" ]; then
  export HF_HOME="${DELTA_DATA}/cache/huggingface"
else
  export HF_HOME="${HF_HOME:-${DELTA_WORK}/hf_cache}"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-${DELTA_WORK}/uv_cache}"
export WANDB_DIR="${WANDB_DIR:-${DELTA_WORK}/wandb}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DELTA_WORK}/triton_cache}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$WANDB_DIR" "$TRITON_CACHE_DIR" "$DELTA_DATA"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-$DELTA_ACCOUNT}"
export PATH="$HOME/.local/bin:$PATH"
cd "$GEOAE_ROOT"
