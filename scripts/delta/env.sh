# Source this on NCSA Delta (login shell AND every job):  source scripts/delta/env.sh
#
# Delta facts this relies on (docs.ncsa.illinois.edu/systems/delta, checked 2026-09-18):
#   $HOME = /u/$USER, 100 GB / 750k files quota  -> keep big things OFF home
#   /work/hdd  per-project 1 TB default, not purged, no backups -> repo, data, checkpoints
#   /tmp on GPU nodes: ~1.5 TB local NVMe, wiped after each job -> per-job data staging
#
# Set DELTA_PROJECT to your allocation's directory name under /work/hdd
# (run `accounts` to see your projects; `ls /work/hdd` to see the directory).
: "${DELTA_PROJECT:?export DELTA_PROJECT=<allocation dir under /work/hdd> first}"

export DELTA_WORK="/work/hdd/${DELTA_PROJECT}/${USER}"
export GEOAE_ROOT="${GEOAE_ROOT:-${DELTA_WORK}/GeoAE}"

# Caches that would otherwise land in $HOME and blow its 100 GB quota:
# HF models alone were 57 GB on the old box, the uv cache 18 GB.
export HF_HOME="${DELTA_WORK}/hf_cache"
export UV_CACHE_DIR="${DELTA_WORK}/uv_cache"
export WANDB_DIR="${DELTA_WORK}/wandb"
export TRITON_CACHE_DIR="${DELTA_WORK}/triton_cache"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$WANDB_DIR" "$TRITON_CACHE_DIR"

export PATH="$HOME/.local/bin:$PATH"   # where the uv installer puts `uv`
cd "$GEOAE_ROOT"
