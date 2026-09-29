# Source this on NCSA Delta (login shell AND every job):  source scripts/delta/env.sh
#
# Delta facts this relies on (docs.ncsa.illinois.edu/systems/delta, checked 2026-09-18,
# re-verified against `quota`/`accounts` on 2026-09-26):
#   $HOME = /u/$USER, 100 GB / 750k files quota  -> keep big things OFF home
#   /projects/<proj>   500 GB-1 TB, shared by the whole allocation -> CODE ONLY
#   /work/hdd/<proj>   1 TB, not purged, no backups -> dumps, checkpoints, caches
#   /work/nvme/<proj>  500 GB, fast metadata       -> the venv (see VENV below)
#   /tmp on GPU nodes: ~1.5 TB local NVMe, wiped after each job -> per-job staging
#
# THREE things this fixes relative to the 2026-09-18 version, all of which
# assumed code and data share one root under /work/hdd. On this account they
# do not (docs/notes/delta-filesystem-layout.md):
#
#   GEOAE_ROOT is derived from THIS SCRIPT'S OWN LOCATION, not from
#       $DELTA_WORK/GeoAE. The repo lives on /projects and the data on
#       /work/hdd, so hardcoding the data root made `cd "$GEOAE_ROOT"` land in
#       a directory with no code in it.
#   VENV: UV_PROJECT_ENVIRONMENT points at /work/nvme. Left unset, `uv run`
#       creates .venv inside the repo, i.e. on /projects: `import torch` there
#       is metadata-bound and took over 8 MINUTES on the Lustre HDD tier
#       against 21 s on nvme.
#   HF_HOME points at the EXISTING 14 GB cache under the data root, which holds
#       the gated meta-llama/Llama-3.2-3B weights and every eval dataset. The
#       old default ($DELTA_WORK/hf_cache) is an empty directory, so jobs
#       re-downloaded 14 GB and 401'd on the gated repo.
#
# `module reset` is deliberately NOT called: verified 2026-09-26 that the torch
# 2.13+cu130 wheel imports fine with Delta's default cudatoolkit/26.5_13.2 on
# LD_LIBRARY_PATH. Add it back only if a CUDA symbol clash reappears.

_geoae_self="${BASH_SOURCE[0]:-$0}"
export GEOAE_ROOT="${GEOAE_ROOT:-$(cd "$(dirname "$_geoae_self")/../.." && pwd)}"

# Which allocation. Edit scripts/delta/site.sh, or export DELTA_PROJECT first.
[ -f "$GEOAE_ROOT/scripts/delta/site.sh" ] && . "$GEOAE_ROOT/scripts/delta/site.sh"
: "${DELTA_PROJECT:?export DELTA_PROJECT=<allocation dir under /work/hdd>, or set it in scripts/delta/site.sh}"

export DELTA_WORK="/work/hdd/${DELTA_PROJECT}/${USER}"      # bulk, 1 TB
export DELTA_FAST="/work/nvme/${DELTA_PROJECT}/${USER}"     # fast metadata, 500 GB
export DELTA_DATA="${DELTA_DATA:-${DELTA_WORK}/GeoAE}"      # dumps/checkpoints/caches root

# The venv: on nvme, OUTSIDE the repo, so it is never on /projects and never
# committed. scripts/delta/setup.sh creates it.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${DELTA_FAST}/venvs/geoae}"

# Caches that would otherwise land in $HOME and blow its 100 GB quota
# (HF models alone are 14 GB here, the uv cache was 18 GB on the old box).
# HF_HOME: prefer the populated cache under the data root; fall back to a fresh
# one only if that does not exist, so a new account still works.
if [ -d "${DELTA_DATA}/cache/huggingface" ]; then
  export HF_HOME="${DELTA_DATA}/cache/huggingface"
else
  export HF_HOME="${HF_HOME:-${DELTA_WORK}/hf_cache}"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-${DELTA_WORK}/uv_cache}"
export WANDB_DIR="${WANDB_DIR:-${DELTA_WORK}/wandb}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DELTA_WORK}/triton_cache}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$WANDB_DIR" "$TRITON_CACHE_DIR" "$DELTA_DATA"

# Charge every `sbatch` in this shell to the GPU allocation without -A. sbatch
# reads SBATCH_ACCOUNT from the environment, and an explicit `-A` still wins —
# which is how the CPU-only corpus job is charged to DELTA_CPU_ACCOUNT instead.
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-$DELTA_ACCOUNT}"

export PATH="$HOME/.local/bin:$PATH"   # where the uv installer puts `uv`
cd "$GEOAE_ROOT"
