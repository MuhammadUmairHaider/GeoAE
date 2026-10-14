#!/bin/bash
# Train (or continue) one GeoAE run on ACENET Argo, one H100 on the group partition:
#   cd /home/dogar/projects/def-hsajjad/dogar/GeoAE/repo
#   sbatch scripts/argo/train.sbatch configs/base/<run>.yaml [extra train args]
#   sbatch --mem=240G scripts/argo/train.sbatch configs/base/<4B run>.yaml     # 102 GB float32 dump (Gemma 3 4B)
#   sbatch --time=1-00:00:00 ...                                               # any #SBATCH line can be overridden
#
# NEVER request more than 1 GPU: the group's argo73 (4x H100) is shared with other users.
#
# STAGE THE DUMP INTO RAM (/dev/shm). /scratch is GPFS, which does NOT use the Linux page cache: a memmap of a
# GPFS file serves every random row read from the network filesystem, and a `cat` prewarm caches nothing.
# Measured 2026-10-03 on the 4B dump: loader workers stuck in D state, GPU 0%, ~25 s/step (box: 2.1 s/step).
# So the dump dir is copied into /dev/shm/geoae_<jobid>/ (tmpfs, ~500 GB on argo73; GPFS reads it sequentially
# at several GB/s) and training memory-maps that copy; it is deleted on exit. tmpfs pages are charged to the
# job's cgroup: --mem must exceed dump size + ~30 GB. 160G covers the 10M-row Llama/Gemma-1B dumps (~46-61 GB);
# the 4B float32 dump (102 GB) needs --mem=240G.
#
# If the run already has checkpoints it continues with --resume latest, so the same command resubmits a run
# that hit the walltime. Add `--no_wandb` to skip W&B (without `wandb login` it falls back to stdout anyway).
#SBATCH --job-name=geoae-train
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/argo/train.sbatch <config.yaml> [extra args]}"; shift
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
mkdir -p logs

read -r ACT_DIR CKPT_DIR LAYER < <(uv run --no-sync python - "$CONFIG" <<'PY'
import sys
from geoae.config import Config
c = Config.from_yaml(sys.argv[1])
print(c.data.activations_dir, c.train.checkpoints_dir, c.data.target_layer)
PY
)
SRC="$GEOAE_ROOT/$ACT_DIR"
# Check for CONTENT, not just the directory: scripts/argo/link_data.sh creates the dump directory so its symlink
# resolves, and `-d` follows symlinks, so an empty dump would otherwise pass and fail later inside the loader.
if [ ! -d "$SRC" ] || [ -z "$(ls "$SRC"/layer_*.npy 2>/dev/null)" ]; then
  echo "no extracted dump at $SRC (no layer_*.npy)."
  echo "Copy it from Delta (rsync -a over the ControlMaster) or re-extract it first."
  exit 1
fi

SHM="/dev/shm/geoae_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$SHM"' EXIT
trap 'exit 143' TERM INT
DUMP="$SHM/$(basename "$ACT_DIR")"
mkdir -p "$DUMP"
echo "[stage] $SRC -> $DUMP (tmpfs; GPFS has no page cache)"
time cp -rL "$SRC/." "$DUMP/"
SRC="$DUMP"

RESUME=()
ls "$GEOAE_ROOT/$CKPT_DIR"/step_*.pt >/dev/null 2>&1 && RESUME=(--resume latest)

NAME=$(basename "$CONFIG" .yaml)
set +e
uv run --no-sync python -u -m geoae.train --config "$CONFIG" --activations_dir "$SRC" \
  "${RESUME[@]}" "$@" 2>&1 | tee -a "logs/${NAME}.log"
RC=${PIPESTATUS[0]}
set -e
exit "$RC"
