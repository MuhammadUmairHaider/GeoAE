#!/bin/bash
# Train (or continue) one GeoAE run:
#   sbatch -A <account> scripts/delta/train.sbatch configs/base/<run>.yaml [extra train args]
#
# Stages the activation dump onto node-local NVMe /tmp before training. Reading a
# 60 GB dump with random access from shared storage is what made epoch 1 take
# 44 min on the old box vs 7 min once cached; local NVMe avoids that entirely.
# If the run already has checkpoints it continues with --resume latest, so the
# same command resubmits a run that hit the walltime.
#SBATCH --job-name=geoae-train
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=100g
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
CONFIG="${1:?usage: sbatch train.sbatch <config.yaml> [extra args]}"; shift
source scripts/delta/env.sh

read -r ACT_DIR CKPT_DIR < <(uv run python - "$CONFIG" <<'PY'
import sys
from geoae.config import Config
c = Config.from_yaml(sys.argv[1])
print(c.data.activations_dir, c.train.checkpoints_dir)
PY
)
SRC="$GEOAE_ROOT/$ACT_DIR"
[ -d "$SRC" ] || { echo "missing $SRC — run extract.sbatch first"; exit 1; }
STAGE="/tmp/${SLURM_JOB_ID:-local}/$(basename "$ACT_DIR")"
mkdir -p "$STAGE"
echo "[stage] $SRC -> $STAGE"
time rsync -a "$SRC/" "$STAGE/"

RESUME=()
ls "$GEOAE_ROOT/$CKPT_DIR"/step_*.pt >/dev/null 2>&1 && RESUME=(--resume latest)

NAME=$(basename "$CONFIG" .yaml)
set +e
uv run python -u -m geoae.train --config "$CONFIG" --activations_dir "$STAGE" \
  "${RESUME[@]}" "$@" 2>&1 | tee -a "logs/${NAME}.log"
RC=${PIPESTATUS[0]}
set -e

# The first job on a fresh dump computes the normalisation stats inside the staged
# copy; keep them, so later jobs (and --resume's norm check) reuse the same file.
for f in "$STAGE"/norm_params_layer*.npz; do
  [ -e "$f" ] && [ ! -e "$SRC/$(basename "$f")" ] && cp -v "$f" "$SRC/"
done
exit "$RC"
