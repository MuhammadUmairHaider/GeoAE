#!/bin/bash
# One array task per block. Stage only that block's 30.72 GB dump and sidecars.
#SBATCH --job-name=gpt2-2x
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=96g
#SBATCH --time=24:00:00
#SBATCH --output=logs/gpt2/slurm-%x-%A_%a.out
set -euo pipefail
source scripts/gpt2/env.sh
L="${SLURM_ARRAY_TASK_ID:?Submit with --array=0-11%2 (or a single layer)}"
[[ "$L" =~ ^[0-9]+$ ]] && [ "$L" -le 11 ] || { echo "Invalid layer: $L" >&2; exit 1; }
CFG="configs/base/gpt2/layer${L}_2x.yaml"
ACT="$GEOAE_ROOT/activations_gpt2_sampled_struct5_10M"
CKPT="$GEOAE_ROOT/checkpoints/gpt2/layer${L}/k2000_bnh_b32k_lam1_d1536_dpc_sampled_struct5_tokbias"
"$GEOAE_PYTHON" scripts/gpt2/lifecycle.py extraction --layer "$L"
"$GEOAE_PYTHON" scripts/gpt2/lifecycle.py table --layer "$L"

if "$GEOAE_PYTHON" scripts/gpt2/lifecycle.py complete --layer "$L"; then
    echo "[skip] Layer $L already has a verified final checkpoint"
    exit 0
fi

STAGE="${GEOAE_STAGE_ROOT:-/tmp}/${SLURM_JOB_ID:-local}_gpt2_layer${L}"
mkdir -p "$STAGE"
trap 'rm -rf -- "$STAGE"' EXIT
FILES=("layer_${L}.npy" meta.json docs.npy "norm_params_layer${L}.npz")
for f in "$ACT"/rows_*.npy; do FILES+=("$(basename "$f")"); done
echo "[stage] Layer $L and shared sidecars -> $STAGE"
for f in "${FILES[@]}"; do
    [ -f "$ACT/$f" ] || { echo "Missing required extraction file: $ACT/$f" >&2; exit 1; }
    rsync -a "$ACT/$f" "$STAGE/"
done

RESUME=()
compgen -G "$CKPT/step_*.pt" >/dev/null && RESUME=(--resume latest)
"$GEOAE_PYTHON" -u -m geoae.train --config "$CFG" --activations_dir "$STAGE" \
    --no_wandb "${RESUME[@]}" 2>&1 | tee -a "logs/gpt2/train_layer${L}.log"
# A zero exit code alone is insufficient: check epoch/config, then atomically
# publish the marker consumed by the serial remote-evaluation watcher.
"$GEOAE_PYTHON" scripts/gpt2/lifecycle.py complete --layer "$L"
