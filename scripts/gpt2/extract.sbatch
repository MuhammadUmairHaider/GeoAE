#!/bin/bash
# One GPT-2 pass captures all 12 layers, followed by 12 fixed token-mean tables.
# The dependent CPU job builds the corpus; GPU time only covers extraction/tables.
#SBATCH --job-name=gpt2-extract
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=logs/gpt2/slurm-%x-%j.out
set -euo pipefail
source scripts/gpt2/env.sh
CFG=configs/base/gpt2/extract_all.yaml
ACT=activations_gpt2_sampled_struct5_10M

if "$GEOAE_PYTHON" scripts/gpt2/lifecycle.py extraction --config "$CFG"; then
    echo "[skip] Complete all-layer extraction"
else
    # A restarted partial extraction must never inherit an old completion file.
    rm -f "$ACT/meta.json"
    "$GEOAE_PYTHON" -u -m geoae.extract --config "$CFG" \
        2>&1 | tee -a logs/gpt2/extraction.log
    "$GEOAE_PYTHON" scripts/gpt2/lifecycle.py extraction --config "$CFG"
fi

for L in {0..11}; do
    TABLE="e2e/checkpoints/general/gpt2/layer${L}/token_bias_sampled_struct5.npz"
    if "$GEOAE_PYTHON" scripts/gpt2/lifecycle.py table --layer "$L"; then
        echo "[skip] Token table layer $L is complete and matches normalization"
    else
        mkdir -p "$(dirname "$TABLE")"
        # Write to a temporary table and rename only after a successful process.
        "$GEOAE_PYTHON" -u -m geoae.token_bias --activations_dir "$ACT" --layer "$L" \
            --val_frac 0.05 --out "${TABLE%.npz}.tmp.npz" \
            2>&1 | tee -a "logs/gpt2/token_bias_layer${L}.log"
        mv -f "${TABLE%.npz}.tmp.npz" "$TABLE"
        "$GEOAE_PYTHON" scripts/gpt2/lifecycle.py table --layer "$L"
    fi
done
echo "[done] GPT-2 all-layer extraction and token tables"
