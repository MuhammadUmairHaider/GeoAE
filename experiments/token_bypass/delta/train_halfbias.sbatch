#!/bin/bash
# Prepare the alpha=0.5 fixed table on a compute node, then use the normal trainer.
#SBATCH --job-name=geoae-halfbias
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=100g
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source scripts/env.sh
BASE=e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz
HALF=e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled_alpha0p5.npz
if [ ! -f "$HALF" ]; then
    scripts/py geoae/extraction/token_bias_scale.py --source "$BASE" --out "$HALF" --alpha 0.5
fi
# Validate an existing table too, so a resubmission cannot silently reuse another alpha.
scripts/py - "$BASE" "$HALF" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=True) as base, np.load(sys.argv[2], allow_pickle=True) as half:
    assert float(half['bias_scale']) == 0.5
    for key in ('token_ids', 'counts', 'norm_mean', 'norm_std', 'vocab_size'):
        assert np.array_equal(base[key], half[key]), key
    expected = (base['table'].astype(np.float32) * 0.5).astype(base['table'].dtype)
    assert np.array_equal(half['table'], expected), 'scaled table differs'
print('[token-bias-scale] verified table, token IDs and normalization', flush=True)
PY
exec bash scripts/slurm/delta_train.sbatch \
    experiments/token_bypass/configs/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_alpha0p5.yaml \
    "$@"
