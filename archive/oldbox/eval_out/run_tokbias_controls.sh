#!/usr/bin/env bash
# MATCHED CONTROL for the token-bypass AE: balanced k-means on x - b[current token] with
# the SAME table (e2e/.../token_bias_sampled.npz), same dpc init / peaks reinit / K /
# balancing as km_new, no encoder. tokbias − km_tokmean is then the encoder's contribution.
#
# Two seeds (42, 7) for every k-means arm (the seed-7 raw / tok64 / pca64 fits already
# exist from run_token_split.sh), so each contrast is read against seed noise.
#
#   ./eval_out/run_tokbias_controls.sh           # ~35 min: 2 fits (~10 min each), probe, token structure, summary
#   ./eval_out/run_tokbias_controls.sh summary
# Fits ~20 GB of GPU memory; fine next to the interventions run (~13 GB). FORCE=1 redoes steps.
set -o pipefail
# Delta: env.sh resolves GEOAE_ROOT from its own location, exports HF_HOME /
# UV_PROJECT_ENVIRONMENT / SBATCH_ACCOUNT, and cds to the repo root.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/delta/env.sh"
PY=scripts/delta/py
E=e2e/checkpoints/general/llama3.2-3B/layer27
B=checkpoints/llama3.2-3B/layer27
PARENT=$B/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt
TB=$B/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
TABLE=$E/token_bias_sampled.npz
KM=$E/balanced_kmeans_k2000_dpc_sampled
STEPS=("$@"); [ $# -eq 0 ] && STEPS=(fits probe tokstruct summary)
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
have() { [ -z "$FORCE" ] && [ -f "$1" ] && echo "[skip] $1 exists" && return 0; return 1; }

if want fits; then
    for seed in 42 7; do
        out=${KM}_tokmean$([ $seed = 7 ] && echo _s7).npz
        have $out || $PY -u -m geoae.interp.fit_balanced_kmeans --checkpoint $PARENT \
            --activations activations_sampled_10M/layer_27.npy --init dpc --reinit peaks \
            --n_clusters 2000 --seed $seed --token_bias $TABLE \
            --out $out 2>&1 | tee logs/fit_balanced_$(basename $out .npz).log || exit 1
    done
fi

if want probe && ! have eval_out/probe_tokbias_controls.json; then
    BASE="km_tokmean=${KM}_tokmean.npz,km_tokmean_s7=${KM}_tokmean_s7.npz"
    BASE+=",km_tok64=${KM}_tok64.npz,km_tok64_s7=${KM}_tok64_s7.npz"
    BASE+=",km_pca64=${KM}_pca64.npz,km_pca64_s7=${KM}_pca64_s7.npz"
    BASE+=",km_new=${KM}.npz,km_new_s7=${KM}_s7.npz"
    $PY -u eval_out/probe_chance_corrected.py --cache cache_ids \
        --models "d6144_new=$PARENT,tokbias=$TB" --baselines "$BASE" \
        --out eval_out/probe_tokbias_controls.json 2>&1 | tee logs/probe_tokbias_controls.log || exit 1
fi

if want tokstruct && ! have eval_out/token_structure_tokbias_controls.json; then
    $PY -u eval_out/token_structure.py \
        --arms "d6144_new=$PARENT,tokbias=$TB,km_tokmean=${KM}_tokmean.npz,km_tok64=${KM}_tok64.npz,km_new=${KM}.npz" \
        --out eval_out/token_structure_tokbias_controls.json 2>&1 | tee logs/token_structure_tokbias_controls.log || exit 1
fi

want summary && $PY eval_out/summarize_tokbias_controls.py | tee logs/summary_tokbias_controls.log
