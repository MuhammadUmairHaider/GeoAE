#!/bin/bash
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=. HF_HUB_OFFLINE=1
PAIRS="res_plain:nobypass,tok_plain:geoae"
L=e2e/checkpoints/general/llama3.2-3B/layer27/delta
.venv/bin/python -u geoae/evaluation/alignment.py --cache cache_ids --fp_min 6 --pairs $PAIRS --pair_min 10 \
    --arms "res_plain=km:$L/plain_kmeans_k2000_dpc_sampled_seed42.npz,tok_plain=km:$L/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz,geoae=ae:checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt,nobypass=ae:checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt" \
    --out results/delta/d6144_nobypass/evals/probe_dense_subsets.json || echo "FAILED llama"
echo LLAMA_PAIR_DONE
