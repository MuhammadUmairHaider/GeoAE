#!/bin/bash
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=. HF_HUB_OFFLINE=1
D=e2e/checkpoints/general/gemma3-4B/layer33/delta
.venv/bin/python -u geoae/evaluation/alignment.py --cache cache/gemma3-4b/l33 --fp_min 6 --pairs res_plain:nobypass,tok_plain:geoae --pair_min 10 \
    --arms "res_plain=km:$D/plain_kmeans_k2000_dpc_sampled_struct5_seed42.npz,tok_plain=km:$D/plain_kmeans_k2000_dpc_sampled_struct5_tokmean_seed42.npz,geoae=ae:checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt,nobypass=ae:checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_train10M/step_0015250.pt" \
    --out results/delta/g4b_l33_nobypass/evals/probe_dense_subsets.json || echo FAILED
echo G4B_PAIR_DONE
