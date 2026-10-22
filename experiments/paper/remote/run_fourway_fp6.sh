#!/bin/bash
# Four-way probes with the reporting rule, the fixed-point subset defined by EXACTLY the four compared spaces
# (plain residual, plain token-removed, bypass AE, no-bypass AE), as the mid-layer final_probes. Waits for the
# Gemma no-bypass evaluation run to finish first (same GPU).
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=. HF_HUB_OFFLINE=1
until grep -q G4B_NOBYPASS_EVALS_DONE logs/g4b_nobypass_evals.log; do sleep 30; done
D=e2e/checkpoints/general/gemma3-4B/layer33/delta
O=results/delta/g4b_l33_nobypass/evals
.venv/bin/python -u geoae/evaluation/alignment.py --cache cache/gemma3-4b/l33 --fp_min 6 \
    --arms "res_plain=km:$D/plain_kmeans_k2000_dpc_sampled_struct5_seed42.npz,tok_plain=km:$D/plain_kmeans_k2000_dpc_sampled_struct5_tokmean_seed42.npz,geoae=ae:checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt,nobypass=ae:checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_train10M/step_0015250.pt" \
    --out $O/probe_dense_subsets.json || echo "FAILED gemma probes"
.venv/bin/python experiments/paper/analysis/summarize_probe_raw_hc.py $O/probe_dense_subsets.json common_fp6 all > $O/probe_raw_hc.md
L=e2e/checkpoints/general/llama3.2-3B/layer27/delta
NL=results/delta/d6144_nobypass/evals
.venv/bin/python -u geoae/evaluation/alignment.py --cache cache_ids --fp_min 6 \
    --arms "res_plain=km:$L/plain_kmeans_k2000_dpc_sampled_seed42.npz,tok_plain=km:$L/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz,geoae=ae:checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt,nobypass=ae:checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt" \
    --out $NL/probe_dense_subsets.json || echo "FAILED llama probes"
.venv/bin/python experiments/paper/analysis/summarize_probe_raw_hc.py $NL/probe_dense_subsets.json common_fp6 all > $NL/probe_raw_hc.md
echo FOURWAY_FP6_DONE
