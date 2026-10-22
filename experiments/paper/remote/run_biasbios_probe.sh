#!/bin/bash
# Bias in Bios clustering probe (profession, gender) for the four last-layer spaces, both models; paired filter.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
LB=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
LN=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt
GB=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
GN=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_train10M/step_0015250.pt
L=e2e/checkpoints/general/llama3.2-3B/layer27/delta
G=e2e/checkpoints/general/gemma3-4B/layer33/delta
.venv/bin/python -u -m geoae.extraction.concept_suite --checkpoint $LB --out cache_ids --only biasbios || echo "FAILED llama cache"
.venv/bin/python -u -m geoae.extraction.concept_suite --checkpoint $GB --out cache/gemma3-4b/l33 --only biasbios || echo "FAILED gemma cache"
R="--rungs biasbios_profession,biasbios_gender --no_anchor_exclusion --fp_min 6 --pairs res_plain:nobypass,tok_plain:geoae --pair_min 10"
.venv/bin/python -u geoae/evaluation/alignment.py --cache cache_ids $R \
  --arms "res_plain=km:$L/plain_kmeans_k2000_dpc_sampled_seed42.npz,tok_plain=km:$L/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz,geoae=ae:$LB,nobypass=ae:$LN" \
  --out results/delta/d6144_nobypass/evals/probe_biasbios.json || echo "FAILED llama probe"
.venv/bin/python -u geoae/evaluation/alignment.py --cache cache/gemma3-4b/l33 $R \
  --arms "res_plain=km:$G/plain_kmeans_k2000_dpc_sampled_struct5_seed42.npz,tok_plain=km:$G/plain_kmeans_k2000_dpc_sampled_struct5_tokmean_seed42.npz,geoae=ae:$GB,nobypass=ae:$GN" \
  --out results/delta/g4b_l33_nobypass/evals/probe_biasbios.json || echo "FAILED gemma probe"
echo BIASBIOS_DONE
