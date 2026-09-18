#!/usr/bin/env bash
# LLM-judge auto-interp: BASE (balanced k-means, no encoder) vs GELU AE (dpc), L27.
#
# Balanced k-means is the fair base control. Do NOT use plain k-means
# (baseline_kmeans_k2000_refit.npz): it has ~1099 live clusters vs ~1990, so its
# intruders/hard negatives are easier — a capacity confound, not a win.
#
# Step 1 exists because closest_tokens stores only the RENDERED context, so a
# context window cannot be changed after the fact. closest_tokens_dpc.json is
# context_window 6; the base file is 128. A short-context judge run previously
# gave a +0.15 semantic win that did NOT replicate at long context — so both arms
# must be regenerated at the same window before judging.
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE

# ---- 1. dpc closest tokens at context 128, everything else identical to the base file
uv run python -u -m geoae.interp.closest_tokens \
  --checkpoint checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt \
  --n_tokens 500000 --top_n 30 --spectrum_n 6 --context_window 128 --seed 42 \
  --out results/ct_dpc_fullseq.json | tee logs/ct_dpc_fullseq.log

# ---- 2. single judge, same protocol as the earlier full run (n=200/arm, gemini-flash-lite)
#         ct_ae_fullseq.json (kmeans++ parent) is included as an AE-vs-AE reference.
uv run python -u -m geoae.interp.llm_judge \
  results/ct_balanced_fullseq.json results/ct_dpc_fullseq.json results/ct_ae_fullseq.json \
  --provider openrouter --model google/gemini-2.5-flash-lite \
  --n_clusters 200 --sample stratified --seed 0 \
  --out results/judge_base_vs_dpc.json | tee logs/judge_base_vs_dpc.log

# ---- 3. three judges — intrusion is the only metric with real inter-judge agreement
uv run python -u -m geoae.interp.judge_agreement \
  results/ct_balanced_fullseq.json results/ct_dpc_fullseq.json \
  --judges google/gemini-2.5-flash-lite,openai/gpt-4o-mini,deepseek/deepseek-chat-v3.1 \
  --provider openrouter --n_clusters 200 --sample stratified --seed 0 \
  --out results/judge_agreement_base_vs_dpc.json | tee logs/judge_agreement_base_vs_dpc.log
