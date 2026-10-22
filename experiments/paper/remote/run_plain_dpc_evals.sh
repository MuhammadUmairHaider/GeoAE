#!/bin/bash
# Plain k-means (density-peaks seeds) through the same probe and token-structure code as the paper's baselines.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=.
D=e2e/checkpoints/general/llama3.2-3B/layer27/delta
.venv/bin/python -u geoae/evaluation/chance_corrected_probe.py --cache cache_ids \
  --baselines "base=$D/balanced_kmeans_k2000_dpc_sampled.npz,plain_dpc=$D/plain_kmeans_k2000_density_peaks_sampled.npz" \
  --atlas_last "" --out results/delta/d6144/evals/probe_plain_kmeans_dpc.json
.venv/bin/python -u geoae/evaluation/token_structure.py \
  --arms "base=$D/balanced_kmeans_k2000_dpc_sampled.npz,plain_dpc=$D/plain_kmeans_k2000_density_peaks_sampled.npz" \
  --activations_dir activations_sampled_10M --layer 27 \
  --out results/delta/d6144/evals/token_structure_plain_dpc.json
echo PLAIN_DPC_DONE
