#!/bin/bash
# Plain token-removed k-means (density-peaks seeds): fit + usage + geometry, then the same probe and token-structure code as the paper.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=.
D=e2e/checkpoints/general/llama3.2-3B/layer27/delta
.venv/bin/python -u experiments/paper/analysis/paper_plain_kmeans_tokmean.py || { echo FAILED_FIT; exit 1; }
.venv/bin/python -u geoae/evaluation/chance_corrected_probe.py --cache cache_ids \
  --baselines "base_tokmean=$D/balanced_kmeans_k2000_dpc_sampled_tokmean.npz,plain_tokmean=$D/plain_kmeans_k2000_density_peaks_sampled_tokmean.npz" \
  --atlas_last "" --out results/delta/d6144/evals/probe_plain_kmeans_tokmean.json
.venv/bin/python -u geoae/evaluation/token_structure.py \
  --arms "base_tokmean=$D/balanced_kmeans_k2000_dpc_sampled_tokmean.npz,plain_tokmean=$D/plain_kmeans_k2000_density_peaks_sampled_tokmean.npz" \
  --activations_dir activations_sampled_10M --layer 27 \
  --out results/delta/d6144/evals/token_structure_plain_tokmean.json
echo PLAIN_TOKMEAN_DONE
