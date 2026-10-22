#!/bin/bash
# Plain k-means (density-peaks seeds) on Llama L27 with seed 42 (the seed of the trained models and balanced baselines), then probes + token structure.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=.
D=e2e/checkpoints/general/llama3.2-3B/layer27/delta
E=results/delta/d6144/evals
.venv/bin/python -u experiments/paper/analysis/paper_plain_kmeans_grid.py --activations_dir activations_sampled_10M --layer 27 \
  --km $D/balanced_kmeans_k2000_dpc_sampled.npz --kmt $D/balanced_kmeans_k2000_dpc_sampled_tokmean.npz \
  --seed 42 --tag _seed42 --out $E/plain_kmeans_grid_seed42.json || { echo FAILED_FIT; exit 1; }
ARMS="base=$D/balanced_kmeans_k2000_dpc_sampled.npz,plain_seed42=$D/plain_kmeans_k2000_dpc_sampled_seed42.npz,plain_tokmean_seed42=$D/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz"
.venv/bin/python -u geoae/evaluation/chance_corrected_probe.py --cache cache_ids --baselines "$ARMS" --atlas_last "" --out $E/probe_plain_kmeans_seed42.json
.venv/bin/python -u geoae/evaluation/token_structure.py --arms "$ARMS" --activations_dir activations_sampled_10M --layer 27 --out $E/token_structure_plain_kmeans_seed42.json
echo SEED42_DONE
