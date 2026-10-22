#!/bin/bash
# Refit plain k-means (saving centroids), then score both fits with the exact cq.json code path and sample.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=.
.venv/bin/python -u experiments/paper/analysis/paper_plain_kmeans_collapse.py
for init in kmeanspp density_peaks; do
  .venv/bin/python -u -m geoae.evaluation.geometry \
    --baseline e2e/checkpoints/general/llama3.2-3B/layer27/delta/plain_kmeans_k2000_${init}_sampled.npz \
    --activations_dir activations_sampled_10M --layer 27 --n_sample 1000000 --seed 0 \
    --names plain_${init} --out results/delta/d6144/evals/cq_plain_kmeans_${init}.json
done
echo PLAIN_CQ_DONE
