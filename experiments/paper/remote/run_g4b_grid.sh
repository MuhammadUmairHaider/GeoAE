#!/bin/bash
# Gemma 3 4B L33: plain k-means (density-peaks seeds) on the residual and the token-removed residual, then the paper's probe and
# token-structure code on all four k-means codebooks (the balanced pair doubles as a check against the saved results).
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export PYTHONPATH=.
D=e2e/checkpoints/general/gemma3-4B/layer33/delta
ACT=activations_gemma3_4b_sampled_struct5_10M
E=results/delta/g4b_l33_d5120/evals
.venv/bin/python -u experiments/paper/analysis/paper_plain_kmeans_grid.py --activations_dir $ACT --layer 33 \
  --km $D/balanced_kmeans_k2000_dpc_sampled_struct5.npz --kmt $D/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz \
  --out $E/plain_kmeans_grid.json || { echo FAILED_FIT; exit 1; }
ARMS="base=$D/balanced_kmeans_k2000_dpc_sampled_struct5.npz,base_tokmean=$D/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz,plain=$D/plain_kmeans_k2000_dpc_sampled_struct5.npz,plain_tokmean=$D/plain_kmeans_k2000_dpc_sampled_struct5_tokmean.npz"
.venv/bin/python -u geoae/evaluation/chance_corrected_probe.py --cache cache/gemma3-4b/l33 --baselines "$ARMS" \
  --atlas_last "" --out $E/probe_plain_kmeans_grid.json
.venv/bin/python -u geoae/evaluation/token_structure.py --arms "$ARMS" --activations_dir $ACT --layer 33 \
  --out $E/token_structure_plain_kmeans_grid.json
echo G4B_GRID_DONE
