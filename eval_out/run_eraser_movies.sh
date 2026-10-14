#!/usr/bin/env bash
# ERASER Movie Reviews rationale benchmark: base h vs bypass-AE z vs k-means codebooks (Llama-3.2-3B L27).
# Extra args are appended (smoke: --limit_train 100 --limit_val 40 --limit_test 20 --n_random 2 --n_boot 200
# --out ... ). Data: data/eraser/movies (official ERASER release).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/remote/env.sh"   # paths + cd to repo root (box)
PY="${PY:-.venv/bin/python}"
OUTDIR=results/delta/d6144/evals/eraser_movies
CB=e2e/checkpoints/general/llama3.2-3B/layer27/delta
mkdir -p "$OUTDIR" cache/eraser_movies/d6144
"$PY" -u -m geoae.interp.eraser_movies \
  --data_dir data/eraser/movies --model_name meta-llama/Llama-3.2-3B --layer 27 \
  --ae checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt \
  --codebooks km_plain=$CB/plain_kmeans_k2000_density_peaks_sampled.npz,kmt_plain=$CB/plain_kmeans_k2000_density_peaks_sampled_tokmean.npz,km_bal=$CB/balanced_kmeans_k2000_dpc_sampled.npz,kmt_bal=$CB/balanced_kmeans_k2000_dpc_sampled_tokmean.npz \
  --cache_dir cache/eraser_movies/d6144 --out "$OUTDIR/eraser_movies.json" "$@" \
  2>&1 | tee "${OUTDIR}/eraser_movies.log"
