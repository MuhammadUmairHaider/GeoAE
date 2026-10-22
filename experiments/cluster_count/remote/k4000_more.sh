#!/usr/bin/env bash
# More K-sensitivity analyses on the box (sequential; each step continues on failure).
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
export GEOAE_REMOTE=1
PY=.venv/bin/python
KMD=e2e/checkpoints/general/llama3.2-3B/layer27/delta
CKD=checkpoints/llama3.2-3B/layer27
step() { local name=$1; shift; echo "[k4000more $(date +%H:%M:%S)] $name start"; "$@"; echo "[k4000more $(date +%H:%M:%S)] $name rc=$?"; }
km() { echo $KMD/balanced_kmeans_k$1_dpc_sampled.npz; }
kmt() { echo $KMD/balanced_kmeans_k$1_dpc_sampled_tokmean.npz; }
ck() { echo $CKD/k$1_bnh_b32k_lam1_d$2_dpc_sampled_tokbias/step_0014450.pt; }
mkdir -p results/delta/d12288/evals results/delta/d12288k4000/evals

# 1-3: d6144 K4000 counterparts of the K2000 range-hypothesis analyses
O=results/delta/d6144k4000/evals
step dvr_d6144k4000 env DVR_KM=$(km 4000) DVR_KMT=$(kmt 4000) DVR_CK=$(ck 4000 6144) DVR_OUT=$O/direction_vs_radius.json $PY experiments/paper/analysis/paper_direction_vs_radius.py
step counterparts_d6144k4000 env BC_CKPT=$(ck 4000 6144) BC_KM=$(km 4000) BC_KMT=$(kmt 4000) BC_REMOVAL=$O/same_direction_removal.json BC_RANGES=$O/ranges_along_directions.json \
  $PY experiments/paper/analysis/paper_base_counterparts.py --cache $O/same_direction_rows.npz --out $O/base_counterparts.json
step base_selected_d6144k4000 env SDR_CKPT=$(ck 4000 6144) SDR_RANGES=$O/ranges_along_directions.json SDR_KM=$(km 4000) SDR_KMT=$(kmt 4000) \
  $PY experiments/paper/analysis/paper_same_direction_removal.py --selection_space residual --no_showcase --min_dprime 3.01 --cache $O/base_selected_rows.npz --out $O/base_selected_removal.json

# 4-9: d12288 at both K (no range analyses exist at 4x width yet)
for K in 2000 4000; do
  if [ $K = 2000 ]; then R=d12288; else R=d12288k4000; fi
  O=results/delta/$R/evals
  step ranges_$R env RANGES_KM=$(km $K) RANGES_KMT=$(kmt $K) RANGES_CK=$(ck $K 12288) RANGES_OUT=$O/ranges_along_directions.json $PY experiments/paper/analysis/paper_ranges_along_directions.py
  step removal_$R env SDR_CKPT=$(ck $K 12288) SDR_RANGES=$O/ranges_along_directions.json SDR_KM=$(km $K) SDR_KMT=$(kmt $K) \
    $PY experiments/paper/analysis/paper_same_direction_removal.py --cache $O/same_direction_rows.npz --out $O/same_direction_removal.json
  step dvr_$R env DVR_KM=$(km $K) DVR_KMT=$(kmt $K) DVR_CK=$(ck $K 12288) DVR_OUT=$O/direction_vs_radius.json $PY experiments/paper/analysis/paper_direction_vs_radius.py
done
echo K4000_MORE_DONE
