#!/bin/bash
# Plain (standard) k-means at K = 4,000 on the Llama-3.2-3B L27 residual stream, for the K = 4,000 row of the paper's
# ablation table (Appendix C), which until now had only a balanced k-means reference at that K. Same recipe as the
# K = 2,000 plain baseline of Table 1: experiments/paper/analysis/paper_plain_kmeans_grid.py (density-peaks seeds, seed 42, Lloyd on every
# 6th training row, geometry on the 1M-row cq.json sample), then the single-token share (trivial_cur_80) on the held-out
# rows with geoae/evaluation/token_structure.py.
# The balanced K = 4,000 codebook is passed for its standardization and re-scored as a check:
#   geometry balanced_residual CH should reproduce results/delta/d6144k4000/evals/cq.json "base" (109.77);
#   token structure base4k trivial_cur_80 should reproduce results/delta/d6144k4000/evals/token_structure.json "base" (0.1162).
#   sbatch experiments/cluster_count/argo/llama_plain_k4000.sbatch
# Output: results/delta/d6144k4000/evals/{plain_kmeans_grid_seed42.json,token_structure_plain_kmeans_seed42.json},
#         e2e/checkpoints/general/llama3.2-3B/layer27/delta/plain_kmeans_k4000_dpc_sampled_seed42.npz
#SBATCH --job-name=llama-plain-k4000
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=8
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
ACT=activations_sampled_10M; LAYER=27
E=e2e/checkpoints/general/llama3.2-3B/layer27/delta
KM=$E/balanced_kmeans_k4000_dpc_sampled.npz; KMT=$E/balanced_kmeans_k4000_dpc_sampled_tokmean.npz
P=$E/plain_kmeans_k4000_dpc_sampled_seed42.npz
O=results/delta/d6144k4000/evals; mkdir -p "$O" logs
for f in "$KM" "$KMT" "$ACT/layer_$LAYER.npy" "$ACT/rows_tok.npy"; do
  [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
echo "[plain-k4000] host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# /scratch is GPFS without page cache: random rows are served from the network, so stage the dump in RAM first.
SHM="/dev/shm/geoae_${SLURM_JOB_ID:-$$}"; trap 'rm -rf "$SHM"' EXIT; trap 'exit 143' TERM INT
A="$SHM/$ACT"; mkdir -p "$A"; echo "[stage] $ACT -> $A"; time cp -rL "$ACT/." "$A/"
"$PY" -u experiments/paper/analysis/paper_plain_kmeans_grid.py --activations_dir "$A" --layer $LAYER --km "$KM" --kmt "$KMT" \
    --spaces residual --seed 42 --tag _seed42 --out "$O/plain_kmeans_grid_seed42.json" \
    || { echo "[abort] plain k-means fit failed"; exit 1; }
[ -s "$P" ] || { echo "[abort] $P was not written"; exit 1; }
"$PY" -u geoae/evaluation/token_structure.py --arms "base4k=$KM,plain4k_seed42=$P" --activations_dir "$A" --layer $LAYER \
    --skip_next --out "$O/token_structure_plain_kmeans_seed42.json" \
    || { echo "[abort] token structure failed"; exit 1; }
echo "[plain-k4000] finished $(date)"
