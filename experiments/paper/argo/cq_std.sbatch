#!/bin/bash
# Geometry with the width-neutral metrics (clustering_quality's silhouette_full, *_var_share, separability_radii,
# inter_centroid_dist_rel) for the two paper models whose dumps live on Argo:
#   sbatch experiments/paper/argo/cq_std.sbatch g4b      -> results/delta/g4b_l33_d5120/evals/cq_std.json
#   sbatch experiments/paper/argo/cq_std.sbatch qwen     -> results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/bypass/evals/cq_std.json
# Same sample as the existing cq.json (1M rows, seed 0, same codebook and checkpoint), written to a NEW file:
# cq.json is never touched. The dump is staged into tmpfs first (GPFS has no page cache; see evals_g4b.sbatch).
#SBATCH --job-name=cq-std
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
WHICH="${1:?usage: sbatch experiments/paper/argo/cq_std.sbatch <g4b|qwen>}"
PY=scripts/py
case "$WHICH" in
    g4b)  DUMP=activations_gemma3_4b_sampled_struct5_10M ;;
    qwen) DUMP=activations_qwen2_5_3b_l35_struct5_train10M ;;
    *)    echo "unknown model: $WHICH (g4b|qwen)" >&2; exit 2 ;;
esac
mkdir -p logs
SHM="/dev/shm/geoae_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$SHM"' EXIT
trap 'exit 143' TERM INT
export ACT_DIR="$SHM/$DUMP"
mkdir -p "$ACT_DIR"; echo "[stage] dump -> $ACT_DIR"; time cp -rL "$DUMP/." "$ACT_DIR/"
if [ "$WHICH" = g4b ]; then
    bash experiments/paper/scripts/run_delta_evals.sh g4b_l33_d5120 cq_std
else
    R=results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/bypass
    "$PY" -u -m geoae.evaluation.geometry \
        --baseline e2e/checkpoints/general/qwen2.5-3B/layer35/delta/balanced_kmeans_k2000_dpc_struct5_train10M.npz \
        --checkpoints checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt \
        --names base bypass --activations_dir "$ACT_DIR" --layer 35 --n_sample 1000000 --seed 0 \
        --out "$R/evals/cq_std.json" 2>&1 | tee "$R/logs/cq_std.log"
fi
echo "[cq_std] done rc=$?"
