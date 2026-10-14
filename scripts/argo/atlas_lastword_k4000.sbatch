#!/bin/bash
# K = 4,000 (or 8,000) token-bypass AEs on the last-word Atlas rows (caches of atlas_lastword.sbatch): codebook labels (Llama also the
# balanced K = 4,000 residual / token-removed k-means) and the cross-document kNN graph of the K = 4,000 latent at k = 10
# and k = 20, into <atlas_lastword>/k4000/ (statistics are run on Delta against the K = 2,000 files).
#   sbatch scripts/argo/atlas_lastword_k4000.sbatch [K=4000] [models="g4b llama"]   e.g. ... 8000 g4b (arm bypass8k, dir k8000/)
#SBATCH --job-name=atlas-lastword-k4000
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=argo76
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
KK="${1:-4000}"; MODELS="${2:-g4b llama}"; ARM="bypass$((KK / 1000))k"
for MODEL in $MODELS; do
  case "$MODEL" in
    g4b)   CACHE=/scratch/dogar/geoae/atlas_g4b_l33_lastword; OUT=results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/k$KK
           TB4=checkpoints/gemma3-4B/layer33/k${KK}_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
           BASE4="" ;;
    llama) CACHE=/scratch/dogar/geoae/atlas_l27_lastword; OUT=results/delta/d6144/evals/atlas_lastword/k$KK
           TB4=checkpoints/llama3.2-3B/layer27/k${KK}_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
           E=e2e/checkpoints/general/llama3.2-3B/layer27/delta
           BASE4=""; [ "$KK" = 4000 ] && BASE4="base4k=$E/balanced_kmeans_k4000_dpc_sampled.npz,base_tokmean4k=$E/balanced_kmeans_k4000_dpc_sampled_tokmean.npz" ;;
  esac
  mkdir -p "$OUT/k20"
  "$PY" -u eval_out/atlas_assign_extra.py --cache "$CACHE" --models "$ARM=$TB4" --baselines "$BASE4" \
      --out "$OUT/assign.npz" 2>&1 | tee "$OUT/assign.log"
  "$PY" -u eval_out/atlas_knn_graph.py --cache "$CACHE" --models "$ARM=$TB4" --space_arms "$ARM" --all_rows --k 10 \
      --out_dir "$OUT" 2>&1 | tee "$OUT/knn_graph.log"
  "$PY" -u eval_out/atlas_knn_graph.py --cache "$CACHE" --models "$ARM=$TB4" --space_arms "$ARM" --all_rows --k 20 \
      --out_dir "$OUT/k20" 2>&1 | tee "$OUT/k20/knn_graph.log"
done
echo "[atlas-lastword-k$KK] done $(date)"
