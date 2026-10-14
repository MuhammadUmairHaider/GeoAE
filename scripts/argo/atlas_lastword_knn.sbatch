#!/bin/bash
# Re-build the cross-document kNN graph of the last-word Atlas rows with another k (base / base_tokmean / bypass spaces),
# for both models, into <atlas_lastword>/k<K>/ (the k=10 graph of atlas_lastword.sbatch stays in place).
#   sbatch scripts/argo/atlas_lastword_knn.sbatch 20
#SBATCH --job-name=atlas-lastword-knn
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=argo73
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
K="${1:?usage: sbatch scripts/argo/atlas_lastword_knn.sbatch <k>}"
for MODEL in g4b llama; do
  case "$MODEL" in
    g4b)   CACHE=/scratch/dogar/geoae/atlas_g4b_l33_lastword; OUT=results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/k$K
           E=e2e/checkpoints/general/gemma3-4B/layer33/delta
           TB=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
           KM=$E/balanced_kmeans_k2000_dpc_sampled_struct5.npz; KMT=$E/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz ;;
    llama) CACHE=/scratch/dogar/geoae/atlas_l27_lastword; OUT=results/delta/d6144/evals/atlas_lastword/k$K
           E=e2e/checkpoints/general/llama3.2-3B/layer27/delta
           TB=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
           KM=$E/balanced_kmeans_k2000_dpc_sampled.npz; KMT=$E/balanced_kmeans_k2000_dpc_sampled_tokmean.npz ;;
  esac
  mkdir -p "$OUT"
  "$PY" -u eval_out/atlas_knn_graph.py --cache "$CACHE" --models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT" \
      --space_arms base,base_tokmean,bypass --all_rows --k "$K" --out_dir "$OUT" 2>&1 | tee "$OUT/knn_graph.log"
done
echo "[atlas-lastword-knn] done $(date)"
