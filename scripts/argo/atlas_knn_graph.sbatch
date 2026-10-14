#!/bin/bash
# Cross-document kNN graph over a fixed sample of Atlas cache rows (Gemma 3 4B layer 33): 3 non-final rows per chunk
# (300,000 rows), k=10 nearest rows by squared Euclidean distance with the query's own DOCUMENT excluded, one graph per
# space (base / base_tokmean / bypass / Matryoshka-10k codes). Neighbour indices only; the per-concept statistics are
# computed on the CPU afterwards. eval_out/atlas_knn_graph.py.
#   sbatch scripts/argo/atlas_knn_graph.sbatch               (H100; on argo76 add --gres=gpu:l40:1 --nodelist=argo76)
#   sbatch scripts/argo/atlas_knn_graph.sbatch --per_chunk 3 --k 10 --seed 0 --space_arms base,base_tokmean,bypass,mat10k
# Output: results/delta/g4b_l33_d5120/evals/sae_suite/atlas_concepts/{knn_graph.npz,knn_graph_meta.json}
#SBATCH --job-name=atlas-knngraph
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

ARM=g4b_l33_d5120; LAYER=33; K=2000
CACHE=/scratch/dogar/geoae/atlas_g4b_l33
OUT=results/delta/$ARM/evals/sae_suite/atlas_concepts
E=e2e/checkpoints/general/gemma3-4B/layer$LAYER
B=checkpoints/gemma3-4B/layer$LAYER
TB=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
KM=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5.npz
KMT=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
TRAIN=results/delta/$ARM/evals/gemma_scope_compare/matched_topk/training
CTL=results/delta/$ARM/evals/sae_suite/controls
MAT=$TRAIN/matryoshka_10240/last.pt
BTK=$TRAIN/batch_topk_10240/last.pt
for f in "$TB" "$KM" "$KMT" "$CACHE/rows.npy" "$CACHE/rows_tok.npy" "$CACHE/rows_nxt.npy" "$CACHE/rows_chunk.npy" \
         "$CACHE/rows_pos.npy" "$CACHE/chunks.npz" "$MAT" "$BTK" "$CTL/kmeans_mat10k.npz" "$CTL/kmeans_bt10k.npz"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
echo "[atlas-knngraph] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SHM="/dev/shm/geoae_atlasknn_${SLURM_JOB_ID:-$$}"
mkdir -p "$SHM"; trap 'rm -rf "$SHM"' EXIT; trap 'exit 143' TERM INT
time cp "$CACHE"/rows.npy "$CACHE"/rows_tok.npy "$CACHE"/rows_nxt.npy "$CACHE"/rows_chunk.npy "$CACHE"/rows_pos.npy "$SHM"/
"$PY" -u eval_out/atlas_knn_graph.py --cache "$CACHE" --rows_dir "$SHM" \
    --models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT" \
    --saes "mat10k=matched:$MAT,bt10k=matched:$BTK" \
    --sae_kmeans "mat10k_km=matched:$MAT@$CTL/kmeans_mat10k.npz,bt10k_km=matched:$BTK@$CTL/kmeans_bt10k.npz" \
    --out_dir "$OUT" "$@" 2>&1 | tee logs/atlas_knn_graph.log
exit "${PIPESTATUS[0]}"
