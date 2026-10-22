#!/bin/bash
# Token-level Atlas in detail (Gemma 3 4B layer 33): per-row labels of base / base_tokmean / bypass / SAE arms on the
# whole Atlas cache + knn and linear-probe scores per continuous space. geoae/analysis/atlas_token_detail.py.
#   sbatch experiments/paper/argo/atlas_token_detail.sbatch            (H100; on argo76 add --gres=gpu:l40:1 --nodelist=argo76)
#   sbatch experiments/paper/argo/atlas_token_detail.sbatch --tok_knn_only   (same-token / other-token knn -> knn_tok.json)
# Output: results/delta/g4b_l33_d5120/evals/sae_suite/atlas_token/{assign.npz,knn_rows.npz,space_scores.json}
#SBATCH --job-name=atlas-tokdetail
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

ARM=g4b_l33_d5120; LAYER=33; K=2000
ATLAS=/scratch/dogar/geoae/atlas
CACHE=/scratch/dogar/geoae/atlas_g4b_l33
DRAWS=results/delta/d6144/interventions/atlas_classes/draws.json
OUT=results/delta/$ARM/evals/sae_suite/atlas_token
E=e2e/checkpoints/general/gemma3-4B/layer$LAYER
B=checkpoints/gemma3-4B/layer$LAYER
TB=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
KM=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5.npz
KMT=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
TRAIN=results/delta/$ARM/evals/gemma_scope_compare/matched_topk/training
CTL=results/delta/$ARM/evals/sae_suite/controls
MAT=$TRAIN/matryoshka_10240/last.pt
BTK=$TRAIN/batch_topk_10240/last.pt
for f in "$TB" "$KM" "$KMT" "$DRAWS" "$CACHE/rows.npy" "$MAT" "$BTK" "$CTL/kmeans_mat10k.npz" "$CTL/kmeans_bt10k.npz"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
echo "[atlas-tokdetail] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SHM="/dev/shm/geoae_atlastok_${SLURM_JOB_ID:-$$}"
mkdir -p "$SHM"; trap 'rm -rf "$SHM"' EXIT; trap 'exit 143' TERM INT
time cp "$CACHE"/rows.npy "$CACHE"/rows_tok.npy "$CACHE"/rows_nxt.npy "$CACHE"/rows_chunk.npy "$CACHE"/rows_pos.npy "$SHM"/
"$PY" -u geoae/analysis/atlas_token_detail.py --cache "$CACHE" --rows_dir "$SHM" --draws "$DRAWS" --atlas_dir "$ATLAS" \
    --models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT" \
    --saes "mat10k=matched:$MAT,bt10k=matched:$BTK" \
    --sae_kmeans "mat10k_km=matched:$MAT@$CTL/kmeans_mat10k.npz,bt10k_km=matched:$BTK@$CTL/kmeans_bt10k.npz" \
    --out_dir "$OUT" "$@" 2>&1 | tee "logs/atlas_token_detail${1:+_tok}.log"
exit "${PIPESTATUS[0]}"
