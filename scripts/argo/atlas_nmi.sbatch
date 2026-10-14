#!/bin/bash
# Atlas NMI for Gemma 3 4B layer 33 (last decoder block, d=2560): chance-corrected NMI of every space's assignment
# (base / base_tokmean k-means, bypass AE, six SAEs) vs FineWeb-Atlas labels. Two stages, one job each:
#   extract  100k Atlas chunks x (8 sampled positions + the final token), float32 -> $CACHE (~9 GB)
#   score    eval_out/atlas_nmi.py on that cache (rows staged to /dev/shm; /scratch has no page cache)
#   JID=$(sbatch --parsable scripts/argo/atlas_nmi.sbatch extract)
#   sbatch --dependency=afterok:$JID scripts/argo/atlas_nmi.sbatch score
#   on argo76 (L40): add  --gres=gpu:l40:1 --nodelist=argo76  to either sbatch (command-line flags override the defaults below)
# Output: results/delta/g4b_l33_d5120/evals/sae_suite/atlas_nmi.json
#SBATCH --job-name=atlas-nmi
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
STAGE="${1:?usage: sbatch scripts/argo/atlas_nmi.sbatch extract|score}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

ARM=g4b_l33_d5120; LAYER=33; K=2000
ATLAS=/scratch/dogar/geoae/atlas
CACHE=/scratch/dogar/geoae/atlas_g4b_l33
DRAWS=results/delta/d6144/interventions/atlas_classes/draws.json
OUT=results/delta/$ARM/evals/sae_suite
mkdir -p "$OUT"
echo "[atlas-nmi] stage=$STAGE host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /scratch/dogar/geoae | tail -1

case "$STAGE" in
extract)
  "$PY" -u eval_out/atlas_siblings_extract.py --model_name google/gemma-3-4b-pt --layer "$LAYER" \
      --n_chunks 100000 --batch 16 --positions 8 --include_last --dtype float32 --model_dtype float32 --sanity_warn_only \
      --atlas_dir "$ATLAS" --out_dir "$CACHE" 2>&1 | tee "logs/atlas_nmi_extract.log"
  exit "${PIPESTATUS[0]}" ;;
score)
  E=e2e/checkpoints/general/gemma3-4B/layer$LAYER
  B=checkpoints/gemma3-4B/layer$LAYER
  TB=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
  KM=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5.npz
  KMT=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
  TRAIN=results/delta/$ARM/evals/gemma_scope_compare/matched_topk/training
  SAES="bt10k=matched:$TRAIN/batch_topk_10240/last.pt,bt20k=matched:$TRAIN/batch_topk_20480/last.pt"
  SAES="$SAES,mat10k=matched:$TRAIN/matryoshka_10240/last.pt,mat20k=matched:$TRAIN/matryoshka_20480/last.pt"
  SAES="$SAES,gs16k=scope:16k,gs262k=scope:262k"
  for f in "$TB" "$KM" "$KMT" "$DRAWS" "$CACHE/rows.npy" "$CACHE/meta.json" "$TRAIN"/{batch_topk,matryoshka}_{10240,20480}/last.pt; do
      [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
  done
  SHM="/dev/shm/geoae_atlasnmi_${SLURM_JOB_ID:-$$}"
  mkdir -p "$SHM"; trap 'rm -rf "$SHM"' EXIT; trap 'exit 143' TERM INT
  echo "[stage] row arrays -> $SHM"; time cp "$CACHE"/rows.npy "$CACHE"/rows_tok.npy "$CACHE"/rows_nxt.npy \
      "$CACHE"/rows_chunk.npy "$CACHE"/rows_pos.npy "$SHM"/
  "$PY" -u eval_out/atlas_nmi.py --cache "$CACHE" --rows_dir "$SHM" --draws "$DRAWS" --atlas_dir "$ATLAS" \
      --models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT" --saes "$SAES" \
      --out "$OUT/atlas_nmi.json" 2>&1 | tee "logs/atlas_nmi_score.log"
  exit "${PIPESTATUS[0]}" ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
