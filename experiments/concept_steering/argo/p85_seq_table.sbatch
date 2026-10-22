#!/bin/bash
# 85%-depth bypass arms (run p85_bypass_20261009): the paper's sequence-level bypass table (tab:seq-bypass), token-bypass
# AE vs plain k-means on the token-removed residual, K = 2000. Same recipe as experiments/model_transfer/argo/qwen_seq_table.sbatch
# (Qwen L35), one model per job:
#   1 plain    experiments/paper/analysis/paper_plain_kmeans_grid.py: Lloyd, density-peaks seeds, seed 42, residual + token-removed
#   2 toxicity concept_suite --only toxicity (Jigsaw 1,500 / 1,500; BOS as in each model's training dump)
#   3 probes   geoae/evaluation/alignment.py --pairs tok_plain:geoae --pair_min 10 (with and without anchor exclusion)
#   4 atlas    last word of each of the first 100k FineWeb-Atlas chunks (extraction settings of the model's last-layer
#              cache; --ref_cache asserts the same chunk stream), then atlas_nmi_filtered.py --min_rows 10
#   sbatch --job-name=p85-seqtab-llama --mem=200G experiments/concept_steering/argo/p85_seq_table.sbatch llama
#   sbatch --job-name=p85-seqtab-gemma --mem=300G experiments/concept_steering/argo/p85_seq_table.sbatch gemma
#   sbatch --job-name=p85-seqtab-qwen  --mem=150G experiments/concept_steering/argo/p85_seq_table.sbatch qwen
# Output: results/p85_bypass_20261009/<family>/layer<L>/seq_table/
#SBATCH --job-name=p85-seqtab
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=argo76
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=0-10:00:00
#SBATCH --output=logs/p85_bypass_20261009/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 GEOAE_NLTK_PATH=/scratch/dogar/geoae/pylib_nltk PYTHONPATH=.
PY="$UV_PROJECT_ENVIRONMENT/bin/python"

MODEL="${1:?llama|gemma|qwen}"
A=/scratch/dogar/geoae/activations
case "$MODEL" in
  llama) FAM=llama3.2-3B; LAYER=24; LM=meta-llama/Llama-3.2-3B; TAG=dpc_sampled_struct5_train10M
         AE=checkpoints/$FAM/layer$LAYER/k2000_bnh_b32k_lam1_d6144_${TAG}_tokbias/step_0015250.pt
         SRC=$A/activations_llama3_3b_l24_sampled_struct5_train10M
         ATL=/scratch/dogar/geoae/atlas_llama_l24_lastword; REF=/scratch/dogar/geoae/atlas_l27
         XARGS="--batch 128 --dtype float16 --model_dtype bfloat16" ;;
  gemma) FAM=gemma3-4B; LAYER=29; LM=google/gemma-3-4b-pt; TAG=dpc_sampled_struct5_train10M
         AE=checkpoints/$FAM/layer$LAYER/k2000_bnh_b32k_lam1_d5120_${TAG}_tokbias/step_0015250.pt
         SRC=$A/activations_gemma3_4b_l29_sampled_struct5_train10M
         ATL=/scratch/dogar/geoae/atlas_g4b_l29_lastword; REF=/scratch/dogar/geoae/atlas_g4b_l33
         XARGS="--batch 16 --dtype float32 --model_dtype float32" ;;            # Gemma overflows fp16
  qwen)  FAM=qwen2.5-3B; LAYER=31; LM=Qwen/Qwen2.5-3B; TAG=dpc_struct5_train10M
         AE=checkpoints/$FAM/layer$LAYER/k2000_bnh_b32k_lam1_d4096_${TAG}_tokbias/step_0015250.pt
         SRC=$A/activations_qwen2_5_3b_l31_struct5_train10M
         ATL=/scratch/dogar/geoae/atlas_qwen25_l31_lastword; REF=
         XARGS="--batch 64 --dtype float32 --model_dtype bfloat16" ;;           # as atlas_qwen25_l35_lastword
  *) echo "[abort] model must be llama, gemma or qwen"; exit 1 ;;
esac
E=e2e/checkpoints/general/$FAM/layer$LAYER/delta
KM=$E/balanced_kmeans_k2000_$TAG.npz; KMT=$E/balanced_kmeans_k2000_${TAG}_tokmean.npz
PKM=$E/plain_kmeans_k2000_${TAG}_seed42.npz; PKMT=$E/plain_kmeans_k2000_${TAG}_tokmean_seed42.npz
CACHE=cache/$FAM/layer$LAYER
META=/scratch/dogar/geoae/atlas_concept_meta.json
OUT=results/p85_bypass_20261009/$FAM/layer$LAYER/seq_table
for f in "$AE" "$KM" "$KMT" "$SRC/layer_$LAYER.npy" "$SRC/rows_tok.npy" "$META" "$GEOAE_NLTK_PATH/nltk" \
         "$CACHE/topic14.npz" "$CACHE/domain.npz" /scratch/dogar/geoae/atlas ${REF:+"$REF/chunks.npz"}; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
mkdir -p "$OUT/atlas_lastword"
echo "[p85-seqtab $MODEL] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1 plain k-means codebooks (dump copied to /dev/shm: /scratch is GPFS without page cache)
if [ ! -f "$PKM" ] || [ ! -f "$PKMT" ]; then
    STAGE=/dev/shm/p85_seqtab_${MODEL}_${SLURM_JOB_ID:-$$}
    trap 'rm -rf "$STAGE"' EXIT
    mkdir -p "$STAGE/act"
    echo "[p85-seqtab $MODEL] staging dump -> $STAGE $(date)"
    cp "$SRC/layer_$LAYER.npy" "$SRC/rows_tok.npy" "$SRC/meta.json" "$STAGE/act/"
    "$PY" -u experiments/paper/analysis/paper_plain_kmeans_grid.py --activations_dir "$STAGE/act" --layer $LAYER --km "$KM" --kmt "$KMT" \
        --seed 42 --tag _seed42 --out "$OUT/plain_kmeans_grid_seed42.json" 2>&1 | tee "$OUT/plain_kmeans_grid.log"
    rm -rf "$STAGE"
fi

# 2 Jigsaw toxicity cache (the CSV is read through the hub cache)
if [ ! -f "$CACHE/toxicity.npz" ]; then
    HF_HUB_OFFLINE=0 "$PY" -u -m geoae.extraction.concept_suite --model_name $LM --layer $LAYER --out "$CACHE" \
        --only toxicity 2>&1 | tee "$OUT/toxicity_cache.log"
fi

# 3 probes, paired filter over the two bypass spaces (plain residual k-means also scored, for reference)
ARMS="res_plain=km:$PKM,tok_plain=km:$PKMT,geoae=ae:$AE"
"$PY" -u geoae/evaluation/alignment.py --cache "$CACHE" --arms "$ARMS" --fp_min 6 \
    --pairs tok_plain:geoae --pair_min 10 --out "$OUT/probe_dense_subsets.json" 2>&1 | tee "$OUT/probe_dense_subsets.log"
"$PY" -u geoae/evaluation/alignment.py --cache "$CACHE" --arms "$ARMS" --fp_min 6 --no_anchor_exclusion \
    --pairs tok_plain:geoae --pair_min 10 --out "$OUT/probe_dense_subsets_allrows.json" 2>&1 | tee "$OUT/probe_dense_subsets_allrows.log"

# 4 FineWeb-Atlas, last word of each chunk
if [ ! -f "$ATL/meta.json" ]; then
    "$PY" -u geoae/extraction/atlas_siblings_extract.py --n_chunks 100000 --out_dir "$ATL" --model_name $LM --layer $LAYER \
        --max_len 256 $XARGS --sanity_warn_only --last_word ${REF:+--ref_cache "$REF"} \
        2>&1 | tee "$OUT/atlas_lastword/extract.log"
fi
"$PY" -u geoae/evaluation/atlas_assign.py --cache "$ATL" --models "bypass=$AE" \
    --baselines "plain42=$PKM,plain42_tokmean=$PKMT" --out "$OUT/atlas_lastword/assign.npz" \
    2>&1 | tee "$OUT/atlas_lastword/assign.log"
OMP_NUM_THREADS=8 "$PY" -u geoae/evaluation/atlas_alignment.py --assign "$OUT/atlas_lastword/assign.npz" \
    --chunks "$ATL/chunks.npz" --concepts "$META" --arms plain42_tokmean,bypass --ref bypass --min_rows 10 \
    --n_boot 300 --seed 0 --out "$OUT/atlas_lastword/nmi_pair10_bypass.json" 2>&1 | tee "$OUT/atlas_lastword/nmi_pair10_bypass.log"
echo "[p85-seqtab $MODEL] done $(date)"
