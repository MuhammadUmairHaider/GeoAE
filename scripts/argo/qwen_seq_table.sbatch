#!/bin/bash
# Qwen2.5-3B layer 35 (last block): the paper's sequence-level bypass table (tab:seq-bypass), token-bypass AE vs plain
# k-means on the token-removed residual (same token-mean table, no encoder), K = 2000, as for Llama L27 / Gemma L33.
#   1 plain    eval_out/paper_plain_kmeans_grid.py: Lloyd, density-peaks seeds, seed 42, residual + token-removed
#              (dump copied to /dev/shm: /scratch is GPFS without page cache)
#   2 toxicity concept_suite --only toxicity (Jigsaw 1,500 toxic / 1,500 clean; Qwen2.5 has no BOS, so none is added,
#              exactly as in its training dump)
#   3 probes   eval_out/probe_dense_subsets.py --pairs tok_plain:geoae --pair_min 10 (rows whose cluster holds >= 10
#              kept rows in BOTH bypass spaces, fixed point); A = anchor exclusion as the Llama/Gemma topic rows,
#              B = every row (as their toxicity row)
#   4 atlas    last word of each of the first 100k FineWeb-Atlas chunks (same chunk stream and labels as atlas_l27 /
#              atlas_g4b_l33; Qwen tokens), labels for plain / plain token-removed / bypass, then
#              eval_out/atlas_nmi_filtered.py --arms plain42_tokmean,bypass --min_rows 10 (3 draws of 14 topics)
#   sbatch scripts/argo/qwen_seq_table.sbatch
# Output: results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/seq_table/
#SBATCH --job-name=qwen-seqtab
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=argo76
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=0-10:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 GEOAE_NLTK_PATH=/scratch/dogar/geoae/pylib_nltk PYTHONPATH=.
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

LAYER=35; LM=Qwen/Qwen2.5-3B
E=e2e/checkpoints/general/qwen2.5-3B/layer$LAYER/delta
KM=$E/balanced_kmeans_k2000_dpc_struct5_train10M.npz; KMT=$E/balanced_kmeans_k2000_dpc_struct5_train10M_tokmean.npz
PKM=$E/plain_kmeans_k2000_dpc_struct5_train10M_seed42.npz; PKMT=$E/plain_kmeans_k2000_dpc_struct5_train10M_tokmean_seed42.npz
AE=checkpoints/qwen2.5-3B/layer$LAYER/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt
SRC=/scratch/dogar/geoae/activations/activations_qwen2_5_3b_l35_struct5_train10M
CACHE=cache/qwen2.5-3B/layer$LAYER
ATL=/scratch/dogar/geoae/atlas_qwen25_l35_lastword
META=/scratch/dogar/geoae/atlas_concept_meta.json
OUT=results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer$LAYER/seq_table
for f in "$AE" "$KM" "$KMT" "$SRC/layer_$LAYER.npy" "$SRC/rows_tok.npy" "$META" "$GEOAE_NLTK_PATH/nltk" \
         "$CACHE/topic14.npz" "$CACHE/domain.npz" /scratch/dogar/geoae/atlas; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
mkdir -p "$OUT/atlas_lastword"
echo "[qwen-seqtab] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1 plain k-means codebooks
if [ ! -f "$PKM" ] || [ ! -f "$PKMT" ]; then
    STAGE=/dev/shm/qwen_seqtab_${SLURM_JOB_ID:-$$}
    trap 'rm -rf "$STAGE"' EXIT
    mkdir -p "$STAGE/act"
    echo "[qwen-seqtab] staging dump -> $STAGE $(date)"
    cp "$SRC/layer_$LAYER.npy" "$SRC/rows_tok.npy" "$SRC/meta.json" "$STAGE/act/"
    "$PY" -u eval_out/paper_plain_kmeans_grid.py --activations_dir "$STAGE/act" --layer $LAYER --km "$KM" --kmt "$KMT" \
        --seed 42 --tag _seed42 --out "$OUT/plain_kmeans_grid_seed42.json" 2>&1 | tee "$OUT/plain_kmeans_grid.log"
    rm -rf "$STAGE"
fi

# 2 Jigsaw toxicity cache (the CSV is fetched through the hub cache: not cached on Argo yet)
if [ ! -f "$CACHE/toxicity.npz" ]; then
    HF_HUB_OFFLINE=0 "$PY" -u -m geoae.interp.concept_suite --model_name $LM --layer $LAYER --out "$CACHE" \
        --only toxicity 2>&1 | tee "$OUT/toxicity_cache.log"
fi

# 3 probes, paired filter over the two bypass spaces (plain residual k-means also scored, for reference)
ARMS="res_plain=km:$PKM,tok_plain=km:$PKMT,geoae=ae:$AE"
"$PY" -u eval_out/probe_dense_subsets.py --cache "$CACHE" --arms "$ARMS" --fp_min 6 \
    --pairs tok_plain:geoae --pair_min 10 --out "$OUT/probe_dense_subsets.json" 2>&1 | tee "$OUT/probe_dense_subsets.log"
"$PY" -u eval_out/probe_dense_subsets.py --cache "$CACHE" --arms "$ARMS" --fp_min 6 --no_anchor_exclusion \
    --pairs tok_plain:geoae --pair_min 10 --out "$OUT/probe_dense_subsets_allrows.json" 2>&1 | tee "$OUT/probe_dense_subsets_allrows.log"

# 4 FineWeb-Atlas, last word of each chunk
if [ ! -f "$ATL/meta.json" ]; then
    "$PY" -u eval_out/atlas_siblings_extract.py --n_chunks 100000 --out_dir "$ATL" --model_name $LM --layer $LAYER \
        --max_len 256 --batch 64 --dtype float32 --model_dtype bfloat16 --sanity_warn_only --last_word \
        2>&1 | tee "$OUT/atlas_lastword/extract.log"
fi
"$PY" -u eval_out/atlas_assign_extra.py --cache "$ATL" --models "bypass=$AE" \
    --baselines "plain42=$PKM,plain42_tokmean=$PKMT" --out "$OUT/atlas_lastword/assign.npz" \
    2>&1 | tee "$OUT/atlas_lastword/assign.log"
OMP_NUM_THREADS=8 "$PY" -u eval_out/atlas_nmi_filtered.py --assign "$OUT/atlas_lastword/assign.npz" \
    --chunks "$ATL/chunks.npz" --concepts "$META" --arms plain42_tokmean,bypass --ref bypass --min_rows 10 \
    --n_boot 300 --seed 0 --out "$OUT/atlas_lastword/nmi_pair10_bypass.json" 2>&1 | tee "$OUT/atlas_lastword/nmi_pair10_bypass.log"
echo "[qwen-seqtab] done $(date)"
