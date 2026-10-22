#!/bin/bash
# Atlas cross-document concept test at ONE row per chunk: the last token of the chunk's last word (NLTK word tokens,
# trailing punctuation dropped), i.e. the position that has read the whole chunk (Gemma 3 4B layer 33, 100k chunks).
#   1 extract  geoae/extraction/atlas_siblings_extract.py --last_word (same stream, tokens and fp32 settings as atlas_g4b_l33;
#              --ref_cache asserts it)                                     -> /scratch/dogar/geoae/atlas_g4b_l33_lastword
#   2 labels   geoae/evaluation/atlas_assign.py: base / base_tokmean (balanced), plain / plain_tokmean, bypass codebook
#   3 graph    geoae/analysis/atlas_knn_graph.py --all_rows: k=10 nearest rows of other documents in base / base_tokmean / bypass
#   4 stats    geoae/analysis/atlas_concept_stats.py --rows all: cluster lift + neighbour enrichment, 1,283 primary concepts
#   5 nmi      geoae/evaluation/atlas_alignment.py: homogeneity / completeness / raw NMI, clusters > 5 rows in every codebook
#   sbatch experiments/paper/argo/atlas_lastword.sbatch [g4b|llama]      (L40 on argo76; Gemma extraction ~1 h, fp32)
#   sbatch --gres=gpu:h100:1 --nodelist=argo73 experiments/paper/argo/atlas_lastword.sbatch llama
# Output: g4b results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/, llama results/delta/d6144/evals/atlas_lastword/
#SBATCH --job-name=atlas-lastword
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=argo76
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 GEOAE_NLTK_PATH=/scratch/dogar/geoae/pylib_nltk
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

MODEL="${1:-g4b}"; K=2000
case "$MODEL" in
  g4b)   ARM=g4b_l33_d5120; LAYER=33; LM=google/gemma-3-4b-pt
         REF=/scratch/dogar/geoae/atlas_g4b_l33; CACHE=/scratch/dogar/geoae/atlas_g4b_l33_lastword
         OUT=results/delta/$ARM/evals/sae_suite/atlas_lastword
         E=e2e/checkpoints/general/gemma3-4B/layer$LAYER/delta
         TB=checkpoints/gemma3-4B/layer$LAYER/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
         KM=$E/balanced_kmeans_k${K}_dpc_sampled_struct5.npz; KMT=$E/balanced_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
         PKM=$E/plain_kmeans_k${K}_dpc_sampled_struct5.npz; PKMT=$E/plain_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
         XARGS="--batch 16 --dtype float32 --model_dtype float32" ;;            # as atlas_g4b_l33 (Gemma overflows fp16)
  llama) ARM=d6144; LAYER=27; LM=meta-llama/Llama-3.2-3B                    # same first 100k chunks as Gemma
         REF=/scratch/dogar/geoae/atlas_l27; CACHE=/scratch/dogar/geoae/atlas_l27_lastword
         OUT=results/delta/$ARM/evals/atlas_lastword
         E=e2e/checkpoints/general/llama3.2-3B/layer$LAYER/delta
         TB=checkpoints/llama3.2-3B/layer$LAYER/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
         KM=$E/balanced_kmeans_k${K}_dpc_sampled.npz; KMT=$E/balanced_kmeans_k${K}_dpc_sampled_tokmean.npz
         PKM=$E/plain_kmeans_k${K}_density_peaks_sampled.npz; PKMT=$E/plain_kmeans_k${K}_density_peaks_sampled_tokmean.npz
         XARGS="--batch 128 --dtype float16 --model_dtype bfloat16" ;;          # as atlas_l27 and the training dump
  *) echo "[abort] model must be g4b or llama"; exit 1 ;;
esac
META=/scratch/dogar/geoae/atlas_concept_meta.json
for f in "$TB" "$KM" "$KMT" "$PKM" "$PKMT" "$META" "$REF/chunks.npz" "$GEOAE_NLTK_PATH/nltk"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
mkdir -p "$OUT"
echo "[atlas-lastword $MODEL] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$CACHE/meta.json" ]; then
    "$PY" -u geoae/extraction/atlas_siblings_extract.py --n_chunks 100000 --out_dir "$CACHE" --model_name "$LM" \
        --layer $LAYER --max_len 256 $XARGS --sanity_warn_only --last_word --ref_cache "$REF" 2>&1 | tee "$OUT/extract.log"
fi
"$PY" -u geoae/evaluation/atlas_assign.py --cache "$CACHE" --models "bypass=$TB" \
    --baselines "base=$KM,base_tokmean=$KMT,plain=$PKM,plain_tokmean=$PKMT" --out "$OUT/assign.npz" 2>&1 | tee "$OUT/assign.log"
"$PY" -u geoae/analysis/atlas_knn_graph.py --cache "$CACHE" --models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT" \
    --space_arms base,base_tokmean,bypass --all_rows --k 10 --out_dir "$OUT" 2>&1 | tee "$OUT/knn_graph.log"
OMP_NUM_THREADS=8 "$PY" -u geoae/analysis/atlas_concept_stats.py --assign "$OUT/assign.npz" --chunks "$CACHE/chunks.npz" \
    --concepts "$META" --graph "$OUT/knn_graph.npz" --rows all \
    --cluster_arms base,base_tokmean,bypass,plain,plain_tokmean --knn_arms base,base_tokmean,bypass \
    --m1_compare "bypass_vs_base=bypass:base,bypass_vs_tokrm=bypass:base_tokmean,bypass_vs_plain=bypass:plain,bypass_vs_plain_tokrm=bypass:plain_tokmean,plain_vs_base=plain:base,plain_tokrm_vs_tokrm=plain_tokmean:base_tokmean" \
    --m2_compare "bypass_vs_base=bypass:base,bypass_vs_tokrm=bypass:base_tokmean" \
    --copy_pairs exclude --n_boot 300 --seed 0 --out_dir "$OUT" 2>&1 | tee "$OUT/concept_stats.log"
OMP_NUM_THREADS=8 "$PY" -u geoae/evaluation/atlas_alignment.py --assign "$OUT/assign.npz" --chunks "$CACHE/chunks.npz" \
    --concepts "$META" --out "$OUT/nmi_filtered.json" 2>&1 | tee "$OUT/nmi_filtered.log"
echo "[atlas-lastword] done $(date)"
