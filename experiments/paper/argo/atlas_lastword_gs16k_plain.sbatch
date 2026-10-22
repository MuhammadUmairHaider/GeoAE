#!/bin/bash
# Gemma Scope 16k read as clusters by PLAIN Lloyd k-means (K=2000) on its native codes (all 16,384 features), fitted on the
# training rows of the plain residual baselines (every 6th row of the training 95% of the Gemma 4B dump, density-peaks
# seeds, seed 0; geoae/baselines/sae_kmeans.py), then the last-word Atlas tests as atlas_lastword_gs16k.sbatch:
#   1 fit      geoae/baselines/sae_kmeans.py              -> controls/plain_kmeans_gs16k.npz (+ _fit.json); skipped if present
#   2 labels   geoae/evaluation/atlas_assign.py --sae_kmeans   -> OUT/assign_gs16k_plain.npz (nearest centroid of the codes)
#   3 stats    geoae/analysis/atlas_concept_stats.py               -> OUT/gs16k_plain_concept_stats.json (cluster lift)
#   4 nmi      geoae/evaluation/atlas_alignment.py (purity / homogeneity / completeness / raw NMI): AE vs it, and all codebooks
#   sbatch experiments/paper/argo/atlas_lastword_gs16k_plain.sbatch    (H100 on argo73, ~10 min; needs the atlas_lastword outputs)
# Output: results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/
#SBATCH --job-name=atlas-lastword-gs16k-plain
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=argo73
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

ARM=g4b_l33_d5120; LAYER=33
ACT=activations_gemma3_4b_sampled_struct5_10M
CACHE=/scratch/dogar/geoae/atlas_g4b_l33_lastword
OUT=results/delta/$ARM/evals/sae_suite/atlas_lastword
CTL=results/delta/$ARM/evals/sae_suite/controls
CB=$CTL/plain_kmeans_gs16k.npz
META=/scratch/dogar/geoae/atlas_concept_meta.json
for f in "$ACT/layer_$LAYER.npy" "$ACT/rows_tok.npy" "$CTL/kmeans_gs16k.npz" "$META" "$CACHE/chunks.npz" \
         "$OUT/assign.npz" "$OUT/assign_gs16k.npz" "$OUT/knn_graph.npz" "$OUT/gs16k/knn_graph.npz"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
echo "[atlas-lastword-gs16k-plain] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$CB" ]; then
    "$PY" -u geoae/baselines/sae_kmeans.py --activations_dir "$ACT" --layer $LAYER --sae scope:16k \
        --codebook "$CB" --balanced "$CTL/kmeans_gs16k.npz" 2>&1 | tee "$OUT/gs16k/plain_kmeans_fit.log"
fi
"$PY" -u geoae/evaluation/atlas_assign.py --cache "$CACHE" --sae_kmeans "gs16k_plain=scope:16k@$CB" \
    --out "$OUT/assign_gs16k_plain.npz" 2>&1 | tee "$OUT/assign_gs16k_plain.log"
OMP_NUM_THREADS=8 "$PY" -u geoae/analysis/atlas_concept_stats.py --assign "$OUT/assign.npz,$OUT/assign_gs16k.npz,$OUT/assign_gs16k_plain.npz" \
    --chunks "$CACHE/chunks.npz" --concepts "$META" --graph "$OUT/knn_graph.npz,$OUT/gs16k/knn_graph.npz" \
    --graph_prefer first --rows all --unassigned_as_cluster \
    --cluster_arms base,base_tokmean,bypass,plain,plain_tokmean,gs16k_km,gs16k,gs16k_plain \
    --knn_arms base,base_tokmean,bypass,gs16k \
    --m1_compare "bypass_vs_gs16k_plain=bypass:gs16k_plain,gs16k_plain_vs_plain=gs16k_plain:plain,gs16k_plain_vs_gs16k_km=gs16k_plain:gs16k_km,bypass_vs_gs16k_native=bypass:gs16k,bypass_vs_base=bypass:base" \
    --m2_compare "bypass_vs_gs16k=bypass:gs16k,bypass_vs_base=bypass:base" \
    --copy_pairs exclude --n_boot 300 --seed 0 --out_dir "$OUT" --out_prefix gs16k_plain_ 2>&1 | tee "$OUT/gs16k_plain_concept_stats.log"
NMI=(--assign "$OUT/assign.npz,$OUT/assign_gs16k.npz,$OUT/assign_gs16k_plain.npz" --chunks "$CACHE/chunks.npz" --concepts "$META")
OMP_NUM_THREADS=8 "$PY" -u geoae/evaluation/atlas_alignment.py "${NMI[@]}" --arms bypass,gs16k_plain \
    --out "$OUT/nmi_filtered_gs16k_pair_plain.json" 2>&1 | tee "$OUT/nmi_filtered_gs16k_pair_plain.log"
OMP_NUM_THREADS=8 "$PY" -u geoae/evaluation/atlas_alignment.py "${NMI[@]}" \
    --arms base,plain,base_tokmean,plain_tokmean,bypass,gs16k_plain \
    --out "$OUT/nmi_filtered_gs16k_plain.json" 2>&1 | tee "$OUT/nmi_filtered_gs16k_plain.log"
echo "[atlas-lastword-gs16k-plain] done $(date)"
