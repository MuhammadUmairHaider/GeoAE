#!/bin/bash
# Gemma Scope 16k (google/gemma-scope-2-4b-pt resid_post_all/layer_33_width_16k_l0_small, JumpReLU, L0 ~20; reads the
# raw residual, no token bypass) added to the last-word Atlas tests of scripts/argo/atlas_lastword.sbatch (Gemma 3 4B L33,
# one row per chunk = last token of the chunk's last word; same cache, same concepts, same statistics).
# Readouts: gs16k_top2000 = strongest feature among the 2,000 most used on training-dump rows (ARR readout, counts from
# sae_controls); gs16k_km = nearest of 2,000 k-means centroids fitted on its codes; gs16k = strongest of all 16,384.
# kNN space = its dense codes, squared Euclidean (as every other space).
#   1 labels   eval_out/atlas_assign_extra.py --saes/--sae_kmeans                   -> OUT/assign_gs16k.npz
#   2 graph    eval_out/atlas_knn_graph.py --all_rows, spaces base,bypass,gs16k     -> OUT/gs16k/knn_graph.npz
#   3 stats    eval_out/atlas_concept_stats.py (cluster lift + neighbour enrichment) -> OUT/gs16k_concept_stats.json
#   4 nmi      eval_out/atlas_nmi_filtered.py: all codebooks jointly + bypass vs each GS readout
#   5 knn      eval_out/atlas_knn_precision.py (raw cross-document neighbour precision, base / bypass / GS)
#   sbatch scripts/argo/atlas_lastword_gs16k.sbatch          (L40 on argo76, ~15 min; needs atlas_lastword outputs)
# Output: results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/
#SBATCH --job-name=atlas-lastword-gs16k
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=argo76
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

ARM=g4b_l33_d5120; LAYER=33; K=2000
CACHE=/scratch/dogar/geoae/atlas_g4b_l33_lastword
OUT=results/delta/$ARM/evals/sae_suite/atlas_lastword
E=e2e/checkpoints/general/gemma3-4B/layer$LAYER/delta
TB=checkpoints/gemma3-4B/layer$LAYER/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
KM=$E/balanced_kmeans_k${K}_dpc_sampled_struct5.npz
CTL=results/delta/$ARM/evals/sae_suite/controls
META=/scratch/dogar/geoae/atlas_concept_meta.json
for f in "$TB" "$KM" "$META" "$CACHE/chunks.npz" "$CACHE/rows.npy" "$OUT/assign.npz" "$OUT/knn_graph.npz" \
         "$CTL/counts_gs16k.npz" "$CTL/kmeans_gs16k.npz"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
echo "[atlas-lastword-gs16k] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" -u eval_out/atlas_assign_extra.py --cache "$CACHE" \
    --saes "gs16k=scope:16k,gs16k_top2000=scope:16k#top2000=$CTL/counts_gs16k.npz" \
    --sae_kmeans "gs16k_km=scope:16k@$CTL/kmeans_gs16k.npz" --out "$OUT/assign_gs16k.npz" 2>&1 | tee "$OUT/assign_gs16k.log"
mkdir -p "$OUT/gs16k"
"$PY" -u eval_out/atlas_knn_graph.py --cache "$CACHE" --models "bypass=$TB" --baselines "base=$KM" \
    --saes "gs16k=scope:16k" --space_arms base,bypass,gs16k --all_rows --k 10 --out_dir "$OUT/gs16k" \
    2>&1 | tee "$OUT/gs16k/knn_graph.log"
OMP_NUM_THREADS=8 "$PY" -u eval_out/atlas_concept_stats.py --assign "$OUT/assign.npz,$OUT/assign_gs16k.npz" \
    --chunks "$CACHE/chunks.npz" --concepts "$META" --graph "$OUT/knn_graph.npz,$OUT/gs16k/knn_graph.npz" \
    --graph_prefer first --rows all --unassigned_as_cluster \
    --cluster_arms base,base_tokmean,bypass,plain,plain_tokmean,gs16k_top2000,gs16k_km,gs16k \
    --knn_arms base,base_tokmean,bypass,gs16k \
    --m1_compare "bypass_vs_gs16k_top2000=bypass:gs16k_top2000,bypass_vs_gs16k_km=bypass:gs16k_km,bypass_vs_gs16k_native=bypass:gs16k,gs16k_top2000_vs_base=gs16k_top2000:base,bypass_vs_base=bypass:base" \
    --m2_compare "bypass_vs_gs16k=bypass:gs16k,gs16k_vs_base=gs16k:base,bypass_vs_base=bypass:base" \
    --copy_pairs exclude --n_boot 300 --seed 0 --out_dir "$OUT" --out_prefix gs16k_ 2>&1 | tee "$OUT/gs16k_concept_stats.log"
NMI=(--assign "$OUT/assign.npz,$OUT/assign_gs16k.npz" --chunks "$CACHE/chunks.npz" --concepts "$META")
OMP_NUM_THREADS=8 "$PY" -u eval_out/atlas_nmi_filtered.py "${NMI[@]}" \
    --arms base,plain,base_tokmean,plain_tokmean,bypass,gs16k_top2000 \
    --out "$OUT/nmi_filtered_gs16k.json" 2>&1 | tee "$OUT/nmi_filtered_gs16k.log"
for R in top2000:gs16k_top2000 km:gs16k_km native:gs16k; do
    OMP_NUM_THREADS=8 "$PY" -u eval_out/atlas_nmi_filtered.py "${NMI[@]}" --arms "bypass,${R#*:}" \
        --out "$OUT/nmi_filtered_gs16k_pair_${R%%:*}.json" 2>&1 | tee "$OUT/nmi_filtered_gs16k_pair_${R%%:*}.log"
done
OMP_NUM_THREADS=8 "$PY" -u eval_out/atlas_knn_precision.py --graph "$OUT/gs16k/knn_graph.npz" --chunks "$CACHE/chunks.npz" \
    --table "$OUT/concept_table.csv" --arms base,bypass,gs16k --out "$OUT/knn_precision_gs16k.json" \
    2>&1 | tee "$OUT/knn_precision_gs16k.log"
echo "[atlas-lastword-gs16k] done $(date)"
