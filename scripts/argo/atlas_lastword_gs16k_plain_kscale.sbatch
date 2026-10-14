#!/bin/bash
# K scaling of the Gemma Scope 16k readout on the last-word Atlas topics: the SAE is read exactly as at K=2000
# (atlas_lastword_gs16k_plain.sbatch: PLAIN Lloyd k-means on its native codes, same training rows, density-peaks seeds from
# the same 32,768-row pool, seed 0), now with K = 4,000 and 8,000, next to the plain residual / token-removed k-means and the
# bypass AEs trained with the same K.
#   1 fit      eval_out/sae_plain_kmeans_fit.py --n_clusters K   -> controls/plain_kmeans_gs16k_k{4000,8000}.npz (+ _fit.json)
#   2 labels   eval_out/atlas_assign_extra.py --sae_kmeans       -> OUT/kscale/assign_gs16k_plainK.npz (gs16k_plain4k / 8k)
#   3 scores   eval_out/atlas_nmi_filtered.py per K, rows kept iff their cluster holds > 5 rows in every compared space:
#                nmi_filtered4_k{K}.json     residual, token-removed, Gemma Scope, bypass AE
#                nmi_filtered3sae_k{K}.json  residual, Gemma Scope, bypass AE
#   sbatch scripts/argo/atlas_lastword_gs16k_plain_kscale.sbatch   (H100 on argo73, ~20 min)
# Output: results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/kscale/
#SBATCH --job-name=atlas-gs16k-kscale
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=argo73
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-01:30:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

ARM=g4b_l33_d5120; LAYER=33
ACT=activations_gemma3_4b_sampled_struct5_10M
CACHE=/scratch/dogar/geoae/atlas_g4b_l33_lastword
BASE=results/delta/$ARM/evals/sae_suite/atlas_lastword
OUT=$BASE/kscale
CTL=results/delta/$ARM/evals/sae_suite/controls
META=/scratch/dogar/geoae/atlas_concept_meta.json
for f in "$ACT/layer_$LAYER.npy" "$ACT/rows_tok.npy" "$META" "$CACHE/chunks.npz" "$BASE/assign.npz" \
         "$BASE/assign_gs16k_plain.npz" "$BASE/k4000/assign.npz" "$BASE/k8000/assign.npz" \
         "$BASE/k8000/assign_plainK.npz" "$BASE/k8000/assign_plainKt.npz"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
mkdir -p "$OUT"
echo "[atlas-gs16k-kscale] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for K in 4000 8000; do
    CB=$CTL/plain_kmeans_gs16k_k$K.npz
    if [ ! -f "$CB" ]; then
        "$PY" -u eval_out/sae_plain_kmeans_fit.py --activations_dir "$ACT" --layer $LAYER --sae scope:16k \
            --n_clusters $K --codebook "$CB" 2>&1 | tee "$OUT/plain_kmeans_fit_k$K.log"
    fi
done
"$PY" -u eval_out/atlas_assign_extra.py --cache "$CACHE" \
    --sae_kmeans "gs16k_plain4k=scope:16k@$CTL/plain_kmeans_gs16k_k4000.npz,gs16k_plain8k=scope:16k@$CTL/plain_kmeans_gs16k_k8000.npz" \
    --out "$OUT/assign_gs16k_plainK.npz" 2>&1 | tee "$OUT/assign_gs16k_plainK.log"

NMI=(--assign "$BASE/assign.npz,$BASE/k4000/assign.npz,$BASE/k8000/assign.npz,$BASE/k8000/assign_plainK.npz,$BASE/k8000/assign_plainKt.npz,$BASE/assign_gs16k_plain.npz,$OUT/assign_gs16k_plainK.npz"
     --chunks "$CACHE/chunks.npz" --concepts "$META")
for KS in "2000:" "4000:4k" "8000:8k"; do
    K=${KS%%:*}; S=${KS##*:}
    OMP_NUM_THREADS=8 "$PY" -u eval_out/atlas_nmi_filtered.py "${NMI[@]}" \
        --arms "plain$S,plain_tokmean$S,gs16k_plain$S,bypass$S" --ref "bypass$S" \
        --out "$OUT/nmi_filtered4_k$K.json" 2>&1 | tee "$OUT/nmi_filtered4_k$K.log"
    OMP_NUM_THREADS=8 "$PY" -u eval_out/atlas_nmi_filtered.py "${NMI[@]}" \
        --arms "plain$S,gs16k_plain$S,bypass$S" --ref "bypass$S" \
        --out "$OUT/nmi_filtered3sae_k$K.json" 2>&1 | tee "$OUT/nmi_filtered3sae_k$K.log"
done
echo "[atlas-gs16k-kscale] done $(date)"
