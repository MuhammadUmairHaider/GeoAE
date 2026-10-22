#!/bin/bash
# Plain Lloyd k-means re-fit on the final Gemma 4B L33 token-bypass GeoAE latent (density-peaks seeds, as the plain
# baselines), then nearest-centroid labels of every FineWeb-Atlas cache row. geoae/baselines/atlas_latent_kmeans.py.
#   sbatch experiments/paper/argo/atlas_latent_plain_kmeans.sbatch
# Output: e2e/.../plain_kmeans_k2000_dpc_sampled_struct5_latent_bypass.npz and
#         results/delta/g4b_l33_d5120/evals/sae_suite/atlas_concepts/plain_kmeans/{assign_latent_plain.npz,latent_plain_fit.json}
# The dump is read in contiguous slabs (every 6th row kept), so it is not staged in /dev/shm. ONE GPU only.
#SBATCH --job-name=atlas-latplain
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

ARM=g4b_l33_d5120; LAYER=33
E=e2e/checkpoints/general/gemma3-4B/layer$LAYER/delta
TB=checkpoints/gemma3-4B/layer$LAYER/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
ACT_SRC="${ACT_DIR:-activations_gemma3_4b_sampled_struct5_10M}"
[ -e "$ACT_SRC/layer_$LAYER.npy" ] || ACT_SRC="$DELTA_DATA/activations/activations_gemma3_4b_sampled_struct5_10M"
for f in "$TB" "$E/balanced_kmeans_k2000_dpc_sampled_struct5.npz" "$ACT_SRC/layer_$LAYER.npy" "$ACT_SRC/rows_tok.npy" \
         /scratch/dogar/geoae/atlas_g4b_l33/rows.npy; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
echo "[atlas-latplain] host $(hostname -f) $(date)  job ${SLURM_JOB_ID:-none}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -u geoae/baselines/atlas_latent_kmeans.py --activations_dir "$ACT_SRC" --layer $LAYER --ckpt "$TB" \
    --km "$E/balanced_kmeans_k2000_dpc_sampled_struct5.npz" \
    --codebook "$E/plain_kmeans_k2000_dpc_sampled_struct5_latent_bypass.npz" \
    --cache /scratch/dogar/geoae/atlas_g4b_l33 \
    --out_dir results/delta/$ARM/evals/sae_suite/atlas_concepts/plain_kmeans "$@" 2>&1 | tee -a logs/atlas_latent_plain_kmeans.log
exit "${PIPESTATUS[0]}"
