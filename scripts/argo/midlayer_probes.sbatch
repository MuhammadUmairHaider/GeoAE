#!/bin/bash
# Mid-layer probes with the CURRENT paper setting (user 2026-10-07), Llama L14 or Gemma 3 4B L17:
#   1. plain Lloyd k-means, density-peaks seeds, seed 42, on the residual and the token-removed residual
#      (eval_out/paper_plain_kmeans_grid.py; standardisation + token table of the balanced pair the campaign fitted)
#   2. chance-corrected NMI + homogeneity / completeness on every probe, all rows and clusters >= 5 / 10 rows
#      (eval_out/probe_dense_subsets.py) for balanced / plain residual k-means, plain token-removed k-means, the mid-layer
#      bypass GeoAE (50 epochs) and the mid-layer NO-bypass AE (cancelled mid-training: Llama epoch 28, Gemma epoch 18)
#   3. Markdown tables (eval_out/summarize_probe_hc.py)
#   sbatch --mem=120G scripts/argo/midlayer_probes.sbatch llama
#   sbatch --mem=200G scripts/argo/midlayer_probes.sbatch gemma
# The dump (layer file + rows_tok) is staged into /dev/shm (GPFS has no page cache). ONE GPU only.
#SBATCH --job-name=mid-probes
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
MODEL="${1:?usage: sbatch scripts/argo/midlayer_probes.sbatch llama|gemma}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=.
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
A=/scratch/dogar/geoae/activations
case "$MODEL" in
  llama) L=14; M=llama3.2-3B; DUMP=$A/activations_llama3_3b_l14_sampled_struct5_train10M; D=6144 ;;
  gemma) L=17; M=gemma3-4B;   DUMP=$A/activations_gemma3_4b_l17_sampled_struct5_train10M; D=5120 ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
E=e2e/checkpoints/general/$M/layer$L/delta
KM=$E/balanced_kmeans_k2000_dpc_sampled_struct5_train10M.npz
KMT=$E/balanced_kmeans_k2000_dpc_sampled_struct5_train10M_tokmean.npz
CK=checkpoints/$M/layer$L
BYP=$CK/k2000_bnh_b32k_lam1_d${D}_dpc_sampled_struct5_train10M_tokbias/step_0015250.pt
NOB=$(ls $CK/k2000_bnh_b32k_lam1_d${D}_dpc_sampled_struct5_train10M/step_*.pt | sort | tail -1)
CACHE=/scratch/dogar/geoae/cache_ids/$M/layer$L
OUT=results/midlast_ablation_20261006/$M/layer$L/current_probes
for f in "$KM" "$KMT" "$BYP" "$NOB" "$DUMP/layer_$L.npy" "$DUMP/rows_tok.npy" "$CACHE/topic14.npz"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
mkdir -p "$OUT" logs
echo "[mid-probes] $MODEL layer $L  host $(hostname -f)  job ${SLURM_JOB_ID:-none}  no-bypass checkpoint $NOB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

SHM="/dev/shm/geoae_midprobe_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$SHM"' EXIT; trap 'exit 143' TERM INT
mkdir -p "$SHM"
time cp "$DUMP/layer_$L.npy" "$DUMP/rows_tok.npy" "$SHM/" || { echo "[abort] staging failed"; exit 1; }

"$PY" -u eval_out/paper_plain_kmeans_grid.py --activations_dir "$SHM" --layer $L --km "$KM" --kmt "$KMT" \
    --seed 42 --tag _seed42 --out "$OUT/plain_kmeans_grid_seed42.json" || { echo "[abort] plain k-means fit failed"; exit 1; }
PL=$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_seed42.npz
PLT=$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_tokmean_seed42.npz
rm -rf "$SHM"   # the probes read only the concept cache

"$PY" -u eval_out/probe_dense_subsets.py --cache "$CACHE" \
    --arms "res_balanced=km:$KM,res_plain=km:$PL,tok_plain=km:$PLT,geoae=ae:$BYP,nobypass_partial=ae:$NOB" \
    --out "$OUT/probe_dense_subsets.json" || { echo "[abort] probes failed"; exit 1; }
"$PY" eval_out/summarize_probe_hc.py "$OUT/probe_dense_subsets.json" min10 common_min10 all > "$OUT/probe_homogeneity_min10.md"
echo "[mid-probes] done -> $OUT"
