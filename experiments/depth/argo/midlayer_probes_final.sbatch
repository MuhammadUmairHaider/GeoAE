#!/bin/bash
# Mid-layer concept probes with the FINISHED no-bypass AEs (epoch 50), Llama L14 or Gemma 3 4B L17.
# Four spaces: plain Lloyd k-means (density-peaks seeds, seed 42) on the residual and on the token-removed residual
# (fitted by experiments/depth/argo/midlayer_probes.sbatch, job 292163/292164), the bypass GeoAE and the no-bypass AE, all K = 2000.
# Scores on every probe row and on the rows whose cluster holds >= 6 kept rows in all four spaces (fixed point,
# reporting rule "more than 5"); summary = raw homogeneity / completeness / NMI (experiments/paper/analysis/summarize_probe_raw_hc.py).
#   sbatch experiments/depth/argo/midlayer_probes_final.sbatch llama|gemma
#SBATCH --job-name=mid-probes-final
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
MODEL="${1:?usage: sbatch experiments/depth/argo/midlayer_probes_final.sbatch llama|gemma}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=.
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
case "$MODEL" in
  llama) L=14; M=llama3.2-3B; D=6144 ;;
  gemma) L=17; M=gemma3-4B;   D=5120 ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
E=e2e/checkpoints/general/$M/layer$L/delta
PL=$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_seed42.npz
PLT=$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_tokmean_seed42.npz
CK=checkpoints/$M/layer$L
BYP=$CK/k2000_bnh_b32k_lam1_d${D}_dpc_sampled_struct5_train10M_tokbias/step_0015250.pt
NOB=$CK/k2000_bnh_b32k_lam1_d${D}_dpc_sampled_struct5_train10M/step_0015250.pt
CACHE=/scratch/dogar/geoae/cache_ids/$M/layer$L
OUT=results/midlast_ablation_20261006/$M/layer$L/final_probes
for f in "$PL" "$PLT" "$BYP" "$NOB" "$CK/k2000_bnh_b32k_lam1_d${D}_dpc_sampled_struct5_train10M/training_complete.json" "$CACHE/topic14.npz"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
mkdir -p "$OUT" logs
echo "[mid-probes-final] $MODEL layer $L  host $(hostname -f)  job ${SLURM_JOB_ID:-none}"
"$PY" -u geoae/evaluation/alignment.py --cache "$CACHE" --fp_min 6 \
    --arms "res_plain=km:$PL,tok_plain=km:$PLT,geoae=ae:$BYP,nobypass=ae:$NOB" \
    --out "$OUT/probe_dense_subsets.json" || { echo "[abort] probes failed"; exit 1; }
"$PY" experiments/paper/analysis/summarize_probe_raw_hc.py "$OUT/probe_dense_subsets.json" common_fp6 all > "$OUT/probe_raw_hc.md"
echo "[mid-probes-final] done -> $OUT"
