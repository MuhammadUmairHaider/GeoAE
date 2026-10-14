#!/bin/bash
# Mid-layer audit, Llama L14 or Gemma 3 4B L17: does a post-hoc plain Lloyd k-means on each AE latent recover the probes?
#   1. eval_out/latent_plain_kmeans.py for the bypass and the no-bypass AE (plain Lloyd on z, density-peaks seeds,
#      seed 42 as the mid-layer residual fits; held-out usage; 1M-row geometry of trained vs re-fit codebook)
#   2. eval_out/probe_ood_check.py: are probe rows in distribution for each of the six spaces?
#   3. probes on six spaces (plain residual / token-removed k-means, bypass AE + its latent re-fit, no-bypass AE + its
#      latent re-fit), raw homogeneity / completeness / NMI on rows in clusters with >5 rows in all six (fixed point)
#   sbatch --gres=gpu:l40:1 scripts/argo/midlayer_latent_plain.sbatch llama|gemma
# The dump (layer file + rows_tok) is staged into /dev/shm (GPFS has no page cache). ONE GPU only.
#SBATCH --job-name=mid-latplain
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
MODEL="${1:?usage: sbatch scripts/argo/midlayer_latent_plain.sbatch llama|gemma}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
A=/scratch/dogar/geoae/activations
case "$MODEL" in
  llama) L=14; M=llama3.2-3B; DUMP=$A/activations_llama3_3b_l14_sampled_struct5_train10M; D=6144 ;;
  gemma) L=17; M=gemma3-4B;   DUMP=$A/activations_gemma3_4b_l17_sampled_struct5_train10M; D=5120 ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
E=e2e/checkpoints/general/$M/layer$L/delta
PL=$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_seed42.npz
PLT=$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_tokmean_seed42.npz
CK=checkpoints/$M/layer$L
BYP=$CK/k2000_bnh_b32k_lam1_d${D}_dpc_sampled_struct5_train10M_tokbias/step_0015250.pt
NOB=$CK/k2000_bnh_b32k_lam1_d${D}_dpc_sampled_struct5_train10M/step_0015250.pt
LB=$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_latent_bypass_seed42.npz
LN=$E/plain_kmeans_k2000_dpc_sampled_struct5_train10M_latent_nobypass_seed42.npz
CACHE=/scratch/dogar/geoae/cache_ids/$M/layer$L
OUT=results/midlast_ablation_20261006/$M/layer$L/latent_plain
for f in "$PL" "$PLT" "$BYP" "$NOB" "$DUMP/layer_$L.npy" "$DUMP/rows_tok.npy" "$CACHE/topic14.npz"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
mkdir -p "$OUT" logs
echo "[mid-latplain] $MODEL layer $L  host $(hostname -f)  job ${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

SHM="/dev/shm/geoae_midlatplain_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$SHM"' EXIT; trap 'exit 143' TERM INT
mkdir -p "$SHM"
time cp "$DUMP/layer_$L.npy" "$DUMP/rows_tok.npy" "$SHM/" || { echo "[abort] staging failed"; exit 1; }

for arm in bypass nobypass; do
    if [ $arm = bypass ]; then ck=$BYP; cb=$LB; else ck=$NOB; cb=$LN; fi
    "$PY" -u eval_out/latent_plain_kmeans.py --activations_dir "$SHM" --layer $L --ckpt "$ck" --codebook "$cb" \
        --seed 42 --out "$OUT/latent_plain_fit_$arm.json" || { echo "[abort] latent fit $arm failed"; exit 1; }
done
ARMS="res_plain=km:$PL,tok_plain=km:$PLT,geoae=ae:$BYP,geoae_lplain=aekm:$BYP+$LB,nobypass=ae:$NOB,nobypass_lplain=aekm:$NOB+$LN"
"$PY" -u eval_out/probe_ood_check.py --activations_dir "$SHM" --layer $L --cache "$CACHE" --arms "$ARMS" \
    --out "$OUT/probe_ood_check.json" || echo "[warn] OOD check failed (probes still run)"
rm -rf "$SHM"
"$PY" -u eval_out/probe_dense_subsets.py --cache "$CACHE" --fp_min 6 --arms "$ARMS" \
    --out "$OUT/probe_dense_subsets.json" || { echo "[abort] probes failed"; exit 1; }
"$PY" eval_out/summarize_probe_raw_hc.py "$OUT/probe_dense_subsets.json" common_fp6 all > "$OUT/probe_raw_hc.md"
echo "[mid-latplain] done -> $OUT"
