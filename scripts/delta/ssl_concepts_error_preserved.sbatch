#!/usr/bin/env bash
# Diagnostic: keep original residual reconstruction error during latent edits.
#SBATCH --job-name=ssl-error-preserved
#SBATCH --account=bbyl-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-ssl-error-preserved-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/delta/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
EXP_DIR=results/ssl_concepts_20261009_v1
test ! -e "$EXP_DIR/controls_error_preserved.json"
echo "[ssl-error-preserved] job=$SLURM_JOB_ID host=$(hostname) started=$(date --iso-8601=seconds)"
scripts/delta/py -u -m geoae.interp.ssl_concept_controls \
  --data "$EXP_DIR/capture/activations.npz" --metadata "$EXP_DIR/capture/capture.json" \
  --checkpoint checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt \
  --model meta-llama/Llama-3.2-3B --output "$EXP_DIR/controls_error_preserved.json" \
  --substrates base,ae_delta --direction-kinds difference,shared \
  --target-nll .1,.25,.5 --alpha-grid 0,.125,.25,.5,1,2,4 \
  --min-docs 24 --bootstrap 2000 --chunk-size 128 --test-match-tolerance .02 \
  2>&1 | tee "$EXP_DIR/logs/error_preserved.log"
echo "[ssl-error-preserved] complete $(date --iso-8601=seconds)"
