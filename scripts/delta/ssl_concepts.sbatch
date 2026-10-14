#!/usr/bin/env bash
# Frozen fine-grained geometry/control confirmation; one GPU, new outputs only.
#SBATCH --job-name=ssl-concepts
#SBATCH --account=bbyl-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-ssl-concepts-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/delta/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
EXP_DIR=results/ssl_concepts_20261009_v1
FRESH_DIR="$EXP_DIR/fresh_scan2"
CAPTURE_DIR="$EXP_DIR/capture"
CKPT=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
MODEL=meta-llama/Llama-3.2-3B
test -f "$FRESH_DIR/dataset.jsonl"
test -f "$FRESH_DIR/prep.json"
test -f "$CKPT"
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/source"
test ! -e "$EXP_DIR/controls.json"
cp eval_out/ssl_concept_capture.py geoae/interp/ssl_concept_controls.py eval_out/ssl_concept_prep.py "$EXP_DIR/source/"
sha256sum "$EXP_DIR/manifest.json" "$FRESH_DIR/dataset.jsonl" "$EXP_DIR/source/"*.py > "$EXP_DIR/source/sha256.txt"
echo "[ssl-concepts] job=$SLURM_JOB_ID host=$(hostname) started=$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f "$CAPTURE_DIR/capture.json" ]; then
  scripts/delta/py -u eval_out/ssl_concept_capture.py \
    --dataset "$FRESH_DIR/dataset.jsonl" --prep "$FRESH_DIR/prep.json" \
    --manifest "$EXP_DIR/manifest.json" --out-dir "$CAPTURE_DIR" \
    --model "$MODEL" --layer 27 --batch-size 16 --tail-tokens 8 \
    2>&1 | tee "$EXP_DIR/logs/capture.log"
fi
scripts/delta/py -u -m geoae.interp.ssl_concept_controls \
  --data "$CAPTURE_DIR/activations.npz" --metadata "$CAPTURE_DIR/capture.json" \
  --checkpoint "$CKPT" --model "$MODEL" --output "$EXP_DIR/controls.json" \
  --substrates base,hb,ae_recon --direction-kinds difference,shared \
  --target-nll .1,.25,.5 --alpha-grid 0,.125,.25,.5,1,2,4 \
  --min-docs 24 --bootstrap 2000 --chunk-size 128 --test-match-tolerance .02 \
  2>&1 | tee "$EXP_DIR/logs/controls.log"
echo "[ssl-concepts] complete $(date --iso-8601=seconds)"
