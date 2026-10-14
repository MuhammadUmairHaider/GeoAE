#!/usr/bin/env bash
#SBATCH --job-name=sugarcrepe-g4b-k8000
#SBATCH --account=bbyl-delta-gpu
#SBATCH --partition=gpuA40x4-interactive,gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-sugarcrepe-gemma-k8000-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/delta/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SUGARCREPE_GEMMA_OUT="${1:-results/sugarcrepe_gemma3_4b_l33_k8000_20261010_fp32}"
SUGARCREPE_GEMMA_CKPT=checkpoints/gemma3-4B/layer33/k8000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
scripts/delta/py -u eval_out/sugarcrepe_geometry.py \
    --model google/gemma-3-4b-pt --layer 33 --checkpoint "$SUGARCREPE_GEMMA_CKPT" \
    --add-special-tokens --lm-precision float32 --out-dir "$SUGARCREPE_GEMMA_OUT"
