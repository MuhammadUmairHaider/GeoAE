#!/usr/bin/env bash
#SBATCH --job-name=sugarcrepe-qwen
#SBATCH --account=bbyl-delta-gpu
#SBATCH --partition=gpuA40x4-interactive,gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm-sugarcrepe-qwen-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
SUGARCREPE_OUT="${1:-results/sugarcrepe_qwen25_3b_l35_20261010_fp32}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
scripts/py -u experiments/compositionality/analysis/sugarcrepe_geometry.py --out-dir "$SUGARCREPE_OUT"
