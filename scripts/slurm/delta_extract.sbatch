#!/bin/bash
# Re-extract a residual-stream dump (the L27 one took 28 min on an A100 40 GB).
#   sbatch scripts/slurm/delta_extract.sbatch                          # L27
#   sbatch scripts/slurm/delta_extract.sbatch --layers 14 \
#          --out_dir activations_diverse_10M_l14                              # L14
#SBATCH --job-name=geoae-extract
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source scripts/env.sh
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(--out_dir activations_diverse_10M)
uv run --no-sync python -u -m geoae.extraction.prefix --config experiments/data_sampling/configs/llama3.2-3b_extract.yaml "${ARGS[@]}"
