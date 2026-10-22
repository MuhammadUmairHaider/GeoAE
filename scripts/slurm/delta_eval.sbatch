#!/bin/bash
# Run any evaluation module on one GPU, e.g.:
#   sbatch scripts/slurm/delta_eval.sbatch geoae.evaluation.control \
#     --checkpoint checkpoints/llama3.2-3B/layer27/<run>/best_val.pt --dataset db14 \
#     --saliency dprime --out results/range_intervention_db14_<run>_dprime.json
# The joint-correct document sets (dbpedia/joint_correct_*.json) are in git, so
# results reproduce on the same documents without rebuilding them.
#SBATCH --job-name=geoae-eval
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
MODULE="${1:?usage: sbatch eval.sbatch <python.module> [args]}"; shift
source scripts/env.sh
uv run --no-sync python -u -m "$MODULE" "$@"
