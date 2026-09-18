#!/bin/bash
# Run any evaluation module on one GPU, e.g.:
#   sbatch -A <account> scripts/delta/eval.sbatch geoae.interp.range_intervention_compare \
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
set -euo pipefail
MODULE="${1:?usage: sbatch eval.sbatch <python.module> [args]}"; shift
source scripts/delta/env.sh
uv run python -u -m "$MODULE" "$@"
