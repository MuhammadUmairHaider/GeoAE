#!/bin/bash
# Cluster geometry (cq.json metrics, same 1M-row sample) of Gemma Scope 16k codes under their plain k-means codebook
# (controls/plain_kmeans_gs16k.npz), plus the balanced residual codebook through the same code as a check against cq.json.
#   sbatch experiments/sae_comparison/argo/sae_geometry.sbatch          -> results/delta/g4b_l33_d5120/evals/sae_suite/cq_gs16k_plain.json
#SBATCH --job-name=sae-geometry
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=argo73
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
ARM=g4b_l33_d5120
"$UV_PROJECT_ENVIRONMENT/bin/python" -u geoae/evaluation/sae_geometry.py --activations_dir activations_gemma3_4b_sampled_struct5_10M \
    --layer 33 --sae scope:16k --codebook results/delta/$ARM/evals/sae_suite/controls/plain_kmeans_gs16k.npz \
    --km e2e/checkpoints/general/gemma3-4B/layer33/delta/balanced_kmeans_k2000_dpc_sampled_struct5.npz \
    --out results/delta/$ARM/evals/sae_suite/cq_gs16k_plain.json
