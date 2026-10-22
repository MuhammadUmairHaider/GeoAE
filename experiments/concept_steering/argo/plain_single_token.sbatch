#!/bin/bash
# Single-token cluster share (trivial_cur_80) of the plain k-means codebooks behind Table 1 (seed 42, K = 2,000) for
# Gemma 3 4B L33 and Qwen2.5-3B L35, on the held-out rows (geoae/evaluation/token_structure.py; Llama's value already exists in
# results/delta/d6144/evals/token_structure_plain_kmeans_seed42.json). Each run adds one codebook whose value is already
# on file, as a check that the evaluation rows are the same:
#   Gemma  plain_untagged  -> 0.115 (results/delta/g4b_l33_d5120/evals/token_structure_plain_kmeans_grid.json "plain")
#   Qwen   reference       -> 0.116 (results/qwen25_3b_l35_2x_k2000_20261008/.../bypass/evals/token_structure.json "base")
#   sbatch experiments/concept_steering/argo/plain_single_token.sbatch
#SBATCH --job-name=plain-single-token
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=8
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
G=e2e/checkpoints/general/gemma3-4B/layer33/delta
Q=e2e/checkpoints/general/qwen2.5-3B/layer35/delta
mkdir -p results/delta/g4b_l33_d5120/evals results/qwen25_l35_d4096/evals logs
rc=0
echo "[plain-single-token] host $(hostname) $(date)"
"$PY" -u geoae/evaluation/token_structure.py \
    --arms "plain_seed42=$G/plain_kmeans_k2000_dpc_sampled_struct5_seed42.npz,plain_untagged=$G/plain_kmeans_k2000_dpc_sampled_struct5.npz" \
    --activations_dir activations_gemma3_4b_sampled_struct5_10M --layer 33 --skip_next \
    --out results/delta/g4b_l33_d5120/evals/token_structure_plain_seed42.json || rc=1
"$PY" -u geoae/evaluation/token_structure.py \
    --arms "plain_seed42=$Q/plain_kmeans_k2000_dpc_struct5_train10M_seed42.npz,reference=$Q/balanced_kmeans_k2000_dpc_struct5_train10M.npz" \
    --activations_dir activations_qwen2_5_3b_l35_struct5_train10M --layer 35 --skip_next \
    --out results/qwen25_l35_d4096/evals/token_structure_plain_seed42.json || rc=1
echo "[plain-single-token] finished rc=$rc $(date)"
exit "$rc"
