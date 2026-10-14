#!/bin/bash
# Four-space labels of the last-word FineWeb-Atlas caches (one row per chunk), both models, for the paired tables:
# plain residual / plain token-removed k-means (Lloyd, density peaks, seed 42), bypass AE (GeoAE; reproduces the
# original assign.npz) and the no-bypass AE. eval_out/atlas_assign_extra.py; then eval_out/atlas_nmi_filtered.py on Delta.
#   sbatch scripts/argo/atlas_lastword_fourway.sbatch
#SBATCH --job-name=atlas-lw-4way
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
L=e2e/checkpoints/general/llama3.2-3B/layer27/delta
G=e2e/checkpoints/general/gemma3-4B/layer33/delta
"$PY" -u eval_out/atlas_assign_extra.py --cache /scratch/dogar/geoae/atlas_l27_lastword \
    --models "bypass=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt,nobypass=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt" \
    --baselines "plain42=$L/plain_kmeans_k2000_dpc_sampled_seed42.npz,plain42_tokmean=$L/plain_kmeans_k2000_dpc_sampled_tokmean_seed42.npz" \
    --out results/delta/d6144/evals/atlas_lastword/assign_fourway.npz || exit 1
"$PY" -u eval_out/atlas_assign_extra.py --cache /scratch/dogar/geoae/atlas_g4b_l33_lastword \
    --models "bypass=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt,nobypass=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_train10M/step_0015250.pt" \
    --baselines "plain42=$G/plain_kmeans_k2000_dpc_sampled_struct5_seed42.npz,plain42_tokmean=$G/plain_kmeans_k2000_dpc_sampled_struct5_tokmean_seed42.npz" \
    --out results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/assign_fourway.npz || exit 1
echo "[atlas-lw-4way] done"
