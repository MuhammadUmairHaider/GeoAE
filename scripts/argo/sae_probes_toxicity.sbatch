#!/bin/bash
# Jigsaw toxicity rung only (the rung was cached after the first run). Probe ladder (eval_out/probe_dense_subsets.py, Gemma 3 4B L33) with Gemma Scope 16k read through its plain k-means codebook
# (controls/plain_kmeans_gs16k.npz) next to the paper arms (which must reproduce g4b_l33_nobypass/evals/probe_dense_subsets.json).
#   sbatch scripts/argo/sae_probes_toxicity.sbatch   -> .../sae_suite/probe_dense_subsets_gs16k_plain_toxicity.json
#SBATCH --job-name=sae-probes-tox
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=argo76
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=.
E=e2e/checkpoints/general/gemma3-4B/layer33/delta; B=checkpoints/gemma3-4B/layer33
S=results/delta/g4b_l33_d5120/evals/sae_suite
ARMS="res_plain=km:$E/plain_kmeans_k2000_dpc_sampled_struct5.npz,tok_plain=km:$E/plain_kmeans_k2000_dpc_sampled_struct5_tokmean.npz"
ARMS="$ARMS,geoae=ae:$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt"
ARMS="$ARMS,nobypass=ae:$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_train10M/step_0015250.pt"
ARMS="$ARMS,sae=saekm:scope:16k@$S/controls/plain_kmeans_gs16k.npz"
"$UV_PROJECT_ENVIRONMENT/bin/python" -u eval_out/probe_dense_subsets.py --cache cache/gemma3-4b/l33 --arms "$ARMS" \
    --pairs sae:geoae,sae:nobypass,tok_plain:geoae --pair_min 10 --rungs toxicity --out "$S/probe_dense_subsets_gs16k_plain_toxicity.json" 2>&1 | tee "$S/probe_dense_subsets_gs16k_plain_toxicity.log"
