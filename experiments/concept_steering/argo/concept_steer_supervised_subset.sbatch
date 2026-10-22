#!/bin/bash
# Add label_z on the frozen Atlas subset: save 2 prompts/concept, alpha 0.5, 32 new tokens.
# Preserve the original 20-prompt batch for exact baseline and residual-label replay.
# Residual supervised label outputs are reused exactly; save only the 28 selected new rows.
#SBATCH --job-name=concept-supervised-subset
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=argo73
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0
OUT=results/delta/g4b_l33_d5120/concept/codex_judge_atlas_s0_a05_subset
mkdir -p "$OUT" logs
"$UV_PROJECT_ENVIRONMENT/bin/python" -u -m geoae.analysis.concept_steer_supervised_subset \
    --gen results/delta/g4b_l33_d5120/concept/concept_steer_generate_sae_atlas_s0.json \
    --sample_design "$OUT/sample_design.json" \
    --out "$OUT/concept_steer_generate_supervised_subset.json" \
    2>&1 | tee "$OUT/concept_steer_generate_supervised_subset.log"
