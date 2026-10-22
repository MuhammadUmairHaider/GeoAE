#!/usr/bin/env bash
# sbatch experiments/sae_comparison/argo/gemma_scope.sbatch prepare|analyze|all [extra CLI options]
# Each job gets one H100. Independent SAE widths can use the prepared cache.
#SBATCH --job-name=gemma-scope
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1
STAGE="${1:-all}"
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p logs
PROVENANCE="$DELTA_DATA/runs/gemma_scope_compare/${SLURM_JOB_ID:-local}"
mkdir -p "$PROVENANCE"
tar --exclude=__pycache__ --exclude='*.pyc' -czf "$PROVENANCE/source.tar.gz" \
    geoae pyproject.toml experiments/sae_comparison/argo/gemma_scope.sbatch
sha256sum "$PROVENANCE/source.tar.gz" > "$PROVENANCE/source.sha256"
git rev-parse HEAD > "$PROVENANCE/git_commit.txt"
"$UV_PROJECT_ENVIRONMENT/bin/python" -u -m geoae.evaluation.sae_compare \
    --stage "$STAGE" --cache_dir "$DELTA_DATA/cache/gemma_scope_compare" \
    --atlas_dir "$DELTA_DATA/atlas" "$@" \
    2>&1 | tee "logs/gemma_scope_${STAGE}_${SLURM_JOB_ID:-local}.log"
