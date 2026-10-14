#!/bin/bash
# Document redraws of a frozen control corpus, CPU part (eval_out/control_datasets_redraw.py): same classes and prompt,
# a new random sample of up to 600 documents per class for every seed.
#   sbatch scripts/argo/control_redraw_prep.sbatch <news20|emotion6> <seed> [<seed> ...]        seeds above 0
# Output per seed: results/delta/control_datasets/<name>_s<seed>/{cfg.json,data.parquet}; an existing seed is skipped.
# Normally started by scripts/argo/control_new_draws.sh.
#SBATCH --job-name=control-redraw-prep
#SBATCH --account=def-hsajjad
#SBATCH --partition=cpu_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1
NAME="${1:?dataset}"; shift
[ $# -gt 0 ] || { echo "usage: sbatch scripts/argo/control_redraw_prep.sbatch <dataset> <seed> [<seed> ...]"; exit 2; }
echo "[redraw-prep] $NAME seeds $* host $(hostname) $(date)"
"$UV_PROJECT_ENVIRONMENT/bin/python" -u eval_out/control_datasets_redraw.py "$NAME" --seeds "$(IFS=,; echo "$*")"
echo "[redraw-prep] finished $(date)"
