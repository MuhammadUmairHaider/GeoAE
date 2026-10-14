#!/bin/bash
# Additional FineWeb-Atlas class draws, CPU part: freeze the draw of every seed (same rules as the original three,
# geoae/interp/atlas_classes.RULES) and write its corpus (few-shot / fit / test split by document, prompt, relatedness).
#   sbatch scripts/argo/atlas_draw_prep.sbatch <seed> [<seed> ...]        seeds other than 0,1,2
# Output per seed: results/delta/d6144/interventions/atlas_classes/s<seed>/{draw.json,cfg.json,data.parquet,relatedness.json}.
# A seed whose files exist is skipped, so the same line can be resubmitted. Normally started by scripts/argo/atlas_new_draws.sh.
#SBATCH --job-name=atlas-draw-prep
#SBATCH --account=def-hsajjad
#SBATCH --partition=cpu_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
AD=results/delta/d6144/interventions/atlas_classes
[ $# -gt 0 ] || { echo "usage: sbatch scripts/argo/atlas_draw_prep.sbatch <seed> [<seed> ...]"; exit 2; }
SEEDS=$(IFS=,; echo "$*")
echo "[draw-prep] seeds $SEEDS host $(hostname) $(date)"
"$PY" -u eval_out/atlas_classes_prep.py draw --seeds "$SEEDS" || exit 1
rc=0
for S in "$@"; do
  SD=$AD/s$S; mkdir -p "$SD/logs"
  if [ -f "$SD/cfg.json" ] && [ -f "$SD/data.parquet" ] && [ -f "$SD/relatedness.json" ]; then echo "[draw-prep] skip prep $S (done)"; continue; fi
  "$PY" -u eval_out/atlas_classes_prep.py prep --seed "$S" 2>&1 | tee "$SD/logs/prep.log"
  r=${PIPESTATUS[0]}; [ "$r" -ne 0 ] && rc=$r
done
echo "[draw-prep] finished rc=$rc $(date)"
exit "$rc"
