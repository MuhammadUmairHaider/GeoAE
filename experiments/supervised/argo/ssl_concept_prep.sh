#!/usr/bin/env bash
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd /home/dogar/projects/def-hsajjad/dogar/GeoAE/repo
source configs/sites/argo.sh
EXP_DIR="$GEOAE_ROOT/results/ssl_concepts_20261009_v1"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$GEOAE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$UV_PROJECT_ENVIRONMENT/bin/python" -u "$EXP_DIR/code/ssl_concept_prep.py" \
  --manifest "$EXP_DIR/manifest.json" \
  --discovery-cache /scratch/dogar/geoae/atlas_l27 \
  --atlas-dir /scratch/dogar/geoae/atlas \
  --out-dir "$EXP_DIR/fresh_scan2" \
  --start-index 300000 --max-source-chunks 1200000 \
  --fit-cap 50 --calibration-cap 30 --test-cap 50 --control-cap 64 \
  --min-per-label-split 24 --min-control-split 32 \
  --seed 20261009 --download-missing
