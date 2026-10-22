#!/usr/bin/env bash
# Gemma-3-4B L17: corpus, extraction, token bypass, training, detailed evaluations.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
midlayer_root="$GEOAE_ROOT"
exec "$midlayer_root/scripts/py" "$midlayer_root/experiments/depth/run.py" --arm gemma4b "$@"
