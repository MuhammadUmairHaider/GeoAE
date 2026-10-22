#!/usr/bin/env bash
# Portable, layer-specific GPT-2 evaluation entry point.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source "$GEOAE_ROOT/scripts/env.sh"
# Existing research sheets use these credentials for the optional API stages.
# Shell tracing is deliberately disabled before reading secrets.
set +x
set -a
[ ! -f "${GEOAE_ENV_FILE:-$GEOAE_ROOT/.env}" ] || source "${GEOAE_ENV_FILE:-$GEOAE_ROOT/.env}"
set +a
exec "${GEOAE_PYTHON:-$GEOAE_ROOT/scripts/py}" -u "$GEOAE_ROOT/experiments/gpt2/evaluate.py" "$@"
