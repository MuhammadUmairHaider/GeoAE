#!/usr/bin/env bash
# Portable, layer-specific GPT-2 evaluation entry point.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../delta/env.sh"
# Existing research sheets use these credentials for the optional API stages.
# Shell tracing is deliberately disabled before reading secrets.
set +x
set -a
[ ! -f "${GEOAE_ENV_FILE:-$GEOAE_ROOT/.env}" ] || source "${GEOAE_ENV_FILE:-$GEOAE_ROOT/.env}"
set +a
exec "${GEOAE_PYTHON:-$GEOAE_ROOT/scripts/delta/py}" -u "$GEOAE_ROOT/scripts/gpt2/evaluate.py" "$@"
