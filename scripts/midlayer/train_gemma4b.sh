#!/usr/bin/env bash
# Gemma-3-4B L17: corpus, extraction, token bypass, training, detailed evaluations.
set -euo pipefail
midlayer_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$midlayer_root/scripts/delta/py" "$midlayer_root/scripts/midlayer/run.py" --arm gemma4b "$@"
