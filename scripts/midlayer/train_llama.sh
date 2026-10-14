#!/usr/bin/env bash
# Llama-3.2-3B L14: corpus, extraction, token bypass, training, detailed evaluations.
set -euo pipefail
midlayer_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$midlayer_root/scripts/delta/py" "$midlayer_root/scripts/midlayer/run.py" --arm llama "$@"
