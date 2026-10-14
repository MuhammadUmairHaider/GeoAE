#!/usr/bin/env bash
# Argo wrapper around scripts/delta/link_data.sh: same link list (activations_*, checkpoints, e2e, corpus, cache,
# cache_ids, dbpedia/*, biasbios/activations), but the data root is /scratch/$USER/geoae.
#   bash scripts/argo/link_data.sh [--check]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source scripts/argo/env.sh
exec bash scripts/delta/link_data.sh "$@"
