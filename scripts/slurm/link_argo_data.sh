#!/usr/bin/env bash
# Argo wrapper around scripts/link_data.sh: same link list (activations_*, checkpoints, e2e, corpus, cache,
# cache_ids, dbpedia/*, biasbios/activations), but the data root is /scratch/$USER/geoae.
#   bash scripts/slurm/link_argo_data.sh [--check]
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "$GEOAE_ROOT"
source configs/sites/argo.sh
exec bash scripts/link_data.sh "$@"
