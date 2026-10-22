#!/usr/bin/env bash
# Run explicitly after the current training job; this script does not stop it.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source "$GEOAE_ROOT/scripts/env.sh"   # paths + cd to repo root
NUMBER_CHECKPOINT="${NUMBER_CHECKPOINT:-checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt}"
NUMBER_OUTPUT="${NUMBER_OUTPUT:-results/range_number_dpc_control.json}"
scripts/py -u -m geoae.evaluation.number_control \
  --checkpoint "$NUMBER_CHECKPOINT" \
  --tao 2 --percent 0.3 --alphas 0.5 1 2 \
  --rotation_seeds 0 1 2 --shuffle_labels \
  --out "$NUMBER_OUTPUT" "$@"
