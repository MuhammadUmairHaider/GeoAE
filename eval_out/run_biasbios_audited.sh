#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
.venv/bin/python -u -m geoae.bias.probe \
  --act_dir biasbios/activations/llama3.2-3B/last \
  --source_hashes eval_out/biasbios_source_hashes.npz \
  --layer 27 \
  --ae_checkpoint checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt \
  --spaces raw geoae --device "${BIOS_DEVICE:-cpu}" \
  --out "${BIOS_OUTPUT:-results/biasbios_dpc_audited.json}" \
  --plot_dir "${BIOS_PLOTS:-figures/biasbios_dpc_audited}" "$@"
