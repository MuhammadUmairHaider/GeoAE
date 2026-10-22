#!/usr/bin/env bash
set -euo pipefail
RECOVERY="${1:?recovery directory}"
TASK="${2:?task key}"
set +eu
if [ -r /etc/profile ]; then source /etc/profile; fi
if ! type module >/dev/null 2>&1 && [ -r /etc/bash.bashrc.local ]; then source /etc/bash.bashrc.local; fi
module load python/miniforge3_pytorch/2.12.0 || exit 1
set -eu
export SLURM_EXPORT_ENV=ALL
for option in ${!SBATCH_@}; do unset "$option"; done
PYTHON="/projects/bbyl/mhaider/GeoAE/nonlinear_qwen_20261018_deltaai/evaluation/env/bin/python"
exec "$PYTHON" -u "$RECOVERY/orchestration/nonlinear_paper_recovery.py" run --root "$RECOVERY" --task "$TASK"
