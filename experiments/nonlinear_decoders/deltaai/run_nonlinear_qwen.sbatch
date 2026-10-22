#!/usr/bin/env bash
# All Slurm resources are explicit arguments from submit_nonlinear_qwen.py.
set -euo pipefail
RUN_DIR="${1:?usage: run_nonlinear_qwen.sbatch <run-dir> <arm>}"
ARM="${2:?missing arm}"
case "$ARM" in mlp|residual_mlp) ;; *) echo "Invalid arm: $ARM" >&2; exit 2 ;; esac
SOURCE="$RUN_DIR/source"
# Load the native ARM PyTorch module in a fresh, non-exported job environment.
# SUSE profile contains nonzero-status probes and unset-variable reads.
# The Cray programming environment supplies dependencies of the PyTorch module.
set +eu
if [ -r /etc/profile ]; then source /etc/profile; fi
if ! type module >/dev/null 2>&1 && [ -r /etc/bash.bashrc.local ]; then
    source /etc/bash.bashrc.local
fi
readarray -t RUNTIME < <(/usr/bin/python3 - "$RUN_DIR/manifest.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))['runtime']
print(r['python'])
print(r.get('setup_script') or '')
for name in r['modules']:
    print(name)
PY
)
if [ "${#RUNTIME[@]}" -lt 2 ]; then
    echo "Cannot read runtime from manifest" >&2; exit 1
fi
if [ -n "${RUNTIME[1]}" ]; then
    source "${RUNTIME[1]}" || { echo "Runtime setup script failed" >&2; exit 1; }
fi
if [ "${#RUNTIME[@]}" -gt 2 ]; then
    type module >/dev/null 2>&1 || { echo "Cray module environment unavailable" >&2; exit 1; }
    # Keep default compiler/Cray modules: the ARM PyTorch module requires them.
    module load "${RUNTIME[@]:2}" || { echo "ARM PyTorch module load failed" >&2; exit 1; }
fi
set -eu
PYTHON="${RUNTIME[0]}"
[ -x "$PYTHON" ] || { echo "Python not executable: $PYTHON" >&2; exit 1; }
cd "$SOURCE"
export PYTHONPATH="$SOURCE" GEOAE_ROOT="$SOURCE"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export GEOAE_TRAIN_WORKERS=2 GEOAE_VAL_WORKERS=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLURM_EXPORT_ENV=ALL
for option in ${!SBATCH_@}; do unset "$option"; done
exec "$PYTHON" -u experiments/nonlinear_decoders/deltaai/run_nonlinear_qwen.py --run-dir "$RUN_DIR" --arm "$ARM"
