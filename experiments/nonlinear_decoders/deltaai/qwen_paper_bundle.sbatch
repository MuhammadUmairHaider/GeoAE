#!/usr/bin/env bash
# One GPU allocation runs every evaluation group for this arm.
set -euo pipefail
EVAL_DIR="${1:?evaluation directory}"
ARM="${2:?arm}"
set +eu
if [ -r /etc/profile ]; then source /etc/profile; fi
if ! type module >/dev/null 2>&1 && [ -r /etc/bash.bashrc.local ]; then
    source /etc/bash.bashrc.local
fi
readarray -t RUNTIME < <(/usr/bin/python3 - "$EVAL_DIR/manifest.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))['runtime']
print(r['python'])
print(r.get('setup_script') or '')
for name in r['modules']:
    print(name)
PY
)
if [ "${#RUNTIME[@]}" -lt 2 ]; then echo 'Cannot read runtime' >&2; exit 1; fi
if [ -n "${RUNTIME[1]}" ]; then source "${RUNTIME[1]}" || exit 1; fi
if [ "${#RUNTIME[@]}" -gt 2 ]; then
    type module >/dev/null 2>&1 || exit 1
    module load "${RUNTIME[@]:2}" || exit 1
fi
set -eu
PYTHON="${RUNTIME[0]}"
SOURCE="$EVAL_DIR/source"
cd "$SOURCE"
export PYTHONPATH="$SOURCE" GEOAE_ROOT="$SOURCE" SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for option in ${!SBATCH_@}; do unset "$option"; done
exec "$PYTHON" -u "$EVAL_DIR/orchestration/qwen_paper_bundle.py" run --eval-dir "$EVAL_DIR" --arm "$ARM"
