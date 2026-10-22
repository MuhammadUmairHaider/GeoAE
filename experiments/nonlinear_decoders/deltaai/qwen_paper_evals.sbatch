#!/usr/bin/env bash
# Resources and dependencies are explicit in qwen_paper_evals.py submit.
set -euo pipefail
EVAL_DIR="${1:?evaluation directory}"
ARM="${2:?arm}"
GROUP="${3:?evaluation group}"
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
for option in ${!SBATCH_@}; do unset "$option"; done
if [ "$GROUP" = preflight ]; then
    exec "$PYTHON" -u experiments/nonlinear_decoders/deltaai/qwen_paper_evals.py preflight --eval-dir "$EVAL_DIR"
fi
exec "$PYTHON" -u experiments/nonlinear_decoders/deltaai/qwen_paper_evals.py run --eval-dir "$EVAL_DIR" --arm "$ARM" --group "$GROUP"
