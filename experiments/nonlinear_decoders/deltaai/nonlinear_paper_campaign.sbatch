#!/usr/bin/env bash
# The campaign submitter supplies all allocation resources explicitly.
set -euo pipefail
CAMPAIGN="${1:?campaign directory}"
MODEL="${2:?model key}"
ARM="${3:?decoder arm}"
set +eu
if [ -r /etc/profile ]; then source /etc/profile; fi
if ! type module >/dev/null 2>&1 && [ -r /etc/bash.bashrc.local ]; then
    source /etc/bash.bashrc.local
fi
readarray -t RUNTIME < <(/usr/bin/python3 - "$CAMPAIGN/campaign_manifest.json" "$MODEL" <<'PY'
import json, sys
runtime = json.load(open(sys.argv[1]))['models'][sys.argv[2]]['runtime']
print(runtime['python'])
print(runtime.get('setup_script') or '')
for name in runtime.get('modules', []):
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
[ -x "$PYTHON" ] || { echo 'Runtime interpreter is unavailable' >&2; exit 1; }
export SLURM_EXPORT_ENV=ALL
for option in ${!SBATCH_@}; do unset "$option"; done
exec "$PYTHON" -u "$CAMPAIGN/orchestration/nonlinear_paper_campaign.py" run --campaign "$CAMPAIGN" --model "$MODEL" --arm "$ARM"
