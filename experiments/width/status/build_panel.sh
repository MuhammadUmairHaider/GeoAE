#!/bin/bash
# Build the status panel page: snapshot (panel_data.py) injected into panel_template.html.
#   bash experiments/width/status/build_panel.sh <out.html>
# Claude publishes <out.html> to the "GeoAE width series" artifact:
#   https://claude.ai/artifact/9qCykNmAMax6Kqpuh3QdQQ   (republish with url= from a new session)
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
OUT="${1:?usage: build_panel.sh <out.html>}"
cd "$GEOAE_ROOT"
python3 experiments/width/status/panel_data.py | python3 -c '
import sys, json
data = json.dumps(json.load(sys.stdin)).replace("</", "<\\/")
tpl = open("experiments/width/status/panel_template.html").read()
open(sys.argv[1], "w").write(tpl.replace("__PANEL_DATA__", data))
print("wrote", sys.argv[1])' "$OUT"
