#!/bin/bash
# Build the status panel page: snapshot (panel_data.py) injected into panel_template.html.
#   bash scripts/delta/build_panel.sh <out.html>
# Claude publishes <out.html> to the "GeoAE width series" artifact:
#   https://claude.ai/artifact/9qCykNmAMax6Kqpuh3QdQQ   (republish with url= from a new session)
set -euo pipefail
OUT="${1:?usage: build_panel.sh <out.html>}"
cd "$(dirname "$0")/../.."
python3 scripts/delta/panel_data.py | python3 -c '
import sys, json
data = json.dumps(json.load(sys.stdin)).replace("</", "<\\/")
tpl = open("scripts/delta/panel_template.html").read()
open(sys.argv[1], "w").write(tpl.replace("__PANEL_DATA__", data))
print("wrote", sys.argv[1])' "$OUT"
