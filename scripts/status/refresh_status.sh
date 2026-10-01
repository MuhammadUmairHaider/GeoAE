#!/usr/bin/env bash
# Rewrite the Live block of STATUS.md (repo root), keeping the hand-written Now block. Written to a temp
# file and moved into place, so an open editor never sees a half-written file. Cheap: one ssh call, one
# squeue, a few greps. Run by scripts/status/status_loop.sh every 60 s.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
F="$ROOT/STATUS.md"
NOW=$( [ -f "$F" ] && awk '/<!-- now:start -->/{p=1} p{print} /<!-- now:end -->/{p=0}' "$F" )
[ -n "$NOW" ] || NOW=$'<!-- now:start -->\n_(nothing written yet)_\n<!-- now:end -->'

box=$(ssh -o BatchMode=yes -o ConnectTimeout=8 geoae-box '
  cd ~/GeoAE 2>/dev/null || { echo "no ~/GeoAE on the box"; exit 0; }
  for LOG in $(ls -t logs/*.log results/delta/*/logs/interp/api_chain.log 2>/dev/null | head -3); do
    echo "== $LOG (written $(date -r "$LOG" +%H:%M:%S) UTC)"
    grep -aE "^\[(run |skip|FAIL|done|abort)|EXIT" "$LOG" | tail -4
    echo "   now: $(tail -c 3000 "$LOG" | tr "\r" "\n" | grep -av "^ *$" | tail -1 | cut -c1-140)"
  done
  echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader)"
  echo "disk: GeoAE $(du -sh ~/GeoAE 2>/dev/null | cut -f1), free $(df -h --output=avail / | tail -1 | tr -d " ")"
  tmux ls 2>/dev/null | sed "s/^/tmux: /"' 2>&1 | head -26)
[ -n "$box" ] || box="(box unreachable)"

copy="(no copy running: ~/copy_more.log absent)"
[ -f "$HOME/copy_more.log" ] && copy=$(tail -n 8 "$HOME/copy_more.log")

gemma=""
for f in "$ROOT"/logs/gemma3-1b_l*_k2000_*_tokbias.log; do
  [ -f "$f" ] || continue
  arm=$(basename "$f" | sed -E 's/gemma3-1b_(l[0-9]+)_k2000_bnh_b32k_lam1_(d[0-9]+)_.*/\1 \2/')
  ep=$(grep -oE "Epoch [0-9]+/[0-9]+" "$f" | tail -1)
  gemma+="| $arm | ${ep:-?} | $(date -r "$f" +%H:%M) |"$'\n'
done

jobs=$(squeue -u "$USER" -h -t RUNNING -o "%.10i %.34j %.10M %.20R" 2>/dev/null)
pend=$(squeue -u "$USER" -h -t PENDING -o "%r" 2>/dev/null | sort | uniq -c | sed 's/^ */pending: /')

{
  echo "# GeoAE status"
  echo
  echo "_Live block refreshed every 60 s by \`scripts/status/status_loop.sh\` · last refresh $(date '+%a %H:%M:%S %Z')_"
  echo
  echo "## Now"
  echo
  echo "$NOW"
  echo
  echo "## Live"
  echo
  echo "### A100 box (geoae-box) — all interp / eval / judge work runs here"
  echo '```text'; echo "$box"; echo '```'
  echo
  echo "### Copy Delta → box (~/copy_more.log)"
  echo '```text'; echo "$copy"; echo '```'
  echo
  echo "### Gemma 3 1B training (Delta)"
  echo
  echo "| arm | last epoch logged | log written |"; echo "| --- | --- | --- |"
  printf "%s" "$gemma"
  echo
  echo "### Delta jobs"
  echo '```text'; echo "${jobs:-(none running)}"; echo "${pend:-}"; echo '```'
} > "$F.tmp" && mv "$F.tmp" "$F"
