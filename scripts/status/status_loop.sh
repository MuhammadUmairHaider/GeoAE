#!/usr/bin/env bash
# Keep STATUS.md's Live block fresh: refresh every 60 s, forever. One instance per checkout.
#   nohup bash scripts/status/status_loop.sh > logs/status_loop.log 2>&1 &
# The pid file records host + pid; login nodes differ, so a loop on another node is reported, not duplicated.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIDF="$ROOT/logs/status_loop.pid"
if [ -f "$PIDF" ]; then
  read -r host pid < "$PIDF"
  if [ "$host" = "$(hostname -s)" ] && kill -0 "$pid" 2>/dev/null; then echo "already running: $host $pid"; exit 0; fi
  if [ "$host" != "$(hostname -s)" ] && [ -z "${FORCE:-}" ]; then
    echo "a loop was started on $host (pid $pid); if it is dead, rerun with FORCE=1"; exit 0
  fi
fi
echo "$(hostname -s) $$" > "$PIDF"
while true; do
  nice bash "$ROOT/scripts/status/refresh_status.sh" || true
  sleep 60
done
