#!/usr/bin/env bash
# Wire the repo (on /projects) to the bulk data root (on /work/hdd), so that no
# config, eval module, eval_out sheet or sbatch script has to carry a
# Delta-specific path:
#
#   bash scripts/delta/link_data.sh          # create/repair the links
#   bash scripts/delta/link_data.sh --check  # report only, change nothing
#
# Each name below becomes a symlink in $GEOAE_ROOT pointing into $DELTA_DATA,
# and the target directory is created if it does not exist. Idempotent: an
# already-correct link is left alone, a link pointing somewhere else is
# repaired, and a real (non-symlink) directory is NEVER touched — it is
# reported as a conflict, because silently replacing one would lose data.
#
# WHY the activation dumps get a flattened name: $DELTA_DATA keeps them under
# activations/<dump>, while every config names the dump at the repo root
# (data.activations_dir: "activations_sampled_10M"). The link bridges the two.
#
# .gitignore NOTE: git treats a symlink as a file, so a pattern with a trailing
# slash ("checkpoints/") does NOT match the link. .gitignore carries slashless
# twins ("/checkpoints") for exactly these names; without them `git add -A`
# would commit dangling links into the repo.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source scripts/delta/env.sh

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

# ACTIVE: part of the current campaign. The target directory is created if absent.
# link-name-in-repo  ->  path-under-$DELTA_DATA
declare -a LINKS=(
  "activations_sampled_10M:activations/activations_sampled_10M"
  "activations_sampled_struct5_10M:activations/activations_sampled_struct5_10M"
  "checkpoints:checkpoints"
  "e2e:e2e"
  "corpus:corpus"
  "cache:cache"
  "cache_ids:cache_ids"
  "dbpedia/activations:dbpedia/activations"
  "dbpedia/checkpoints:dbpedia/checkpoints"
)

# LEGACY: linked ONLY if the target already exists. These are the pre-2026-09-26
# data distribution (activations_diverse_*, every position 4..255 of an equal
# 5-way mix) and the layer-14 track. Creating empty targets for them would put
# directories that look like dumps but hold no rows in the repo, and
# train.sbatch's "missing dump" check only tests the dump a config names — so an
# empty one would sail past it and fail later inside the loader.
declare -a OPTIONAL=(
  "activations_diverse_10M:activations/activations_diverse_10M"
  "activations_diverse_l14_10M:activations/activations_diverse_l14_10M"
  "cache_l14:cache_l14"
)
for spec in "${OPTIONAL[@]}"; do
  [ -d "$DELTA_DATA/${spec#*:}" ] && LINKS+=("$spec")
done

problems=0
for spec in "${LINKS[@]}"; do
  name="${spec%%:*}"
  rel="${spec#*:}"
  target="$DELTA_DATA/$rel"

  if [ -e "$name" ] && [ ! -L "$name" ]; then
    echo "[CONFLICT] $name is a real directory in the repo, not a link."
    echo "           Move its contents into $target and remove it, then re-run."
    problems=$((problems + 1))
    continue
  fi

  current=""
  [ -L "$name" ] && current="$(readlink "$name")"

  if [ "$current" = "$target" ]; then
    status="ok"
  elif [ -n "$current" ]; then
    status="repair (was $current)"
  else
    status="create"
  fi

  if [ "$CHECK" -eq 1 ]; then
    printf '%-34s %-8s -> %s\n' "$name" "$status" "$target"
    [ -d "$target" ] || echo "    (target does not exist yet)"
    continue
  fi

  mkdir -p "$target" "$(dirname "$name")"
  if [ "$status" != "ok" ]; then
    ln -sfn "$target" "$name"
  fi
  printf '%-34s %-8s -> %s\n' "$name" "$status" "$target"
done

# Directories the repo owns itself (tracked, or small enough for /projects).
[ "$CHECK" -eq 0 ] && mkdir -p logs results eval_out figures

if [ "$problems" -gt 0 ]; then
  echo
  echo "$problems conflict(s) — nothing was changed for those names."
  exit 1
fi
echo
echo "Data root: $DELTA_DATA"
df -h "$DELTA_DATA" 2>/dev/null | tail -1
