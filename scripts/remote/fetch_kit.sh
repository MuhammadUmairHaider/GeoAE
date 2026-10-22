#!/usr/bin/env bash
# Put the portable kit's files at their repo-relative paths and verify them (sha256).
#   bash scripts/remote/fetch_kit.sh hf:<user>/geoae_kit_d6144     # private HF repo (needs `hf auth login`)
#   bash scripts/remote/fetch_kit.sh /path/to/geoae_kit_d6144.tar
#   bash scripts/remote/fetch_kit.sh /path/to/geoae_kit_d6144      # an unpacked kit directory
# Existing identical files are left alone; a different file at a target path aborts (never overwritten).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${1:?usage: fetch_kit.sh <hf:repo | kit.tar | kit dir>}"
PY="${PY:-$(command -v python3)}"; [ -x .venv/bin/python ] && PY=.venv/bin/python
TMP=""
case "$SRC" in
  hf:*)
    KIT=.kit_cache/$(basename "${SRC#hf:}")
    "$PY" -c "from huggingface_hub import snapshot_download as s; s(repo_id='${SRC#hf:}', local_dir='$KIT')"
    ;;
  *.tar)
    TMP=$(mktemp -d); tar -xf "$SRC" -C "$TMP"; KIT=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -1) ;;
  *)
    KIT="$SRC" ;;
esac
[ -f "$KIT/MANIFEST.sha256" ] || { echo "[abort] no MANIFEST.sha256 in $KIT"; exit 1; }
echo "[fetch] verifying $(wc -l < "$KIT/MANIFEST.sha256") files in $KIT"
(cd "$KIT" && sha256sum --quiet -c MANIFEST.sha256) || { echo "[abort] checksum mismatch in the kit"; exit 1; }
while read -r sum path; do
  if [ -e "$path" ]; then
    if cmp -s "$path" "$KIT/$path"; then continue; fi
    echo "[abort] $path exists and differs from the kit — move it aside first"; exit 1
  fi
  mkdir -p "$(dirname "$path")"
  cp "$KIT/$path" "$path"
done < "$KIT/MANIFEST.sha256"
[ -n "$TMP" ] && rm -rf "$TMP"
echo "[fetch] kit in place: $(awk '{print $2}' "$KIT/MANIFEST.sha256" | cut -d/ -f1 | sort -u | tr '\n' ' ')"
