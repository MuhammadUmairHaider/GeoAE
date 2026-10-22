# Shared environment entry point. Set GEOAE_SITE=delta, argo, or local to choose explicitly.
_geoae_self="${BASH_SOURCE[0]:-$0}"
export GEOAE_ROOT="${GEOAE_ROOT:-$(cd "$(dirname "$_geoae_self")/.." && pwd)}"
_geoae_site="${GEOAE_SITE:-}"
if [ -z "$_geoae_site" ]; then
  case "${HOSTNAME:-$(hostname -s)}" in
    *argo*) _geoae_site=argo ;;
    *)
      if [ -z "${GEOAE_REMOTE:-}" ] && [ -d /work/hdd ] && [ -d /work/nvme ]; then
        _geoae_site=delta
      else
        _geoae_site=local
      fi ;;
  esac
fi
case "$_geoae_site" in
  delta|argo|local) source "$GEOAE_ROOT/configs/sites/$_geoae_site.sh" ;;
  *) echo "Unknown GEOAE_SITE: $_geoae_site (choose delta, argo, or local)" >&2; return 2 2>/dev/null || exit 2 ;;
esac
unset _geoae_site
