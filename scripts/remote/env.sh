# Environment for running GeoAE OFF Delta (any Linux box with a GPU, or a laptop for the API steps).
# scripts/delta/env.sh hands over to this file whenever Delta's /work/hdd and /work/nvme are absent (or
# GEOAE_REMOTE=1 is set), so every sheet (eval_out/run_*.sh) and scripts/delta/py run unchanged here.
_geoae_self="${BASH_SOURCE[0]:-$0}"
export GEOAE_ROOT="${GEOAE_ROOT:-$(cd "$(dirname "$_geoae_self")/../.." && pwd)}"
export GEOAE_REMOTE=1
# Site overlay: on ACENET Argo the data root, venv, HF cache and uv dirs differ (scripts/argo/env.sh). It is a no-op
# elsewhere (the Jetstream box, laptops), keyed on the host name.
if [ -z "${GEOAE_SITE:-}" ] && [ -f "$GEOAE_ROOT/scripts/argo/env.sh" ]; then
    case "$(hostname -f 2>/dev/null)" in *argo*) . "$GEOAE_ROOT/scripts/argo/env.sh"; return 0 2>/dev/null || exit 0;; esac
fi
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$GEOAE_ROOT/.venv}"   # made by scripts/remote/setup.sh
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export DELTA_DATA="${DELTA_DATA:-$GEOAE_ROOT}"      # sheets that write under $DELTA_DATA stay inside the checkout
export PATH="$HOME/.local/bin:$PATH"
cd "$GEOAE_ROOT"
