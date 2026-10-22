# Local CPU or explicitly configured remote GPU environment. Source through scripts/env.sh.
_geoae_self="${BASH_SOURCE[0]:-$0}"
export GEOAE_ROOT="${GEOAE_ROOT:-$(cd "$(dirname "$_geoae_self")/../.." && pwd)}"
export GEOAE_REMOTE=1
export GEOAE_SITE=local
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$GEOAE_ROOT/.venv}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export DELTA_DATA="${DELTA_DATA:-$GEOAE_ROOT}"
export PATH="$HOME/.local/bin:$PATH"
cd "$GEOAE_ROOT"
