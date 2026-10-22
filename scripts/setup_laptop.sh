#!/usr/bin/env bash
# Create a native Mac environment from the checked-in lock, then run CPU checks.
# No activation dumps, model weights, datasets, or API calls are downloaded.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/setup_laptop.sh [--offline] [--smoke-only | --source-only]

  default        Install locked dependencies on Apple Silicon/macOS 14+, then test.
  --offline      Install only from uv's cache; never download Python or packages.
  --smoke-only   Test an existing checkout-local .venv; do not install anything.
  --source-only  Parse Python sources with python3; no third-party packages needed.
  --help         Show this help.

The first default run needs uv and internet access for Python/package downloads.
The environment is created locally in .venv; do not copy a Linux .venv onto a Mac.
Intel Macs can use --source-only and submit GPU work directly to Delta over SSH.
HELP
}

mode=setup
offline=0
for arg in "$@"; do
  case "$arg" in
    --offline) offline=1 ;;
    --source-only|--smoke-only)
      if [ "$mode" != setup ]; then
        echo 'Choose only one of --source-only and --smoke-only.' >&2
        exit 2
      fi
      mode="${arg#--}"
      ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

repo_root="$GEOAE_ROOT"
cd "$repo_root"

if [ "$mode" = source-only ]; then
  command -v python3 >/dev/null || { echo 'Install Python 3.10+ for source checks.' >&2; exit 1; }
  python3 - <<'PY'
import ast
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit('Source checks require Python 3.10 or newer.')
files = sorted(p for folder in ('geoae', 'experiments', 'scripts', 'tests', 'tools')
               for p in Path(folder).rglob('*.py'))
if not files:
    raise SystemExit('No Python sources found; check the extracted bundle layout.')
for path in files:
    ast.parse(path.read_bytes(), filename=str(path))
print(f'Source syntax passed: {len(files)} Python files. No dependencies installed.')
PY
  exit 0
fi

if [ "$mode" = setup ]; then
  if [ "$(uname -s)" != Darwin ]; then
    echo 'Dependency installation here targets macOS. Use --smoke-only for an existing environment.' >&2
    exit 1
  fi
  if [ "$(uname -m)" != arm64 ]; then
    cat >&2 <<'INTEL'
The current uv.lock has no torch wheel for Intel macOS.
On Apple Silicon, open a native ARM64 terminal rather than a Rosetta terminal.
On Intel, use --source-only and Delta for Python/GPU work. A separate compatible
Intel environment needs its own dependency choices; this script preserves uv.lock.
INTEL
    exit 1
  fi
  macos_major="$(sw_vers -productVersion | cut -d. -f1)"
  if [ "$macos_major" -lt 14 ]; then
    echo 'The locked torch wheel requires macOS 14 or newer. Use --source-only and Delta.' >&2
    exit 1
  fi
  command -v uv >/dev/null || {
    echo 'Install uv first (for example: brew install uv), then rerun this script.' >&2
    exit 1
  }
  # Refuse a copied Linux or Intel interpreter before uv considers reusing it.
  if [ -e .venv ] || [ -L .venv ]; then
    if ! .venv/bin/python -c 'import platform, sys; assert sys.platform == "darwin" and platform.machine() == "arm64"' 2>/dev/null; then
      echo 'Existing .venv is not a usable native Apple Silicon environment.' >&2
      echo 'Move it aside, then rerun this script to create a fresh .venv.' >&2
      exit 1
    fi
  fi
  sync_args=(sync --frozen --extra dev --extra sae --python 3.12)
  if [ "$offline" = 1 ]; then
    sync_args+=(--offline --no-python-downloads)
  fi
  UV_PROJECT_ENVIRONMENT="$repo_root/.venv" uv "${sync_args[@]}"
fi

if [ ! -x .venv/bin/python ]; then
  echo 'No usable .venv/bin/python. Run setup on a supported Mac first.' >&2
  exit 1
fi

# These checks use small synthetic tensors and temporary fixture files only.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
.venv/bin/python - <<'PY'
import platform
from pathlib import Path
import torch
import geoae

assert Path(geoae.__file__).resolve().parent == Path('geoae').resolve(), 'Wrong checkout imported'
print(f'Python host: {platform.system()} {platform.machine()} | torch {torch.__version__}')
print(f'GeoAE imported from {Path(geoae.__file__).resolve()}')
PY
.venv/bin/python -m pytest -p no:cacheprovider \
  tests/unit/test_model.py tests/unit/test_nonlinear_decoders.py tests/workflows/test_deltaai_nonlinear.py
echo 'Laptop CPU smoke checks passed. See docs/history/migration/LAPTOP_HANDOFF.md for the Delta workflow.'
