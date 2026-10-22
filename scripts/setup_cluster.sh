#!/usr/bin/env bash
# One-time setup for a GeoAE checkout on an NCSA Delta LOGIN node:
#
#   cd <the checkout>              # e.g. /projects/bbyl/mhaider/GeoAE
#   bash scripts/setup_cluster.sh
#
# Run it FROM the checkout you want to use. Unlike the 2026-09-18 version this
# does not clone anything: the repo belongs on /projects (code tier) while
# $DELTA_WORK/GeoAE is the DATA root, and cloning into the data root produced a
# directory that was half repo and half dump (docs/notes/delta-filesystem-layout.md).
#
# What it does:
#   1. creates the uv venv on /work/nvme (NOT in the repo, NOT on /work/hdd:
#      `import torch` off the Lustre HDD tier took over 8 minutes)
#   2. symlinks the bulk data directories into the repo (scripts/link_data.sh)
#   3. prints the toolchain versions and the GPU-hour balances
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "$GEOAE_ROOT"
if [ "${1:-}" = --site ]; then
  export GEOAE_SITE="${2:?usage: setup_cluster.sh [--site delta|argo]}"
  shift 2
fi
source scripts/env.sh
case "$GEOAE_SITE" in
  delta|argo) ;;
  *) echo 'Run cluster setup on Delta or Argo; use scripts/setup_laptop.sh for the Mac.' >&2; exit 1 ;;
esac

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

echo "== venv =================================================================="
echo "UV_PROJECT_ENVIRONMENT = $UV_PROJECT_ENVIRONMENT"
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"
uv sync --frozen --extra dev

echo
echo "== data links ==========================================================="
if [ "$GEOAE_SITE" = argo ]; then
  bash scripts/slurm/link_argo_data.sh
else
  bash scripts/link_data.sh
fi

echo
echo "== toolchain ============================================================"
uv run --no-sync python - <<'PY'
import torch, transformers, numpy
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| transformers", transformers.__version__, "| numpy", numpy.__version__)
PY

echo
echo "== allocations =========================================================="
accounts 2>/dev/null || echo "(accounts unavailable)"

cat <<MSG

Setup done. Still to do by hand (they need your credentials):

  # Gated Llama-3.2-3B. NOT required if HF_HOME already holds the weights —
  # verified 2026-09-26 that the cached gated repo loads with no token at all.
  # Needed only for a NEW gated download. Either log in:
  uv run --no-sync hf auth login
  # or drop the token in the repo root as hf_tokken.txt (git-ignored;
  # geoae/hf_auth.py picks it up automatically).

  # Weights & Biases, or pass --no_wandb to every training job:
  uv run --no-sync wandb login

Then follow RUNBOOK_DELTA_ROUND2.md.
MSG
