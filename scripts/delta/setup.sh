#!/usr/bin/env bash
# One-time setup on a Delta LOGIN node:
#   export DELTA_PROJECT=<alloc>; bash scripts/delta/setup.sh
# (run from anywhere; clones into /work/hdd/$DELTA_PROJECT/$USER/GeoAE)
set -euo pipefail
: "${DELTA_PROJECT:?export DELTA_PROJECT=<allocation dir under /work/hdd> first}"
DELTA_WORK="/work/hdd/${DELTA_PROJECT}/${USER}"
mkdir -p "$DELTA_WORK"

if [ ! -d "$DELTA_WORK/GeoAE/.git" ]; then
  git clone -b clean https://github.com/MuhammadUmairHaider/GeoAE.git "$DELTA_WORK/GeoAE"
fi
cd "$DELTA_WORK/GeoAE"
source scripts/delta/env.sh

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                                     # creates .venv on /work/hdd, not $HOME
mkdir -p logs results eval_out

uv run python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| transformers", transformers.__version__)
PY

cat <<'MSG'

Setup done. Still to do by hand (need your credentials):
  uv run huggingface-cli login     # Llama-3.2-3B is gated; accept the license on HF first
  uv run wandb login               # or pass --no_wandb to training
Then follow HANDOFF_DELTA.md section 3.
MSG
