#!/usr/bin/env bash
# One-shot setup of a GeoAE checkout on a new machine (no Slurm, no Delta paths needed).
#   bash scripts/remote/setup.sh <kit>        # kit: hf:<user>/geoae_kit_d6144 | geoae_kit_d6144.tar | kit dir
# 1. venv (.venv; uv if present, else python -m venv + pip)   2. kit -> repo paths (sha256-checked)
# 3. word-sense datasets   4. CPU unit tests of the interp tools   5. prints what to run next.
# Needs: a Hugging Face login with access to meta-llama/Llama-3.2-3B (gated) for the GPU steps, and
# OPENROUTER_API_KEY in .env for the judge steps.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
source scripts/remote/env.sh
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then
    uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[dev]" openai pyarrow
  else
    python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]" openai pyarrow
  fi
fi
.venv/bin/python -c "import torch; print('[setup] torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
[ -n "${1:-}" ] && bash scripts/remote/fetch_kit.sh "$1" || echo "[setup] no kit given: skipping artifacts"
bash eval_out/run_interp.sh d6144 wsd_data
.venv/bin/python -m pytest tests/test_word_sense.py tests/test_atlas_summary.py -x
cat <<'NEXT'
[setup] done. Next (see HANDOFF_REMOTE.md):
  bash eval_out/run_interp.sh d6144 gpu      # GPU: base + bypass cluster examples, word senses (~1-1.5 h)
  bash eval_out/run_interp.sh d6144 api      # judges + atlas (API; needs OPENROUTER_API_KEY in .env)
NEXT
