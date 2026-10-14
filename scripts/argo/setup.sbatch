#!/bin/bash
# One-time venv build on a CPU node (login nodes kill long CPU time):
#   sbatch scripts/argo/setup.sbatch
# Builds the venv at $UV_PROJECT_ENVIRONMENT from uv.lock (--frozen) + the extras the pipeline needs that are not in
# pyproject (openai for the judges, umap-learn for the atlas plots). Re-run `uv pip install` for extras after any
# later `uv sync` (a plain sync removes them; `uv run --no-sync` / scripts/delta/py never syncs).
#SBATCH --job-name=geoae-setup
#SBATCH --partition=cpu_general
#SBATCH --account=def-hsajjad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
source scripts/argo/env.sh
command -v uv >/dev/null || { echo "uv missing: curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=\$HOME/.local/bin UV_NO_MODIFY_PATH=1 sh"; exit 1; }
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")"
time uv sync --frozen --extra dev
time uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" openai umap-learn
uv pip list --python "$UV_PROJECT_ENVIRONMENT/bin/python" | grep -i -E "^(torch|transformers|numpy|openai|umap-learn|pytest|geoae|accelerate|datasets|scikit-learn) "
"$UV_PROJECT_ENVIRONMENT/bin/python" --version
