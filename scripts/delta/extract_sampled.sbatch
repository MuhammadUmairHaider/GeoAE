#!/bin/bash
# Build the sampled-position L27 dump + token-bias table (corpus -> extract -> table).
#   sbatch -A <account> scripts/delta/extract_sampled.sbatch [extract_config.yaml]
#
# Three steps in one job, each skipped when its output is already complete, so
# resubmitting the same command resumes:
#   1. corpus     geoae.build_corpus  -> corpus/sampled_v1/     always run (incremental,
#                                                                reuses finished source files)
#   2. extract    geoae.extract       -> $ACT/                  skip if $ACT/meta.json is
#                                                                complete (mode=sampled,
#                                                                n_tokens matches config)
#   3. token-bias geoae.token_bias    -> $TABLE                 skip if $TABLE exists;
#                                                                also writes $ACT/norm_params_layer<L>.npz
#
# Timings from the old box (one A100 40 GB):
#   corpus      ~40 min   0.3 GB  (needs outbound internet to the HF hub; compute nodes have it)
#               ~0 min if corpus/sampled_v1 was copied from the old box (HANDOFF_DELTA.md s.0)
#   extract     ~1.5 h    61 GB
#   token-bias  ~15 min   ~300 MB
#
# Llama-3.2-3B is gated: log in to HF on the LOGIN node first (`huggingface-cli login`
# or `hf auth login`) before submitting. HF_HOME is set by env.sh, so the cached token
# is shared with the job.
#
# `set -o pipefail` (part of `set -euo pipefail` below) is what makes a failing python
# step fail the job despite piping through `tee`.
#SBATCH --job-name=geoae-extract-sampled
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=96g
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
source scripts/delta/env.sh

CFG="${1:-configs/base/llama3.2-3b_extract_sampled.yaml}"
if [ "$CFG" != "configs/base/llama3.2-3b_extract_sampled.yaml" ] && [ -z "${TABLE:-}" ]; then
  echo "ABORT: a non-default config was passed but TABLE is unset — the default" \
       "table name belongs to the default dump only. Set TABLE=... for this config."
  exit 1
fi
TABLE="${TABLE:-e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz}"
mkdir -p logs

read -r ACT N_TOKENS TARGET_LAYER < <(uv run python - "$CFG" <<'PY'
import sys
from geoae.config import Config
c = Config.from_yaml(sys.argv[1])
print(c.extraction.activations_dir, c.extraction.n_tokens, c.extraction.target_layer)
PY
)

# ---- 1. corpus --------------------------------------------------------------
# Always run — build_corpus is incremental and reuses finished source files.
uv run python -u -m geoae.build_corpus --config "$CFG" --workers 3 \
  2>&1 | tee -a "logs/corpus_$(basename "$CFG" .yaml).log"

# ---- 2. extract ---------------------------------------------------------------
extract_done() {
  uv run python - "$ACT" "$N_TOKENS" <<'PY'
import json, sys
from pathlib import Path
m = Path(sys.argv[1]) / "meta.json"
n_tokens = int(sys.argv[2])
ok = m.exists() and (lambda d: d.get("mode") == "sampled" and d.get("n_tokens") == n_tokens)(json.load(open(m)))
sys.exit(0 if ok else 1)
PY
}
if extract_done; then
  echo "[skip] $ACT/meta.json already complete (mode=sampled, n_tokens=$N_TOKENS)"
else
  uv run python -u -m geoae.extract --config "$CFG" \
    2>&1 | tee "logs/extract_$(basename "$CFG" .yaml).log"
  extract_done || { echo "ABORT: extraction finished but $ACT/meta.json is not complete"; exit 1; }
fi

# ---- 3. token-bias table ------------------------------------------------------
if [ -e "$TABLE" ]; then
  echo "[skip] $TABLE already exists"
else
  uv run python -u -m geoae.token_bias --activations_dir "$ACT" --layer "$TARGET_LAYER" --out "$TABLE" \
    2>&1 | tee "logs/token_bias_$(basename "$ACT").log"
  NORM_CACHE="$ACT/norm_params_layer${TARGET_LAYER}.npz"
  [ -e "$NORM_CACHE" ] || { echo "ABORT: expected norm cache $NORM_CACHE was not written"; exit 1; }
fi

echo "Next: sbatch -A <account> scripts/delta/train.sbatch configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias.yaml"
echo "  and: sbatch -A <account> scripts/delta/train.sbatch configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias.yaml"
