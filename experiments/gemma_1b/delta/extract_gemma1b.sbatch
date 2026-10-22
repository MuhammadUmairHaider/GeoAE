#!/bin/bash
# Gemma 3 1B PT: corpus -> sampled extraction of layers 13 + 25 -> token-bias table per layer.
#   sbatch experiments/gemma_1b/delta/extract_gemma1b.sbatch
#
# Same three steps as extract_sampled.sbatch, each skipped when its output is already
# complete, so resubmitting resumes. Differences: its own Gemma-tokenised corpus dir,
# two layers in one pass (fp32, ~92 GB), and one token-bias table PER layer:
#   e2e/checkpoints/general/gemma3-1B/layer{13,25}/token_bias_sampled_struct5.npz
# (token_bias also writes $ACT/norm_params_layer<L>.npz, which training reuses).
#SBATCH --job-name=geoae-g1b-extract
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x4-preempt,gpuA40x4-preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source scripts/env.sh

CFG=experiments/gemma_1b/configs/gemma3-1b_extract_sampled_struct5.yaml
ACT=activations_gemma3_1b_sampled_struct5_10M
N_TOKENS=10000000
LAYERS=(13 25)
mkdir -p logs

# ---- 1. corpus (incremental; reuses finished source files) -------------------
uv run --no-sync python -u -m geoae.extraction.corpus --config "$CFG" --workers 3 \
  2>&1 | tee -a "logs/corpus_$(basename "$CFG" .yaml).log"

# ---- 2. extract ----------------------------------------------------------------
extract_done() {
  uv run --no-sync python - "$ACT" "$N_TOKENS" <<'PY'
import json, sys
from pathlib import Path
m = Path(sys.argv[1]) / "meta.json"
ok = m.exists() and (lambda d: d.get("mode") == "sampled" and d.get("n_tokens") == int(sys.argv[2])
                     and sorted(d.get("layers", [])) == [13, 25])(json.load(open(m)))
sys.exit(0 if ok else 1)
PY
}
if extract_done; then
  echo "[skip] $ACT/meta.json already complete"
else
  uv run --no-sync python -u -m geoae.extraction.prefix --config "$CFG" \
    2>&1 | tee "logs/extract_$(basename "$CFG" .yaml).log"
  extract_done || { echo "ABORT: extraction finished but $ACT/meta.json is not complete"; exit 1; }
fi

# ---- 3. token-bias table per layer ------------------------------------------------
for L in "${LAYERS[@]}"; do
  TABLE="e2e/checkpoints/general/gemma3-1B/layer${L}/token_bias_sampled_struct5.npz"
  mkdir -p "$(dirname "$TABLE")"
  if [ -e "$TABLE" ]; then
    echo "[skip] $TABLE already exists"
  else
    uv run --no-sync python -u -m geoae.token_bias --activations_dir "$ACT" --layer "$L" --out "$TABLE" \
      2>&1 | tee "logs/token_bias_${ACT}_l${L}.log"
    [ -e "$ACT/norm_params_layer${L}.npz" ] || { echo "ABORT: norm cache for layer $L not written"; exit 1; }
  fi
done
echo "[done] Gemma 3 1B extraction + token-bias tables"
