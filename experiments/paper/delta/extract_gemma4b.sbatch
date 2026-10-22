#!/bin/bash
# Gemma 3 4B PT: corpus -> sampled extraction of layer 33 (the LAST of 34 blocks) -> token-bias table.
#   sbatch experiments/paper/delta/extract_gemma4b.sbatch
#
# Mirror of extract_gemma1b.sbatch, one layer. Each step is skipped when its output is already complete,
# so resubmitting resumes. Differences from the 1B job:
#   * corpus: corpus/gemma3_4b_sampled_struct5_v1 is a COPY of the 1B corpus with manifest model_tokenizer
#     rewritten (the two tokenizers are byte-identical), so build_corpus reuses every source and streams nothing.
#   * extraction is float32 (dim 443 overflows fp16), 10M x 2560 x 4 B = 102 GB under $DELTA_DATA.
#   * the token-bias step STAGES layer_33.npy (+ sidecars) to node-local /tmp. On the 1B job each token-bias step
#     took 4-5 h reading the fp32 dump from /work/hdd (3 sequential passes: 2 for norm stats, 1 for the table); the 4B
#     file is 2.2x larger. Staging falls back to reading in place if /tmp has < (file + 15 GB) free. Outputs always end
#     up in the REAL dump dir ($ACT/norm_params_layer33.npz) and the real table path.
# Resource notes: --time 36 h is a ceiling, expected ~3-8 h extraction (compute-bound at 4B, not input-bound like the 1B)
# + ~1 h token bias. A preempted/requeued job restarts extraction from scratch (the extractor has no row-level resume).
#SBATCH --job-name=geoae-g4b-extract
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x4-preempt,gpuA40x4-preempt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=96g
#SBATCH --time=36:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source scripts/env.sh

CFG=experiments/paper/configs/gemma3-4b_extract_sampled_struct5.yaml
ACT=activations_gemma3_4b_sampled_struct5_10M
N_TOKENS=10000000
LAYERS=(33)
mkdir -p logs
bash scripts/link_data.sh >/dev/null   # make sure $ACT resolves to $DELTA_DATA/activations/$ACT (never /projects)

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
                     and sorted(d.get("layers", [])) == [33])(json.load(open(m)))
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

# ---- 3. token-bias table (staged to node-local /tmp when it fits) -------------------
REAL="$(readlink -f "$ACT")"
for L in "${LAYERS[@]}"; do
  TABLE="e2e/checkpoints/general/gemma3-4B/layer${L}/token_bias_sampled_struct5.npz"
  mkdir -p "$(dirname "$TABLE")"
  if [ -e "$TABLE" ]; then
    echo "[skip] $TABLE already exists"; continue
  fi
  SRC_ACT="$REAL"; STAGE=""
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    need_kb=$(( $(stat -c %s "$REAL/layer_${L}.npy") / 1024 + 15*1024*1024 ))
    avail_kb=$(df -Pk /tmp | awk 'NR==2{print $4}')
    if [ "$avail_kb" -gt "$need_kb" ]; then
      STAGE="/tmp/${SLURM_JOB_ID}/${ACT}"
      trap '[ -n "$STAGE" ] && rm -rf "/tmp/${SLURM_JOB_ID}"' EXIT
      mkdir -p "$STAGE"
      echo "[stage] $REAL -> $STAGE (layer $L only + sidecars); /tmp free $((avail_kb/1048576)) GB"
      time rsync -a --include="layer_${L}.npy" --exclude="layer_*.npy" "$REAL/" "$STAGE/"
      SRC_ACT="$STAGE"
    else
      echo "[stage] /tmp has only $((avail_kb/1048576)) GB free (< $((need_kb/1048576)) needed): reading $REAL in place"
    fi
  fi
  uv run --no-sync python -u -m geoae.token_bias --activations_dir "$SRC_ACT" --layer "$L" --out "$TABLE" \
    2>&1 | tee "logs/token_bias_${ACT}_l${L}.log"
  # token_bias computes the normalisation stats inside the (possibly staged) dir; keep them in the real one.
  if [ "$SRC_ACT" != "$REAL" ] && [ -e "$SRC_ACT/norm_params_layer${L}.npz" ] && [ ! -e "$REAL/norm_params_layer${L}.npz" ]; then
    cp -v "$SRC_ACT/norm_params_layer${L}.npz" "$REAL/"
  fi
  [ -e "$REAL/norm_params_layer${L}.npz" ] || { echo "ABORT: norm cache for layer $L not written"; exit 1; }
  [ -n "$STAGE" ] && rm -rf "/tmp/${SLURM_JOB_ID}" && STAGE=""
done
echo "[done] Gemma 3 4B extraction + token-bias table"
