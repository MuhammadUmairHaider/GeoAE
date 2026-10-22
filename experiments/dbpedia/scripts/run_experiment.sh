#!/bin/bash
# DBpedia-14 clustering experiment — extract → train → evaluate.
#
# Parameterised by env vars (defaults target the current task: Llama, layer 27,
# unprompted, MEAN pooling, MSE pipeline):
#
#   MODEL=llama3.2-3B  LAYER=27  MODE=unprompted  POOLING=mean \
#   VARIANTS="gelu semisup"  bash experiments/dbpedia/scripts/run_experiment.sh
#
# Tree convention (explicit pooling axis):
#   dbpedia/activations/<MODEL>/<POOLING>/<MODE>/
#   experiments/dbpedia/configs/<MODEL>/layer<LAYER>/<MODE>_<POOLING>_<ENC>.yaml
#   dbpedia/checkpoints/<MODEL>/layer<LAYER>/<MODE>_<POOLING>_<ENC>/
set -euo pipefail

MODEL="${MODEL:-llama3.2-3B}"
LAYER="${LAYER:-27}"
MODE="${MODE:-unprompted}"
POOLING="${POOLING:-mean}"
VARIANTS="${VARIANTS:-gelu semisup}"
# HuggingFace id (extraction only); derive from MODEL short name by default.
case "$MODEL" in
  llama3.2-3B) MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-3B}" ;;
  Qwen3.5-9B)  MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}" ;;
  *)           MODEL_ID="${MODEL_ID:?set MODEL_ID for unknown MODEL}" ;;
esac

# Anchor to this checkout, including when invoked outside the project root.
export GEOAE_ROOT="${GEOAE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$GEOAE_ROOT"

CFG_DIR="experiments/dbpedia/configs/${MODEL}/layer${LAYER}"
RES_DIR="dbpedia/results/${MODEL}/layer${LAYER}"
for ENC in $VARIANTS; do
    CFG="${CFG_DIR}/${MODE}_${POOLING}_${ENC}.yaml"
    [[ -f "$CFG" ]] || { echo "MISSING config: $CFG" >&2; exit 1; }
done
mkdir -p "$RES_DIR"

echo "=== Step 1: Extract (${MODE}, pooling=${POOLING}, layer ${LAYER}) ==="
"$GEOAE_ROOT/scripts/py" -m geoae.extraction.dbpedia \
    --model_name "$MODEL_ID" --mode "$MODE" --pooling "$POOLING" --layer "$LAYER" \
    2>&1 | tee "${RES_DIR}/extract_${MODE}_${POOLING}.log"

for ENC in $VARIANTS; do
    CFG="${CFG_DIR}/${MODE}_${POOLING}_${ENC}.yaml"
    echo ""
    echo "=== Step 2: Train AE — ${MODE}_${POOLING}_${ENC} ==="
    [[ -f "$CFG" ]] || { echo "  MISSING config: $CFG"; exit 1; }
    "$GEOAE_ROOT/scripts/py" -m geoae.training.mse --config "$CFG" --no_wandb \
        2>&1 | tee "${RES_DIR}/train_${MODE}_${POOLING}_${ENC}.log"
done

echo ""
echo "=== Step 3: Evaluate clustering vs ground truth (pooling=${POOLING}) ==="
"$GEOAE_ROOT/scripts/py" -m geoae.evaluation.dbpedia --compare \
    --model "$MODEL" --layer "$LAYER" --pooling "$POOLING" \
    2>&1 | tee "${RES_DIR}/eval_${POOLING}.log"

echo ""
echo "=== Done. Results in ${RES_DIR}/ ==="
