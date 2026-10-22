#!/usr/bin/env bash
# Source from the checkout on Argo. Bulk GPT-2 artifacts live on /scratch.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
source "$GEOAE_ROOT/configs/sites/argo.sh"
export GEOAE_PYTHON="${GEOAE_PYTHON:-${UV_PROJECT_ENVIRONMENT}/bin/python}"
export OMP_NUM_THREADS="${GPT_RUN_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export GEOAE_TRAIN_WORKERS="${GEOAE_TRAIN_WORKERS:-1}"
export GEOAE_VAL_WORKERS="${GEOAE_VAL_WORKERS:-1}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
[ -x "$GEOAE_PYTHON" ] || { echo "Python not executable: $GEOAE_PYTHON" >&2; return 1; }

# Preserve existing artifacts and shared Gemma paths. Refuse conflicting links.
gpt_run_link() {
    local relative="$1" target="$DELTA_DATA/$1" existing="$GEOAE_ROOT/$1"
    mkdir -p "$target" "$(dirname "$existing")"
    if [ -e "$existing" ] || [ -L "$existing" ]; then
        [ "$(readlink -f "$existing")" = "$(readlink -f "$target")" ] || {
            echo "Existing GPT-2 path resolves outside DELTA_DATA: $existing" >&2
            return 1
        }
    else
        ln -s "$target" "$existing"
    fi
}
gpt_run_link activations_gpt2_sampled_struct5_10M || return 1
gpt_run_link corpus/gpt2_sampled_struct5_v1 || return 1
gpt_run_link checkpoints/gpt2 || return 1
gpt_run_link e2e/checkpoints/general/gpt2 || return 1
gpt_run_link cache/gpt2 || return 1
gpt_run_link cache_ids/gpt2 || return 1
gpt_run_link biasbios/activations/gpt2 || return 1
gpt_run_link dbpedia/activations/gpt2 || return 1
gpt_run_link results/gpt_run || return 1
unset -f gpt_run_link
mkdir -p "$GEOAE_ROOT/logs/gpt_run" "$GEOAE_ROOT/runs/gpt_run"
