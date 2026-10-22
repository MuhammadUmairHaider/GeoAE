#!/bin/bash
# Shared Delta environment and GPT-2 bulk-data links. Source from the repo root.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
cd "$GEOAE_ROOT"
source scripts/env.sh
export GEOAE_PYTHON="${GEOAE_PYTHON:-${UV_PROJECT_ENVIRONMENT}/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
[ -x "$GEOAE_PYTHON" ] || { echo "Python not executable: $GEOAE_PYTHON" >&2; return 1; }

# Only this campaign's paths are created. Refuse to replace an existing path
# pointing elsewhere; other experiment artifacts and links remain intact.
gpt2_link() {
    local relative="$1" target="$DELTA_DATA/$1" existing="$GEOAE_ROOT/$1"
    mkdir -p "$target" "$(dirname "$existing")"
    if [ -e "$existing" ] || [ -L "$existing" ]; then
        [ "$(readlink -f "$existing")" = "$(readlink -f "$target")" ] || {
            echo "Existing campaign path resolves outside DELTA_DATA: $existing" >&2
            return 1
        }
    else
        ln -s "$target" "$existing"
    fi
}
gpt2_link activations_gpt2_sampled_struct5_10M
gpt2_link corpus/gpt2_sampled_struct5_v1
gpt2_link checkpoints/gpt2
gpt2_link e2e/checkpoints/general/gpt2
unset -f gpt2_link
mkdir -p logs/gpt2 runs/gpt2_2x
