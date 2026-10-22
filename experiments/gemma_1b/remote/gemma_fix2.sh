#!/usr/bin/env bash
# Serial rerun of the Gemma steps that hit CUDA OOM while four chains shared the GPU (finished steps skip).
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT" && export GEOAE_REMOTE=1 && source configs/sites/local.sh >/dev/null
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
bash experiments/paper/scripts/run_delta_evals.sh g1b_l13_d1152 mmlu        > logs/evals_fix2_g1b_l13_d1152.log 2>&1
bash experiments/paper/scripts/run_delta_evals.sh g1b_l25_d1152 csteer cgen > logs/evals_fix2_g1b_l25_d1152.log 2>&1
bash experiments/paper/scripts/run_delta_evals.sh g1b_l25_d4608 csteer cgen > logs/evals_fix2_g1b_l25_d4608.log 2>&1
echo "GEM FIX2 EXIT" | tee logs/gemma_fix2_done.log
