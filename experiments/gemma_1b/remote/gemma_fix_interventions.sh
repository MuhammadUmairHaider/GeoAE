#!/usr/bin/env bash
# Rerun the Gemma 3 1B intervention steps that failed under HF_HUB_OFFLINE=1 in gemma_evals.sh:
# steer / range_db14 / range_bios stream wikimedia/wikipedia, which needs the network. Offline mode
# stays off here (verified: cached gated Gemma still loads without a token). Finished steps
# (number, number_hb) are skipped by run_delta_evals.sh. One serial chain per layer (joint-correct
# pools are per layer); each arm waits until its gemma_evals.sh chain is past interventions.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT" && export GEOAE_REMOTE=1 && source configs/sites/local.sh >/dev/null
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
fix() { for a in "$@"; do
    until grep -aq "csteer" logs/evals_$a.log 2>/dev/null; do sleep 60; done
    bash experiments/paper/scripts/run_delta_evals.sh $a interventions > logs/evals_fix_$a.log 2>&1
  done; }
fix g1b_l13_d2304 g1b_l13_d1152 g1b_l13_d4608 &
fix g1b_l25_d2304 g1b_l25_d1152 g1b_l25_d4608 &
wait
echo "GEM FIX EXIT" | tee logs/gemma_fix_done.log
