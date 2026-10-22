#!/usr/bin/env bash
# gpu | api : Gemma 3 1B interp pipeline on the box (d2304 first: 2x hidden, like Llama d6144)
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT" && export GEOAE_REMOTE=1 HF_HUB_OFFLINE=1 JUDGE_CONCURRENCY=16 && source configs/sites/local.sh >/dev/null
if [ "$1" = gpu ]; then
  for a in g1b_l13_d2304 g1b_l25_d2304 g1b_l13_d1152 g1b_l25_d1152 g1b_l13_d4608 g1b_l25_d4608; do
    bash experiments/paper/scripts/run_interp.sh $a gpu 2>&1 | tee logs/interp_${a}_gpu.log
  done
  echo "GEM GPU EXIT" | tee -a logs/interp_g1b_l25_d4608_gpu.log
else
  set -a; source .env; set +a
  for a in g1b_l13_d2304 g1b_l25_d2304; do
    until grep -aq "^\[done\]" logs/interp_${a}_gpu.log 2>/dev/null; do sleep 60; done
    bash experiments/paper/scripts/run_interp.sh $a judge judge_tok judge3 atlas atlas_tok 2>&1 | tee logs/interp_${a}_api.log
  done
  echo "GEM API EXIT" | tee -a logs/interp_g1b_l25_d2304_api.log
fi
