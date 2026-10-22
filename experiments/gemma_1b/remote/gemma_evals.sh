#!/usr/bin/env bash
# Gemma 3 1B non-judge evals on the box. The four probe-act extractions run in parallel (the 1B model
# leaves the A100 mostly idle at one example per forward), then layer-13 and layer-25 arms run as two
# parallel chains, d2304 first (2x hidden, like Llama d6144). LLM-judge steps (cjudge*) are left out.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT" && export GEOAE_REMOTE=1 HF_HUB_OFFLINE=1 && source configs/sites/local.sh >/dev/null
M=google/gemma-3-1b-pt; PY=.venv/bin/python
for L in 13 25; do
  $PY -u -m geoae.extraction.biasbios --layer $L --model_name $M --pooling last --max_length 256 --n_train 50000 \
      --n_test 20000 --out_base biasbios/activations/g1b_l$L > logs/probe_acts_biasbios_g1b_l$L.log 2>&1 &
  $PY -u -m geoae.extraction.dbpedia --mode unprompted --pooling last --layer $L --model_name $M --max_length 256 \
      --n_train 50000 --n_test 10000 --dir_suffix _ids --out_base dbpedia/activations/g1b_l$L \
      > logs/probe_acts_dbpedia_g1b_l$L.log 2>&1 &
done
wait
bash experiments/paper/scripts/run_delta_prep.sh g1b_probe_acts 2>&1 | tee logs/g1b_prep_probe_acts.log   # verifies; skips complete ones
chain() { for a in "$@"; do bash experiments/paper/scripts/run_delta_evals.sh $a evals interventions steering concept geometry probes > logs/evals_$a.log 2>&1; done; }
chain g1b_l13_d2304 g1b_l13_d1152 g1b_l13_d4608 &
chain g1b_l25_d2304 g1b_l25_d1152 g1b_l25_d4608 &
wait
echo "GEM EVALS EXIT" | tee logs/gemma_evals_done.log
