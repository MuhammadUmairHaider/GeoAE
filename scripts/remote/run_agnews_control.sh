#!/bin/bash
# AG News control runs (range_intervention_compare), same settings as the DBpedia-14 / Bias in Bios runs:
# d-prime saliency, top 30% dims, tao 2, n_fit 80 / n_eval 50 / n_comp 80 / n_ppl 40, alphas 0.5 1 2, seed 42, all 4 classes.
cd ~/GeoAE && source scripts/remote/env.sh
run() {  # arm checkpoint layer
  local arm=$1 ck=$2 L=$3
  local out=results/delta/$arm/interventions/range_intervention_ag_news_dprime.json
  echo "=== $arm start $(date)"
  scripts/delta/py -u -m geoae.interp.range_intervention_compare --checkpoint "$ck" --dataset ag_news \
    --saliency dprime --tao 2.0 \
    --correct_json results/delta/joint_correct/joint_correct_ag_news_l${L}_$arm.json \
    --out "$out" > results/delta/$arm/logs/interventions/range_ag_news_dprime.log 2>&1
  echo "=== $arm exit $? $(date)"
}
run d6144         checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt 27
run g4b_l33_d5120 checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt 33
echo "=== all done $(date)"
