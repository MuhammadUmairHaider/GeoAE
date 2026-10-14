#!/bin/bash
# Control reruns that also record confidence in the gold label (first-token softmax prob) and per-doc outcomes.
# Standard operators only (range removal with other-class mean, range steering, plain steering), alphas 0.5 1 2,
# d-prime top 30%, tao 2, n_fit 80 / n_eval 50 / n_comp 80, seed 42 -- same docs as the earlier *_dprime.json runs
# (same joint-correct files), so accuracies must reproduce them. Writes *_dprime_conf.json; never overwrites old files.
# Usage (box): bash scripts/remote/run_control_conf.sh 2>&1 | tee results/delta/run_control_conf.out
cd ~/GeoAE && source scripts/remote/env.sh
BIOS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27   # no "teacher" (0% few-shot)
run() {  # arm checkpoint layer dataset
  local arm=$1 ck=$2 L=$3 ds=$4 extra=()
  [ "$ds" = biasbios ] && extra=(--concepts "$BIOS")
  local out=results/delta/$arm/interventions/range_intervention_${ds}_dprime_conf.json
  mkdir -p results/delta/$arm/interventions results/delta/$arm/logs/interventions
  if [ -f "$out" ] && python3 -c "import json,sys; sys.exit(0 if 'summary' in json.load(open('$out')) else 1)"; then
    echo "=== $arm $ds already done"; return 0; fi
  rm -f "$out"
  echo "=== $arm $ds start $(date)"
  scripts/delta/py -u -m geoae.interp.range_intervention_compare --checkpoint "$ck" --dataset "$ds" "${extra[@]}" \
    --saliency dprime --tao 2.0 --removal_modes range --replace comp --steer_modes global,range \
    --correct_json results/delta/joint_correct/joint_correct_${ds}_l${L}_$arm.json \
    --out "$out" > results/delta/$arm/logs/interventions/range_${ds}_dprime_conf.log 2>&1
  echo "=== $arm $ds exit $? $(date)"
}
LL=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
GM=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
for ds in db14 biasbios ag_news; do run d6144 "$LL" 27 "$ds"; done
for ds in db14 biasbios ag_news; do run g4b_l33_d5120 "$GM" 33 "$ds"; done
echo "=== all done $(date)"
