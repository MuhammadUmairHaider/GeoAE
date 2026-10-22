#!/bin/bash
# One Qwen2.5-3B L35 bypass-AE control dataset (same settings as run_control_conf.sh). Usage: qwen_one.sh <db14|biasbios|ag_news>
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT" && source configs/sites/local.sh
ds=$1; ARM=qwen25_l35_d4096
TB=checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt
I=results/delta/$ARM/interventions; L=results/delta/$ARM/logs/interventions; mkdir -p $I $L results/delta/joint_correct
extra=(); [ "$ds" = biasbios ] && extra=(--concepts 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27)
out=$I/range_intervention_${ds}_dprime_conf.json; rm -f $out
echo "=== qwen $ds start $(date)"
scripts/py -u -m geoae.evaluation.control --checkpoint $TB --dataset $ds "${extra[@]}" \
  --saliency dprime --tao 2.0 --removal_modes range --replace comp --steer_modes global,range \
  --correct_json results/delta/joint_correct/joint_correct_${ds}_l35_$ARM.json --out $out > $L/range_${ds}_dprime_conf.log 2>&1
echo "=== qwen $ds exit $? $(date)"
