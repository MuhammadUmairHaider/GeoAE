#!/bin/bash
# Box twin of scripts/delta/control_conf_one.sbatch: one control dataset with confidence records.
#   bash scripts/remote/control_conf_one.sh <arm> <checkpoint> <layer> <db14|biasbios|ag_news>
#   env: ALPHAS (default "0.5 1 2"), REMOVAL (default range; empty = skip range removal), TAG (default conf)
cd ~/GeoAE && source scripts/remote/env.sh
ARM=${1:?arm}; TB=${2:?checkpoint}; LY=${3:?layer}; ds=${4:?dataset}
ALPHAS=${ALPHAS:-0.5 1 2}; REMOVAL=${REMOVAL-range}; TAG=${TAG:-conf}
I=results/delta/$ARM/interventions; L=results/delta/$ARM/logs/interventions; JCD=results/delta/joint_correct
mkdir -p "$I" "$L" "$JCD"
extra=(); [ "$ds" = biasbios ] && extra=(--concepts 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27)
out=$I/range_intervention_${ds}_dprime_${TAG}.json; rm -f "$out"
echo "=== $ARM $ds alphas=[$ALPHAS] removal=[$REMOVAL] tag=$TAG start $(date)"
scripts/delta/py -u -m geoae.interp.range_intervention_compare --checkpoint "$TB" --dataset "$ds" "${extra[@]}" \
  --saliency dprime --tao 2.0 --removal_modes "$REMOVAL" --replace comp --steer_modes global,range --alphas $ALPHAS \
  --correct_json "$JCD/joint_correct_${ds}_l${LY}_$ARM.json" --out "$out" > "$L/range_${ds}_dprime_${TAG}.log" 2>&1
echo "=== $ARM $ds exit $? $( grep -q '"summary"' "$out" 2>/dev/null && echo ok || echo INCOMPLETE) $(date)"
