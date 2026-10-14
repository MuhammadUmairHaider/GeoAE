#!/bin/bash
# Control-dataset range removal/steering (operators only) + Gemma FineWeb-Atlas draws, on the box.
#   bash scripts/remote/run_control_extra.sh [smoke]
# Operators: --removal_modes range --replace comp --steer_modes global,range --saliency dprime --tao 2.0
# (alphas 0.5 1 2, n_fit 80, n_eval 50, n_comp 80, n_ppl 40, seed 42 = defaults).
# Full : arms d6144 (Llama L27) and g4b_l33_d5120 (Gemma L33) x {jigsaw2 trec6 emotion6 news20}, then Gemma x Atlas s0..s2.
#        Outputs under results/delta/<arm>/interventions/ ; Gemma Atlas jc files get the tag _g4b (Llama files untouched).
# Smoke: 4 datasets x 2 arms with --smoke --concepts all, plus Gemma Atlas s0 (classes 0,1, tag _smoke_g4b);
#        everything under results/control_extra_smoke/.
# Each run is skipped when its output already holds "summary"; a failure never aborts the job.
# Overrides (all optional): CTRL_ROOT (repo dir), CTRL_ENV (env file), CTRL_ARMS, CTRL_DATASETS, CTRL_ATLAS_SEEDS
# (space lists; empty = none), CTRL_LOG. Used to split the full job across machines (box / Argo / Delta).
cd "${CTRL_ROOT:-$HOME/GeoAE}" && source "${CTRL_ENV:-scripts/remote/env.sh}"
PY=scripts/delta/py
MODE="${1:-full}"
OPS=(--removal_modes range --replace comp --steer_modes global,range --saliency dprime --tao 2.0)
declare -A CK=([d6144]=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
               [g4b_l33_d5120]=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt)
declare -A LY=([d6144]=27 [g4b_l33_d5120]=33)
ARMS=(${CTRL_ARMS-d6144 g4b_l33_d5120})
DATASETS=(${CTRL_DATASETS-jigsaw2 trec6 emotion6 news20})
LOG=${CTRL_LOG:-results/delta/run_control_extra.out}
AD=results/delta/d6144/interventions/atlas_classes       # frozen Atlas draws (cfg/parquet) + Llama jc files live here
if [ "$MODE" = smoke ]; then
  R=results/control_extra_smoke; LOG=$R/run_control_extra.out; mkdir -p $R
  JD=$R/joint_correct; ATLAS_JC_DIR=$R/atlas_classes; ATAG=_smoke_g4b
  SM=(--smoke)   # --smoke rewrites '--concepts all' to a 2-class default, so all classes are passed explicitly below
  AT_JC=(--n_fit 20 --n_eval 10 --max_try 48 --min_test 5 --classes 0,1)
  AT_IV=(--alphas 1.0 --n_fit 20 --n_eval 10 --n_comp 24 --n_ppl 8 --n_wiki 20 --max_try 48)
  ATLAS_SEEDS=(0)
else
  JD=results/delta/joint_correct; ATLAS_JC_DIR=$AD; ATAG=_g4b
  SM=()
  AT_JC=(--n_fit 80 --n_eval 50 --max_try 600 --min_test 40 --classes all)
  AT_IV=(--n_comp 260 --max_try 600)
  ATLAS_SEEDS=(${CTRL_ATLAS_SEEDS-0 1 2})
fi
mkdir -p "$(dirname "$LOG")" "$JD"
say() { echo "$*" | tee -a "$LOG"; }
done_ok() { [ -f "$1" ] && grep -q '"summary"' "$1"; }

for arm in "${ARMS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    if [ "$MODE" = smoke ]; then
      out=$R/$arm/range_intervention_${ds}_dprime_smoke.json; lg=$R/$arm/logs/range_${ds}_dprime.log
    else
      out=results/delta/$arm/interventions/range_intervention_${ds}_dprime.json
      lg=results/delta/$arm/logs/interventions/range_${ds}_dprime.log
    fi
    mkdir -p "$(dirname "$out")" "$(dirname "$lg")"
    if done_ok "$out"; then say "=== $arm $ds skip (done) $(date)"; continue; fi
    rm -f "$out"
    CON=all
    if [ "$MODE" = smoke ]; then CON=$($PY -c "import json;print(','.join(map(str,range(len(json.load(open('results/delta/control_datasets/$ds/cfg.json'))['classes'])))))"); fi
    say "=== $arm $ds start $(date) concepts=$CON"
    $PY -u -m geoae.interp.range_intervention_compare --checkpoint "${CK[$arm]}" --dataset "$ds" "${OPS[@]}" \
      "${SM[@]}" --concepts "$CON" --correct_json "$JD/joint_correct_${ds}_l${LY[$arm]}_${arm}.json" --out "$out" > "$lg" 2>&1
    say "=== $arm $ds exit $? $(date)"
  done
done

# Gemma only: FineWeb-Atlas draws (stages mirror scripts/argo/atlas_classes.sbatch)
arm=g4b_l33_d5120
for s in "${ATLAS_SEEDS[@]}"; do
  SD=$ATLAS_JC_DIR/s$s
  if [ "$MODE" = smoke ]; then
    out=$R/$arm/range_intervention_atlas_s${s}_dprime_smoke.json; lg=$R/$arm/logs
  else
    out=results/delta/$arm/interventions/atlas_classes/range_intervention_atlas_s${s}_dprime.json
    lg=results/delta/$arm/logs/interventions
  fi
  mkdir -p "$SD" "$(dirname "$out")" "$lg"
  if [ ! -f "$AD/s$s/cfg.json" ] || [ ! -f "$AD/s$s/data.parquet" ]; then
    say "=== $arm atlas_s$s exit 3 (frozen draw $AD/s$s/{cfg.json,data.parquet} missing) $(date)"; continue
  fi
  if done_ok "$out"; then say "=== $arm atlas_s$s skip (done) $(date)"; continue; fi
  if [ ! -f "$SD/jc_report$ATAG.json" ]; then
    say "=== $arm atlas_s$s jc start $(date)"
    $PY -u eval_out/atlas_classes_joint_correct.py --seed "$s" --checkpoint "${CK[$arm]}" --out_dir "$ATLAS_JC_DIR" \
      "${AT_JC[@]}" --tag "$ATAG" > "$lg/atlas_s${s}_jc${ATAG}.log" 2>&1
    say "=== $arm atlas_s$s jc exit $? $(date)"
  fi
  if [ ! -f "$SD/jc_report$ATAG.json" ]; then say "=== $arm atlas_s$s exit 4 (no jc report) $(date)"; continue; fi
  KEPT=$($PY -c "import json;print(json.load(open('$SD/jc_report$ATAG.json'))['concepts_arg'])")
  rm -f "$out"
  say "=== $arm atlas_s$s start $(date) concepts=$KEPT"
  $PY -u -m geoae.interp.range_intervention_compare --checkpoint "${CK[$arm]}" --dataset "atlas_s$s" "${OPS[@]}" \
    --concepts "$KEPT" --correct_json "$SD/joint_correct_kept$ATAG.json" "${AT_IV[@]}" --out "$out" \
    > "$lg/range_atlas_s${s}_dprime.log" 2>&1
  say "=== $arm atlas_s$s exit $? $(date)"
done
say "=== all done $(date)"
