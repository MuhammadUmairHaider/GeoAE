#!/bin/bash
# FineWeb-Atlas class corpus, range removal/steering at the classification token, ONE draw per job (1 H100, group partition).
#   sbatch scripts/argo/atlas_classes.sbatch <seed> [stages] [key=value ...]
# stages (comma list, default prep,joint_correct,interventions; each one SKIPS when its output already exists):
#   prep           decode eligible chunks, document-level few-shot/fit/test split, cfg.json + relatedness.json
#                  (needs the frozen results/delta/d6144/interventions/atlas_classes/draws.json; CPU only)
#   joint_correct  base-LM and AE-splice accuracy per class, joint-correct pools, kept/dropped classes (>= 40 test docs)
#   interventions  geoae.interp.range_intervention_compare on the kept classes (db14 settings: --saliency dprime --tao 2.0,
#                  --alphas 0.5 1.0 2.0), writes range_intervention_atlas_s<seed>_dprime.json incl. comp_by_class
# key=value: ckpt= | n_fit=80 | n_eval=50 | n_comp=80 | n_ppl=40 | n_wiki=100 | max_try=600 | min_test=40 | alphas="0.5 1.0 2.0"
#            | classes=all (jc stage; smoke: 0,1) | concepts= (override the kept list) | otag= (suffix of the interventions output/log) | smoke=1 (tiny n, file tag _smoke, alpha 1.0)
# Smoke:  sbatch --time=00:40:00 scripts/argo/atlas_classes.sbatch 0 joint_correct,interventions smoke=1
# Full:   for s in 0 1 2; do sbatch scripts/argo/atlas_classes.sbatch $s; done    (each is an independent single-GPU job)
#SBATCH --job-name=atlas-classes
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
SEED="${1:?usage: sbatch scripts/argo/atlas_classes.sbatch <seed> [stages] [key=value ...]}"
STAGES="${2:-prep,joint_correct,interventions}"
shift; [ $# -gt 0 ] && shift
declare -A OPT=([ckpt]="checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt"
  [n_fit]=80 [n_eval]=50 [n_comp]=80 [n_ppl]=40 [n_wiki]=100 [max_try]=600 [min_test]=40 [alphas]="0.5 1.0 2.0"
  [classes]=all [concepts]="" [smoke]=0 [otag]="")
for kv in "$@"; do OPT[${kv%%=*}]="${kv#*=}"; done
TAG=""
if [ "${OPT[smoke]}" = "1" ]; then
  TAG="_smoke"; OPT[n_fit]=20; OPT[n_eval]=10; OPT[n_comp]=24; OPT[n_ppl]=8; OPT[n_wiki]=20; OPT[max_try]=48; OPT[min_test]=5
  OPT[alphas]="1.0"; [ "${OPT[classes]}" = "all" ] && OPT[classes]="0,1"
fi
OTAG="${OPT[otag]:-$TAG}"      # suffix of the interventions output only (timing runs on the full-size pools: otag=_time concepts=0,1)
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
AD=results/delta/d6144/interventions/atlas_classes
SD=$AD/s$SEED
OUT=$AD/range_intervention_atlas_s${SEED}_dprime${OTAG}.json
mkdir -p "$SD/logs" logs
has() { [[ ",$STAGES," == *",$1,"* ]]; }
echo "[atlas-classes] seed=$SEED stages=$STAGES opts: $(for k in "${!OPT[@]}"; do printf '%s=%s ' "$k" "${OPT[$k]}"; done)"
echo "[atlas-classes] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
run() { local name="$1"; shift; echo "[atlas-classes $(date +%H:%M:%S)] start $name"; local t0=$SECONDS
        "$@" 2>&1 | tee "$SD/logs/$name${OTAG:-$TAG}.log"; local r=${PIPESTATUS[0]}
        echo "[atlas-classes $(date +%H:%M:%S)] done $name rc=$r in $((SECONDS - t0))s"; [ "$r" -ne 0 ] && rc=$r; return "$r"; }

if has prep; then
  if [ -f "$SD/cfg.json" ] && [ -f "$SD/data.parquet" ] && [ -f "$SD/relatedness.json" ]; then echo "[skip] prep (done)"
  else run prep "$PY" -u eval_out/atlas_classes_prep.py prep --seed "$SEED" || exit "$rc"; fi
fi
if has joint_correct; then
  if [ -f "$SD/jc_report$TAG.json" ]; then echo "[skip] joint_correct (done)"
  else run joint_correct "$PY" -u eval_out/atlas_classes_joint_correct.py --seed "$SEED" --checkpoint "${OPT[ckpt]}" \
        --classes "${OPT[classes]}" --n_fit "${OPT[n_fit]}" --n_eval "${OPT[n_eval]}" --max_try "${OPT[max_try]}" \
        --min_test "${OPT[min_test]}" ${TAG:+--tag $TAG} || exit "$rc"; fi
fi
if has interventions; then
  KEPT="${OPT[concepts]}"
  [ -z "$KEPT" ] && KEPT=$("$PY" -c "import json;print(json.load(open('$SD/jc_report$TAG.json'))['concepts_arg'])")
  if [ -f "$OUT" ] && "$PY" -c "import json,sys;sys.exit(0 if 'summary' in json.load(open('$OUT')) else 1)"; then echo "[skip] interventions (done: $OUT)"
  else
    rm -f "$OUT"     # an interrupted run leaves a partial file; the tool restarts from the (cached) pools
    # wikimedia/wikipedia (ppl texts, wiki-mean) is streamed from the Hub: not cached, so offline mode must be off here
    run interventions env HF_HUB_OFFLINE=0 "$PY" -u -m geoae.interp.range_intervention_compare --checkpoint "${OPT[ckpt]}" \
        --dataset "atlas_s$SEED" --concepts "$KEPT" --correct_json "$SD/joint_correct_kept$TAG.json" \
        --saliency dprime --tao 2.0 --alphas ${OPT[alphas]} --n_fit "${OPT[n_fit]}" --n_eval "${OPT[n_eval]}" \
        --n_comp "${OPT[n_comp]}" --n_ppl "${OPT[n_ppl]}" --n_wiki "${OPT[n_wiki]}" --max_try "${OPT[max_try]}" \
        --out "$OUT" || exit "$rc"
  fi
fi
echo "[atlas-classes] finished rc=$rc $(date)"
exit "$rc"
