#!/bin/bash
# Plain steering on TWO-WAY TRUE-POSITIVE documents (correct under the base model AND under the AE splice) for the models
# that have no SAE (Llama-3.2-3B L27, Qwen2.5-3B L35), under the rules of the Gemma three-way run
# (TP3=1 experiments/paper/argo/control_three_way.sbatch): up to 80 fit + N_EVAL=100 evaluation documents per class, concepts with
# fewer than MIN_EVAL=20 evaluation documents skipped, at most MAX_TRY=3000 documents tried per class on the Hub / control
# datasets (AG News: test + train); Atlas draws use the model's kept classes, 260 other-class documents and 600 documents
# tried per (class, split). Edit: plain steering only (st_global: a - alpha * (class mean - other-class mean) on every
# coordinate) at the strengths in ALPHAS (default 0.4 0.5). geoae/evaluation/control.py --substrates h,z.
#   sbatch experiments/paper/argo/control_tp2_plain.sbatch <d6144|d6144k256|qwen25_l35_d4096> <dataset> [<dataset> ...]
#   d6144k256 = the K=256 codebook arm (11 Oct); STEP=step_00NNNNN evaluates an earlier checkpoint of it into tp2_<STEP>/
#   datasets: db14 biasbios ag_news emotion6 news20 atlas_s0 atlas_s1 atlas_s2
#   an L40 instead of an H100:  sbatch --gres=gpu:l40:1 --nodelist=argo76 experiments/paper/argo/control_tp2_plain.sbatch ...
# Output per dataset: results/delta/<arm>/interventions/tp2/range_intervention_<ds>_dprime_hz_tp2.json (+ .log). A dataset
# whose output already holds "summary" is skipped, so the same line can be resubmitted after an interruption.
#SBATCH --job-name=control-tp2
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0      # wikimedia/wikipedia (perplexity texts) is streamed from the Hub
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
ARM="${1:?arm}"; shift
case "$ARM" in
  d6144)            CK=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt; LY=27; ATAG="" ;;
  d6144k256)        CK=checkpoints/llama3.2-3B/layer27/k256_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/${STEP:-step_0014450}.pt; LY=27; ATAG=_k256 ;;
  # 12 Oct no-balancing / 100-epoch models (configs/nobal_e100): same kept Atlas classes per model, own true-positive lists
  d6144_ne100)            CK=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_nobal_e100/step_0028900.pt; LY=27; ATAG="" ;;
  qwen25_l35_d4096_ne100) CK=checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias_nobal_e100/step_0030500.pt; LY=35; ATAG=_qwen ;;
  # 12 Oct no-staging models (configs/nobal_nostage_e100)
  d6144_ns100)            CK=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_nobal_nostage_e100/step_0028900.pt; LY=27; ATAG="" ;;
  qwen25_l35_d4096_ns100) CK=checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias_nobal_nostage_e100/step_0030500.pt; LY=35; ATAG=_qwen ;;
  # 12 Oct no-staging models, 50 epochs (configs/nobal_nostage_e50)
  d6144_ns50)            CK=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_nobal_nostage_e50/step_0014450.pt; LY=27; ATAG="" ;;
  qwen25_l35_d4096_ns50) CK=checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias_nobal_nostage_e50/step_0015250.pt; LY=35; ATAG=_qwen ;;
  qwen25_l35_d4096) CK=checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt; LY=35; ATAG=_qwen ;;
  *) echo "[tp2] unknown arm $ARM"; exit 2 ;;
esac
[ -f "$CK" ] || { echo "[tp2] missing $CK"; exit 1; }
SFX="${STEP:+_$STEP}"       # a non-final checkpoint keeps its own outputs and its own true-positive list
OUT=results/delta/$ARM/interventions/tp2$SFX; mkdir -p "$OUT" logs results/delta/joint_correct
AD=results/delta/d6144/interventions/atlas_classes      # the frozen Atlas draws (cfg.json, data.parquet) of every model
echo "[tp2] $ARM datasets: $* host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
rc=0
for DS in "$@"; do
  out=$OUT/range_intervention_${DS}_dprime_hz_tp2.json
  if [ -f "$out" ] && grep -q '"summary"' "$out"; then echo "[tp2] skip $DS (done)"; continue; fi
  EXTRA=()
  case "$DS" in
    atlas_s*)
      S=${DS#atlas_s}; SD=$AD/s$S
      if [ ! -f "$SD/jc_report$ATAG.json" ]; then       # this model's kept classes (>= 40 jointly-correct test documents)
        echo "[tp2 $(date +%H:%M:%S)] $ARM $DS: building the kept-class list"
        "$PY" -u geoae/extraction/atlas_joint_correct.py --seed "$S" --checkpoint "$CK" --out_dir "$AD" --n_fit 80 --n_eval 50 \
            --max_try 600 --min_test 40 --classes all ${ATAG:+--tag $ATAG} > "$OUT/jc_${DS}.log" 2>&1 || { echo "[tp2] $DS jc failed"; rc=1; continue; }
      fi
      JC=$SD/joint_correct_tp2_$ARM.json
      EXTRA=(--concepts "$("$PY" -c "import json;print(json.load(open('$SD/jc_report$ATAG.json'))['concepts_arg'])")" --n_comp 260 --max_try 600) ;;
    *)
      JC=results/delta/joint_correct/joint_correct_${DS}_l${LY}_${ARM}${SFX}_tp2.json
      EXTRA=(--max_try "${MAX_TRY:-3000}")
      [ "$DS" = biasbios ] && EXTRA+=(--concepts 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27)   # no "teacher"
      [ "$DS" = ag_news ] && EXTRA+=(--extra_split train) ;;
  esac
  rm -f "$out"
  echo "[tp2 $(date +%H:%M:%S)] start $ARM $DS"
  # shellcheck disable=SC2086
  "$PY" -u -m geoae.evaluation.control --checkpoint "$CK" --dataset "$DS" "${EXTRA[@]}" \
      --n_eval "${N_EVAL:-100}" --min_eval "${MIN_EVAL:-20}" --saliency dprime --tao 2.0 --removal_modes "" \
      --steer_modes global --alphas ${ALPHAS:-0.4 0.5} --substrates h,z --correct_json "$JC" \
      --out "$out" > "$OUT/range_${DS}_dprime_hz_tp2.log" 2>&1
  r=$?; [ "$r" -ne 0 ] && rc=$r
  echo "[tp2 $(date +%H:%M:%S)] $ARM $DS exit $r $( [ -f "$out" ] && grep -q '"summary"' "$out" && echo ok || echo INCOMPLETE)"
done
echo "[tp2] finished rc=$rc $(date)"
exit "$rc"
