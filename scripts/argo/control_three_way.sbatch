#!/bin/bash
# Control edits in THREE spaces on the same documents (Gemma 3 4B L33): base residual (h), token-bypass AE latent (z, its
# reconstruction spliced in) and Gemma Scope 16k codes (s, its reconstruction spliced in, BOS kept). Plain steering and
# range steering at the strengths in ALPHAS (default 0.4 0.5); d-prime top 30% of each space's coordinates, tao 2.
# geoae/interp/range_intervention_compare.py --substrates h,z,s. Per-document confidences are stored, so the confidence
# tables (relative and absolute) and three-way paired differences come from one file.
#   sbatch scripts/argo/control_three_way.sbatch <atlas_s0|atlas_s1|atlas_s2|db14|biasbios|ag_news|emotion6|news20> [--smoke]
#   ALPHAS="0.4 0.5" (default); atlas draws use the Gemma pools of the Atlas control rows and 260 other-class documents.
#   TP3=1 sbatch --export=ALL,TP3=1 ... : three-way TRUE-POSITIVE documents (correct under base, AE splice and SAE splice);
#   builds its own pool (up to 80 fit + N_EVAL=100 evaluation documents per class, concepts with < MIN_EVAL=20 skipped)
#   and writes ..._hzs_tp3.json. The five paper datasets try at most MAX_TRY=3000 documents per class (AG News: test + train).
#   <db14|biasbios|ag_news>_s<N>: a DOCUMENT REDRAW of that Hub dataset (the test split is shuffled with seed N instead of
#   42, so other documents are tried first); own pool and output files under the full name, e.g. ..._db14_s1_dprime_hzs_tp3.json.
# Output: results/delta/g4b_l33_d5120/interventions/sae/range_intervention_<ds>_dprime_hzs.json
#SBATCH --job-name=control-3way
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0      # wikimedia/wikipedia (perplexity texts) is streamed from the Hub
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
DS="${1:?dataset}"; shift || true
DSET=$DS; SEEDARG=()
if [[ "$DS" =~ ^(db14|biasbios|ag_news)_s([0-9]+)$ ]]; then DSET=${BASH_REMATCH[1]}; SEEDARG=(--seed "${BASH_REMATCH[2]}"); fi
ARM="${ARM:-g4b_l33_d5120}"     # ARM=g4b_l33_d5120_ne100 CK=<its checkpoint>: another main AE with its own pools and output folder
ATP=g4b; [ "$ARM" = g4b_l33_d5120 ] || ATP=$ARM      # tag of the Atlas three-way true-positive list
# CK=<other AE checkpoint> AE_TAG=_nobypass: another AE as the z substrate (own pool and output files, tagged AE_TAG)
CK="${CK:-checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt}"
AE_TAG="${AE_TAG:-}"
JC=results/delta/joint_correct/joint_correct_${DS}_l33_$ARM.json
EXTRA=(); TAG=""
[ "$DSET" = biasbios ] && EXTRA=(--concepts 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27)
case "$DS" in atlas_s*)
    AD=results/delta/d6144/interventions/atlas_classes/s${DS#atlas_s}; JC=$AD/joint_correct_kept_g4b.json
    EXTRA=(--concepts "$("$PY" -c "import json;print(json.load(open('$AD/jc_report_g4b.json'))['concepts_arg'])")" --n_comp 260 --max_try 600) ;;
esac
[ "${1:-}" = --smoke ] && { EXTRA=(--smoke); TAG=_smoke; }
if [ "${TP3:-0}" = 1 ]; then      # documents that are true positives under base, AE splice AND SAE splice; own pool file
    case "$DS" in atlas_s*) JC=results/delta/d6144/interventions/atlas_classes/s${DS#atlas_s}/joint_correct_tp3_${ATP}_gs16k$TAG.json ;;
                  *)        JC=results/delta/joint_correct/joint_correct_${DS}_l33_${ARM}${AE_TAG}_tp3_gs16k$TAG.json ;; esac
    if [ -n "$TAG" ]; then EXTRA+=(--tp_all --min_eval 1)                 # smoke: tiny pool, keep every concept
    else EXTRA+=(--tp_all --n_eval "${N_EVAL:-100}" --min_eval "${MIN_EVAL:-20}"); fi
    case "$DS" in atlas_s*) ;; *)      # Hub datasets: at most MAX_TRY documents tried per class; AG News also draws on train
        if [ -z "$TAG" ]; then EXTRA+=(--max_try "${MAX_TRY:-3000}"); fi
        if [ "$DSET" = ag_news ]; then EXTRA+=(--extra_split train); fi ;;
    esac
    TAG=_tp3$TAG
fi
SUBS="${SUBS:-h,z,s}"; SUBTAG="${SUBS//,/}"       # SUBS=s1,s5: only the SAE top-1 / top-5 feature arms (file ..._s1s5...)
if [ "${EDIT_PPL:-1}" = 0 ]; then EXTRA+=(--edit_ppl 0); fi
# PERCENT=1.0 VTAG=_alldims STEER=range: range steering with NO coordinate selection (the range gate on every coordinate)
if [ -n "${PERCENT:-}" ]; then EXTRA+=(--percent "$PERCENT"); fi
SUBTAG="$SUBTAG${VTAG:-}$AE_TAG"; STEER="${STEER:-global,range}"
if [ -n "$AE_TAG" ] && [ "${TP3:-0}" != 1 ]; then echo "AE_TAG needs TP3=1 (the default pools belong to the bypass AE)"; exit 2; fi
case "$DS" in atlas_s*) if [ -n "$AE_TAG" ]; then echo "AE_TAG: Atlas pools are not tagged per AE yet"; exit 2; fi ;; esac
OUT=results/delta/$ARM/interventions/sae; mkdir -p "$OUT"
# shellcheck disable=SC2086
"$PY" -u -m geoae.interp.range_intervention_compare --checkpoint "$CK" --dataset "$DSET" "${SEEDARG[@]}" "${EXTRA[@]}" \
    --saliency dprime --tao 2.0 --removal_modes "" --steer_modes "$STEER" --alphas ${ALPHAS:-0.4 0.5} \
    --sae scope:16k --substrates "$SUBS" --correct_json "$JC" \
    --out "$OUT/range_intervention_${DS}_dprime_$SUBTAG$TAG.json" 2>&1 | tee "$OUT/range_${DS}_dprime_$SUBTAG$TAG.log"
