#!/usr/bin/env bash
# Interventions for the TOKEN-BYPASS AE (d6144 dpc sampled + model.token_bias), with
# the exact commands and settings used for its parent d6144_new
# (eval_out/run_d6144_sampled_evals.sh), so the two compare as z − h head to head.
#
#   NEW   d6144_new  k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt  (results exist)
#   TOK   tokbias    k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
#
# The tools feed the bypass AE its token ids: fit-time activations are captured with
# the id at the same position, and every splice reads ids off the input embedding
# (geoae.hooks.TokenIdTap). Joint-correct doc sets are rebuilt for this AE
# (its weight fingerprint differs), so each tool first spends a while predicting.
#
#   ./eval_out/run_tokbias_interventions.sh            # all, fast -> slow, ~5 h
#   ./eval_out/run_tokbias_interventions.sh number     # only these steps
#   ./eval_out/run_tokbias_interventions.sh summary
# Steps: number (~5 min) | steer (~35 min) | range_db14 (~45 min) | range_bios (~3.5 h) | summary
# Finished steps are skipped; FORCE=1 redoes them. A failed step is reported at the end.
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE
PY=.venv/bin/python
TOK=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
TAG=d6144_tokbias
JC_DB14=dbpedia/joint_correct_db14_l27_$TAG.json
JC_BIOS=dbpedia/joint_correct_biasbios_l27_$TAG.json
BIOS_CONCEPTS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27  # no "teacher" (0% few-shot)
FAILED=()
ALL_STEPS=(number steer range_db14 range_bios number_hb summary)
REQUESTED=("$@"); [ $# -eq 0 ] && REQUESTED=("${ALL_STEPS[@]}")
want() { for s in "${REQUESTED[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
[ -f "$TOK" ] || { echo "[abort] missing $TOK"; exit 1; }

step() {   # step <name> <done-file> <function>
    want "$1" || return
    if [ -z "$FORCE" ] && [ -f "$2" ] && grep -qs '"status": "complete"\|"summary"\|"concepts"' "$2"; then
        echo "[skip] $1 — $2 exists (FORCE=1 to redo)"; return
    fi
    echo -e "\n[run ] $1  ($(date +%H:%M))"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

number() {
    NUMBER_CHECKPOINT="$TOK" NUMBER_OUTPUT=results/range_number_${TAG}_dprime.json \
        bash eval_out/run_number_control.sh --saliency dprime | tee logs/range_number_${TAG}_dprime.log || return 1
    $PY eval_out/review_number_control.py --source results/range_number_${TAG}_dprime.json \
        --review_out eval_out/number_${TAG}_review.json --figure_prefix figures/number_control/${TAG}_review
}
steer() {
    $PY -u -m geoae.interp.steering_concept_compare --checkpoint "$TOK" --dataset db14 \
        --correct_json $JC_DB14 --out results/steer_db14_$TAG.json | tee logs/steer_db14_$TAG.log
}
range_db14() {
    $PY -u -m geoae.interp.range_intervention_compare --checkpoint "$TOK" --dataset db14 \
        --correct_json $JC_DB14 --saliency dprime --tao 2.0 \
        --out results/range_intervention_db14_${TAG}_dprime.json | tee logs/range_db14_${TAG}_dprime.log
}
range_bios() {
    $PY -u -m geoae.interp.range_intervention_compare --checkpoint "$TOK" --dataset biasbios \
        --concepts $BIOS_CONCEPTS --correct_json $JC_BIOS --saliency dprime --tao 2.0 --alphas 1.0 2.0 \
        --out results/range_intervention_biasbios_${TAG}_dprime.json | tee logs/range_biasbios_${TAG}_dprime.log
}
# Matched base arm (hb): edit h - b[tok], add b[tok] back. Only the number control can
# differ from h: DB14 / bias_in_bios prompts all end in the same token ('":'), where b is a
# constant and every edit rule is shift-invariant, so hb == h at their decision position.
number_hb() {
    rm -f results/range_number_${TAG}_dprime_hb.json   # the tool refuses to overwrite; an incomplete file is stale
    NUMBER_CHECKPOINT="$TOK" NUMBER_OUTPUT=results/range_number_${TAG}_dprime_hb.json \
        bash eval_out/run_number_control.sh --saliency dprime \
        --hb_table e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz \
        | tee logs/range_number_${TAG}_dprime_hb.log
}
summary() { $PY -u eval_out/summarize_tokbias_interventions.py | tee logs/summary_tokbias_interventions.log; }

step number     results/range_number_${TAG}_dprime.json                 number
step steer      results/steer_db14_$TAG.json                            steer
step range_db14 results/range_intervention_db14_${TAG}_dprime.json      range_db14
step range_bios results/range_intervention_biasbios_${TAG}_dprime.json  range_bios
step number_hb  results/range_number_${TAG}_dprime_hb.json              number_hb
want summary && summary

[ ${#FAILED[@]} -gt 0 ] && { echo -e "\n[done] FAILED steps: ${FAILED[*]}"; exit 1; }
echo -e "\n[done] $(date +%H:%M)"
