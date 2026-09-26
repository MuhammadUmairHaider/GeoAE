#!/usr/bin/env bash
# MATCHED BASE for the token-bypass AE, in the same style as the AE: every intervention is
# applied to (h - mean)/std - b[current token] and b[tok] is added back afterwards (arm `hb`),
# next to the plain base `h`. The bypass AE's `z` results already exist on the SAME doc sets
# (its joint-correct caches), so z is not re-run; the re-run h must reproduce the old h
# exactly, which the summary checks before pairing.
#
#   z − hb   the encoder's contribution to edit quality
#   hb − h   what the token-mean removal alone does to a residual edit
#
# Expect hb ≈ h on DB14 / bias_in_bios: every prompt ends in '":', where b is a constant and
# the edit rules are shift-invariant; hb can differ only through the generated answer tokens
# and perplexity. The number control (edit at the attractor noun) is where it can differ.
#
#   ./eval_out/run_tokbias_hb.sh              # ~5 h: steer ~35 min, range_db14 ~45 min, number ~6 min, range_bios ~3.5 h
#   ./eval_out/run_tokbias_hb.sh summary
# Finished steps are skipped; FORCE=1 redoes them.
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE
PY=.venv/bin/python
TOK=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
TABLE=e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz
TAG=d6144_tokbias
JC_DB14=dbpedia/joint_correct_db14_l27_$TAG.json          # reused as-is: same docs as the z results
JC_BIOS=dbpedia/joint_correct_biasbios_l27_$TAG.json
BIOS_CONCEPTS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27
FAILED=()
ALL_STEPS=(steer range_db14 number range_bios summary)
REQUESTED=("$@"); [ $# -eq 0 ] && REQUESTED=("${ALL_STEPS[@]}")
want() { for s in "${REQUESTED[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
for f in "$TOK" "$TABLE" "$JC_DB14" "$JC_BIOS"; do [ -f "$f" ] || { echo "[abort] missing $f"; exit 1; }; done

step() {   # step <name> <done-file> <function>
    want "$1" || return
    if [ -z "$FORCE" ] && [ -f "$2" ] && grep -qs '"status": "complete"\|"summary"\|"summary_selectivity"' "$2"; then
        echo "[skip] $1 — $2 exists (FORCE=1 to redo)"; return
    fi
    echo -e "\n[run ] $1  ($(date +%H:%M))"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

steer() {
    $PY -u -m geoae.interp.steering_concept_compare --checkpoint "$TOK" --dataset db14 \
        --correct_json $JC_DB14 --hb_table $TABLE --skip_z \
        --out results/steer_db14_${TAG}_hb.json | tee logs/steer_db14_${TAG}_hb.log
}
range_db14() {
    $PY -u -m geoae.interp.range_intervention_compare --checkpoint "$TOK" --dataset db14 \
        --correct_json $JC_DB14 --saliency dprime --tao 2.0 --hb_table $TABLE --substrates h,hb \
        --out results/range_intervention_db14_${TAG}_dprime_hb.json | tee logs/range_db14_${TAG}_dprime_hb.log
}
number() {
    rm -f results/range_number_${TAG}_dprime_hb.json      # the tool refuses to overwrite
    NUMBER_CHECKPOINT="$TOK" NUMBER_OUTPUT=results/range_number_${TAG}_dprime_hb.json \
        bash eval_out/run_number_control.sh --saliency dprime --hb_table $TABLE \
        | tee logs/range_number_${TAG}_dprime_hb.log
}
range_bios() {
    $PY -u -m geoae.interp.range_intervention_compare --checkpoint "$TOK" --dataset biasbios \
        --concepts $BIOS_CONCEPTS --correct_json $JC_BIOS --saliency dprime --tao 2.0 --alphas 1.0 2.0 \
        --hb_table $TABLE --substrates h,hb \
        --out results/range_intervention_biasbios_${TAG}_dprime_hb.json | tee logs/range_biasbios_${TAG}_dprime_hb.log
}
summary() { $PY -u eval_out/summarize_tokbias_hb.py | tee logs/summary_tokbias_hb.log; }

step steer      results/steer_db14_${TAG}_hb.json                        steer
step range_db14 results/range_intervention_db14_${TAG}_dprime_hb.json    range_db14
step number     results/range_number_${TAG}_dprime_hb.json               number
step range_bios results/range_intervention_biasbios_${TAG}_dprime_hb.json range_bios
want summary && summary

[ ${#FAILED[@]} -gt 0 ] && { echo -e "\n[done] FAILED steps: ${FAILED[*]}"; exit 1; }
echo -e "\n[done] $(date +%H:%M)"
