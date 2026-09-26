#!/usr/bin/env bash
# Re-run the d6144 INTERVENTION evals at step_0014200 (epoch 50).
#
# Why: the d6144 steering and range JSONs on disk were run on best_val.pt (epoch 47):
#   results/steer_db14_dpc.json
#   results/range_intervention_db14_b32k_dpc_dprime.json
#   results/range_intervention_biasbios_b32k_dpc_dprime.json
# Every other eval in the width series (MMLU, clustering quality, probes, kNN) is already
# epoch-matched, so this is the last thing between the series and a clean 1x/2x/4x contrast.
# Nothing here overwrites the epoch-47 files; the new ones carry a _d6144_50 suffix, and
# eval_out/summarize_width_series.py picks them up automatically once they exist.
#
# Usage:
#   ./eval_out/run_d6144_ep50_evals.sh                  # all three, ~4.5-5 h
#   ./eval_out/run_d6144_ep50_evals.sh steer range_db14 # DB14 only, ~1.5 h
#   FORCE=1 ./eval_out/run_d6144_ep50_evals.sh steer    # redo a finished step
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE

D6144=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc/step_0014200.pt
JC_DB14=dbpedia/joint_correct_db14_l27_d6144_50.json
JC_BIOS=dbpedia/joint_correct_biasbios_l27_d6144_50.json
BIOS_CONCEPTS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27  # no "teacher" (0% few-shot)
FAILED=()

ALL_STEPS=(steer range_db14 range_bios)
REQUESTED=("$@"); [ $# -eq 0 ] && REQUESTED=("${ALL_STEPS[@]}")
for r in "${REQUESTED[@]}"; do
    case " ${ALL_STEPS[*]} " in *" $r "*) ;; *) echo "[abort] unknown step '$r' (have: ${ALL_STEPS[*]})"; exit 2;; esac
done
want() { for s in "${REQUESTED[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

uv run python - "$D6144" <<'EOF' || { echo "[abort] checkpoint guard failed"; exit 1; }
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ok = bool(c["model_state"]["centroids_initialized"])
print(f"[guard] {sys.argv[1]}: epoch {c['epoch']}, val_mse {c['val_mse']:.5f}, centroids_initialized={ok}")
sys.exit(0 if ok and c["epoch"] == 50 else 1)
EOF

step() {   # step <name> <done-check> <function>
    want "$1" || return
    if [ -z "$FORCE" ] && eval "$2"; then echo "[skip] $1 — already done (FORCE=1 to redo)"; return; fi
    echo -e "\n[run ] $1  ($(date +%H:%M))"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

steer() {       # ~30-40 min incl. building this checkpoint's own joint-correct doc set
    uv run python -u -m geoae.interp.steering_concept_compare --checkpoint $D6144 --dataset db14 \
        --correct_json $JC_DB14 --out results/steer_db14_d6144_50.json \
        | tee logs/steer_db14_d6144_50.log
}
range_db14() {  # ~45 min, reuses the DB14 doc set
    uv run python -u -m geoae.interp.range_intervention_compare --checkpoint $D6144 --dataset db14 \
        --correct_json $JC_DB14 --saliency dprime --tao 2.0 \
        --out results/range_intervention_db14_d6144_50_dprime.json \
        | tee logs/range_db14_d6144_50_dprime.log
}
range_bios() {  # ~3-3.5 h (~1 h doc-set build + ~2 h evals)
    uv run python -u -m geoae.interp.range_intervention_compare --checkpoint $D6144 --dataset biasbios \
        --concepts $BIOS_CONCEPTS --correct_json $JC_BIOS --saliency dprime --tao 2.0 --alphas 1.0 2.0 \
        --out results/range_intervention_biasbios_d6144_50_dprime.json \
        | tee logs/range_biasbios_d6144_50_dprime.log
}

done_range() { grep -qs '"summary"' "$1"; }

step "steer"       "[ -f results/steer_db14_d6144_50.json ]"                                 steer
step "range_db14"  "done_range results/range_intervention_db14_d6144_50_dprime.json"         range_db14
step "range_bios"  "done_range results/range_intervention_biasbios_d6144_50_dprime.json"     range_bios

echo
echo "[next] uv run python eval_out/summarize_width_series.py   # now epoch-matched throughout"
if [ ${#FAILED[@]} -eq 0 ]; then echo "[done] all requested evals complete ($(date +%H:%M))"; else echo "[done] FAILED: ${FAILED[*]}"; exit 1; fi
