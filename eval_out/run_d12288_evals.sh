#!/usr/bin/env bash
# Eval suite for the d12288 dpc run — the 4x point of the L27 K=2000 width series
# (d3072 -> d6144 -> d12288; all bnh / b32k / lam1 / dpc-init, all 50 epochs).
#
# Checkpoint facts, verified 2026-09-20 after training finished:
#   d12288/step_0014200.pt   epoch 50, val_mse 0.01927, centroids_initialized=True   <- used here
#   d12288/best_val.pt       epoch 49, val_mse 0.01924, centroids_initialized=True
# Unlike d3072 (whose best_val is the pre-clustering epoch 10), d12288's best_val is a
# real checkpoint — but every step below uses step_0014200 so the comparison against
# d6144 and d3072 is epoch-matched.
#
# Reference arms already on disk:
#   d6144/step_0014200.pt (ep50)  — probe / geom / cq compare against this
#   d6144/best_val.pt     (ep47)  — the checkpoint the EXISTING d6144 range + steer JSONs
#                                   were run on; the summary step flags that mismatch
#   d3072/step_0014200.pt (ep50)  — eval_out/*d3072_50*, results/*d3072_50*
#
# Usage:
#   ./eval_out/run_d12288_evals.sh                  # all steps, fast -> slow, ~5-6 h
#   ./eval_out/run_d12288_evals.sh cq range_db14    # only these steps
#   FORCE=1 ./eval_out/run_d12288_evals.sh probe    # re-run an overwrite-safe step (number is immutable)
#   ./eval_out/run_d12288_evals.sh summary          # just reprint the width-series tables
# Steps: mmlu probe geom cq steer range_db14 range_bios number summary
#
# Re-runnable: a step whose output already exists is skipped, and a failing step is
# reported at the end instead of silently ending the chain. The three LM-in-the-loop
# runs build their own joint-correct doc sets for this AE (different ae_sha); that
# build time is included in the estimates below.
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE

B=checkpoints/llama3.2-3B/layer27
D12288=$B/k2000_bnh_b32k_lam1_d12288_dpc/step_0014200.pt   # epoch 50
D6144=$B/k2000_bnh_b32k_lam1_d6144_dpc/step_0014200.pt     # epoch 50, matched
D3072=$B/k2000_bnh_b32k_lam1_d3072_dpc/step_0014200.pt     # epoch 50, matched
JC_DB14=dbpedia/joint_correct_db14_l27_d12288_dpc.json
JC_BIOS=dbpedia/joint_correct_biasbios_l27_d12288_dpc.json
BIOS_CONCEPTS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27  # no "teacher" (0% few-shot)
FAILED=()

ALL_STEPS=(mmlu probe geom cq steer range_db14 range_bios number summary)
REQUESTED=("$@"); [ $# -eq 0 ] && REQUESTED=("${ALL_STEPS[@]}")
for r in "${REQUESTED[@]}"; do
    case " ${ALL_STEPS[*]} " in *" $r "*) ;; *) echo "[abort] unknown step '$r' (have: ${ALL_STEPS[*]})"; exit 2;; esac
done
want() { for s in "${REQUESTED[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

# Guard: refuse to evaluate a checkpoint whose centroids were never initialised.
uv run python - "$D12288" "$D6144" <<'EOF' || { echo "[abort] checkpoint guard failed"; exit 1; }
import sys, torch
for p in sys.argv[1:]:
    c = torch.load(p, map_location="cpu", weights_only=False)
    ok = bool(c["model_state"]["centroids_initialized"])
    print(f"[guard] {p}: epoch {c['epoch']}, val_mse {c['val_mse']:.5f}, centroids_initialized={ok}")
    if not ok:
        sys.exit(1)
EOF

step() {   # step <name> <done-check> <function>
    want "$1" || return
    if [ -z "$FORCE" ] && eval "$2"; then echo "[skip] $1 — already done"; return; fi
    echo -e "\n[run ] $1  ($(date +%H:%M))"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

mmlu() {   # ~2 min. d6144 ep50 already exists: eval_out/mmlu_gelu50.json (delta -0.0265)
    uv run python -u -m geoae.interp.causal_concept_compare --checkpoint "$D12288" \
        --layer 27 --mmlu 2000 --seed 42 | tee logs/mmlu_d12288_50.log || return 1
    mv results_ccc_mmlu.json eval_out/mmlu_d12288_50.json   # --mmlu ignores --out, it always writes this name
}
probe() {  # ~3 min, same anchor holdout as probe_init_arms.json / probe_d3072_50.json
    uv run python -u -m geoae.interp.concept_probe --cache cache \
        --models "d12288=$D12288,d6144=$D6144,d3072=$D3072" --baselines "" \
        --exclude_anchors --anchor_seed 42 --anchor_per_class 25 --anchor_min_examples 5 \
        --atlas_last cache/atlas8k_last.npz --atlas_min_examples 25 \
        --out eval_out/probe_d12288_50.json | tee logs/probe_d12288_50.log
}
geom() {   # ~3 min
    uv run python -u -m geoae.interp.concept_geometry --cache cache \
        --rungs pos_coarse,pos_fine,ravel_country,topic14,language \
        --models "d12288=$D12288,d6144=$D6144,d3072=$D3072" --baselines "" \
        --projections tsne,pca --n_points 6000 --perplexity 30 --max_classes 8 \
        --outdir figures/d12288_50 | tee logs/geom_d12288_50.log
}
cq() {     # ~50-60 min for all three, same 1M sample / seed 0 as cq_d3072_50.json.
           # RAM: the latents are held dense, 1e6 x 12288 x fp32 = 49 GB for d12288 alone
           # (freed before the next arm is encoded). d12288 goes first so it fails fast.
    local avail; avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    echo "[cq] MemAvailable ${avail} GB; d12288 latents need ~49 GB"
    [ "$avail" -lt 60 ] && echo "[cq] WARNING: under 60 GB free — stop other jobs or drop --n_sample to 500000"
    uv run python -u -m geoae.interp.clustering_quality \
        --checkpoints $D12288 $D6144 $D3072 --names d12288 d6144 d3072 \
        --activations_dir activations_diverse_10M --layer 27 --n_sample 1000000 --seed 0 \
        --out eval_out/cq_d12288_50.json | tee logs/cq_d12288_50.log
}
steer() {  # ~30-40 min incl. building the DB14 doc set
    uv run python -u -m geoae.interp.steering_concept_compare --checkpoint $D12288 --dataset db14 \
        --correct_json $JC_DB14 --out results/steer_db14_d12288_50.json \
        | tee logs/steer_db14_d12288_50.log
}
range_db14() {  # ~45-60 min, reuses the DB14 doc set; same protocol as range_intervention_db14_b32k_dpc_dprime.json
    uv run python -u -m geoae.interp.range_intervention_compare --checkpoint $D12288 --dataset db14 \
        --correct_json $JC_DB14 --saliency dprime --tao 2.0 \
        --out results/range_intervention_db14_d12288_50_dprime.json \
        | tee logs/range_db14_d12288_50_dprime.log
}
range_bios() {  # ~3-3.5 h (~1 h doc-set build + ~2 h evals); same protocol as the d6144 bias_in_bios run
    uv run python -u -m geoae.interp.range_intervention_compare --checkpoint $D12288 --dataset biasbios \
        --concepts $BIOS_CONCEPTS --correct_json $JC_BIOS --saliency dprime --tao 2.0 --alphas 1.0 2.0 \
        --out results/range_intervention_biasbios_d12288_50_dprime.json \
        | tee logs/range_biasbios_d12288_50_dprime.log
}
number() {  # ~3 min on A100; top-30% d-prime edits, with rotations + shuffled labels
    NUMBER_CHECKPOINT="$D12288" \
    NUMBER_OUTPUT=results/range_number_d12288_50_dprime.json \
        bash eval_out/run_number_control.sh --saliency dprime \
        | tee logs/range_number_d12288_50_dprime.log || return 1
    uv run python eval_out/review_number_control.py \
        --source results/range_number_d12288_50_dprime.json \
        --review_out eval_out/number_d12288_50_review.json \
        --figure_prefix figures/number_control/d12288_50_review
}
summary() {  # seconds — d3072 / d6144 / d12288 side by side, paired stats, from whatever exists
    uv run python -u eval_out/summarize_width_series.py | tee logs/summary_width_series.log
}

done_range() { grep -qs '"summary"' "$1"; }   # partial range files exist mid-run; only a summary means done

step "mmlu"        "[ -f eval_out/mmlu_d12288_50.json ]"                                  mmlu
step "probe"       "[ -f eval_out/probe_d12288_50.json ]"                                 probe
step "geom"        "[ -d figures/d12288_50/tsne ]"                                        geom
step "cq"          "[ -f eval_out/cq_d12288_50.json ]"                                    cq
step "steer"       "[ -f results/steer_db14_d12288_50.json ]"                             steer
step "range_db14"  "done_range results/range_intervention_db14_d12288_50_dprime.json"     range_db14
step "range_bios"  "done_range results/range_intervention_biasbios_d12288_50_dprime.json" range_bios
step "number"      "grep -qs '\"status\": \"complete\"' results/range_number_d12288_50_dprime.json && [ -f eval_out/number_d12288_50_review.json ]" number
step "summary"     "false"                                                                summary

echo
echo "[note] checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d12288_dpc holds 39 GB of"
echo "       intermediate steps. Once these evals are done only best_val.pt and step_0014200.pt"
echo "       are needed — the d6144/d3072 arms were trimmed the same way."
if [ ${#FAILED[@]} -eq 0 ]; then echo "[done] all requested evals complete ($(date +%H:%M))"; else echo "[done] FAILED: ${FAILED[*]}"; exit 1; fi
