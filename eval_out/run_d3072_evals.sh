#!/usr/bin/env bash
# Full eval suite for the d3072 dpc run (latent_dim 3072, finished ep50), against the
# d6144 dpc run at the SAME epoch (both step_0014200.pt) — a pure width comparison.
#
# DO NOT use d3072/best_val.pt: it is epoch 10, pre-clustering, centroids never
# initialised. val_mse rises once clustering starts, so best_val never moves past it.
#
# Re-runnable: a step whose output already exists is skipped, and a failing step is
# reported at the end instead of silently ending the chain. Ordered fast -> slow,
# ~4.5-5 h total. The three LM-in-the-loop runs build their own joint-correct doc
# sets for this AE (different ae_sha), which is included in the times below.
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE
B=checkpoints/llama3.2-3B/layer27
D3072=$B/k2000_bnh_b32k_lam1_d3072_dpc/step_0014200.pt   # epoch 50
D6144=$B/k2000_bnh_b32k_lam1_d6144_dpc/step_0014200.pt   # epoch 50, matched
JC_DB14=dbpedia/joint_correct_db14_l27_d3072_dpc.json
JC_BIOS=dbpedia/joint_correct_biasbios_l27_d3072_dpc.json
BIOS_CONCEPTS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27  # no "teacher" (0% few-shot)
FAILED=()

# Guard: refuse to evaluate a checkpoint whose centroids were never initialised.
uv run python - "$D3072" "$D6144" <<'EOF' || { echo "[abort] checkpoint guard failed"; exit 1; }
import sys, torch
for p in sys.argv[1:]:
    c = torch.load(p, map_location="cpu", weights_only=False)
    ok = bool(c["model_state"]["centroids_initialized"])
    print(f"[guard] {p}: epoch {c['epoch']}, centroids_initialized={ok}")
    if not ok:
        sys.exit(1)
EOF

step() {   # step <name> <done-check> <function>
    if eval "$2"; then echo "[skip] $1 — already done"; return; fi
    echo -e "\n[run ] $1  ($(date +%H:%M))"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

mmlu() {   # ~2 min. The d6144 ep50 number already exists: eval_out/mmlu_gelu50.json
    uv run python -u -m geoae.interp.causal_concept_compare --checkpoint "$D3072" \
        --layer 27 --mmlu 2000 --seed 42 | tee logs/mmlu_d3072_50.log || return 1
    mv results_ccc_mmlu.json eval_out/mmlu_d3072_50.json
}
probe() {  # ~2 min, same anchor holdout as probe_init_arms.json
    uv run python -u -m geoae.interp.concept_probe --cache cache \
        --models "d3072=$D3072,d6144=$D6144" --baselines "" \
        --exclude_anchors --anchor_seed 42 --anchor_per_class 25 --anchor_min_examples 5 \
        --atlas_last cache/atlas8k_last.npz --atlas_min_examples 25 \
        --out eval_out/probe_d3072_50.json | tee logs/probe_d3072_50.log
}
geom() {   # ~2 min
    uv run python -u -m geoae.interp.concept_geometry --cache cache \
        --rungs pos_coarse,pos_fine,ravel_country,topic14,language \
        --models "d3072=$D3072,d6144=$D6144" --baselines "" \
        --projections tsne,pca --n_points 6000 --perplexity 30 --max_classes 8 \
        --outdir figures/d3072_50 | tee logs/geom_d3072_50.log
}
cq() {     # ~25 min, same 1M sample as cq_init_arms.json / cq_tanh50.json
    uv run python -u -m geoae.interp.clustering_quality \
        --checkpoints $D3072 $D6144 --names d3072 d6144 \
        --activations_dir activations_diverse_10M --layer 27 --n_sample 1000000 --seed 0 \
        --out eval_out/cq_d3072_50.json | tee logs/cq_d3072_50.log
}
steer() {  # ~30 min incl. building the DB14 doc set
    uv run python -u -m geoae.interp.steering_concept_compare --checkpoint $D3072 --dataset db14 \
        --correct_json $JC_DB14 --out results/steer_db14_d3072_50.json \
        | tee logs/steer_db14_d3072_50.log
}
range_db14() {  # ~45 min, reuses the DB14 set; same protocol as range_intervention_db14_b32k_dpc_dprime.json
    uv run python -u -m geoae.interp.range_intervention_compare --checkpoint $D3072 --dataset db14 \
        --correct_json $JC_DB14 --saliency dprime --tao 2.0 \
        --out results/range_intervention_db14_d3072_50_dprime.json \
        | tee logs/range_db14_d3072_50_dprime.log
}
range_bios() {  # ~3 h (~1 h doc-set build + ~2 h evals); same protocol as the d6144 bias_in_bios run
    uv run python -u -m geoae.interp.range_intervention_compare --checkpoint $D3072 --dataset biasbios \
        --concepts $BIOS_CONCEPTS --correct_json $JC_BIOS --saliency dprime --tao 2.0 --alphas 1.0 2.0 \
        --out results/range_intervention_biasbios_d3072_50_dprime.json \
        | tee logs/range_biasbios_d3072_50_dprime.log
}

done_range() { grep -qs '"summary"' "$1"; }   # partial range files exist mid-run; only a summary means done

step "MMLU"                 "[ -f eval_out/mmlu_d3072_50.json ]"                                      mmlu
step "concept probe"        "[ -f eval_out/probe_d3072_50.json ]"                                     probe
step "geometry"             "[ -d figures/d3072_50/tsne ]"                                            geom
step "clustering quality"   "[ -f eval_out/cq_d3072_50.json ]"                                        cq
step "DB14 steering"        "[ -f results/steer_db14_d3072_50.json ]"                                 steer
step "range DB14 (d')"      "done_range results/range_intervention_db14_d3072_50_dprime.json"         range_db14
step "range bias_in_bios"   "done_range results/range_intervention_biasbios_d3072_50_dprime.json"     range_bios

echo
if [ ${#FAILED[@]} -eq 0 ]; then echo "[done] all evals complete ($(date +%H:%M))"; else echo "[done] FAILED: ${FAILED[*]}"; exit 1; fi
