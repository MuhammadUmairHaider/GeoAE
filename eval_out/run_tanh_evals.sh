#!/usr/bin/env bash
# dpc_tanh finished (epoch 50, resumed from 28). Evaluate its FINAL checkpoint against
# the gelu dpc run at the SAME epoch (both step_0014200.pt), so tanh-vs-gelu is not
# confounded with a checkpoint-age difference. The *_tanh50 outputs from the
# mid-training snapshot are kept separately.
#
# DO NOT use dpc_tanh/best_val.pt: it is epoch 5, pre-clustering, centroids all zero
# (val_mse rises once clustering starts, so best_val never advanced past epoch 5).
#
# Re-runnable: a step whose output already exists is skipped, and a failing step is
# reported at the end instead of silently ending the chain. (Sep 17 09:43: systemd-oomd
# killed the whole tmux scope during clustering_quality; its loader now peaks at
# ~25 GB instead of ~107 GB — see clustering_quality.read_rows.)
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE
B=checkpoints/llama3.2-3B/layer27
TANH=$B/k2000_bnh_b32k_lam1_d6144_dpc_tanh/step_0014200.pt   # epoch 50 (NOT best_val: that is epoch 5)
GELU=$B/k2000_bnh_b32k_lam1_d6144_dpc/step_0014200.pt        # epoch 50, matched
JC=dbpedia/joint_correct_db14_l27_dpc_tanh50.json
FAILED=()

step() {   # step <name> <done-check> <function>
    if eval "$2"; then echo "[skip] $1 — already done"; return; fi
    echo -e "\n[run ] $1"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

mmlu() {   # ~2 min
    for arm in "$TANH:tanh50" "$GELU:gelu50"; do
        uv run python -u -m geoae.interp.causal_concept_compare --checkpoint "${arm%%:*}" \
            --layer 27 --mmlu 2000 --seed 42 || return 1
        mv results_ccc_mmlu.json eval_out/mmlu_${arm##*:}.json
    done | tee logs/mmlu_tanh50.log
}
probe() {  # ~2 min, same anchor holdout as probe_init_arms.json
    uv run python -u -m geoae.interp.concept_probe --cache cache \
        --models "tanh50=$TANH,gelu50=$GELU" --baselines "" \
        --exclude_anchors --anchor_seed 42 --anchor_per_class 25 --anchor_min_examples 5 \
        --atlas_last cache/atlas8k_last.npz --atlas_min_examples 25 \
        --out eval_out/probe_tanh50.json | tee logs/probe_tanh50.log
}
geom() {   # ~2 min
    uv run python -u -m geoae.interp.concept_geometry --cache cache \
        --rungs pos_coarse,pos_fine,ravel_country,topic14,language \
        --models "tanh50=$TANH,gelu50=$GELU" --baselines "" \
        --projections tsne,pca --n_points 6000 --perplexity 30 --max_classes 8 \
        --outdir figures/tanh50 | tee logs/geom_tanh50.log
}
cq() {     # ~25 min, same 1M sample as cq_init_arms.json
    uv run python -u -m geoae.interp.clustering_quality \
        --checkpoints $TANH $GELU --names tanh50 gelu50 \
        --activations_dir activations_diverse_10M --layer 27 --n_sample 1000000 --seed 0 \
        --out eval_out/cq_tanh50.json | tee logs/cq_tanh50.log
}
steer() {  # ~30 min; builds the tanh joint-correct set
    uv run python -u -m geoae.interp.steering_concept_compare --checkpoint $TANH --dataset db14 \
        --correct_json $JC --out results/steer_db14_dpc_tanh50.json | tee logs/steer_db14_tanh50.log
}
range_() { # ~45 min; reuses the set from steer
    uv run python -u -m geoae.interp.range_intervention_compare --checkpoint $TANH --dataset db14 \
        --correct_json $JC --saliency abs \
        --out results/range_intervention_db14_dpc_tanh50_abs.json | tee logs/range_db14_tanh50_abs.log
}

step "MMLU"               "[ -f eval_out/mmlu_gelu50.json ]"                                   mmlu
step "concept probe"      "[ -f eval_out/probe_tanh50.json ]"                                  probe
step "geometry"           "[ -d figures/tanh50/tsne ]"                                         geom
step "clustering quality" "[ -f eval_out/cq_tanh50.json ]"                                     cq
step "DB14 steering"      "[ -f results/steer_db14_dpc_tanh50.json ]"                          steer
step "range h vs z"       "grep -qs '\"summary\"' results/range_intervention_db14_dpc_tanh50_abs.json" range_

echo
if [ ${#FAILED[@]} -eq 0 ]; then echo "[done] all evals complete"; else echo "[done] FAILED: ${FAILED[*]}"; exit 1; fi

# ---- then resume training: epochs 29..50 (~3.9 h). -a appends to the existing log.
# uv run python -u -m geoae.train \
#   --config configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_tanh.yaml \
#   --resume latest | tee -a logs/llama_l27_bnh_b32k_lam1_d6144_dpc_tanh.log
