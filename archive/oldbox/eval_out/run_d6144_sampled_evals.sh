#!/usr/bin/env bash
# Full eval sweep for the DATA ABLATION: d6144 dpc trained on the sampled-position
# dump (activations_sampled_10M) vs the identical config on the old dump
# (activations_diverse_10M). Only the training data differs between the two AEs.
#
#   NEW  d6144_dpc_sampled/step_0014450.pt   epoch 50 (289 steps/epoch on the new dump)
#   OLD  d6144_dpc/step_0014200.pt           epoch 50 (284 steps/epoch on the old dump)
#
# Encoder-free controls, one per dump, same dpc init / reinit peaks / K / balancing:
#   km_old  balanced_kmeans_k2000_dpc.npz           fit on the old dump (exists)
#   km_new  balanced_kmeans_k2000_dpc_sampled.npz   fit on the new dump (step km_fit)
#
# Usage:
#   ./eval_out/run_d6144_sampled_evals.sh               # everything, fast -> slow, ~10-11 h
#   ./eval_out/run_d6144_sampled_evals.sh probe cq      # only these steps
#   FORCE=1 ./eval_out/run_d6144_sampled_evals.sh probe # redo a finished step
#   ./eval_out/run_d6144_sampled_evals.sh summary       # just print the old-vs-new tables
#
# Steps (in order; the quick, decisive ones first):
#   km_fit      ~15 min  balanced raw k-means (dpc) on the NEW dump — the data control
#   mmlu        ~2 min   NEW round-trip MMLU (OLD: eval_out/mmlu_gelu50.json)
#   probe       ~6 min   concept probes: OLD, NEW, km_old, km_new in ONE run
#   geom        ~3 min   kNN / PCA / t-SNE: OLD vs NEW
#   cq_new      ~25 min  clustering quality on the NEW dump: km_new, OLD, NEW
#   cq_old      ~25 min  clustering quality on the OLD dump: km_old, OLD, NEW
#   steer       ~35 min  DB14 steering, NEW
#   range_db14  ~45 min  DB14 range interventions (d' saliency), NEW
#   number      ~3 min   grammatical-number control, NEW
#   range_bios  ~3.5 h   bias_in_bios range interventions, NEW
#   old_ep50    ~4.5 h   OLD steer / range_db14 / range_bios at epoch 50 — the existing
#                        OLD intervention files are best_val (epoch 47), so this makes the
#                        comparison epoch-matched (delegates to run_d6144_ep50_evals.sh)
#   summary     seconds  eval_out/summarize_data_ablation.py
#
# INTERVENTION RESULTS ARE COMPARED AS z - h. Each AE builds its OWN joint-correct doc
# set, so absolute selectivities are not comparable across AEs; the summary compares
# the paired AE-minus-base effect, as the width series did.
#
# Re-runnable: finished steps are skipped; a failed step is reported at the end and
# does not stop the rest.
set -o pipefail
# Delta: env.sh resolves GEOAE_ROOT from its own location, exports HF_HOME /
# UV_PROJECT_ENVIRONMENT / SBATCH_ACCOUNT, and cds to the repo root.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/delta/env.sh"

PY=scripts/delta/py
B=checkpoints/llama3.2-3B/layer27
E=e2e/checkpoints/general/llama3.2-3B/layer27
NEW=$B/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt
OLD=$B/k2000_bnh_b32k_lam1_d6144_dpc/step_0014200.pt
KM_OLD=$E/balanced_kmeans_k2000_dpc.npz
KM_NEW=$E/balanced_kmeans_k2000_dpc_sampled.npz
ACTS_NEW=activations_sampled_10M
ACTS_OLD=activations_diverse_10M
TAG=d6144_sampled
JC_DB14=dbpedia/joint_correct_db14_l27_$TAG.json
JC_BIOS=dbpedia/joint_correct_biasbios_l27_$TAG.json
BIOS_CONCEPTS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27  # no "teacher" (0% few-shot)
FAILED=()

ALL_STEPS=(km_fit mmlu probe geom cq_new cq_old steer range_db14 number range_bios old_ep50 summary)
REQUESTED=("$@"); [ $# -eq 0 ] && REQUESTED=("${ALL_STEPS[@]}")
for r in "${REQUESTED[@]}"; do
    case " ${ALL_STEPS[*]} " in *" $r "*) ;; *) echo "[abort] unknown step '$r' (have: ${ALL_STEPS[*]})"; exit 2;; esac
done
want() { for s in "${REQUESTED[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

# ---- guards -----------------------------------------------------------------
if [ "${REQUESTED[*]}" != "summary" ]; then
    if pgrep -f "geoae.train --config configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled" >/dev/null; then
        echo "[abort] the d6144_sampled training is still running — wait for epoch 50"; exit 1
    fi
    [ -f "$KM_OLD" ] || { echo "[abort] missing $KM_OLD"; exit 1; }
    $PY - "$NEW" "$OLD" <<'EOF' || { echo "[abort] checkpoint guard failed"; exit 1; }
import sys, torch
for p in sys.argv[1:]:
    c = torch.load(p, map_location="cpu", weights_only=False)
    ok = bool(c["model_state"]["centroids_initialized"]) and c["epoch"] == 50
    print(f"[guard] {p}: epoch {c['epoch']}, val_mse {c['val_mse']:.5f}, "
          f"centroids_initialized={bool(c['model_state']['centroids_initialized'])}")
    if not ok:
        sys.exit(1)
EOF
fi

step() {   # step <name> <done-check> <function>
    want "$1" || return
    if [ -z "$FORCE" ] && eval "$2"; then echo "[skip] $1 — already done (FORCE=1 to redo)"; return; fi
    echo -e "\n[run ] $1  ($(date +%H:%M))"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

# ---- steps --------------------------------------------------------------------
km_fit() {   # mirrors logs/fit_balanced_dpc_k2000.log (the km_old fit): dpc init, peaks reinit,
             # defaults for K / sample / epochs / seed. Norm comes from NEW (= new-dump stats).
    $PY -u -m geoae.interp.fit_balanced_kmeans --checkpoint "$NEW" --activations $ACTS_NEW/layer_27.npy \
        --init dpc --reinit peaks --n_clusters 2000 --seed 42 \
        --out $KM_NEW | tee logs/fit_balanced_dpc_k2000_sampled.log
}
mmlu() {
    $PY -u -m geoae.interp.causal_concept_compare --checkpoint "$NEW" \
        --layer 27 --mmlu 2000 --seed 42 | tee logs/mmlu_$TAG.log || return 1
    mv results_ccc_mmlu.json eval_out/mmlu_$TAG.json   # --mmlu ignores --out
}
probe() {    # all four arms in ONE run (same anchor holdout as probe_d12288_50.json)
    $PY -u -m geoae.interp.concept_probe --cache cache \
        --models "d6144_old=$OLD,d6144_new=$NEW" \
        --baselines "km_old=$KM_OLD,km_new=$KM_NEW" \
        --exclude_anchors --anchor_seed 42 --anchor_per_class 25 --anchor_min_examples 5 \
        --atlas_last cache/atlas8k_last.npz --atlas_min_examples 25 \
        --out eval_out/probe_$TAG.json | tee logs/probe_$TAG.log
}
geom() {
    $PY -u -m geoae.interp.concept_geometry --cache cache \
        --rungs pos_coarse,pos_fine,ravel_country,topic14,language \
        --models "d6144_old=$OLD,d6144_new=$NEW" --baselines "" \
        --projections tsne,pca --n_points 6000 --perplexity 30 --max_classes 8 \
        --outdir figures/$TAG | tee logs/geom_$TAG.log
}
cq_run() {   # cq_run <acts_dir> <km npz> <km name> <out>. One invocation per dump so the
             # unseeded silhouette/dunn subsamples are shared by the arms compared.
             # RAM: 1M x 6144 x fp32 = 25 GB of latents per AE (freed between arms).
    $PY -u -m geoae.interp.clustering_quality --baseline "$2" \
        --checkpoints "$OLD" "$NEW" --names "$3" d6144_old d6144_new \
        --activations_dir "$1" --layer 27 --n_sample 1000000 --seed 0 \
        --out "$4" | tee "logs/$(basename "$4" .json).log"
}
cq_new() { cq_run $ACTS_NEW $KM_NEW km_new eval_out/cq_${TAG}_newdump.json; }
cq_old() { cq_run $ACTS_OLD $KM_OLD km_old eval_out/cq_${TAG}_olddump.json; }
steer() {
    $PY -u -m geoae.interp.steering_concept_compare --checkpoint "$NEW" --dataset db14 \
        --correct_json $JC_DB14 --out results/steer_db14_$TAG.json | tee logs/steer_db14_$TAG.log
}
range_db14() {
    $PY -u -m geoae.interp.range_intervention_compare --checkpoint "$NEW" --dataset db14 \
        --correct_json $JC_DB14 --saliency dprime --tao 2.0 \
        --out results/range_intervention_db14_${TAG}_dprime.json | tee logs/range_db14_${TAG}_dprime.log
}
range_bios() {
    $PY -u -m geoae.interp.range_intervention_compare --checkpoint "$NEW" --dataset biasbios \
        --concepts $BIOS_CONCEPTS --correct_json $JC_BIOS --saliency dprime --tao 2.0 --alphas 1.0 2.0 \
        --out results/range_intervention_biasbios_${TAG}_dprime.json \
        | tee logs/range_biasbios_${TAG}_dprime.log
}
number() {
    NUMBER_CHECKPOINT="$NEW" NUMBER_OUTPUT=results/range_number_${TAG}_dprime.json \
        bash eval_out/run_number_control.sh --saliency dprime | tee logs/range_number_${TAG}_dprime.log || return 1
    $PY eval_out/review_number_control.py --source results/range_number_${TAG}_dprime.json \
        --review_out eval_out/number_${TAG}_review.json --figure_prefix figures/number_control/${TAG}_review
}
old_ep50() { bash eval_out/run_d6144_ep50_evals.sh; }
summary() { $PY -u eval_out/summarize_data_ablation.py | tee logs/summary_data_ablation.log; }

done_range() { grep -qs '"summary"' "$1"; }   # partial range files exist mid-run

step km_fit     "[ -f $KM_NEW ]"                                                   km_fit
step mmlu       "[ -f eval_out/mmlu_$TAG.json ]"                                   mmlu
step probe      "[ -f eval_out/probe_$TAG.json ]"                                  probe
step geom       "[ -d figures/$TAG/tsne ]"                                         geom
step cq_new     "[ -f eval_out/cq_${TAG}_newdump.json ]"                           cq_new
step cq_old     "[ -f eval_out/cq_${TAG}_olddump.json ]"                           cq_old
step steer      "[ -f results/steer_db14_$TAG.json ]"                              steer
step range_db14 "done_range results/range_intervention_db14_${TAG}_dprime.json"    range_db14
step number     "grep -qs '\"status\": \"complete\"' results/range_number_${TAG}_dprime.json && [ -f eval_out/number_${TAG}_review.json ]" number
step range_bios "done_range results/range_intervention_biasbios_${TAG}_dprime.json" range_bios
step old_ep50   "done_range results/range_intervention_biasbios_d6144_50_dprime.json && done_range results/range_intervention_db14_d6144_50_dprime.json && [ -f results/steer_db14_d6144_50.json ]" old_ep50
step summary    "false"                                                            summary

echo
if [ ${#FAILED[@]} -eq 0 ]; then echo "[done] all requested evals complete ($(date +%H:%M))"; else echo "[done] FAILED: ${FAILED[*]}"; exit 1; fi
