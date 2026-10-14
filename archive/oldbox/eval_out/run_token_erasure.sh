#!/usr/bin/env bash
# Option 2 — TOKEN-ERASED balanced k-means at L27 (new sampled dump), no retraining.
#
# Question: if the directions carrying current-token and predicted-next-token
# identity are projected out before clustering, do the sequence rungs (topic,
# domain, atlas, sentiment) improve, and at what cost to the token rungs?
#
#   km_new     balanced k-means, dpc init, raw normalised space (exists)
#   km_tok64   same, 64 between-token directions erased
#   km_tok256  same, 256 erased
#   km_pca64   same, top-64 TOTAL-variance directions erased: the control. Removing
#              dominant directions changes k-means whether or not they carry tokens,
#              so km_tok64 has to beat km_pca64, not just km_new.
#
# Basis (already fit, logs/token_erasure_sampled.log): on held-out rows, erasing 64
# token directions cuts current-token R^2 0.162 -> 0.062 and predicted-next 0.152 ->
# 0.051, keeps 65% of the between-document variance (pca64: 52%).
#
#   ./eval_out/run_token_erasure.sh            # ~50 min: 3 fits (~15 min each) + probe (~6 min)
#   ./eval_out/run_token_erasure.sh probe      # only the probe
# Finished steps are skipped; FORCE=1 redoes them.
set -o pipefail
# Delta: env.sh resolves GEOAE_ROOT from its own location, exports HF_HOME /
# UV_PROJECT_ENVIRONMENT / SBATCH_ACCOUNT, and cds to the repo root.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/delta/env.sh"
PY=scripts/delta/py
E=e2e/checkpoints/general/llama3.2-3B/layer27
NEW=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt
ACTS=activations_sampled_10M/layer_27.npy
ERASE=$E/token_erasure_sampled.npz
KM_NEW=$E/balanced_kmeans_k2000_dpc_sampled.npz
STEPS=("$@"); [ $# -eq 0 ] && STEPS=(basis tok64 tok256 pca64 probe)
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
done_or_force() { [ -z "$FORCE" ] && [ -f "$1" ] && echo "[skip] $1 exists" && return 0; return 1; }

if want basis && ! done_or_force $ERASE; then
    $PY -u -m geoae.interp.token_erasure --checkpoint $NEW --activations_dir activations_sampled_10M \
        --out $ERASE | tee logs/token_erasure_sampled.log || exit 1
fi

fit() {   # fit <name> <basis> <rank>. Same settings as km_new (logs/fit_balanced_dpc_k2000_sampled.log).
    local out=$E/balanced_kmeans_k2000_dpc_sampled_$1.npz
    done_or_force $out && return
    $PY -u -m geoae.interp.fit_balanced_kmeans --checkpoint $NEW --activations $ACTS \
        --init dpc --reinit peaks --n_clusters 2000 --seed 42 \
        --erase $ERASE --erase_basis $2 --erase_rank $3 \
        --out $out | tee logs/fit_balanced_dpc_k2000_sampled_$1.log
}
want tok64  && fit tok64  token 64
want tok256 && fit tok256 token 256
want pca64  && fit pca64  pca   64

if want probe; then   # one run, same anchor holdout as eval_out/probe_d6144_sampled.json
    $PY -u -m geoae.interp.concept_probe --cache cache \
        --models "d6144_new=$NEW" \
        --baselines "km_new=$KM_NEW,km_tok64=$E/balanced_kmeans_k2000_dpc_sampled_tok64.npz,km_tok256=$E/balanced_kmeans_k2000_dpc_sampled_tok256.npz,km_pca64=$E/balanced_kmeans_k2000_dpc_sampled_pca64.npz" \
        --exclude_anchors --anchor_seed 42 --anchor_per_class 25 --anchor_min_examples 5 \
        --atlas_last cache/atlas8k_last.npz --atlas_min_examples 25 \
        --out eval_out/probe_token_erasure.json | tee logs/probe_token_erasure.log
fi
