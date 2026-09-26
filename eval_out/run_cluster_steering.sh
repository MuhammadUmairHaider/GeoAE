#!/usr/bin/env bash
# CLUSTER STEERING — are the UNSUPERVISED clusters control handles? (geoae/interp/cluster_steering.py)
#
# Move a held-out row from its own cluster to a random target cluster (translation by the
# centroid gap) and measure whether the LM's next-token behaviour moves toward the target
# cluster's signature. L27 is the last block, so no forward pass is needed — dump rows only.
#
#   d6144_new   parent AE                          (ae, and ae_h = its partition edited in base space)
#   tokbias     token-bypass AE                    (ae, ae_h)
#   km_new      balanced k-means on x              encoder-free
#   km_tokmean  balanced k-means on x - b[tok]     encoder-free, matched to tokbias
#   km_tok64    balanced k-means, 64 token dirs projected out
#
# Arms are compared at MATCHED edit size and at MATCHED disruption (KL from the unedited
# prediction), with 95% bootstrap CIs over source documents.
#
#   main      hub clusters (> 3x uniform usage) excluded as sources/targets   (~20 min)
#   allhubs   same with no hub exclusion — the bypass AE has more hub rows, so check that
#             excluding them is not what makes it look better                   (~20 min)
#
#   ./eval_out/run_cluster_steering.sh            # both
#   ./eval_out/run_cluster_steering.sh main
# ~12 GB of GPU; fine next to the bias_in_bios hb run. FORCE=1 redoes a step.
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE
PY=.venv/bin/python
E=e2e/checkpoints/general/llama3.2-3B/layer27
B=checkpoints/llama3.2-3B/layer27
ARMS="d6144_new=$B/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt"
ARMS+=",tokbias=$B/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt"
ARMS+=",km_new=$E/balanced_kmeans_k2000_dpc_sampled.npz"
ARMS+=",km_tokmean=$E/balanced_kmeans_k2000_dpc_sampled_tokmean.npz"
ARMS+=",km_tok64=$E/balanced_kmeans_k2000_dpc_sampled_tok64.npz"
STEPS=("$@"); [ $# -eq 0 ] && STEPS=(main allhubs)
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
have() { [ -z "$FORCE" ] && [ -f "$1" ] && echo "[skip] $1 exists" && return 0; return 1; }

if want main && ! have eval_out/cluster_steering.json; then
    $PY -u -m geoae.interp.cluster_steering --arms "$ARMS" \
        --out eval_out/cluster_steering.json 2>&1 | tee logs/cluster_steering.log || exit 1
fi
if want allhubs && ! have eval_out/cluster_steering_allhubs.json; then
    $PY -u -m geoae.interp.cluster_steering --arms "$ARMS" --hub_x 1e9 \
        --out eval_out/cluster_steering_allhubs.json 2>&1 | tee logs/cluster_steering_allhubs.log || exit 1
fi
