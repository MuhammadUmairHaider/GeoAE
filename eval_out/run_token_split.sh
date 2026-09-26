#!/usr/bin/env bash
# CURRENT-TOKEN vs PREDICTED-NEXT-TOKEN erasure at L27 (new sampled dump), no retraining.
#
# Decides whether design B (a current-token bypass in the AE) is worth training:
# B can bypass the current token safely, but not the predicted next token (its bypass
# term would be computed from the unedited activation and pull edits back). So the
# question is how much of the km_tok64 gain survives with the current token alone.
#
# Arms, all rank 64, all balanced k-means with the km_new settings, 2 seeds each (42, 7):
#   raw     no erasure                         (= km_new at seed 42)
#   cur64   64 current-token directions
#   pred64  64 predicted-next-token directions
#   both64  64 directions of the combined scatter (= km_tok64 at seed 42)
#   pca64   top-64 total-variance directions (All-but-the-Top control)
# plus the d6144_new AE for reference. Scored with chance-corrected NMI.
#
#   ./eval_out/run_token_split.sh           # ~30 min: 2 bases, 7 new fits, probe, summary
#   ./eval_out/run_token_split.sh summary   # just the tables
# Finished steps are skipped; FORCE=1 redoes them.
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE
PY=.venv/bin/python
E=e2e/checkpoints/general/llama3.2-3B/layer27
NEW=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt
ACTS=activations_sampled_10M/layer_27.npy
B_BOTH=$E/token_erasure_sampled.npz
B_CUR=$E/token_erasure_sampled_cur.npz
B_PRED=$E/token_erasure_sampled_pred.npz
KM=$E/balanced_kmeans_k2000_dpc_sampled
STEPS=("$@"); [ $# -eq 0 ] && STEPS=(bases fits probe summary)
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
have() { [ -z "$FORCE" ] && [ -f "$1" ] && echo "[skip] $1 exists" && return 0; return 1; }

if want bases; then
    [ -f $B_BOTH ] || { echo "[abort] missing $B_BOTH — run eval_out/run_token_erasure.sh basis"; exit 1; }
    for k in cur pred; do
        out=$E/token_erasure_sampled_$k.npz
        have $out || $PY -u -m geoae.interp.token_erasure --checkpoint $NEW \
            --activations_dir activations_sampled_10M --keys $k \
            --out $out | tee logs/token_erasure_sampled_$k.log || exit 1
    done
fi

fit() {   # fit <out> <seed> [<basis file> <basis> <rank>]
    have $1 && return
    local erase=(); [ $# -gt 2 ] && erase=(--erase $3 --erase_basis $4 --erase_rank $5)
    $PY -u -m geoae.interp.fit_balanced_kmeans --checkpoint $NEW --activations $ACTS \
        --init dpc --reinit peaks --n_clusters 2000 --seed $2 "${erase[@]}" \
        --out $1 | tee logs/fit_balanced_$(basename $1 .npz).log || exit 1
}
if want fits; then        # seed-42 raw / both64 / pca64 already exist from run_token_erasure.sh
    fit ${KM}_cur64.npz     42 $B_CUR  token 64
    fit ${KM}_pred64.npz    42 $B_PRED token 64
    fit ${KM}_s7.npz        7
    fit ${KM}_cur64_s7.npz  7  $B_CUR  token 64
    fit ${KM}_pred64_s7.npz 7  $B_PRED token 64
    fit ${KM}_tok64_s7.npz  7  $B_BOTH token 64
    fit ${KM}_pca64_s7.npz  7  $B_BOTH pca   64
fi

if want probe && ! have eval_out/probe_token_split.json; then
    $PY -u eval_out/probe_chance_corrected.py --models "d6144_new=$NEW" \
        --baselines "raw_s42=${KM}.npz,raw_s7=${KM}_s7.npz,cur64_s42=${KM}_cur64.npz,cur64_s7=${KM}_cur64_s7.npz,pred64_s42=${KM}_pred64.npz,pred64_s7=${KM}_pred64_s7.npz,both64_s42=${KM}_tok64.npz,both64_s7=${KM}_tok64_s7.npz,pca64_s42=${KM}_pca64.npz,pca64_s7=${KM}_pca64_s7.npz" \
        --out eval_out/probe_token_split.json | tee logs/probe_token_split.log || exit 1
fi

want summary && $PY eval_out/summarize_token_split.py | tee logs/summary_token_split.log
