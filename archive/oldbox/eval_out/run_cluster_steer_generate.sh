#!/usr/bin/env bash
# GENERATION-LEVEL CLUSTER STEERING — base vs bypass AE.
#   base    balanced k-means on the plain residual (K=2000, dpc init, no encoder)
#   bypass  the token-bypass AE's clusters
# At every generated position: h <- h + alpha * (mean of target cluster's members - mean of
# the position's own cluster's members). Same form for both, so only the partition differs.
# 40 random content clusters per codebook x 20 neutral prompts x alpha {0, .1, .2, .3, .5, .75, 1},
# 32 greedy tokens. Scored by (a) automatic forced choice vs 3 other clusters (uses the
# clusters' next-token profiles, so partly circular), (b) an LLM judge doing the same forced
# choice from short cluster profiles — the independent check — both at matched fluency cost.
#
#   ./eval_out/run_cluster_steer_generate.sh            # gen (~20 min GPU) then judge (~10 min, API)
#   for g in pos dim pos+dim; do GATE=$g ./eval_out/run_cluster_steer_generate.sh; done   # range-gated arms
#   ./eval_out/run_cluster_steer_generate.sh gen
#   ./eval_out/run_cluster_steer_generate.sh judge      # needs OPENROUTER_API_KEY (read from .env)
# The judge makes ~9,600 calls to google/gemini-2.5-flash-lite (~6M input tokens, well under $1);
# answers are disk-cached, so re-running is free. FORCE=1 redoes a step.
set -o pipefail
# Delta: env.sh resolves GEOAE_ROOT from its own location, exports HF_HOME /
# UV_PROJECT_ENVIRONMENT / SBATCH_ACCOUNT, and cds to the repo root.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/delta/env.sh"
PY=scripts/delta/py
BASE=e2e/checkpoints/general/llama3.2-3B/layer27/balanced_kmeans_k2000_dpc_sampled.npz
BYPASS=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
# v2 (default): whole-word targets from >= 20 documents. V=v1 reproduces the first run's
# target rule (any alphabetic token), whose files are kept for comparison.
V=${V:-v2}
if [ "$V" = v1 ]; then
    GEN=eval_out/cluster_steer_generate.json; JUDGE=eval_out/cluster_steer_judge.json
    GEN_ARGS=(--word_filter alpha --min_docs 1)
else
    GEN=eval_out/cluster_steer_generate_$V.json; JUDGE=eval_out/cluster_steer_judge_$V.json
    GEN_ARGS=()
fi
# GATE=pos|dim|pos+dim: NeuronLens-style range gating (tao 2.0, top-30% d' coordinates per cluster).
# Gated edits are smaller, so their alpha grid runs further (to 3.0) and the judge scores all of it.
GATE=${GATE:-none}
JUDGE_ARGS=()
if [ "$GATE" != none ]; then
    T=${GATE/+/}
    GEN=eval_out/cluster_steer_generate_${V}_gate_$T.json; JUDGE=eval_out/cluster_steer_judge_${V}_gate_$T.json
    GEN_ARGS+=(--gate "$GATE" --tao 2.0 --percent 0.3 --alphas 0.1 0.2 0.3 0.5 0.75 1.0 1.5 2.0 3.0)
    JUDGE_ARGS=(--max_alpha 3.0)
    V=${V}_gate_$T
fi
STEPS=("$@"); [ $# -eq 0 ] && STEPS=(gen judge)
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
have() { [ -z "$FORCE" ] && [ -f "$1" ] && echo "[skip] $1 exists" && return 0; return 1; }

if want gen && ! have $GEN; then
    $PY -u -m geoae.interp.cluster_steer_generate --base $BASE --bypass $BYPASS "${GEN_ARGS[@]}" \
        --out $GEN 2>&1 | tee logs/cluster_steer_generate_$V.log || exit 1
fi
if want judge && ! have $JUDGE; then
    [ -f $GEN ] || { echo "[abort] run the gen step first"; exit 1; }
    set -a; [ -f .env ] && source .env; set +a
    [ -n "$OPENROUTER_API_KEY" ] || { echo "[abort] OPENROUTER_API_KEY not set (.env)"; exit 1; }
    $PY -u -m geoae.interp.cluster_steer_judge --gen $GEN --provider openrouter \
        --model google/gemini-2.5-flash-lite "${JUDGE_ARGS[@]}" --out $JUDGE 2>&1 | tee logs/cluster_steer_judge_$V.log || exit 1
fi
