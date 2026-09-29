#!/usr/bin/env bash
# ARM-INDEPENDENT PREP for the delta round-2 evals (see eval_out/run_delta_evals.sh).
#
# Builds the shared inputs every arm's eval sheet reads: the K-matched balanced
# k-means codebooks (KM/KMT), the RAVEL/IOI token-level caches, and cache_ids/
# (the sequence-rung concept caches WITH last_token_id, needed by the token-bypass
# AEs). All four steps are ARM-INDEPENDENT and read-only from every arm's point of
# view: they write only into e2e/checkpoints/.../delta/, cache/ and cache_ids/,
# never into eval_out/delta/<arm>/, results/delta/<arm>/ or any old fixed-name file.
#
#   eval_out/run_delta_prep.sh km bench cache atlas   # everything
#   eval_out/run_delta_prep.sh bench cache            # what run_delta_evals.sh needs
#   FORCE=1 eval_out/run_delta_prep.sh cache           # redo one step
#   DRY=1 eval_out/run_delta_prep.sh km bench cache atlas   # print, don't run
#
# ACT_DIR: `km` reads activations_sampled_10M/layer_27.npy (61 GB, symlink into
# /work/hdd Lustre) with scattered random-row access, which is very slow over the
# network filesystem. scripts/delta/sheet_staged.sbatch rsyncs the whole dump
# directory to node-local /tmp/$SLURM_JOB_ID/activations_sampled_10M and exports
# ACT_DIR to that path; `km` resolves its --activations from ACT_DIR, falling back
# to the Lustre path when unset. bench/cache never read the dump (live LM encoding
# of small datasets), so ACT_DIR does not affect them.
#
# Steps
#   km      the 4 balanced-k-means fits (K=2000, K=4000 x plain/tokmean). These are
#           ALSO being run directly as their own Slurm jobs right now (see the
#           handoff) — this step only skips them when their .npz already exists;
#           it does not compete with those jobs for a GPU unless FORCE=1 and the
#           output is genuinely missing.
#   bench   RAVEL + IOI token-level caches -> cache/ravel.npz, cache/ioi.npz
#   cache   cache_ids/: symlink the token rungs from cache/, then build the 9
#           sequence rungs WITH last_token_id via concept_suite
#   atlas   INVESTIGATION ONLY (see docstring below) — prints findings, does not
#           guess-build cache/atlas8k_last.npz
#
# Finished steps are skipped (FORCE=1 redoes one); a failing step is reported at
# the end as FAILED with a non-zero exit.
set -o pipefail
# Delta: env.sh resolves GEOAE_ROOT from its own location, exports HF_HOME /
# UV_PROJECT_ENVIRONMENT / SBATCH_ACCOUNT, and cds to the repo root.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/delta/env.sh"
PY=scripts/delta/py
E=e2e/checkpoints/general/llama3.2-3B/layer27
B=checkpoints/llama3.2-3B/layer27
DELTA=$E/delta
# Reference checkpoint: only its model_name + target_layer are read by
# fit_balanced_kmeans / benchmark_cache / concept_suite. d6144 tokbias is the one
# guaranteed to exist on this machine (d12288 and d6144k4000 are still training).
REFCK=$B/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
# Set by scripts/delta/sheet_staged.sbatch to the node-local rsync'd copy of the dump;
# unset (falls back to the Lustre symlink) on the login node / under plain sheet.sbatch.
# Only `km` reads the dump (fit_balanced_kmeans --activations); bench/cache do live LM
# encoding of small datasets and never touch it.
ACT="${ACT_DIR:-activations_sampled_10M}"
TABLE=$E/token_bias_sampled.npz
SEQ=sentiment,sentiment_long,subjectivity,language,topic4,topic14,topic20,formality,domain

ATOMIC=(km bench cache atlas)
STEPS=("$@"); [ $# -eq 0 ] && { echo "usage: $0 km bench cache atlas"; exit 2; }
for s in "${STEPS[@]}"; do
    ok=0; for a in "${ATOMIC[@]}"; do [ "$a" = "$s" ] && ok=1 && break; done
    [ "$ok" -eq 1 ] || { echo "[abort] unknown step '$s' (have: ${ATOMIC[*]})" >&2; exit 2; }
done
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

FAILED=()
run_or_dry() {
    if [ -n "$DRY" ]; then
        echo "[dry ] $*"
    else
        "$@"
    fi
}
step() {   # step <name> <done-check, bash -c'd> <function>
    want "$1" || return
    if [ -z "$FORCE" ] && eval "$2"; then echo "[skip] $1 — already done (FORCE=1 to redo)"; return; fi
    echo -e "\n[run ] $1  ($(date +%H:%M))"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

# --------------------------------------------------------------------------------- #
# km — the 4 balanced-k-means fits. Exact command per the handoff; --token_bias only
# on the *_tokmean variant (that IS the token-mean subtraction).
# --------------------------------------------------------------------------------- #
km_fit() {   # km_fit <K> <out.npz> [--token_bias TABLE]
    local K="$1" OUT="$2"; shift 2
    mkdir -p "$(dirname "$OUT")"
    run_or_dry "$PY" -u -m geoae.interp.fit_balanced_kmeans \
        --checkpoint "$REFCK" --activations "$ACT/layer_27.npy" \
        --init dpc --reinit peaks --n_clusters "$K" --seed 42 "$@" --out "$OUT"
}
km() {
    local rc=0
    km_fit 2000 "$DELTA/balanced_kmeans_k2000_dpc_sampled.npz" || rc=1
    km_fit 2000 "$DELTA/balanced_kmeans_k2000_dpc_sampled_tokmean.npz" --token_bias "$TABLE" || rc=1
    km_fit 4000 "$DELTA/balanced_kmeans_k4000_dpc_sampled.npz" || rc=1
    km_fit 4000 "$DELTA/balanced_kmeans_k4000_dpc_sampled_tokmean.npz" --token_bias "$TABLE" || rc=1
    return $rc
}
km_done() {
    [ -f "$DELTA/balanced_kmeans_k2000_dpc_sampled.npz" ] &&
    [ -f "$DELTA/balanced_kmeans_k2000_dpc_sampled_tokmean.npz" ] &&
    [ -f "$DELTA/balanced_kmeans_k4000_dpc_sampled.npz" ] &&
    [ -f "$DELTA/balanced_kmeans_k4000_dpc_sampled_tokmean.npz" ]
}

# --------------------------------------------------------------------------------- #
# bench — RAVEL + IOI token caches (geoae.interp.benchmark_cache). --out is a STEM;
# the tool appends .npz itself, so --out cache/ravel writes cache/ravel.npz.
# --------------------------------------------------------------------------------- #
bench() {
    local rc=0
    run_or_dry "$PY" -u -m geoae.interp.benchmark_cache --bench ravel \
        --checkpoint "$REFCK" --n_rows 6000 --out cache/ravel \
        2>&1 | tee logs/bench_ravel_delta.log || rc=1
    run_or_dry "$PY" -u -m geoae.interp.benchmark_cache --bench ioi \
        --checkpoint "$REFCK" --n_rows 6000 --out cache/ioi \
        2>&1 | tee logs/bench_ioi_delta.log || rc=1
    return $rc
}

# --------------------------------------------------------------------------------- #
# cache — cache_ids/: symlink the token rungs (already carry token_id), then build
# the 9 sequence rungs WITH last_token_id.
# --------------------------------------------------------------------------------- #
cache() {
    mkdir -p cache_ids
    for f in pos.npz ner.npz ner_coarse.npy ioi.npz ravel.npz; do
        if [ -n "$DRY" ]; then
            echo "[dry ] ln -s ../cache/$f cache_ids/$f  (skipped if already present)"
        else
            [ -e "cache_ids/$f" ] || ln -s "../cache/$f" "cache_ids/$f"
        fi
    done
    run_or_dry "$PY" -u -m geoae.interp.concept_suite --checkpoint "$REFCK" \
        --out cache_ids --only "$SEQ" 2>&1 | tee logs/concept_suite_cache_ids_delta.log
}
cache_done() {
    [ -n "$DRY" ] && return 1
    [ -f cache_ids/pos.npz ] && [ -f cache_ids/ner.npz ] && [ -f cache_ids/ner_coarse.npy ] \
        && [ -f cache_ids/ioi.npz ] && [ -f cache_ids/ravel.npz ] || return 1
    "$PY" - <<'EOF'
import sys
import numpy as np
rungs = ["sentiment", "sentiment_long", "subjectivity", "language",
         "topic4", "topic14", "topic20", "formality", "domain"]
ok = True
for r in rungs:
    p = f"cache_ids/{r}.npz"
    try:
        d = np.load(p, allow_pickle=True)
    except FileNotFoundError:
        ok = False
        break
    if "last_token_id" not in d.files:
        ok = False
        break
sys.exit(0 if ok else 1)
EOF
}

# --------------------------------------------------------------------------------- #
# atlas — INVESTIGATION, not a build step. The probe (eval_out/probe_chance_corrected.py)
# defaults --atlas_last to cache/atlas8k_last.npz, which does not exist on this
# machine. Reading geoae/seeded_init.py:
#
#   anchor_row_indices(..., atlas_last=<non-empty path>) UNCONDITIONALLY calls
#   load_atlas_anchors(path), whose first line is
#       d = np.load(str(path), allow_pickle=True)
#   with no existence check. probe_chance_corrected.main() calls
#   anchor_row_indices(...) once, near the top, BEFORE the per-rung loop, using
#   whatever --atlas_last resolves to (default: non-empty). So on this machine the
#   probe step CRASHES with FileNotFoundError before scoring a single rung —
#   it does not silently skip the atlas rung, and it does not need the file only
#   for atlas rungs specifically: the crash happens regardless of which rungs are
#   requested via --rungs, because the anchor-exclusion call is unconditional.
#
# Whether an eval NEEDS atlas8k_last.npz for real content: NO, for the tokbias
# family. cache_ids/ is built by concept_suite --only "$SEQ" (see the `cache` step
# above), which does not include atlas_doc / atlas_tone / atlas_content — those
# come from a SEPARATE build (geoae/interp/atlas_cache.py -> cache/atlas8k.npz +
# atlas8k_labels.parquet, chunk-level; then some further last-token-pooling step
# to reach cache/atlas8k_last.npz's expected schema H_last/document_ids/tone_ids/
# content_ids) that this repo does not contain a verified producer for —
# HANDOFF_DELTA.md says as much ("not re-verified in this migration — check
# --help before relying on it"). The committed old-box eval_out/probe_tokbias.json
# (git-tracked, read directly rather than guessed at) has NO atlas_* keys, so the
# atlas rung was never part of the tokbias probe even on the machine that
# produced it — cache/atlas8k_last.npz existed there only to satisfy the
# anchor-exclusion default, not to add scored content.
#
# CONCLUSION, reported rather than silently worked around:
#   - This prep script does NOT attempt to rebuild cache/atlas8k_last.npz — the
#     conversion step from atlas_cache.py's chunk-level output to its expected
#     H_last/document_ids/tone_ids/content_ids schema is not verified anywhere in
#     this repo, and guessing at it risks writing a poisoned anchor cache.
#   - eval_out/run_delta_evals.sh's `probe` step instead passes --atlas_last ""
#     to probe_chance_corrected.py, which short-circuits the crashing branch in
#     anchor_row_indices (the `if atlas_last:` guard) and skips the atlas-anchor
#     exclusion entirely. Since cache_ids/ never carries an atlas_* rung for this
#     arm family, no rung's held-out rows are affected by omitting it.
# --------------------------------------------------------------------------------- #
atlas() {
    echo "[atlas] cache/atlas8k_last.npz: $([ -f cache/atlas8k_last.npz ] && echo present || echo ABSENT)"
    echo "[atlas] probe_chance_corrected.py --atlas_last defaults to cache/atlas8k_last.npz and"
    echo "[atlas] geoae/seeded_init.py:load_atlas_anchors() np.load()s it with NO existence check,"
    echo "[atlas] called UNCONDITIONALLY by anchor_row_indices() before any rung is scored — so the"
    echo "[atlas] probe step CRASHES (FileNotFoundError), it does not silently skip the atlas rung."
    echo "[atlas] The committed eval_out/probe_tokbias.json (old box) has NO atlas_* rungs, so the"
    echo "[atlas] tokbias probe never scored atlas content even where the file existed — it was only"
    echo "[atlas] needed there to satisfy this unconditional call."
    echo "[atlas] This repo has no verified producer for atlas8k_last.npz's H_last/document_ids/"
    echo "[atlas] tone_ids/content_ids schema from atlas_cache.py's chunk-level output (see"
    echo "[atlas] HANDOFF_DELTA.md: 'not re-verified in this migration'). NOT GUESSING a rebuild."
    echo "[atlas] Fix applied in eval_out/run_delta_evals.sh's probe step: --atlas_last \"\" (disables"
    echo "[atlas] the crashing anchor-exclusion branch; cache_ids/ never has an atlas_* rung anyway,"
    echo "[atlas] so no scored rung's held-out rows are affected)."
    return 0
}

# --------------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------------- #
step "km"    "km_done"     km
step "bench" "[ -f cache/ravel.npz ] && [ -f cache/ioi.npz ]" bench
step "cache" "cache_done"  cache
step "atlas" "false"       atlas

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "\n[done] FAILED steps: ${FAILED[*]}"
    exit 1
fi
echo -e "\n[done] $(date +%H:%M)"
