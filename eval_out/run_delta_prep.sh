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
#   probe_acts  last-token layer-27 activations + current-token ids + source-text hashes for the
#           `probes` group of run_delta_evals.sh (GPU; live Llama-3.2-3B forward, one document at a
#           time, max_length 256 — the settings of the existing dbpedia extraction):
#             biasbios/activations/llama3.2-3B/last/            50k train / 20k test bias_in_bios
#             dbpedia/activations/llama3.2-3B/last_ids/unprompted/   50k train / 10k test DBpedia-14
#           each with layer_27{,_test}.npy, labels, meta.json, tokens_{train,test}.npy (int64, the
#           pooled position's token id) and source_hashes.npz (keys train/test = SHA256 of the text).
#           The existing dbpedia/.../last/unprompted is NOT touched (it has no token ids). Not in any
#           default set; submit via `scripts/delta/submit_delta_evals.sh probe_acts` (~1-1.5 h, 1 GPU).
#   g1b_km  Gemma 3 1B: the balanced-k-means fits for layers 13 and 25 (K=2000 x plain/tokmean), via
#           fit_balanced_kmeans --norm_params (no AE checkpoint needed) -> e2e/checkpoints/general/
#           gemma3-1B/layer{13,25}/delta/balanced_kmeans_k2000_dpc_sampled_struct5{,_tokmean}.npz.
#           Reads the Gemma dump (ACT_DIR, else activations_gemma3_1b_sampled_struct5_10M).
#   g1b_cache  Gemma 3 1B probe caches for EACH layer in ONE dir cache/gemma3-1b/l{13,25}/: token rungs
#           (pos, ner, ner_coarse) + the 9 sequence rungs WITH last_token_id (concept_suite --model_name
#           --layer) + RAVEL and IOI (benchmark_cache). That dir plays the role of BOTH cache/ and
#           cache_ids/ for the g1b arms (run_delta_evals.sh: CACHE_IDS).
#   g1b_probe_acts  the `probe_acts` extraction with google/gemma-3-1b-pt, once per layer, into
#           {biasbios,dbpedia}/activations/g1b_l<L>/gemma-3-1b-pt/... (one dir per layer because the
#           probe tools require meta.json layer == --layer). Not in any default set.
#   g4b_km / g4b_cache / g4b_probe_acts   the same three steps for Gemma 3 4B (google/gemma-3-4b-pt, hidden 2560), layer 33 only
#           (the LAST of 34 blocks): e2e/checkpoints/general/gemma3-4B/layer33/delta/..., cache/gemma3-4b/l33/,
#           {biasbios,dbpedia}/activations/g4b_l33/gemma-3-4b-pt/... Dump: activations_gemma3_4b_sampled_struct5_10M.
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

ATOMIC=(km bench cache atlas probe_acts g1b_km g1b_cache g1b_probe_acts g4b_km g4b_cache g4b_probe_acts)
STEPS=("$@"); [ $# -eq 0 ] && { echo "usage: $0 km bench cache atlas probe_acts g1b_km g1b_cache g1b_probe_acts g4b_km g4b_cache g4b_probe_acts"; exit 2; }
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
# probe_acts — extraction for the probes group. Both extractors also save the pooled
# token id and the source-text hashes; --dir_suffix _ids keeps the new DBpedia tree next
# to (not over) the old token-id-less one. biasbios/activations goes on the data root
# (/work/hdd) like dbpedia/activations does, via a symlink made here if it is absent.
# --------------------------------------------------------------------------------- #
PA_BB=biasbios/activations/llama3.2-3B/last
PA_DB=dbpedia/activations/llama3.2-3B/last_ids/unprompted
pa_ok() {   # pa_ok <dir> <n_train> <n_test> [layer=27]: complete only if every file exists and row counts match
    local d="$1" f L="${4:-27}"
    for f in meta.json source_hashes.npz tokens_train.npy tokens_test.npy layer_${L}.npy layer_${L}_test.npy \
             labels_train.npy labels_test.npy; do
        [ -f "$d/$f" ] || return 1
    done
    "$PY" - "$d" "$2" "$3" "$L" <<'EOF'
import sys
import numpy as np
d, ntr, nte, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ok = True
for split, n in (("train", ntr), ("test", nte)):
    x = np.load(f"{d}/layer_{L}{'' if split == 'train' else '_test'}.npy", mmap_mode="r")
    t = np.load(f"{d}/tokens_{split}.npy", mmap_mode="r")
    h = np.load(f"{d}/source_hashes.npz")[split]
    ok &= (len(x) == len(t) == len(h) == n) and t.dtype == np.int64
sys.exit(0 if ok else 1)
EOF
}
probe_acts_done() { [ -n "$DRY" ] && return 1; pa_ok "$PA_BB" 50000 20000 && pa_ok "$PA_DB" 50000 10000; }
probe_acts() {
    local rc=0
    if [ -z "$DRY" ] && [ ! -e biasbios/activations ]; then
        mkdir -p biasbios "$DELTA_DATA/biasbios/activations"
        ln -s "$DELTA_DATA/biasbios/activations" biasbios/activations
    fi
    if [ -n "$DRY" ] || [ -n "$FORCE" ] || ! pa_ok "$PA_BB" 50000 20000; then
        run_or_dry "$PY" -u -m geoae.bias.extract --layer 27 --pooling last --max_length 256 \
            --n_train 50000 --n_test 20000 --out_base biasbios/activations \
            2>&1 | tee logs/probe_acts_biasbios_delta.log || rc=1
    else echo "[skip] $PA_BB complete"; fi
    if [ -n "$DRY" ] || [ -n "$FORCE" ] || ! pa_ok "$PA_DB" 50000 10000; then
        run_or_dry "$PY" -u -m geoae.dbpedia.extract --mode unprompted --pooling last --layer 27 \
            --max_length 256 --n_train 50000 --n_test 10000 --dir_suffix _ids --out_base dbpedia/activations \
            2>&1 | tee logs/probe_acts_dbpedia_delta.log || rc=1
    else echo "[skip] $PA_DB complete"; fi
    return $rc
}

# --------------------------------------------------------------------------------- #
# Gemma arms. One generic implementation (gem_*), parameterised by a small per-family table (gfam); the
# g1b_* / g4b_* step names are thin wrappers that select the family first.
#   g1b: google/gemma-3-1b-pt (26 blocks, hidden 1152), layers 13 (middle) and 25 (LAST block)
#   g4b: google/gemma-3-4b-pt (34 blocks, hidden 2560), layer 33 (LAST block) only
# Own dump variable G_ACT, separate from the Llama ACT (ACT_DIR, when set by the staged sheet, wins).
# --------------------------------------------------------------------------------- #
gfam() {
    case "$1" in
        g1b) G_TAG=g1b; G_DUMP=activations_gemma3_1b_sampled_struct5_10M; G_MODEL=google/gemma-3-1b-pt
             G_LAYERS=(13 25); G_E2E=gemma3-1B; G_CACHE=cache/gemma3-1b; G_PA=gemma-3-1b-pt ;;
        g4b) G_TAG=g4b; G_DUMP=activations_gemma3_4b_sampled_struct5_10M; G_MODEL=google/gemma-3-4b-pt
             G_LAYERS=(33); G_E2E=gemma3-4B; G_CACHE=cache/gemma3-4b; G_PA=gemma-3-4b-pt ;;
        *) echo "[abort] unknown Gemma family '$1'" >&2; return 2 ;;
    esac
    G_ACT="${ACT_DIR:-$G_DUMP}"
}
g_delta() { echo "e2e/checkpoints/general/$G_E2E/layer$1/delta"; }
gem_km_fit() {   # gem_km_fit <layer> <out.npz> [--token_bias TABLE]
    local L="$1" OUT="$2"; shift 2
    mkdir -p "$(dirname "$OUT")"
    run_or_dry "$PY" -u -m geoae.interp.fit_balanced_kmeans \
        --norm_params "$G_ACT/norm_params_layer$L.npz" --model_name "$G_MODEL" --layer "$L" \
        --activations "$G_ACT/layer_$L.npy" \
        --init dpc --reinit peaks --n_clusters 2000 --seed 42 "$@" --out "$OUT"
}
gem_km() {
    local rc=0 L D T
    for L in "${G_LAYERS[@]}"; do
        D=$(g_delta "$L"); T="e2e/checkpoints/general/$G_E2E/layer$L/token_bias_sampled_struct5.npz"
        [ -n "$DRY" ] || [ -f "$T" ] || { echo "[abort] $T missing"; rc=1; continue; }
        [ -f "$D/balanced_kmeans_k2000_dpc_sampled_struct5.npz" ] && [ -z "$FORCE" ] && [ -z "$DRY" ] \
            && echo "[skip] L$L plain fit exists" \
            || gem_km_fit "$L" "$D/balanced_kmeans_k2000_dpc_sampled_struct5.npz" || rc=1
        [ -f "$D/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz" ] && [ -z "$FORCE" ] && [ -z "$DRY" ] \
            && echo "[skip] L$L tokmean fit exists" \
            || gem_km_fit "$L" "$D/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz" --token_bias "$T" || rc=1
    done
    return $rc
}
gem_km_done() {
    local L D
    for L in "${G_LAYERS[@]}"; do
        D=$(g_delta "$L")
        [ -f "$D/balanced_kmeans_k2000_dpc_sampled_struct5.npz" ] &&
        [ -f "$D/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz" ] || return 1
    done
}

# gem_cache: one dir per layer holding everything a probe/geometry sheet reads (see header).
gem_cache() {
    local rc=0 L C
    for L in "${G_LAYERS[@]}"; do
        C="$G_CACHE/l$L"
        [ -n "$DRY" ] || mkdir -p "$C" logs
        # concept_suite skips a rung whose .npz already exists, so this is resumable.
        run_or_dry "$PY" -u -m geoae.interp.concept_suite --model_name "$G_MODEL" --layer "$L" \
            --out "$C" --only "pos,ner,$SEQ" 2>&1 | tee "logs/concept_suite_${G_TAG}_l$L.log" || rc=1
        local b
        for b in ravel ioi; do
            if [ -n "$DRY" ] || [ -n "$FORCE" ] || [ ! -f "$C/$b.npz" ]; then
                run_or_dry "$PY" -u -m geoae.interp.benchmark_cache --bench "$b" \
                    --model_name "$G_MODEL" --layer "$L" --n_rows 6000 --out "$C/$b" \
                    2>&1 | tee "logs/bench_${b}_${G_TAG}_l$L.log" || rc=1
            else echo "[skip] $C/$b.npz exists"; fi
        done
    done
    return $rc
}
gem_cache_done() {
    [ -n "$DRY" ] && return 1
    local L C
    for L in "${G_LAYERS[@]}"; do
        C="$G_CACHE/l$L"
        [ -f "$C/pos.npz" ] && [ -f "$C/ner.npz" ] && [ -f "$C/ner_coarse.npy" ] \
            && [ -f "$C/ioi.npz" ] && [ -f "$C/ravel.npz" ] || return 1
        "$PY" - "$C" <<'EOF' || return 1
import sys
import numpy as np
c = sys.argv[1]
rungs = ["sentiment", "sentiment_long", "subjectivity", "language",
         "topic4", "topic14", "topic20", "formality", "domain"]
try:
    ok = all("last_token_id" in np.load(f"{c}/{r}.npz", allow_pickle=True).files for r in rungs)
except FileNotFoundError:
    ok = False
sys.exit(0 if ok else 1)
EOF
    done
}

# gem_probe_acts: same extractors as probe_acts, Gemma model, one output tree per layer.
gem_pa_bb() { echo "biasbios/activations/${G_TAG}_l$1/$G_PA/last"; }
gem_pa_db() { echo "dbpedia/activations/${G_TAG}_l$1/$G_PA/last_ids/unprompted"; }
gem_probe_acts_done() {
    [ -n "$DRY" ] && return 1
    local L
    for L in "${G_LAYERS[@]}"; do
        pa_ok "$(gem_pa_bb "$L")" 50000 20000 "$L" && pa_ok "$(gem_pa_db "$L")" 50000 10000 "$L" || return 1
    done
}
gem_probe_acts() {
    local rc=0 L
    if [ -z "$DRY" ] && [ ! -e biasbios/activations ]; then
        mkdir -p biasbios "$DELTA_DATA/biasbios/activations"
        ln -s "$DELTA_DATA/biasbios/activations" biasbios/activations
    fi
    for L in "${G_LAYERS[@]}"; do
        if [ -n "$DRY" ] || [ -n "$FORCE" ] || ! pa_ok "$(gem_pa_bb "$L")" 50000 20000 "$L"; then
            run_or_dry "$PY" -u -m geoae.bias.extract --layer "$L" --model_name "$G_MODEL" --pooling last \
                --max_length 256 --n_train 50000 --n_test 20000 --out_base "biasbios/activations/${G_TAG}_l$L" \
                2>&1 | tee "logs/probe_acts_biasbios_${G_TAG}_l$L.log" || rc=1
        else echo "[skip] $(gem_pa_bb "$L") complete"; fi
        if [ -n "$DRY" ] || [ -n "$FORCE" ] || ! pa_ok "$(gem_pa_db "$L")" 50000 10000 "$L"; then
            run_or_dry "$PY" -u -m geoae.dbpedia.extract --mode unprompted --pooling last --layer "$L" \
                --model_name "$G_MODEL" --max_length 256 --n_train 50000 --n_test 10000 --dir_suffix _ids \
                --out_base "dbpedia/activations/${G_TAG}_l$L" \
                2>&1 | tee "logs/probe_acts_dbpedia_${G_TAG}_l$L.log" || rc=1
        else echo "[skip] $(gem_pa_db "$L") complete"; fi
    done
    return $rc
}


# Step wrappers: select the family, then run the generic implementation.
g1b_km() { gfam g1b; gem_km; }
g1b_cache() { gfam g1b; gem_cache; }
g1b_probe_acts() { gfam g1b; gem_probe_acts; }
g1b_km_done() { gfam g1b; gem_km_done; }
g1b_cache_done() { gfam g1b; gem_cache_done; }
g1b_probe_acts_done() { gfam g1b; gem_probe_acts_done; }
g4b_km() { gfam g4b; gem_km; }
g4b_cache() { gfam g4b; gem_cache; }
g4b_probe_acts() { gfam g4b; gem_probe_acts; }
g4b_km_done() { gfam g4b; gem_km_done; }
g4b_cache_done() { gfam g4b; gem_cache_done; }
g4b_probe_acts_done() { gfam g4b; gem_probe_acts_done; }

# --------------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------------- #
step "km"    "km_done"     km
step "bench" "[ -f cache/ravel.npz ] && [ -f cache/ioi.npz ]" bench
step "cache" "cache_done"  cache
step "atlas" "false"       atlas
step "probe_acts" "probe_acts_done" probe_acts
step "g1b_km"    "g1b_km_done"    g1b_km
step "g1b_cache" "g1b_cache_done" g1b_cache
step "g1b_probe_acts" "g1b_probe_acts_done" g1b_probe_acts
step "g4b_km"    "g4b_km_done"    g4b_km
step "g4b_cache" "g4b_cache_done" g4b_cache
step "g4b_probe_acts" "g4b_probe_acts_done" g4b_probe_acts

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "\n[done] FAILED steps: ${FAILED[*]}"
    exit 1
fi
echo -e "\n[done] $(date +%H:%M)"
