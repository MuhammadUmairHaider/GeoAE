#!/usr/bin/env bash
# PER-ARM evals for the delta round-2 token-bypass AEs. Modeled closely on
# eval_out/run_tokbias_d12288.sh, generalised over 5 arms and rewritten so that
# EVERY output lands under the single browsable tree results/delta/<arm>/{evals,
# interventions,steering,concept,figures,logs}/ or results/delta/joint_correct/ —
# never at a fixed eval_out/*.json / results/*.json path, which on this machine
# already holds STALE numbers copied over from the old box (see the handoff).
# Nothing here reads or skips against those old paths. See results/delta/README.md
# for the full tree and what each file measures.
#
#   eval_out/run_delta_evals.sh <arm> <steps|groups...>
#   arm: d768 | d3072 | d6144 | d12288 | d6144k4000 | d12288k4000     (Llama-3.2-3B, layer 27)
#        g1b_l{13,25}_d{1152,2304,4608}                            (Gemma 3 1B, K=2000; see results/delta/README.md)
#        g4b_l33_d5120                                             (Gemma 3 4B, layer 33 = last of 34 blocks, K=2000)
#
#   eval_out/run_delta_evals.sh d6144 evals
#   eval_out/run_delta_evals.sh d6144 all
#   eval_out/run_delta_evals.sh d6144 guard          # just run the checkpoint guard
#   DRY=1 eval_out/run_delta_evals.sh d6144 all       # print resolved commands, run nothing
#   FORCE=1 eval_out/run_delta_evals.sh d6144 mmlu    # redo one step
#
# ACT_DIR: scattered random-row reads into activations_sampled_10M/layer_27.npy (61 GB,
# a symlink into /work/hdd Lustre) are extremely slow over the network filesystem.
# scripts/delta/sheet_staged.sbatch rsyncs the whole dump directory to node-local
# /tmp/$SLURM_JOB_ID/activations_sampled_10M first and exports ACT_DIR to that path;
# every step below that actually reads the dump (tokstruct, cq, csteer, cgen,
# cgen_concept) resolves its --activations_dir from ACT_DIR, defaulting to the
# Lustre path when unset (e.g. a login-node DRY run or a plain sheet.sbatch job).
# The other steps (probe, mmlu, number*, steer, range_*, cjudge*) never touch the
# dump at all — see the per-tool sweep in this session's report — so they ignore
# ACT_DIR entirely and are unaffected either way.
#
# Steps: tokstruct probe mmlu cq cq_std | number steer range_db14 range_bios number_hb |
#        csteer cgen cjudge | cgen_concept cjudge_concept | geometry | biasprobe tpp | guard
# Group aliases: evals interventions steering concept judge geometry probes all
#   evals          = tokstruct probe mmlu cq                     -> results/delta/<arm>/evals/
#   (cq_std = cq with the width-neutral metrics -> evals/cq_std.json; in no group, name it explicitly)
#   interventions  = number steer range_db14 range_bios number_hb -> results/delta/<arm>/interventions/
#   steering       = csteer cgen                  (GPU only — no judge) -> results/delta/<arm>/steering/
#   concept        = cgen_concept                 (GPU only — no judge) -> results/delta/<arm>/concept/
#   judge          = cjudge cjudge_concept       (API only; run SERIALLY across arms from the login
#                                                  node: llm_judge's disk cache writes are not atomic,
#                                                  and base prompts are shared between arms)
#   geometry       = geometry   (PCA / t-SNE / UMAP / variance panels, base | base_tokmean | bypass,
#                                 reads the small cache_ids/ rung caches only, NOT the dump)
#                                                                    -> results/delta/<arm>/figures/geometry/
#   probes         = biasprobe tpp   (fixed-probe coordinate-removal, raw | tokmean | bypass,
#                                 mean-fill ablation, l1=0; GPU; reads the small last-token act dirs
#                                 written by `run_delta_prep.sh probe_acts`, NOT the dump)
#                                                                    -> results/delta/<arm>/probes/
#                                    biasbios_probe.json   geoae.bias.probe: gender/profession probes on BiasBios
#                                    tpp_dbpedia.json      geoae.interp.probe_perturbation on DBpedia-14
#                                    tpp_biasbios.json     the same TPP on the 28 bias_in_bios professions
#                                  plots -> figures/probes/{biasbios,tpp_dbpedia,tpp_biasbios}/
#   all            = evals interventions steering concept judge   (never implies train; does NOT
#                                                                  include geometry or probes — ask for them)
#
# A checkpoint GUARD runs automatically before any eval/intervention/steering/concept
# step (see guard() below): loads the arm's checkpoint on CPU and refuses to score an
# AE whose centroids were never initialised, whose width/K is wrong, or that is
# missing its token-bias table; also requires the K-matched KM/KMT codebooks to exist
# for steps that use them, and cache_ids/ for probe. Finished steps are skipped;
# FORCE=1 redoes one. A failing step is reported at the end (FAILED), not left silent.
set -o pipefail
# Delta: env.sh resolves GEOAE_ROOT from its own location, exports HF_HOME /
# UV_PROJECT_ENVIRONMENT / SBATCH_ACCOUNT, and cds to the repo root.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/delta/env.sh"
PY=scripts/delta/py
E=e2e/checkpoints/general/llama3.2-3B/layer27
B=checkpoints/llama3.2-3B/layer27
DELTA=$E/delta
TABLE=$E/token_bias_sampled.npz
# Set by scripts/delta/sheet_staged.sbatch to the node-local rsync'd copy; unset
# (falls back to the Lustre symlink) on the login node / under plain sheet.sbatch.
ACT="${ACT_DIR:-activations_sampled_10M}"
# Per-arm-family settings. Llama values are the historical hard-coded ones (every
# command printed for a Llama arm is unchanged). The g1b block below overrides them
# once the arm is known. LAYER_OPT / LAYER_ARR carry an explicit --layer ONLY for g1b.
FAM=llama; LAYER=27; MODEL_NAME=meta-llama/Llama-3.2-3B; N_DEC_LAYERS=28
CACHE_IDS=cache_ids
ACT_BB=biasbios/activations/llama3.2-3B/last
ACT_DB=dbpedia/activations/llama3.2-3B/last_ids/unprompted
LAYER_OPT=""; LAYER_ARR=(); TS_EXTRA=()

ARM="${1:-}"; shift || true
case "$ARM" in
    d768)       CKDIR=k2000_bnh_b32k_lam1_d768_dpc_sampled_tokbias;  K=2000; DIM=768   ;;
    d3072)      CKDIR=k2000_bnh_b32k_lam1_d3072_dpc_sampled_tokbias; K=2000; DIM=3072  ;;
    d6144)      CKDIR=k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias; K=2000; DIM=6144  ;;
    d12288)     CKDIR=k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias;K=2000; DIM=12288 ;;
    d6144k4000) CKDIR=k4000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias; K=4000; DIM=6144  ;;
    d12288k4000) CKDIR=k4000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias; K=4000; DIM=12288 ;;
    g1b_l13_d1152|g1b_l13_d2304|g1b_l13_d4608|g1b_l25_d1152|g1b_l25_d2304|g1b_l25_d4608)
        FAM=g1b; K=2000; DIM=${ARM##*_d}; LAYER=${ARM#g1b_l}; LAYER=${LAYER%%_*}
        CKDIR=k2000_bnh_b32k_lam1_d${DIM}_dpc_sampled_struct5_tokbias ;;
    g4b_l33_d5120)
        FAM=g4b; K=2000; DIM=${ARM##*_d}; LAYER=${ARM#g4b_l}; LAYER=${LAYER%%_*}
        CKDIR=k2000_bnh_b32k_lam1_d${DIM}_dpc_sampled_struct5_tokbias ;;
    *)
        echo "usage: $0 <d768|d3072|d6144|d12288|d6144k4000|d12288k4000|g1b_l{13,25}_d{1152,2304,4608}|g4b_l33_d5120> <steps|groups...>" >&2
        echo "  steps: tokstruct probe mmlu cq cq_std number steer range_db14 range_bios number_hb csteer cgen cjudge cgen_concept cjudge_concept geometry biasprobe tpp guard" >&2
        echo "  groups: evals interventions steering concept judge geometry probes all" >&2
        exit 2
        ;;
esac
if [ "$FAM" != llama ]; then
    # Per-family parameter table for the Gemma arms: e2e/checkpoint family dir, dump, HF model, decoder block
    # count, probe-act model dir, cache dir. Layer 25 (1B) / 33 (4B) is the LAST block.
    case "$FAM" in
        g1b) F_E2E=gemma3-1B; F_DUMP=activations_gemma3_1b_sampled_struct5_10M; MODEL_NAME=google/gemma-3-1b-pt
             N_DEC_LAYERS=26; F_CACHE=cache/gemma3-1b; F_PA=gemma-3-1b-pt ;;   # hidden 1152
        g4b) F_E2E=gemma3-4B; F_DUMP=activations_gemma3_4b_sampled_struct5_10M; MODEL_NAME=google/gemma-3-4b-pt
             N_DEC_LAYERS=34; F_CACHE=cache/gemma3-4b; F_PA=gemma-3-4b-pt ;;   # hidden 2560
    esac
    E=e2e/checkpoints/general/$F_E2E/layer$LAYER
    B=checkpoints/$F_E2E/layer$LAYER
    DELTA=$E/delta
    TABLE=$E/token_bias_sampled_struct5.npz
    ACT="${ACT_DIR:-$F_DUMP}"
    CACHE_IDS=$F_CACHE/l$LAYER      # token rungs + sequence rungs WITH ids + ravel/ioi in ONE dir
    ACT_BB=biasbios/activations/${FAM}_l$LAYER/$F_PA/last
    ACT_DB=dbpedia/activations/${FAM}_l$LAYER/$F_PA/last_ids/unprompted
    LAYER_OPT=" --layer $LAYER"; LAYER_ARR=(--layer "$LAYER")
fi
LAST_BLOCK=0; [ "$LAYER" -eq $((N_DEC_LAYERS - 1)) ] && LAST_BLOCK=1
# cNMI-next / trivial-next apply final norm + lm_head to the dump rows: valid only at the last block.
[ "$FAM" != llama ] && [ "$LAST_BLOCK" -eq 0 ] && TS_EXTRA=(--skip_next)
TB=$B/$CKDIR/step_0014450.pt
if [ "$FAM" != llama ]; then
    KM=$DELTA/balanced_kmeans_k${K}_dpc_sampled_struct5.npz
    KMT=$DELTA/balanced_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
else
    KM=$DELTA/balanced_kmeans_k${K}_dpc_sampled.npz
    KMT=$DELTA/balanced_kmeans_k${K}_dpc_sampled_tokmean.npz
fi
TAG=$ARM
ARMD=results/delta/$ARM
EVALSD=$ARMD/evals
INTERVD=$ARMD/interventions
STEERD=$ARMD/steering
CONCEPTD=$ARMD/concept
PROBED=$ARMD/probes
FIGD=$ARMD/figures
LOGD=$ARMD/logs
JCD=results/delta/joint_correct
mkdir -p "$EVALSD" "$INTERVD" "$STEERD" "$CONCEPTD" "$FIGD" \
         "$LOGD/evals" "$LOGD/interventions" "$LOGD/steering" "$LOGD/concept" "$LOGD/geometry" "$JCD" \
         "$PROBED" "$FIGD/probes" "$LOGD/probes"
JC_DB14=$JCD/joint_correct_db14_l${LAYER}_$TAG.json
JC_BIOS=$JCD/joint_correct_biasbios_l${LAYER}_$TAG.json
BIOS_CONCEPTS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27  # no "teacher" (0% few-shot)

if [ $# -eq 0 ]; then
    sed -n '1,60p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

# --------------------------------------------------------------------------------- #
# Steps and aliases
# --------------------------------------------------------------------------------- #
ATOMIC=(tokstruct probe mmlu cq cq_std number steer range_db14 range_bios number_hb csteer cgen cjudge cgen_concept cjudge_concept geometry biasprobe tpp guard)
EVALS=(tokstruct probe mmlu cq)
INTERVENTIONS=(number steer range_db14 range_bios number_hb)
STEERING=(csteer cgen)
CONCEPT=(cgen_concept)
JUDGE=(cjudge cjudge_concept)
GEOMETRY=(geometry)
PROBES=(biasprobe tpp)
GUARD_STEPS=(tokstruct probe mmlu cq cq_std number steer range_db14 range_bios number_hb csteer cgen cjudge cgen_concept cjudge_concept geometry biasprobe tpp)

expand() {
    local out=()
    for s in "$@"; do
        case "$s" in
            evals)         out+=("${EVALS[@]}") ;;
            interventions) out+=("${INTERVENTIONS[@]}") ;;
            steering)      out+=("${STEERING[@]}") ;;
            concept)       out+=("${CONCEPT[@]}") ;;
            judge)         out+=("${JUDGE[@]}") ;;
            geometry)      out+=("${GEOMETRY[@]}") ;;
            probes)        out+=("${PROBES[@]}") ;;
            all)           out+=("${EVALS[@]}" "${INTERVENTIONS[@]}" "${STEERING[@]}" "${CONCEPT[@]}" "${JUDGE[@]}") ;;
            *)
                local ok=0
                for a in "${ATOMIC[@]}"; do [ "$a" = "$s" ] && ok=1 && break; done
                if [ "$ok" -eq 1 ]; then
                    out+=("$s")
                else
                    echo "[abort] unknown step '$s' (have: ${ATOMIC[*]}, or aliases: evals interventions steering concept judge geometry probes all)" >&2
                    exit 2
                fi
                ;;
        esac
    done
    echo "${out[@]}"
}
REQUESTED=($(expand "$@")) || exit $?
want() { for s in "${REQUESTED[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

FAILED=()
# Steps that SystemExit unless --layer is the LM's last block (next-token behaviour is read off
# lm_head(final_norm(h)) / the edit acts on the next-token choice). Skipped, not FAILED, otherwise.
LAST_BLOCK_ONLY=" csteer cgen cjudge cgen_concept cjudge_concept "
step() {   # step <name> <done-check-string, bash -c'd> <function>
    want "$1" || return
    if [ "$LAST_BLOCK" -eq 0 ] && [[ "$LAST_BLOCK_ONLY" == *" $1 "* ]]; then
        echo "[skip] $1 — needs the last block (layer $((N_DEC_LAYERS - 1)))"; return
    fi
    if [ -z "$FORCE" ] && [ -z "$DRY" ] && eval "$2"; then
        echo "[skip] $1 — already done (FORCE=1 to redo)"; return
    fi
    echo -e "\n[run ] $1  ($(date +%H:%M))"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

# --------------------------------------------------------------------------------- #
# Checkpoint guard
# --------------------------------------------------------------------------------- #
guard() {
    if [ ! -f "$TB" ]; then
        echo "[guard] $TB missing — training not at epoch 50 (or not started)"
        [ -n "$DRY" ] && return 0
        return 1
    fi
    if [ "$FAM" != llama ] && [ ! -f "$TABLE" ]; then
        echo "[guard] token-bias table $TABLE missing"
        [ -n "$DRY" ] && return 0
        return 1
    fi
    "$PY" - "$TB" "$DIM" "$K" "$FAM" "$LAYER" "$MODEL_NAME" <<'EOF'
import sys
import torch

path, want_dim, want_k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fam, want_layer, want_model = sys.argv[4], int(sys.argv[5]), sys.argv[6]
c = torch.load(path, map_location="cpu", weights_only=False)
epoch = c.get("epoch")
val_mse = c.get("val_mse")
model_cfg = c["config"]["model"]
dim = model_cfg.get("latent_dim")
n_clusters = model_cfg.get("n_clusters")
init = bool(c["model_state"].get("centroids_initialized", False))
has_tb = "tb_table" in c["model_state"]
print(f"[guard] {path}: epoch {epoch}, val_mse {val_mse}, latent_dim {dim}, "
      f"n_clusters {n_clusters}, centroids_initialized {init}, token_bias_table {has_tb}")
ok = (epoch == 50 and init and dim == want_dim and n_clusters == want_k and has_tb)
if not ok:
    print(f"[guard] FAIL — expected epoch 50, centroids_initialized True, "
          f"latent_dim {want_dim}, n_clusters {want_k}, a token-bias table present")
if fam != "llama":   # Llama arms keep the original checks only
    layer = c["config"]["data"].get("target_layer")
    model = c["config"]["extraction"].get("model_name")
    print(f"[guard] target_layer {layer} (want {want_layer}), model_name {model} (want {want_model})")
    if layer != want_layer or model != want_model:
        print("[guard] FAIL — checkpoint target_layer/model_name do not match this arm")
        ok = False
sys.exit(0 if ok else 1)
EOF
}
# Under DRY these are no-ops (return success) so the intended command still prints
# with its resolved paths, even before eval_out/run_delta_prep.sh has produced KM/KMT
# or cache_ids/. In a real (non-DRY) run they abort the step with a clear message.
PREP_KM=km; [ "$FAM" != llama ] && PREP_KM=${FAM}_km
PREP_PA=probe_acts; [ "$FAM" != llama ] && PREP_PA=${FAM}_probe_acts
require_km()  { [ -n "$DRY" ] && return 0; [ -f "$KM" ]  || { echo "[abort] $KM missing — run eval_out/run_delta_prep.sh $PREP_KM (K=$K)"; return 1; }; }
require_kmt() { [ -n "$DRY" ] && return 0; [ -f "$KMT" ] || { echo "[abort] $KMT missing — run eval_out/run_delta_prep.sh $PREP_KM (K=$K)"; return 1; }; }
require_cache_ids() {
    [ -n "$DRY" ] && return 0
    [ -d "$CACHE_IDS" ] || {
        if [ "$FAM" != llama ]; then echo "[abort] $CACHE_IDS/ missing — run eval_out/run_delta_prep.sh ${FAM}_cache"
        else echo "[abort] cache_ids/ missing — run eval_out/run_delta_prep.sh bench cache"; fi
        return 1; }
}

NEED_GUARD=0
for s in "${GUARD_STEPS[@]}"; do want "$s" && NEED_GUARD=1; done
want guard && NEED_GUARD=1
if [ "$NEED_GUARD" -eq 1 ]; then
    guard || { echo "[abort] checkpoint guard failed"; [ -z "$DRY" ] && exit 1; }
fi

# --------------------------------------------------------------------------------- #
# helper: run a command for real, or print it (DRY=1) with all paths resolved
# --------------------------------------------------------------------------------- #
X() {   # X <log path> -- <cmd...>     (X /path/to/log.log -- prog --flag val ...)
    local log="$1"; shift
    [ "$1" = "--" ] && shift
    if [ -n "$DRY" ]; then
        echo "[dry ] $* | tee $log"
        return 0
    fi
    "$@" 2>&1 | tee "$log"
}

# --------------------------------------------------------------------------------- #
# evals -> results/delta/<arm>/evals/
# --------------------------------------------------------------------------------- #
tokstruct() {
    require_km && require_kmt || return 1
    X "$LOGD/evals/token_structure.log" -- \
        "$PY" -u eval_out/token_structure.py \
        --arms "base=$KM,base_tokmean=$KMT,bypass=$TB" \
        --activations_dir "$ACT" "${LAYER_ARR[@]}" "${TS_EXTRA[@]}" \
        --out "$EVALSD/token_structure.json"
}
probe() {
    require_km && require_kmt && require_cache_ids || return 1
    # --atlas_last "" : cache/atlas8k_last.npz does not exist on this machine and
    # probe_chance_corrected.py's default crashes on the missing file (see
    # eval_out/run_delta_prep.sh's `atlas` step for the investigation). cache_ids/
    # never carries an atlas_* rung for this arm family, so nothing scored is
    # affected by disabling atlas-anchor exclusion.
    X "$LOGD/evals/probe.log" -- \
        "$PY" -u eval_out/probe_chance_corrected.py --cache "$CACHE_IDS" \
        --models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT" \
        --atlas_last "" --out "$EVALSD/probe.json"
}
mmlu() {
    X "$LOGD/evals/mmlu.log" -- \
        "$PY" -u -m geoae.interp.mmlu_splice --checkpoint "$TB" --out "$EVALSD/mmlu.json"
}
cq() {
    require_km || return 1
    local avail
    avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
    echo "[cq] MemAvailable ${avail} GB; d${DIM} latents need ~$((DIM * 4 / 1000)) GB per 1M rows fp32"
    [ "$avail" -lt 60 ] && echo "[cq] WARNING: under 60 GB free — stop other jobs or drop --n_sample to 500000"
    X "$LOGD/evals/cq.log" -- \
        "$PY" -u -m geoae.interp.clustering_quality --baseline "$KM" --checkpoints "$TB" \
        --names base bypass \
        --activations_dir "$ACT" --layer "$LAYER" --n_sample 1000000 --seed 0 \
        --out "$EVALSD/cq.json"
}
# Same run as cq, written to its own file: the code now also emits the width-neutral
# metrics (silhouette_full, *_share, separability_radii, inter_centroid_dist_rel).
cq_std() {
    require_km || return 1
    local avail
    avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
    echo "[cq_std] MemAvailable ${avail} GB; d${DIM} latents need ~$((DIM * 4 / 1000)) GB per 1M rows fp32"
    [ "$avail" -lt 60 ] && echo "[cq_std] WARNING: under 60 GB free — stop other jobs or drop --n_sample to 500000"
    X "$LOGD/evals/cq_std.log" -- \
        "$PY" -u -m geoae.interp.clustering_quality --baseline "$KM" --checkpoints "$TB" \
        --names base bypass \
        --activations_dir "$ACT" --layer "$LAYER" --n_sample 1000000 --seed 0 \
        --out "$EVALSD/cq_std.json"
}

# --------------------------------------------------------------------------------- #
# interventions (identical settings to eval_out/run_tokbias_interventions.sh /
# run_tokbias_d12288.sh, only the checkpoint and output paths change)
# -> results/delta/<arm>/interventions/ (figures -> .../figures/)
# --------------------------------------------------------------------------------- #
number() {
    if [ -n "$DRY" ]; then
        echo "[dry ] NUMBER_CHECKPOINT=$TB NUMBER_OUTPUT=$INTERVD/range_number_dprime.json bash eval_out/run_number_control.sh --saliency dprime | tee $LOGD/interventions/number_dprime.log"
        echo "[dry ] $PY eval_out/review_number_control.py --source $INTERVD/range_number_dprime.json --review_out $INTERVD/number_review.json --figure_prefix $FIGD/number_review"
        return 0
    fi
    NUMBER_CHECKPOINT="$TB" NUMBER_OUTPUT="$INTERVD/range_number_dprime.json" \
        bash eval_out/run_number_control.sh --saliency dprime | tee "$LOGD/interventions/number_dprime.log" || return 1
    "$PY" eval_out/review_number_control.py --source "$INTERVD/range_number_dprime.json" \
        --review_out "$INTERVD/number_review.json" --figure_prefix "$FIGD/number_review"
}
steer() {
    X "$LOGD/interventions/steer_db14.log" -- \
        "$PY" -u -m geoae.interp.steering_concept_compare --checkpoint "$TB" --dataset db14 \
        --correct_json "$JC_DB14" --out "$INTERVD/steer_db14.json"
}
range_db14() {
    X "$LOGD/interventions/range_db14_dprime.log" -- \
        "$PY" -u -m geoae.interp.range_intervention_compare --checkpoint "$TB" --dataset db14 \
        --correct_json "$JC_DB14" --saliency dprime --tao 2.0 \
        --out "$INTERVD/range_intervention_db14_dprime.json"
}
range_bios() {
    X "$LOGD/interventions/range_biasbios_dprime.log" -- \
        "$PY" -u -m geoae.interp.range_intervention_compare --checkpoint "$TB" --dataset biasbios \
        --concepts "$BIOS_CONCEPTS" --correct_json "$JC_BIOS" --saliency dprime --tao 2.0 --alphas 1.0 2.0 \
        --out "$INTERVD/range_intervention_biasbios_dprime.json"
}
# Matched base arm (hb): edit h - b[tok], add b[tok] back. The tool refuses to overwrite an
# existing output, and an interrupted run leaves an INCOMPLETE file behind, so it is
# deleted here first (this only runs when step() has decided the file is not yet complete).
number_hb() {
    if [ -n "$DRY" ]; then
        echo "[dry ] rm -f $INTERVD/range_number_dprime_hb.json"
        echo "[dry ] NUMBER_CHECKPOINT=$TB NUMBER_OUTPUT=$INTERVD/range_number_dprime_hb.json bash eval_out/run_number_control.sh --saliency dprime --hb_table $TABLE | tee $LOGD/interventions/number_dprime_hb.log"
        return 0
    fi
    rm -f "$INTERVD/range_number_dprime_hb.json"
    NUMBER_CHECKPOINT="$TB" NUMBER_OUTPUT="$INTERVD/range_number_dprime_hb.json" \
        bash eval_out/run_number_control.sh --saliency dprime --hb_table "$TABLE" \
        | tee "$LOGD/interventions/number_dprime_hb.log"
}

# --------------------------------------------------------------------------------- #
# steering (next-token cluster steering + generation + LLM judge)
# -> results/delta/<arm>/steering/
# --------------------------------------------------------------------------------- #
csteer() {
    require_km && require_kmt || return 1
    local ARMS="base=$KM,base_tokmean=$KMT,bypass=$TB"
    if [ -n "$DRY" ]; then
        echo "[dry ] $PY -u -m geoae.interp.cluster_steering --arms $ARMS --activations_dir $ACT$LAYER_OPT --out $STEERD/cluster_steering.json | tee $LOGD/steering/cluster_steering.log"
        echo "[dry ] $PY -u -m geoae.interp.cluster_steering --arms $ARMS --activations_dir $ACT$LAYER_OPT --hub_x 1e9 --out $STEERD/cluster_steering_allhubs.json | tee $LOGD/steering/cluster_steering_allhubs.log"
        return 0
    fi
    "$PY" -u -m geoae.interp.cluster_steering --arms "$ARMS" --activations_dir "$ACT" "${LAYER_ARR[@]}" \
        --out "$STEERD/cluster_steering.json" 2>&1 | tee "$LOGD/steering/cluster_steering.log" || return 1
    "$PY" -u -m geoae.interp.cluster_steering --arms "$ARMS" --activations_dir "$ACT" "${LAYER_ARR[@]}" --hub_x 1e9 \
        --out "$STEERD/cluster_steering_allhubs.json" 2>&1 | tee "$LOGD/steering/cluster_steering_allhubs.log"
    # side file, alongside each --out: <out>.rows.npz (Path.with_suffix), already
    # inside results/delta/<arm>/steering/ since it derives from --out.
}
cgen() {
    require_km || return 1
    X "$LOGD/steering/cluster_steer_generate.log" -- \
        "$PY" -u -m geoae.interp.cluster_steer_generate --base "$KM" --bypass "$TB" \
        --activations_dir "$ACT" "${LAYER_ARR[@]}" \
        --out "$STEERD/cluster_steer_generate.json"
}
cjudge() {
    [ -f "$STEERD/cluster_steer_generate.json" ] || { [ -n "$DRY" ] || { echo "[abort] run cgen first"; return 1; }; }
    if [ -n "$DRY" ]; then
        echo "[dry ] (source .env; require OPENROUTER_API_KEY)"
        echo "[dry ] $PY -u -m geoae.interp.cluster_steer_judge --gen $STEERD/cluster_steer_generate.json --provider openrouter --model google/gemini-2.5-flash-lite --out $STEERD/cluster_steer_judge_llmfluency.json --fluency judge | tee $LOGD/steering/cluster_steer_judge_llmfluency.log"
        return 0
    fi
    set -a; [ -f .env ] && source .env; set +a
    [ -n "$OPENROUTER_API_KEY" ] || { echo "[abort] OPENROUTER_API_KEY not set (.env)"; return 1; }
    "$PY" -u -m geoae.interp.cluster_steer_judge --gen "$STEERD/cluster_steer_generate.json" \
        --provider openrouter --model google/gemini-2.5-flash-lite \
        --fluency judge \
        --out "$STEERD/cluster_steer_judge_llmfluency.json" 2>&1 | tee "$LOGD/steering/cluster_steer_judge_llmfluency.log"
    # cache_dir default cache/llm_judge is a SHARED disk cache of (prompt -> answer)
    # across every arm — intentional and cheap to keep shared (see spec); not redirected.
}

# --------------------------------------------------------------------------------- #
# concept (named-concept steering + LLM judge)
# -> results/delta/<arm>/concept/
# --------------------------------------------------------------------------------- #
cgen_concept() {
    require_km || return 1
    X "$LOGD/concept/concept_steer_generate.log" -- \
        "$PY" -u -m geoae.interp.concept_steer_generate --base "$KM" --bypass "$TB" \
        --datasets db14,biasbios --handles label,label_z,base,bypass,base_dir,bypass_dir \
        --activations_dir "$ACT" "${LAYER_ARR[@]}" \
        --out "$CONCEPTD/concept_steer_generate.json"
}
cjudge_concept() {
    [ -f "$CONCEPTD/concept_steer_generate.json" ] || { [ -n "$DRY" ] || { echo "[abort] run cgen_concept first"; return 1; }; }
    if [ -n "$DRY" ]; then
        echo "[dry ] (source .env; require OPENROUTER_API_KEY)"
        echo "[dry ] $PY -u -m geoae.interp.concept_steer_judge --gen $CONCEPTD/concept_steer_generate.json --provider openrouter --model google/gemini-2.5-flash-lite --out $CONCEPTD/concept_steer_judge_llmfluency.json --fluency judge | tee $LOGD/concept/concept_steer_judge_llmfluency.log"
        return 0
    fi
    set -a; [ -f .env ] && source .env; set +a
    [ -n "$OPENROUTER_API_KEY" ] || { echo "[abort] OPENROUTER_API_KEY not set (.env)"; return 1; }
    "$PY" -u -m geoae.interp.concept_steer_judge --gen "$CONCEPTD/concept_steer_generate.json" \
        --provider openrouter --model google/gemini-2.5-flash-lite \
        --fluency judge \
        --out "$CONCEPTD/concept_steer_judge_llmfluency.json" 2>&1 | tee "$LOGD/concept/concept_steer_judge_llmfluency.log"
}

# --------------------------------------------------------------------------------- #
# geometry -> results/delta/<arm>/figures/geometry/{pca,tsne,umap,variance}/<rung>.png
# Three panels per rung: base (normalised raw residual), base_tokmean (residual minus
# per-token mean), bypass (AE latent z = ae.encode(x, tok)), same points and colours,
# each labelled with its kNN-10 label agreement. Reads ONLY the small rung caches in
# cache_ids/ (sequence rungs need last_token_id, which only cache_ids/ carries; token
# rungs carry token_id in both caches) — not the activation dump, so it needs no staging.
# --------------------------------------------------------------------------------- #
GEO_RUNGS_ALL="topic14:topic14.npz topic4:topic4.npz domain:domain.npz language:language.npz formality:formality.npz subjectivity:subjectivity.npz pos_coarse:pos.npz ner_coarse:ner.npz ioi_role:ioi.npz ravel_continent:ravel.npz"
GEOD=$FIGD/geometry
geo_rungs() {   # rungs whose cache_ids/ file exists
    local r out=()
    for r in $GEO_RUNGS_ALL; do [ -f "$CACHE_IDS/${r#*:}" ] && out+=("${r%%:*}"); done
    echo "${out[@]}"
}
geo_projs() {   # umap only if umap-learn is importable (the tool skips it otherwise)
    local p="pca tsne variance"
    "$PY" -c "import umap" >/dev/null 2>&1 && p="$p umap"
    echo "$p"
}
geo_missing() {   # rungs still lacking any expected PNG
    local r p out=() projs; projs=$(geo_projs)
    for r in $(geo_rungs); do
        for p in $projs; do [ -f "$GEOD/$p/$r.png" ] || { out+=("$r"); break; }
        done
    done
    echo "${out[@]}"
}
geometry() {
    require_km && require_kmt && require_cache_ids || return 1
    local rungs
    if [ -n "$DRY" ]; then rungs=$(echo $GEO_RUNGS_ALL | tr ' ' '\n' | cut -d: -f1 | paste -sd,); else
        if [ -n "$FORCE" ]; then rungs=$(geo_rungs); else rungs=$(geo_missing); fi
        [ -n "$rungs" ] || { echo "[geometry] nothing missing"; return 0; }
        rungs=$(echo $rungs | tr ' ' ',')
    fi
    X "$LOGD/geometry/geometry.log" -- \
        "$PY" -u -m geoae.interp.concept_geometry --cache "$CACHE_IDS" \
        --models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT" \
        --order base,base_tokmean,bypass --rungs "$rungs" \
        --n_points 6000 --max_classes 14 --perplexity 30 \
        --projections pca,tsne,umap,variance --outdir "$GEOD"
}

# --------------------------------------------------------------------------------- #
# probes -> results/delta/<arm>/probes/{biasbios_probe,tpp_dbpedia,tpp_biasbios}.json
# Fixed-probe coordinate-removal evals in three spaces: raw (z-scored residual),
# tokmean (z-scored residual - b[tok], the token-mean control) and geoae (the bypass
# AE's latent, ae.encode(x, tok)). Bypass AEs need the current token id of every row,
# so these read tokens_{train,test}.npy from the act dirs that
# `eval_out/run_delta_prep.sh probe_acts` extracts (last-token pooling, layer 27; NOT the
# dump). Ablation is MEAN-fill of standardized coordinates with lambda_l1 = 0, the
# corrected protocol (zero-fill is an artifact for shifted GELU latents). Never cite
# TPP mean_selectivity. Runs on the GPU; the tools refuse to overwrite an output and an
# interrupted run leaves a status "running" file, so it is removed here before a rerun.
# --------------------------------------------------------------------------------- #
# (ACT_BB / ACT_DB are set per arm family near the top; g1b uses biasbios|dbpedia/activations/g1b_l<L>/gemma-3-1b-pt/...)
probe_acts_ok() {   # probe_acts_ok <act dir>
    local f
    for f in meta.json source_hashes.npz tokens_train.npy tokens_test.npy layer_${LAYER}.npy layer_${LAYER}_test.npy labels_train.npy labels_test.npy; do
        [ -f "$1/$f" ] || return 1
    done
}
require_probe_acts() {
    [ -n "$DRY" ] && return 0
    probe_acts_ok "$1" || { echo "[abort] $1 incomplete (needs tokens_*.npy + source_hashes.npz) — run eval_out/run_delta_prep.sh $PREP_PA"; return 1; }
}
json_complete() { grep -qs '"status": "complete"' "$1"; }
# probe_run <out.json> <log> -- <cmd... without --out>: skip a complete output, drop a stale one
probe_run() {
    local out="$1" log="$2"; shift 2; [ "$1" = "--" ] && shift
    if [ -z "$DRY" ]; then
        if [ -z "$FORCE" ] && json_complete "$out"; then echo "[skip] $out already complete"; return 0; fi
        rm -f "$out" "$out.tmp"
    fi
    X "$log" -- "$@" --out "$out"
}
biasprobe() {
    require_probe_acts "$ACT_BB" || return 1
    probe_run "$PROBED/biasbios_probe.json" "$LOGD/probes/biasbios_probe.log" -- \
        "$PY" -u -m geoae.bias.probe --act_dir "$ACT_BB" --source_hashes "$ACT_BB/source_hashes.npz" \
        --layer "$LAYER" --ae_checkpoint "$TB" --token_bias_table "$TABLE" \
        --spaces raw tokmean geoae --device cuda --plot_dir "$FIGD/probes/biasbios"
}
tpp() {
    require_probe_acts "$ACT_DB" && require_probe_acts "$ACT_BB" || return 1
    local rc=0
    probe_run "$PROBED/tpp_dbpedia.json" "$LOGD/probes/tpp_dbpedia.log" -- \
        "$PY" -u -m geoae.interp.probe_perturbation --act_dir "$ACT_DB" --source_hashes "$ACT_DB/source_hashes.npz" \
        --layer "$LAYER" --ae_checkpoint "$TB" --token_bias_table "$TABLE" \
        --spaces raw tokmean geoae --device cuda --plot_dir "$FIGD/probes/tpp_dbpedia" || rc=1
    # same tool on the 28 bias_in_bios professions (act_dir carries labels_train.npy = profession
    # alias and meta.json "classes" = professions)
    probe_run "$PROBED/tpp_biasbios.json" "$LOGD/probes/tpp_biasbios.log" -- \
        "$PY" -u -m geoae.interp.probe_perturbation --act_dir "$ACT_BB" --source_hashes "$ACT_BB/source_hashes.npz" \
        --layer "$LAYER" --ae_checkpoint "$TB" --token_bias_table "$TABLE" \
        --spaces raw tokmean geoae --device cuda --plot_dir "$FIGD/probes/tpp_biasbios" || rc=1
    return $rc
}

# --------------------------------------------------------------------------------- #
# guard-only pseudo-step (for testing / sanity-checking a checkpoint on its own)
# --------------------------------------------------------------------------------- #
guard_step() { echo "[guard] guard already ran above for this invocation."; return 0; }

# --------------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------------- #
step "tokstruct"      "[ -f $EVALSD/token_structure.json ]"                                  tokstruct
step "probe"          "[ -f $EVALSD/probe.json ]"                                            probe
step "mmlu"           "[ -f $EVALSD/mmlu.json ]"                                              mmlu
step "cq"             "[ -f $EVALSD/cq.json ]"                                                cq
step "cq_std"         "[ -f $EVALSD/cq_std.json ]"                                            cq_std
step "number"         "grep -qs '\"status\": \"complete\"' $INTERVD/range_number_dprime.json && [ -f $INTERVD/number_review.json ]" number
step "steer"          "grep -qs '\"concepts\"' $INTERVD/steer_db14.json"                      steer
step "range_db14"     "grep -qs '\"summary\"' $INTERVD/range_intervention_db14_dprime.json"   range_db14
step "range_bios"     "grep -qs '\"summary\"' $INTERVD/range_intervention_biasbios_dprime.json" range_bios
step "number_hb"      "grep -qs '\"status\": \"complete\"' $INTERVD/range_number_dprime_hb.json" number_hb
step "csteer"         "[ -f $STEERD/cluster_steering.json ] && [ -f $STEERD/cluster_steering_allhubs.json ]" csteer
step "cgen"           "[ -f $STEERD/cluster_steer_generate.json ]"                            cgen
step "cjudge"         "[ -f $STEERD/cluster_steer_judge_llmfluency.json ]"                               cjudge
step "cgen_concept"   "[ -f $CONCEPTD/concept_steer_generate.json ]"                          cgen_concept
step "cjudge_concept" "[ -f $CONCEPTD/concept_steer_judge_llmfluency.json ]"                             cjudge_concept
step "geometry"       "[ -z \"\$(geo_missing)\" ]"                                             geometry
step "biasprobe"      "grep -qs '\"status\": \"complete\"' $PROBED/biasbios_probe.json"      biasprobe
step "tpp"            "grep -qs '\"status\": \"complete\"' $PROBED/tpp_dbpedia.json && grep -qs '\"status\": \"complete\"' $PROBED/tpp_biasbios.json" tpp
step "guard"          "false"                                                                 guard_step

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "\n[done] FAILED steps: ${FAILED[*]}"
    exit 1
fi
echo -e "\n[done] $(date +%H:%M)"
