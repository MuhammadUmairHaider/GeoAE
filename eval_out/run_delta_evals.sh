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
#   arm: d768 | d3072 | d6144 | d12288 | d6144k4000 | d12288k4000
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
# Steps: tokstruct probe mmlu cq | number steer range_db14 range_bios number_hb |
#        csteer cgen cjudge | cgen_concept cjudge_concept | guard
# Group aliases: evals interventions steering concept judge all
#   evals          = tokstruct probe mmlu cq                     -> results/delta/<arm>/evals/
#   interventions  = number steer range_db14 range_bios number_hb -> results/delta/<arm>/interventions/
#   steering       = csteer cgen                  (GPU only — no judge) -> results/delta/<arm>/steering/
#   concept        = cgen_concept                 (GPU only — no judge) -> results/delta/<arm>/concept/
#   judge          = cjudge cjudge_concept       (API only; run SERIALLY across arms from the login
#                                                  node: llm_judge's disk cache writes are not atomic,
#                                                  and base prompts are shared between arms)
#   all            = evals interventions steering concept judge   (never implies train)
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

ARM="${1:-}"; shift || true
case "$ARM" in
    d768)       CKDIR=k2000_bnh_b32k_lam1_d768_dpc_sampled_tokbias;  K=2000; DIM=768   ;;
    d3072)      CKDIR=k2000_bnh_b32k_lam1_d3072_dpc_sampled_tokbias; K=2000; DIM=3072  ;;
    d6144)      CKDIR=k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias; K=2000; DIM=6144  ;;
    d12288)     CKDIR=k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias;K=2000; DIM=12288 ;;
    d6144k4000) CKDIR=k4000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias; K=4000; DIM=6144  ;;
    d12288k4000) CKDIR=k4000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias; K=4000; DIM=12288 ;;
    *)
        echo "usage: $0 <d768|d3072|d6144|d12288|d6144k4000|d12288k4000> <steps|groups...>" >&2
        echo "  steps: tokstruct probe mmlu cq number steer range_db14 range_bios number_hb csteer cgen cjudge cgen_concept cjudge_concept guard" >&2
        echo "  groups: evals interventions steering concept judge all" >&2
        exit 2
        ;;
esac
TB=$B/$CKDIR/step_0014450.pt
KM=$DELTA/balanced_kmeans_k${K}_dpc_sampled.npz
KMT=$DELTA/balanced_kmeans_k${K}_dpc_sampled_tokmean.npz
TAG=$ARM
ARMD=results/delta/$ARM
EVALSD=$ARMD/evals
INTERVD=$ARMD/interventions
STEERD=$ARMD/steering
CONCEPTD=$ARMD/concept
FIGD=$ARMD/figures
LOGD=$ARMD/logs
JCD=results/delta/joint_correct
mkdir -p "$EVALSD" "$INTERVD" "$STEERD" "$CONCEPTD" "$FIGD" \
         "$LOGD/evals" "$LOGD/interventions" "$LOGD/steering" "$LOGD/concept" "$JCD"
JC_DB14=$JCD/joint_correct_db14_l27_$TAG.json
JC_BIOS=$JCD/joint_correct_biasbios_l27_$TAG.json
BIOS_CONCEPTS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27  # no "teacher" (0% few-shot)

if [ $# -eq 0 ]; then
    sed -n '1,38p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

# --------------------------------------------------------------------------------- #
# Steps and aliases
# --------------------------------------------------------------------------------- #
ATOMIC=(tokstruct probe mmlu cq number steer range_db14 range_bios number_hb csteer cgen cjudge cgen_concept cjudge_concept guard)
EVALS=(tokstruct probe mmlu cq)
INTERVENTIONS=(number steer range_db14 range_bios number_hb)
STEERING=(csteer cgen)
CONCEPT=(cgen_concept)
JUDGE=(cjudge cjudge_concept)
GUARD_STEPS=(tokstruct probe mmlu cq number steer range_db14 range_bios number_hb csteer cgen cjudge cgen_concept cjudge_concept)

expand() {
    local out=()
    for s in "$@"; do
        case "$s" in
            evals)         out+=("${EVALS[@]}") ;;
            interventions) out+=("${INTERVENTIONS[@]}") ;;
            steering)      out+=("${STEERING[@]}") ;;
            concept)       out+=("${CONCEPT[@]}") ;;
            judge)         out+=("${JUDGE[@]}") ;;
            all)           out+=("${EVALS[@]}" "${INTERVENTIONS[@]}" "${STEERING[@]}" "${CONCEPT[@]}" "${JUDGE[@]}") ;;
            *)
                local ok=0
                for a in "${ATOMIC[@]}"; do [ "$a" = "$s" ] && ok=1 && break; done
                if [ "$ok" -eq 1 ]; then
                    out+=("$s")
                else
                    echo "[abort] unknown step '$s' (have: ${ATOMIC[*]}, or aliases: evals interventions steering concept judge all)" >&2
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
step() {   # step <name> <done-check-string, bash -c'd> <function>
    want "$1" || return
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
    "$PY" - "$TB" "$DIM" "$K" <<'EOF'
import sys
import torch

path, want_dim, want_k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
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
sys.exit(0 if ok else 1)
EOF
}
# Under DRY these are no-ops (return success) so the intended command still prints
# with its resolved paths, even before eval_out/run_delta_prep.sh has produced KM/KMT
# or cache_ids/. In a real (non-DRY) run they abort the step with a clear message.
require_km()  { [ -n "$DRY" ] && return 0; [ -f "$KM" ]  || { echo "[abort] $KM missing — run eval_out/run_delta_prep.sh km (K=$K)"; return 1; }; }
require_kmt() { [ -n "$DRY" ] && return 0; [ -f "$KMT" ] || { echo "[abort] $KMT missing — run eval_out/run_delta_prep.sh km (K=$K)"; return 1; }; }
require_cache_ids() {
    [ -n "$DRY" ] && return 0
    [ -d cache_ids ] || { echo "[abort] cache_ids/ missing — run eval_out/run_delta_prep.sh bench cache"; return 1; }
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
        --activations_dir "$ACT" \
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
        "$PY" -u eval_out/probe_chance_corrected.py --cache cache_ids \
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
        --activations_dir "$ACT" --layer 27 --n_sample 1000000 --seed 0 \
        --out "$EVALSD/cq.json"
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
        echo "[dry ] $PY -u -m geoae.interp.cluster_steering --arms $ARMS --activations_dir $ACT --out $STEERD/cluster_steering.json | tee $LOGD/steering/cluster_steering.log"
        echo "[dry ] $PY -u -m geoae.interp.cluster_steering --arms $ARMS --activations_dir $ACT --hub_x 1e9 --out $STEERD/cluster_steering_allhubs.json | tee $LOGD/steering/cluster_steering_allhubs.log"
        return 0
    fi
    "$PY" -u -m geoae.interp.cluster_steering --arms "$ARMS" --activations_dir "$ACT" \
        --out "$STEERD/cluster_steering.json" 2>&1 | tee "$LOGD/steering/cluster_steering.log" || return 1
    "$PY" -u -m geoae.interp.cluster_steering --arms "$ARMS" --activations_dir "$ACT" --hub_x 1e9 \
        --out "$STEERD/cluster_steering_allhubs.json" 2>&1 | tee "$LOGD/steering/cluster_steering_allhubs.log"
    # side file, alongside each --out: <out>.rows.npz (Path.with_suffix), already
    # inside results/delta/<arm>/steering/ since it derives from --out.
}
cgen() {
    require_km || return 1
    X "$LOGD/steering/cluster_steer_generate.log" -- \
        "$PY" -u -m geoae.interp.cluster_steer_generate --base "$KM" --bypass "$TB" \
        --activations_dir "$ACT" \
        --out "$STEERD/cluster_steer_generate.json"
}
cjudge() {
    [ -f "$STEERD/cluster_steer_generate.json" ] || { [ -n "$DRY" ] || { echo "[abort] run cgen first"; return 1; }; }
    if [ -n "$DRY" ]; then
        echo "[dry ] (source .env; require OPENROUTER_API_KEY)"
        echo "[dry ] $PY -u -m geoae.interp.cluster_steer_judge --gen $STEERD/cluster_steer_generate.json --provider openrouter --model google/gemini-2.5-flash-lite --out $STEERD/cluster_steer_judge.json | tee $LOGD/steering/cluster_steer_judge.log"
        return 0
    fi
    set -a; [ -f .env ] && source .env; set +a
    [ -n "$OPENROUTER_API_KEY" ] || { echo "[abort] OPENROUTER_API_KEY not set (.env)"; return 1; }
    "$PY" -u -m geoae.interp.cluster_steer_judge --gen "$STEERD/cluster_steer_generate.json" \
        --provider openrouter --model google/gemini-2.5-flash-lite \
        --out "$STEERD/cluster_steer_judge.json" 2>&1 | tee "$LOGD/steering/cluster_steer_judge.log"
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
        --activations_dir "$ACT" \
        --out "$CONCEPTD/concept_steer_generate.json"
}
cjudge_concept() {
    [ -f "$CONCEPTD/concept_steer_generate.json" ] || { [ -n "$DRY" ] || { echo "[abort] run cgen_concept first"; return 1; }; }
    if [ -n "$DRY" ]; then
        echo "[dry ] (source .env; require OPENROUTER_API_KEY)"
        echo "[dry ] $PY -u -m geoae.interp.concept_steer_judge --gen $CONCEPTD/concept_steer_generate.json --provider openrouter --model google/gemini-2.5-flash-lite --out $CONCEPTD/concept_steer_judge.json | tee $LOGD/concept/concept_steer_judge.log"
        return 0
    fi
    set -a; [ -f .env ] && source .env; set +a
    [ -n "$OPENROUTER_API_KEY" ] || { echo "[abort] OPENROUTER_API_KEY not set (.env)"; return 1; }
    "$PY" -u -m geoae.interp.concept_steer_judge --gen "$CONCEPTD/concept_steer_generate.json" \
        --provider openrouter --model google/gemini-2.5-flash-lite \
        --out "$CONCEPTD/concept_steer_judge.json" 2>&1 | tee "$LOGD/concept/concept_steer_judge.log"
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
step "number"         "grep -qs '\"status\": \"complete\"' $INTERVD/range_number_dprime.json && [ -f $INTERVD/number_review.json ]" number
step "steer"          "grep -qs '\"concepts\"' $INTERVD/steer_db14.json"                      steer
step "range_db14"     "grep -qs '\"summary\"' $INTERVD/range_intervention_db14_dprime.json"   range_db14
step "range_bios"     "grep -qs '\"summary\"' $INTERVD/range_intervention_biasbios_dprime.json" range_bios
step "number_hb"      "grep -qs '\"status\": \"complete\"' $INTERVD/range_number_dprime_hb.json" number_hb
step "csteer"         "[ -f $STEERD/cluster_steering.json ] && [ -f $STEERD/cluster_steering_allhubs.json ]" csteer
step "cgen"           "[ -f $STEERD/cluster_steer_generate.json ]"                            cgen
step "cjudge"         "[ -f $STEERD/cluster_steer_judge.json ]"                               cjudge
step "cgen_concept"   "[ -f $CONCEPTD/concept_steer_generate.json ]"                          cgen_concept
step "cjudge_concept" "[ -f $CONCEPTD/concept_steer_judge.json ]"                             cjudge_concept
step "guard"          "false"                                                                 guard_step

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "\n[done] FAILED steps: ${FAILED[*]}"
    exit 1
fi
echo -e "\n[done] $(date +%H:%M)"
