#!/usr/bin/env bash
# INTERPRETATION evals for the delta token-bypass AEs: can an LLM NAME each codebook's clusters, and what
# KIND of concept are they? Per arm: "base" = balanced k-means with the same K on the normalised residual
# (KM, as in experiments/paper/scripts/run_delta_evals.sh), "bypass" = the token-bypass AE (its epoch-50 checkpoint). The arm
# table below is a verbatim copy of run_delta_evals.sh's: keep the two in step.
#
#   experiments/paper/scripts/run_interp.sh <arm> <steps|groups...>
#   arm: d768 | d3072 | d6144 | d12288 | d6144k4000 | d12288k4000     (Llama-3.2-3B, layer 27)
#        d6144decoderonly                                        (full-input encoder, decoder bias)
#        g1b_l{13,25}_d{1152,2304,4608}                            (Gemma 3 1B, K=2000)
#
#   experiments/paper/scripts/run_interp.sh d6144 judge judge3 atlas      # login node, ONE ARM AT A TIME (see judge)
#   DRY=1 experiments/paper/scripts/run_interp.sh d6144 wsd_data ct_base ct wsd judge judge3 atlas   # print, run nothing
#   FORCE=1 experiments/paper/scripts/run_interp.sh d6144 ct              # redo one step
#
# Steps (always run in this order, whatever order they are typed in)
#   wsd_data  login node  download CoarseWSD-20 + WiC: geoae.analysis.word_sense --download_only -> data/wsd/
#                         (done = data/wsd/download_manifest.json exists). Once, for any arm.
#   ct_base   GPU         closest_tokens of the BASE codebook on held-out docs of the arm's own dump
#                         (--sampled_from, +-24-token passages, doc stats, hub flag, 500k tokens)
#                         -> results/delta/base_<base npz stem>/interp/closest_tokens.json
#                         Shared by every arm with the same base npz. Gemma: base_g1b_l<L>_<stem>, because
#                         the stem is the same at layers 13 and 25 and the codebooks are not.
#   ct_tok    GPU         the same for the TOKEN-MEAN control (balanced k-means on x - b[token], i.e. the bypass
#                         AE without its encoder) -> results/delta/base_<tokmean npz stem>/interp/closest_tokens.json
#                         Shared like ct_base (Gemma: base_g1b_l<L>_<stem>).
#   ct        GPU         the same for the bypass AE -> results/delta/<arm>/interp/closest_tokens.json
#   wsd       GPU         geoae.analysis.word_sense: do clusters separate the senses one token carries
#                         (CoarseWSD-20, WiC), base vs bypass (+ token-mean control)
#                         -> results/delta/<arm>/interp/word_sense.json
#   judge     login node  llm_judge on the base and the bypass closest_tokens (gemini-2.5-flash-lite via
#                         OpenRouter: explanation, level, coherence, detection, intrusion; up to 2000
#                         clusters each) -> .../interp/judge_flash_lite.json. API only. Run SERIALLY across
#                         arms to avoid redundant calls for shared base prompts; the cache is atomic.
#   judge_tok login node  llm_judge on the token-mean closest_tokens ALONE, same model and settings as judge
#                         -> results/delta/base_<tokmean stem>/interp/judge_flash_lite.json (shared). Alone because
#                         llm_judge draws every run's samples from one RNG: judged next to the bypass file it would
#                         re-sample the bypass clusters instead of reusing the judgments atlas compares against.
#   judge3    login node  judge_agreement on the same two files: 3 judges x 200 clusters, Kendall's W
#                         -> .../interp/judge_agreement.json (keys closest_tokens_base/_bypass.json). API only.
#   atlas     login node  geoae.analysis.atlas_summary: share of the text in clusters that are nameable, and
#                         nameable as a CONCEPT rather than a string, level mix, bypass - base with bootstrap
#                         CIs, browsable atlas_{base,bypass}.json -> .../interp/atlas_{summary,base,bypass}.json
#   pair      login node  geoae.analysis.pairwise_judge: head-to-head, blind A/B pairs of clusters drawn by usage
#                         from base, token-mean control and bypass (every pairing, both orders, + real-vs-fake
#                         controls): which is the clearer concept, which is more about meaning, per-cluster
#                         coherence / abstraction 1-10 and level -> .../interp/pairwise_<model>.json. API only.
#                         PAIR_MODEL (default anthropic/claude-sonnet-5.5, ~$4/arm at the defaults), PAIR_PAIRS
#                         (100 per pairing), PAIR_CONTROLS (25 per space). Check the controls line first: a judge
#                         whose real clusters do not beat fake ones ~95%+ of the time is too weak to read.
#   pair_export           the same items as blind batch files for Claude agent judges (no API spend):
#                         .../interp/agent_judge/{batches/batch_NN.md, key.json}. The key names the spaces, so
#                         never show it to a judge. Agent prompt: eval_out/agent_judge_prompt.md.
#   pair_import           score the agents' .../agent_judge/answers/answers_*.jsonl against the key
#                         -> .../interp/pairwise_agents.json
#   atlas_tok login node  atlas_summary with the token-mean control as "base": does the ENCODER change what is
#                         nameable, beyond subtracting b[token]? -> .../interp/vs_tokmean/atlas_*.json
# Groups: all = gpu = ct_base ct_tok ct wsd (the GPU steps ONLY)    api = judge judge_tok judge3 atlas atlas_tok pair
#
# Where things run. ct_base / ct / wsd need a GPU (LM forward passes: ~6M tokens per closest_tokens run).
# One job per arm:
#   sbatch scripts/slurm/delta_sheet.sbatch experiments/paper/scripts/run_interp.sh d6144 ct_base ct wsd
# Arms sharing a base (d768 d3072 d6144 d12288; d6144k4000 d12288k4000; each Gemma layer) need ct_base ONCE:
# give it to one arm's job and the others only `ct wsd` (or chain them with --dependency=afterok:<jobid>).
# Two jobs that both start before the shared file exists would each recompute it and write the same path.
# These steps read only <dump>/meta.json, never layer_*.npy, so plain sheet.sbatch is right: no staging, no ACT_DIR.
# wsd_data needs the network. wsd_data / judge / judge3 / atlas run from the login node (judge, judge3 and atlas
# are API / CPU-light; threads are capped so the login-node CPU-time limit is not hit):
#   bash experiments/paper/scripts/run_interp.sh d6144 wsd_data
#   bash experiments/paper/scripts/run_interp.sh d6144 judge judge3 atlas
#
# Two things about the tools reused here:
#  * judge / judge3 need llm_judge.load_centroids to read a token-bypass checkpoint (its hard negatives come from
#    the centroids named in each closest_tokens file's meta.checkpoint). load_ae_checkpoint refuses such a
#    checkpoint unless allow_token_bias=True is passed; llm_judge.py passes it as of 2026-10-01. An older copy
#    dies with a SystemExit "... is a TOKEN-BYPASS AE" only after the base run's API calls are spent.
#  * judge_agreement keys its output by the file's basename and both inputs are called closest_tokens.json, so
#    the second would silently overwrite the first. judge3 passes it two symlinks with distinct names instead.
# Finished steps are skipped (FORCE=1 redoes one); a failing step is reported at the end (FAILED).
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -o pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"   # before env.sh cds
# Delta: env.sh resolves GEOAE_ROOT from its own location, exports HF_HOME / UV_PROJECT_ENVIRONMENT /
# SBATCH_ACCOUNT, and cds to the repo root.
source "$GEOAE_ROOT/scripts/env.sh"
PY=scripts/py

# --------------------------------------------------------------------------------- #
# Arm table: copied from experiments/paper/scripts/run_delta_evals.sh (CKDIR, K, DIM, FAM, LAYER, E/B/DELTA, TB, KM, KMT)
# --------------------------------------------------------------------------------- #
E=e2e/checkpoints/general/llama3.2-3B/layer27
B=checkpoints/llama3.2-3B/layer27
DELTA=$E/delta
DUMP=activations_sampled_10M            # closest_tokens --sampled_from / word_sense --act_meta read its meta.json only
FAM=llama; LAYER=27

ARM="${1:-}"; shift || true
case "$ARM" in
    d768)       CKDIR=k2000_bnh_b32k_lam1_d768_dpc_sampled_tokbias;  K=2000; DIM=768   ;;
    d3072)      CKDIR=k2000_bnh_b32k_lam1_d3072_dpc_sampled_tokbias; K=2000; DIM=3072  ;;
    d6144)      CKDIR=k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias; K=2000; DIM=6144  ;;
    d6144decoderonly) CKDIR=k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly; K=2000; DIM=6144 ;;
    d12288)     CKDIR=k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias;K=2000; DIM=12288 ;;
    d6144k4000) CKDIR=k4000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias; K=4000; DIM=6144  ;;
    d12288k4000) CKDIR=k4000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias; K=4000; DIM=12288 ;;
    g1b_l13_d1152|g1b_l13_d2304|g1b_l13_d4608|g1b_l25_d1152|g1b_l25_d2304|g1b_l25_d4608)
        FAM=g1b; K=2000; DIM=${ARM##*_d}; LAYER=${ARM#g1b_l}; LAYER=${LAYER%%_*}
        CKDIR=k2000_bnh_b32k_lam1_d${DIM}_dpc_sampled_struct5_tokbias ;;
    *)
        echo "usage: $0 <d768|d3072|d6144|d6144decoderonly|d12288|d6144k4000|d12288k4000|g1b_l{13,25}_d{1152,2304,4608}> <steps|groups...>" >&2
        echo "  steps: wsd_data ct_base ct_tok ct wsd judge judge_tok judge3 atlas atlas_tok     groups: all gpu api" >&2
        exit 2
        ;;
esac
if [ "$FAM" = g1b ]; then
    # Gemma 3 1B (google/gemma-3-1b-pt). Layer 25 is the LAST block.
    E=e2e/checkpoints/general/gemma3-1B/layer$LAYER
    B=checkpoints/gemma3-1B/layer$LAYER
    DELTA=$E/delta
    DUMP=activations_gemma3_1b_sampled_struct5_10M
fi
CKPT=$B/$CKDIR/step_0014450.pt          # epoch 50 (289 steps/epoch), "TB" in run_delta_evals.sh
if [ "$FAM" = g1b ]; then
    KM=$DELTA/balanced_kmeans_k${K}_dpc_sampled_struct5.npz
    KM_TOKMEAN=$DELTA/balanced_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
else
    KM=$DELTA/balanced_kmeans_k${K}_dpc_sampled.npz
    KM_TOKMEAN=$DELTA/balanced_kmeans_k${K}_dpc_sampled_tokmean.npz
fi

# --------------------------------------------------------------------------------- #
# Output tree. The base codebook depends on the base npz alone, so its closest_tokens are shared by every arm
# using that npz. The npz STEM alone is not unique for Gemma (layers 13 and 25 both say ..._struct5), so the
# Gemma directory carries the layer.
# --------------------------------------------------------------------------------- #
BASE_STEM=$(basename "$KM" .npz)
BASE_TAG=base_$BASE_STEM
[ "$FAM" = g1b ] && BASE_TAG=base_g1b_l${LAYER}_$BASE_STEM
ARMD=results/delta/$ARM
INTERPD=$ARMD/interp
LOGD=$ARMD/logs/interp
BASED=results/delta/$BASE_TAG/interp
BASE_CT=$BASED/closest_tokens.json
BYP_CT=$INTERPD/closest_tokens.json
# token-mean control: the bypass AE without its encoder, laid out like the base
TOK_TAG=base_$(basename "$KM_TOKMEAN" .npz)
[ "$FAM" = g1b ] && TOK_TAG=base_g1b_l${LAYER}_$(basename "$KM_TOKMEAN" .npz)
TOKD=results/delta/$TOK_TAG/interp
TOK_CT=$TOKD/closest_tokens.json
TOK_JUDGE_OUT=$TOKD/judge_flash_lite.json
ATLAS_TOK_DIR=$INTERPD/vs_tokmean
PAIR_MODEL=${PAIR_MODEL:-anthropic/claude-sonnet-5.5}
PAIR_OUT=$INTERPD/pairwise_$(basename "$PAIR_MODEL").json
PAIR_DIR=$INTERPD/agent_judge
WSD_DIR=data/wsd
WSD_DONE=$WSD_DIR/download_manifest.json
JUDGE_MODEL=google/gemini-2.5-flash-lite
JUDGE_OUT=$INTERPD/judge_flash_lite.json
JUDGE3_OUT=$INTERPD/judge_agreement.json

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$SELF"; }
if [ $# -eq 0 ]; then usage; exit 2; fi

# --------------------------------------------------------------------------------- #
# Steps and groups
# --------------------------------------------------------------------------------- #
ATOMIC=(wsd_data ct_base ct_tok ct wsd judge judge_tok judge3 atlas atlas_tok pair pair_export pair_import)
GPU=(ct_base ct_tok ct wsd)
API=(judge judge_tok judge3 atlas atlas_tok pair)

expand() {
    local out=()
    for s in "$@"; do
        case "$s" in
            all|gpu) out+=("${GPU[@]}") ;;
            api)     out+=("${API[@]}") ;;
            *)
                local ok=0
                for a in "${ATOMIC[@]}"; do [ "$a" = "$s" ] && ok=1 && break; done
                if [ "$ok" -eq 1 ]; then
                    out+=("$s")
                else
                    echo "[abort] unknown step '$s' (have: ${ATOMIC[*]}, or groups: all gpu api)" >&2
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
step() {   # step <name> <done-check-string, eval'd> <function>
    want "$1" || return
    if [ -z "$FORCE" ] && [ -z "$DRY" ] && eval "$2"; then
        echo "[skip] $1 — already done (FORCE=1 to redo)"; return
    fi
    echo -e "\n[run ] $1  ($(date +%H:%M))"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

# helper: run a command for real, or print it (DRY=1) with all paths resolved
X() {   # X <log path> -- <cmd...>     (X /path/to/log.log -- prog --flag val ...)
    local log="$1"; shift
    [ "$1" = "--" ] && shift
    if [ -n "$DRY" ]; then
        echo "[dry ] $* | tee $log"
        return 0
    fi
    mkdir -p "$(dirname "$log")"
    "$@" 2>&1 | tee "$log"
}
mk() { [ -n "$DRY" ] || mkdir -p "$@"; }

# Under DRY the require_* checks are no-ops, so the intended command still prints with its resolved paths even
# before the inputs exist. In a real run they abort the step with a clear message.
PREP_KM=km; [ "$FAM" = g1b ] && PREP_KM=g1b_km
require_km()   { [ -n "$DRY" ] && return 0; [ -f "$KM" ]         || { echo "[abort] $KM missing — run experiments/paper/scripts/run_delta_prep.sh $PREP_KM (K=$K)"; return 1; }; }
require_kmt()  { [ -n "$DRY" ] && return 0; [ -f "$KM_TOKMEAN" ] || { echo "[abort] $KM_TOKMEAN missing — run experiments/paper/scripts/run_delta_prep.sh $PREP_KM (K=$K)"; return 1; }; }
require_ckpt() { [ -n "$DRY" ] && return 0; [ -f "$CKPT" ]       || { echo "[abort] $CKPT missing — the arm has not reached epoch 50 (still training?)"; return 1; }; }
require_cts() {   # both closest_tokens files (the GPU steps ct_base / ct make them)
    [ -n "$DRY" ] && return 0
    local miss=0
    [ -f "$BASE_CT" ] || { echo "[abort] $BASE_CT missing — run: sbatch scripts/slurm/delta_sheet.sbatch experiments/paper/scripts/run_interp.sh $ARM ct_base"; miss=1; }
    [ -f "$BYP_CT" ]  || { echo "[abort] $BYP_CT missing — run: sbatch scripts/slurm/delta_sheet.sbatch experiments/paper/scripts/run_interp.sh $ARM ct"; miss=1; }
    return $miss
}

# --------------------------------------------------------------------------------- #
# GPU steps
# --------------------------------------------------------------------------------- #
wsd_data() {
    mk "$WSD_DIR" "$LOGD"
    X "$LOGD/wsd_data.log" -- \
        "$PY" -u -m geoae.analysis.word_sense --download_only --data_dir "$WSD_DIR"
}
ct_base() {
    require_km || return 1
    mk "$BASED" "$LOGD"
    X "$LOGD/ct_base.log" -- \
        "$PY" -u -m geoae.analysis.closest_tokens --baseline_kmeans "$KM" \
        --sampled_from "$DUMP" --doc_view --n_tokens 500000 --out "$BASE_CT"
}
ct_tok() {
    require_kmt || return 1
    mk "$TOKD" "$LOGD"
    X "$LOGD/ct_tok.log" -- \
        "$PY" -u -m geoae.analysis.closest_tokens --baseline_kmeans "$KM_TOKMEAN" \
        --sampled_from "$DUMP" --doc_view --n_tokens 500000 --out "$TOK_CT"
}
ct() {
    require_ckpt || return 1
    mk "$INTERPD" "$LOGD"
    X "$LOGD/ct.log" -- \
        "$PY" -u -m geoae.analysis.closest_tokens --checkpoint "$CKPT" \
        --sampled_from "$DUMP" --doc_view --n_tokens 500000 --out "$BYP_CT"
}
wsd() {
    require_km && require_kmt && require_ckpt || return 1
    [ -n "$DRY" ] || [ -f "$WSD_DONE" ] || echo "[warn] $WSD_DONE missing — the datasets were not fetched; run the wsd_data step on the login node (a compute node may have no network)"
    mk "$INTERPD" "$LOGD"
    X "$LOGD/wsd.log" -- \
        "$PY" -u -m geoae.analysis.word_sense --checkpoint "$CKPT" --baseline_kmeans "$KM" \
        --tokmean_kmeans "$KM_TOKMEAN" --datasets coarsewsd20,wic --data_dir "$WSD_DIR" \
        --act_meta "$DUMP/meta.json" --out "$INTERPD/word_sense.json"
}

# --------------------------------------------------------------------------------- #
# API / login-node steps
# --------------------------------------------------------------------------------- #
login_node_only() {   # login_node_only <step>: judge / judge3 are API-bound; a GPU job running them idles the GPU
    [ -n "${SLURM_JOB_ID:-}" ] && echo "[warn] $1 is API only and is running inside Slurm job $SLURM_JOB_ID, holding a GPU for nothing — run it from the login node"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"   # login-node CPU-time limit
    return 0
}
judge_env() {   # source .env and require the key; the key is never printed
    set -a; [ -f .env ] && source .env; set +a
    [ -n "$OPENROUTER_API_KEY" ] || { echo "[abort] OPENROUTER_API_KEY not set (.env)"; return 1; }
}
judge() {
    require_cts || return 1
    login_node_only judge
    if [ -n "$DRY" ]; then echo "[dry ] (source .env; require OPENROUTER_API_KEY)"; else judge_env || return 1; fi
    mk "$INTERPD" "$LOGD"
    X "$LOGD/judge.log" -- \
        "$PY" -u -m geoae.analysis.llm_judge "$BASE_CT" "$BYP_CT" --provider openrouter --model "$JUDGE_MODEL" \
        --n_clusters 2000 --sample stratified --min_assigned 50 --concurrency "${JUDGE_CONCURRENCY:-6}" --out "$JUDGE_OUT"
}
judge_tok() {
    [ -n "$DRY" ] || [ -f "$TOK_CT" ] || { echo "[abort] $TOK_CT missing — run the ct_tok step"; return 1; }
    login_node_only judge_tok
    if [ -n "$DRY" ]; then echo "[dry ] (source .env; require OPENROUTER_API_KEY)"; else judge_env || return 1; fi
    mk "$TOKD" "$LOGD"
    X "$LOGD/judge_tok.log" -- \
        "$PY" -u -m geoae.analysis.llm_judge "$TOK_CT" --provider openrouter --model "$JUDGE_MODEL" \
        --n_clusters 2000 --sample stratified --min_assigned 50 --concurrency "${JUDGE_CONCURRENCY:-6}" --out "$TOK_JUDGE_OUT"
}
judge3() {
    require_cts || return 1
    login_node_only judge3
    if [ -n "$DRY" ]; then echo "[dry ] (source .env; require OPENROUTER_API_KEY)"; else judge_env || return 1; fi
    mk "$INTERPD" "$LOGD"
    # judge_agreement keys its output by Path(file).name; both files are closest_tokens.json. Distinct names via symlinks.
    local JD rc
    if [ -n "$DRY" ]; then
        JD='${TMPDIR:-/tmp}/geoae_judge3.XXXXXX'
        echo "[dry ] ln -s $(realpath -m "$BASE_CT") $JD/closest_tokens_base.json"
        echo "[dry ] ln -s $(realpath -m "$BYP_CT") $JD/closest_tokens_bypass.json"
    else
        JD=$(mktemp -d "${TMPDIR:-/tmp}/geoae_judge3.XXXXXX") || return 1
        ln -s "$(realpath "$BASE_CT")" "$JD/closest_tokens_base.json"
        ln -s "$(realpath "$BYP_CT")" "$JD/closest_tokens_bypass.json"
    fi
    X "$LOGD/judge3.log" -- \
        "$PY" -u -m geoae.analysis.judge_agreement "$JD/closest_tokens_base.json" "$JD/closest_tokens_bypass.json" \
        --judges google/gemini-2.5-flash-lite,openai/gpt-4o-mini,deepseek/deepseek-chat-v3.1 \
        --n_clusters 200 --out "$JUDGE3_OUT"
    rc=$?
    [ -n "$DRY" ] || rm -rf "$JD"
    return $rc
}
atlas() {
    require_cts || return 1
    [ -n "$DRY" ] || [ -f "$JUDGE_OUT" ] || { echo "[abort] $JUDGE_OUT missing — run: bash experiments/paper/scripts/run_interp.sh $ARM judge"; return 1; }
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
    mk "$INTERPD" "$LOGD"
    X "$LOGD/atlas.log" -- \
        "$PY" -u -m geoae.analysis.atlas_summary --judge "$JUDGE_OUT" --ct_base "$BASE_CT" \
        --ct_bypass "$BYP_CT" --out_dir "$INTERPD/"
}

atlas_tok() {
    local f
    for f in "$TOK_CT" "$BYP_CT" "$JUDGE_OUT" "$TOK_JUDGE_OUT"; do
        [ -n "$DRY" ] || [ -f "$f" ] || { echo "[abort] $f missing — run ct_tok / ct / judge / judge_tok first"; return 1; }
    done
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
    mk "$ATLAS_TOK_DIR" "$LOGD"
    X "$LOGD/atlas_tok.log" -- \
        "$PY" -u -m geoae.analysis.atlas_summary --judge "$JUDGE_OUT" "$TOK_JUDGE_OUT" --ct_base "$TOK_CT" \
        --ct_bypass "$BYP_CT" --out_dir "$ATLAS_TOK_DIR/"
}

pair_args() {   # the three spaces, base first: win rates are the later space's
    echo --space "base=$BASE_CT" --space "token_removed=$TOK_CT" --space "geoae=$BYP_CT" \
        --pairs "${PAIR_PAIRS:-100}" --control_pairs "${PAIR_CONTROLS:-25}"
}
require_pair_cts() {
    local f
    for f in "$BASE_CT" "$TOK_CT" "$BYP_CT"; do
        [ -n "$DRY" ] || [ -f "$f" ] || { echo "[abort] $f missing — run ct_base / ct_tok / ct first"; return 1; }
    done
}
pair() {
    require_pair_cts || return 1
    login_node_only pair
    if [ -n "$DRY" ]; then echo "[dry ] (source .env; require OPENROUTER_API_KEY)"; else judge_env || return 1; fi
    mk "$INTERPD" "$LOGD"
    X "$LOGD/pair.log" -- \
        "$PY" -u -m geoae.analysis.pairwise_judge $(pair_args) --provider openrouter --model "$PAIR_MODEL" \
        --concurrency "${JUDGE_CONCURRENCY:-6}" --out "$PAIR_OUT"
}
pair_export() {
    require_pair_cts || return 1
    mk "$PAIR_DIR" "$LOGD"
    X "$LOGD/pair_export.log" -- \
        "$PY" -u -m geoae.analysis.pairwise_judge $(pair_args) --export "$PAIR_DIR/batches" --key "$PAIR_DIR/key.json" \
        --batch_size 75
}
pair_import() {
    [ -n "$DRY" ] || [ -f "$PAIR_DIR/key.json" ] || { echo "[abort] $PAIR_DIR/key.json missing — run pair_export"; return 1; }
    mk "$LOGD"
    X "$LOGD/pair_import.log" -- \
        "$PY" -u -m geoae.analysis.pairwise_judge --import_dir "$PAIR_DIR/answers" --key "$PAIR_DIR/key.json" \
        --out "$INTERPD/pairwise_agents.json"
}

# --------------------------------------------------------------------------------- #
# dispatch (fixed order, whatever order the steps were typed in)
# --------------------------------------------------------------------------------- #
echo "[interp] $ARM: $FAM layer $LAYER, K=$K, latent $DIM | bypass $CKPT"
echo "[interp] base $KM | token-mean control $KM_TOKMEAN"
echo "[interp] dump $DUMP | outputs $INTERPD, base shared under results/delta/$BASE_TAG/interp"

step "wsd_data" "[ -f $WSD_DONE ]"                                                        wsd_data
step "ct_base"  "[ -f $BASE_CT ]"                                                          ct_base
step "ct_tok"   "[ -f $TOK_CT ]"                                                           ct_tok
step "ct"       "[ -f $BYP_CT ]"                                                           ct
step "wsd"      "[ -f $INTERPD/word_sense.json ]"                                          wsd
step "judge"    "[ -f $JUDGE_OUT ]"                                                        judge
step "judge_tok" "[ -f $TOK_JUDGE_OUT ]"                                                   judge_tok
step "judge3"   "[ -f $JUDGE3_OUT ]"                                                       judge3
step "atlas"    "[ -f $INTERPD/atlas_summary.json ] && [ -f $INTERPD/atlas_base.json ] && [ -f $INTERPD/atlas_bypass.json ]" atlas
step "atlas_tok" "[ -f $ATLAS_TOK_DIR/atlas_summary.json ]"                                atlas_tok
step "pair"      "[ -f $PAIR_OUT ]"                                                        pair
step "pair_export" "[ -f $PAIR_DIR/key.json ]"                                             pair_export
step "pair_import" "[ -f $INTERPD/pairwise_agents.json ]"                                  pair_import

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "\n[done] FAILED steps: ${FAILED[*]}"
    exit 1
fi
echo -e "\n[done] $(date +%H:%M)"
