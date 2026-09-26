#!/usr/bin/env bash
# DESIGN B, 4x WIDTH — TOKEN-BYPASS AE at L27, d12288, sampled dump.
#
# The encoder sees x - b[current token], the reconstruction is decoder(z) + b[current
# token], b = shrunk per-token mean (geoae/token_bias.py). Parents:
#   bypass6k (d6144 tokbias)   same bypass recipe at 2x width — the results here compare
#                               HEAD TO HEAD against it (does bypass generalise to 4x?)
#   d12288_dpc_sampled         same width, same dump, WITHOUT the token bypass
# Config: configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias.yaml
#
# THE GPU IS FULLY USED DURING TRAINING (~17.5 h). Run `train` on its own, then run the
# eval groups afterwards — do not try to overlap them with training on the same card.
#
#   ./eval_out/run_tokbias_d12288.sh train              # ~17.5 h (~20 min/epoch x 50); resumable
#   ./eval_out/run_tokbias_d12288.sh evals               # ~1.5 h  (tokstruct probe mmlu cq)
#   ./eval_out/run_tokbias_d12288.sh interventions        # ~6 h    (number steer range_db14 range_bios number_hb)
#   ./eval_out/run_tokbias_d12288.sh steering             # ~1.5 h  (csteer cgen cjudge; ~10 min is the API judge, a
#                                                          #          few cents — base answers are cache-shared)
#   ./eval_out/run_tokbias_d12288.sh all                  # evals + interventions + steering + summary (NOT train)
#   ./eval_out/run_tokbias_d12288.sh summary              # just reprint the decision tables
#
# Individual steps also work directly, e.g.:
#   ./eval_out/run_tokbias_d12288.sh cq range_db14
#
# Steps: train | tokstruct probe mmlu cq | number steer range_db14 range_bios number_hb |
#        csteer cgen cjudge | summary
# Group aliases: evals interventions steering all (all never implies train — training is
# 17.5 h and must be started explicitly).
#
# A checkpoint GUARD runs automatically before any eval/intervention/steering step (never
# before train): it loads both TB12 and TB6 on CPU and refuses to score an AE whose
# centroids were never initialised, whose width is wrong, or that is missing its
# token-bias table. Finished steps are skipped; FORCE=1 redoes one. A failing step is
# reported at the end (FAILED), not left silent.
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE
PY=.venv/bin/python
E=e2e/checkpoints/general/llama3.2-3B/layer27
B=checkpoints/llama3.2-3B/layer27
CFG=configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias.yaml
CKDIR=$B/k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias
TB12=$CKDIR/step_0014450.pt                                              # epoch 50: 289 steps/epoch
TB6=$B/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt      # epoch 50, the 2x-width reference
TABLE=$E/token_bias_sampled.npz
KM_NEW=$E/balanced_kmeans_k2000_dpc_sampled.npz            # "base": balanced k-means on the plain residual
KM_TOKMEAN=$E/balanced_kmeans_k2000_dpc_sampled_tokmean.npz
TAG=d12288_tokbias
JC_DB14=dbpedia/joint_correct_db14_l27_$TAG.json
JC_BIOS=dbpedia/joint_correct_biasbios_l27_$TAG.json
BIOS_CONCEPTS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27  # no "teacher" (0% few-shot)

# --------------------------------------------------------------------------------- #
# Steps and aliases
# --------------------------------------------------------------------------------- #
ATOMIC=(train tokstruct probe mmlu cq number steer range_db14 range_bios number_hb csteer cgen cjudge summary)
EVALS=(tokstruct probe mmlu cq)
INTERVENTIONS=(number steer range_db14 range_bios number_hb)
STEERING=(csteer cgen cjudge)
GUARD_STEPS=(tokstruct probe mmlu cq number steer range_db14 range_bios number_hb csteer cgen cjudge)

if [ $# -eq 0 ]; then
    sed -n '1,34p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

expand() {
    local out=()
    for s in "$@"; do
        case "$s" in
            evals)         out+=("${EVALS[@]}") ;;
            interventions) out+=("${INTERVENTIONS[@]}") ;;
            steering)      out+=("${STEERING[@]}") ;;
            all)           out+=("${EVALS[@]}" "${INTERVENTIONS[@]}" "${STEERING[@]}" summary) ;;
            *)
                local ok=0
                for a in "${ATOMIC[@]}"; do [ "$a" = "$s" ] && ok=1 && break; done
                if [ "$ok" -eq 1 ]; then
                    out+=("$s")
                else
                    echo "[abort] unknown step '$s' (have: ${ATOMIC[*]}, or aliases: evals interventions steering all)" >&2
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
    if [ -z "$FORCE" ] && eval "$2"; then echo "[skip] $1 — already done (FORCE=1 to redo)"; return; fi
    echo -e "\n[run ] $1  ($(date +%H:%M))"
    "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }
}

# --------------------------------------------------------------------------------- #
# Checkpoint guard — runs once, before any requested eval/intervention/steering step.
# Mirrors eval_out/run_d12288_evals.sh's guard, extended to also require the
# token-bias table and check BOTH widths are what they claim to be.
# --------------------------------------------------------------------------------- #
guard() {
    [ -f "$TB12" ] || { echo "[abort] $TB12 missing — training not at epoch 50 (or not started)"; return 1; }
    [ -f "$TB6" ]  || { echo "[abort] $TB6 missing — the d6144 tokbias reference checkpoint is required for comparison"; return 1; }
    "$PY" - "$TB12" 12288 "$TB6" 6144 <<'EOF'
import sys
import torch

pairs = [(sys.argv[1], int(sys.argv[2])), (sys.argv[3], int(sys.argv[4]))]
ok = True
for path, want_dim in pairs:
    c = torch.load(path, map_location="cpu", weights_only=False)
    epoch = c.get("epoch")
    val_mse = c.get("val_mse")
    dim = c["config"]["model"].get("latent_dim")
    init = bool(c["model_state"].get("centroids_initialized", False))
    has_tb = "tb_table" in c["model_state"]
    print(f"[guard] {path}: epoch {epoch}, val_mse {val_mse}, latent_dim {dim}, "
          f"centroids_initialized {init}, token_bias_table {has_tb}")
    if epoch != 50 or not init or dim != want_dim or not has_tb:
        print(f"[guard] FAIL — expected epoch 50, centroids_initialized True, "
              f"latent_dim {want_dim}, a token-bias table present")
        ok = False
sys.exit(0 if ok else 1)
EOF
}
NEED_GUARD=0
for s in "${GUARD_STEPS[@]}"; do want "$s" && NEED_GUARD=1; done
if [ "$NEED_GUARD" -eq 1 ]; then
    guard || { echo "[abort] checkpoint guard failed"; exit 1; }
fi

# --------------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------------- #
train() {
    [ -f "$TABLE" ] || { echo "[abort] $TABLE missing — build it via eval_out/run_tokbias.sh table"; return 1; }
    local avail
    avail=$(df --output=avail -BG . 2>/dev/null | tail -1 | tr -dc '0-9')
    echo "[train] free disk on $(pwd): ${avail} GB"
    if [ -z "$avail" ] || [ "$avail" -lt 20 ]; then
        echo "[abort] only ${avail:-?} GB free (< 20 GB) — free disk before starting a 12-checkpoint x ~1.25 GB run"
        return 1
    fi
    resume=()
    ls "$CKDIR"/step_*.pt >/dev/null 2>&1 && resume=(--resume latest)
    "$PY" -u -m geoae.train --config "$CFG" "${resume[@]}" 2>&1 \
        | tee -a logs/llama_l27_d12288_dpc_sampled_tokbias.log
}

# --------------------------------------------------------------------------------- #
# evals
# --------------------------------------------------------------------------------- #
tokstruct() {
    "$PY" -u eval_out/token_structure.py \
        --arms "base=$KM_NEW,km_tokmean=$KM_TOKMEAN,bypass6k=$TB6,bypass12k=$TB12" \
        --out eval_out/token_structure_tokbias12k.json 2>&1 | tee logs/token_structure_tokbias12k.log
}
probe() {
    "$PY" -u eval_out/probe_chance_corrected.py --cache cache_ids \
        --models "bypass6k=$TB6,bypass12k=$TB12" --baselines "base=$KM_NEW,km_tokmean=$KM_TOKMEAN" \
        --out eval_out/probe_tokbias12k.json 2>&1 | tee logs/probe_tokbias12k.log
}
mmlu() {
    "$PY" -u -m geoae.interp.mmlu_splice --checkpoint "$TB12" \
        --out eval_out/mmlu_d12288_sampled_tokbias.json 2>&1 | tee logs/mmlu_d12288_sampled_tokbias.log
}
cq() {
    local avail
    avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    echo "[cq] MemAvailable ${avail} GB; d12288 latents need ~49 GB (1M rows x 12288 x fp32)"
    [ "$avail" -lt 60 ] && echo "[cq] WARNING: under 60 GB free — stop other jobs or drop --n_sample to 500000"
    "$PY" -u -m geoae.interp.clustering_quality --baseline "$KM_NEW" --checkpoints "$TB6" "$TB12" \
        --names base bypass6k bypass12k \
        --activations_dir activations_sampled_10M --layer 27 --n_sample 1000000 --seed 0 \
        --out eval_out/cq_tokbias12k.json 2>&1 | tee logs/cq_tokbias12k.log
}

# --------------------------------------------------------------------------------- #
# interventions (identical settings to eval_out/run_tokbias_interventions.sh)
# --------------------------------------------------------------------------------- #
number() {
    NUMBER_CHECKPOINT="$TB12" NUMBER_OUTPUT=results/range_number_${TAG}_dprime.json \
        bash eval_out/run_number_control.sh --saliency dprime | tee logs/range_number_${TAG}_dprime.log || return 1
    "$PY" eval_out/review_number_control.py --source results/range_number_${TAG}_dprime.json \
        --review_out eval_out/number_${TAG}_review.json --figure_prefix figures/number_control/${TAG}_review
}
steer() {
    "$PY" -u -m geoae.interp.steering_concept_compare --checkpoint "$TB12" --dataset db14 \
        --correct_json "$JC_DB14" --out results/steer_db14_$TAG.json | tee logs/steer_db14_$TAG.log
}
range_db14() {
    "$PY" -u -m geoae.interp.range_intervention_compare --checkpoint "$TB12" --dataset db14 \
        --correct_json "$JC_DB14" --saliency dprime --tao 2.0 \
        --out results/range_intervention_db14_${TAG}_dprime.json | tee logs/range_db14_${TAG}_dprime.log
}
range_bios() {
    "$PY" -u -m geoae.interp.range_intervention_compare --checkpoint "$TB12" --dataset biasbios \
        --concepts "$BIOS_CONCEPTS" --correct_json "$JC_BIOS" --saliency dprime --tao 2.0 --alphas 1.0 2.0 \
        --out results/range_intervention_biasbios_${TAG}_dprime.json | tee logs/range_biasbios_${TAG}_dprime.log
}
# Matched base arm (hb): edit h - b[tok], add b[tok] back. The tool refuses to overwrite an
# existing output, and an interrupted run leaves an INCOMPLETE file behind, so it is
# deleted here first (this only runs when step() has decided the file is not yet complete).
number_hb() {
    rm -f results/range_number_${TAG}_dprime_hb.json
    NUMBER_CHECKPOINT="$TB12" NUMBER_OUTPUT=results/range_number_${TAG}_dprime_hb.json \
        bash eval_out/run_number_control.sh --saliency dprime --hb_table "$TABLE" \
        | tee logs/range_number_${TAG}_dprime_hb.log
}

# --------------------------------------------------------------------------------- #
# steering (next-token cluster steering + generation + LLM judge)
# --------------------------------------------------------------------------------- #
csteer() {
    local ARMS="base=$KM_NEW,km_tokmean=$KM_TOKMEAN,bypass6k=$TB6,bypass12k=$TB12"
    "$PY" -u -m geoae.interp.cluster_steering --arms "$ARMS" \
        --out eval_out/cluster_steering_tokbias12k.json 2>&1 | tee logs/cluster_steering_tokbias12k.log || return 1
    "$PY" -u -m geoae.interp.cluster_steering --arms "$ARMS" --hub_x 1e9 \
        --out eval_out/cluster_steering_tokbias12k_allhubs.json 2>&1 | tee logs/cluster_steering_tokbias12k_allhubs.log
}
cgen() {
    "$PY" -u -m geoae.interp.cluster_steer_generate --base "$KM_NEW" --bypass "$TB12" \
        --out eval_out/cluster_steer_generate_v2_tokbias12k.json 2>&1 | tee logs/cluster_steer_generate_v2_tokbias12k.log
}
cjudge() {
    [ -f eval_out/cluster_steer_generate_v2_tokbias12k.json ] || { echo "[abort] run cgen first"; return 1; }
    set -a; [ -f .env ] && source .env; set +a
    [ -n "$OPENROUTER_API_KEY" ] || { echo "[abort] OPENROUTER_API_KEY not set (.env)"; return 1; }
    "$PY" -u -m geoae.interp.cluster_steer_judge --gen eval_out/cluster_steer_generate_v2_tokbias12k.json \
        --provider openrouter --model google/gemini-2.5-flash-lite \
        --out eval_out/cluster_steer_judge_v2_tokbias12k.json 2>&1 | tee logs/cluster_steer_judge_v2_tokbias12k.log
}

summary() { "$PY" eval_out/summarize_tokbias_d12288.py | tee logs/summary_tokbias_d12288.log; }

# --------------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------------- #
step "train"       "[ -f $CKDIR/step_0014450.pt ]"                                                        train
step "tokstruct"   "[ -f eval_out/token_structure_tokbias12k.json ]"                                      tokstruct
step "probe"       "[ -f eval_out/probe_tokbias12k.json ]"                                                probe
step "mmlu"        "[ -f eval_out/mmlu_d12288_sampled_tokbias.json ]"                                     mmlu
step "cq"          "[ -f eval_out/cq_tokbias12k.json ]"                                                   cq
step "number"      "grep -qs '\"status\": \"complete\"' results/range_number_${TAG}_dprime.json && [ -f eval_out/number_${TAG}_review.json ]" number
step "steer"       "grep -qs '\"concepts\"' results/steer_db14_$TAG.json"                                 steer
step "range_db14"  "grep -qs '\"summary\"' results/range_intervention_db14_${TAG}_dprime.json"            range_db14
step "range_bios"  "grep -qs '\"summary\"' results/range_intervention_biasbios_${TAG}_dprime.json"        range_bios
step "number_hb"   "grep -qs '\"status\": \"complete\"' results/range_number_${TAG}_dprime_hb.json"       number_hb
step "csteer"      "[ -f eval_out/cluster_steering_tokbias12k.json ] && [ -f eval_out/cluster_steering_tokbias12k_allhubs.json ]" csteer
step "cgen"        "[ -f eval_out/cluster_steer_generate_v2_tokbias12k.json ]"                             cgen
step "cjudge"      "[ -f eval_out/cluster_steer_judge_v2_tokbias12k.json ]"                                cjudge
want summary && summary

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "\n[done] FAILED steps: ${FAILED[*]}"
    exit 1
fi
echo -e "\n[done] $(date +%H:%M)"
