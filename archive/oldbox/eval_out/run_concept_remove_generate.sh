#!/usr/bin/env bash
# GENERATION-LEVEL CONCEPT REMOVAL: base residual (h) vs bypass AE latent (z).
# range_intervention_compare.py showed on DB14 (a single forced-choice decision) that
# NeuronLens range edits in the bypass latent remove the target class about as well as in the
# base residual, with ~4-5/100 LESS collateral (rm_range_comp 86/20 -> 89/15; st_range_a1
# 86/24 -> 90/20; st_transport_a1 86/28 -> 88/23). This asks whether that holds in GENERATED
# TEXT: same fit-time ranges/gates/shifts/transports, same operator in both substrates, DB14
# target + complement prompts, judged by an independent LLM forced-choice class check.
#
# --z_mode delta (default) vs splice: delta edits ONLY the difference the op introduces
# (x' = x + W_dec(edit(z)-z)), so an identity edit leaves the z-substrate's unedited text
# BYTE-IDENTICAL to h's -- the fairest comparison, since neither substrate's baseline carries
# AE reconstruction error. splice instead round-trips through the full decoder
# (x' = ae.decode(edit(z), tok)), so its own "unedited" baseline is the AE-recon splice
# (generated once per concept and judged separately) -- this is what a deployed AE-based
# intervention would actually look like, recon error included, but makes h vs z a comparison
# of "clean residual" vs "residual after an AE round trip", not of the edit alone.
#
# --pos_gate llr (default) vs none: a --diag_only run found the per-COORDINATE range gate
# (mu_c +- tao*sd_c on the salient dims) fires on ~95% of positions of EVERY class, target or
# not -- at the TOKEN level, over all non-BOS positions rather than one classification
# decision, a handful of individually-close-to-mu_c coordinates is a nearly universal
# condition, so the ops were degenerating into un-gated, almost-everywhere edits. llr adds a
# genuinely class-selective POSITION gate on top (LLR_c = sum over salient coords of
# log N(mu_c,sd_c) - log N(mu_rest,sd_rest)); none reproduces the old ungated behaviour for
# comparison -- POS_GATE=none appends _nogate to the output/log names so both variants can
# coexist on disk.
#
# GATE_RATE (--gate_target_rate, default 0.8): the LLR threshold is CALIBRATED per concept AND
# per substrate to fire on this share of class c's OWN fit positions, rather than using one
# fixed --llr_thresh for both -- h sums 922 salient coordinates' log-density terms, z sums
# 1843 (2x, since Lz=2*D here), so a raw LLR score is not on the same scale between them: a
# fixed thresh=0.0 made z's gate fire on ~3x more of its COMPLEMENT positions than h's (0.30
# vs 0.10 in a --diag_only check) even though both had target fire rate ~0.8-0.9 -- i.e. the
# substrates were not being tested at comparably strict gates. GATE_RATE<=0 falls back to the
# fixed threshold everywhere.
#
# gen: 14 db14 concepts x 2 substrates x (1 rm_range_comp + 2 ops x 3 alphas = 7 cells) = 196
# generate() calls, ~23 prompts (10 target + 1x13 complement) x 40 greedy tokens each, plus 14
# unsteered baseline calls (+14 more z-recon baselines under --z_mode splice) = ~210-224 calls
# total. Directly measured: ~1.1-2s per generate() call at 23 prompts/40 tokens, ~1.8s per
# class for the FIT capture pass (both substrates in one pass, 14 classes x 150 docs ~= 25s
# total) -> budget ~10-15 min GPU (most of it the generate() calls; model load ~2s).
# judge: every text (unsteered + 196 edited, +14 more under splice) is judged on all ~23
# prompts each -> up to (196+14)*23 ~= 4,830 raw judgements; heavy dedup since the SAME
# candidate options are reused across every substrate/op/alpha for a given prompt position,
# and many unsteered/baseline continuations repeat verbatim -> a few $, well under $5.
# Answers are disk-cached, so re-running is free. FORCE=1 redoes a step.
#
#   ./eval_out/run_concept_remove_generate.sh              # gen (~10-15 min GPU) then judge (API)
#   ./eval_out/run_concept_remove_generate.sh gen
#   ./eval_out/run_concept_remove_generate.sh judge         # needs OPENROUTER_API_KEY (from .env)
#   TAG=splice Z_MODE=splice ./eval_out/run_concept_remove_generate.sh
#   POS_GATE=none TAG=nogate ./eval_out/run_concept_remove_generate.sh gen   # old, ungated comparison
#   scripts/delta/py -m geoae.interp.concept_remove_generate --bypass $BYPASS --diag_only \
#       --out eval_out/concept_remove_diag.json                            # gate check, no generation, ~1-2 min
set -o pipefail
# Delta: env.sh resolves GEOAE_ROOT from its own location, exports HF_HOME /
# UV_PROJECT_ENVIRONMENT / SBATCH_ACCOUNT, and cds to the repo root.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/delta/env.sh"
PY=scripts/delta/py
BYPASS=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
DATASETS=${DATASETS:-db14}
Z_MODE=${Z_MODE:-delta}
POS_GATE=${POS_GATE:-llr}
GATE_RATE=${GATE_RATE:-0.8}
TAG=${TAG:-}
OPS=${OPS:-}          # e.g. OPS=st_range,st_transport ALPHAS="4 8 16" TAG=strong (strength sweep)
ALPHAS=${ALPHAS:-}
EXTRA=(); [ -n "$OPS" ] && EXTRA+=(--ops "$OPS"); [ -n "$ALPHAS" ] && EXTRA+=(--alphas $ALPHAS)
SUF=$([ "$DATASETS" = db14 ] || echo "_${DATASETS}")
[ "$POS_GATE" = none ] && SUF="${SUF}_nogate"
[ -n "$TAG" ] && SUF="${SUF}_${TAG}"
GEN=eval_out/concept_remove_generate$SUF.json
JUDGE=eval_out/concept_remove_judge$SUF.json

STEPS=("$@"); [ $# -eq 0 ] && STEPS=(gen judge)
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
have() { [ -z "$FORCE" ] && [ -f "$1" ] && echo "[skip] $1 exists" && return 0; return 1; }

if want gen && ! have $GEN; then
    $PY -u -m geoae.interp.concept_remove_generate --bypass $BYPASS --datasets "$DATASETS" \
        --z_mode "$Z_MODE" --pos_gate "$POS_GATE" --gate_target_rate "$GATE_RATE" "${EXTRA[@]}" \
        --out $GEN 2>&1 | tee logs/concept_remove_generate$SUF.log || exit 1
fi
if want judge && ! have $JUDGE; then
    [ -f $GEN ] || { echo "[abort] run the gen step first"; exit 1; }
    set -a; [ -f .env ] && source .env; set +a
    [ -n "$OPENROUTER_API_KEY" ] || { echo "[abort] OPENROUTER_API_KEY not set (.env)"; exit 1; }
    $PY -u -m geoae.interp.concept_remove_judge --gen $GEN --provider openrouter \
        --model google/gemini-2.5-flash-lite --out $JUDGE 2>&1 | tee logs/concept_remove_judge$SUF.log || exit 1
fi
