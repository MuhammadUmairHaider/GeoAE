#!/usr/bin/env bash
# NAMED-CONCEPT steering: several handles toward the SAME target, base vs bypass AE vs a
# supervised reference (--handles picks which; default label,base,bypass).
#   label    v_c = class mean - class-balanced grand mean (in normalised residual space,
#            fit on FIT documents of each concept), a fixed direction at every position,
#            scaled to alpha * R_c (R_c = the BASE handle's own typical edit size, so every
#            handle moves a comparable amount at a given alpha).
#   label_z  the SAME direction, computed in the BYPASS AE's latent instead (fit z on the
#            same FIT tokens, v_z = class mean - grand mean in z-space, mapped to residual
#            space via the decoder WEIGHT only: u = ae.decoder(v_z)), then rescaled and
#            edited the same way. Tests whether the AE's latent geometry carries a BETTER
#            supervised direction than the base residual does. cos(label, label_z) is
#            recorded per concept and printed per dataset (mean/min/max).
#   base     balanced k-means codebook (no encoder): the codebook's own clusters that are
#            class-c-informative on FIT tokens (not a hub, >= min_count class-c tokens from
#            >= min_docs distinct documents), ranked by CLASS-BALANCED PRECISION and kept
#            top_k; target = their GENERAL-reference member means, weighted by class-c
#            TOKEN COUNT (not by precision -- precision only ranks eligibility).
#   bypass   the same construction, with the token-bypass AE's clusters.
#   base_dir / bypass_dir
#            the SAME base/bypass cluster target mean m_c^cb, used as a CONSTANT direction
#            (x += alpha * R_c * d/||d||, no per-position cluster assignment) instead of the
#            "move into your own cluster" translation base/bypass actually apply. d_c^cb =
#            m_c^cb - mean over concepts that HAVE a cb handle of m_c'^cb. Isolates whether
#            base/bypass's advantage (if any) is the TARGET GEOMETRY or the per-position
#            assignment mechanism. cos(base_dir,label), cos(bypass_dir,label) and
#            cos(base_dir,bypass_dir) are recorded per concept, printed per dataset.
# cluster_steer_generate.py showed each codebook's OWN clusters are usable handles, but base
# and bypass steered DIFFERENT targets there. This test steers every handle toward the SAME
# 14 DBpedia-14 classes and ~28 bias_in_bios professions, so it can say which handle gets
# more fluent on-target text for a GIVEN concept -- and whether the AE beats a plain
# supervised direction fit on the very same labelled documents, in EITHER its clusters
# (base/bypass), its raw latent geometry (label vs label_z), or the cluster TARGETS used as
# directions rather than per-position translations (base_dir/bypass_dir).
#
# DATASETS=db14 (or biasbios) runs one dataset into its own files; TAG appends a further
# suffix, so a labelled sub-run never collides with the default files, e.g.:
#
#   DATASETS=db14 TAG=labelz HANDLES=label,label_z \
#       ALPHAS="0.1 0.2 0.3 0.4 0.5 0.6 0.75 1.0 1.5" ./eval_out/run_concept_steer_generate.sh
#
# writes eval_out/concept_steer_generate_db14_labelz.json / concept_steer_judge_db14_labelz.json
# and logs/concept_steer_{generate,judge}_db14_labelz.log. That invocation: 14 db14 concepts x
# 2 handles x 9 alphas = 252 generate() calls (~1-2s each measured) + the capture pass (14
# classes x 200 docs, ~1.8s/class, PLUS the label_z bypass-encode pass since label_z is
# requested) + reference stats -> budget ~10-15 min GPU. Judge: the alpha-0 prompt (unsteered
# text + options) is IDENTICAL across every handle, so concept_steer_judge.py dedupes prompt
# strings before calling the LLM: ~n_concepts x n_prompts x (n_handles x n_alphas + 1) UNIQUE
# calls to google/gemini-2.5-flash-lite -- for this invocation ~14 x 20 x (2x9 + 1) = ~5,320
# calls, well under $1; for the DEFAULT full run (both datasets, label/base/bypass, 9 alphas)
# ~42 x 20 x (3x9 + 1) = ~23,500 calls, a few $, well under $5. Answers are disk-cached, so
# re-running is free. FORCE=1 redoes a step.
#
#   DATASETS=db14 TAG=clusterdir HANDLES=label,base,bypass,base_dir,bypass_dir \
#       ALPHAS="0.1 0.2 0.3 0.4 0.5 0.6 0.75 1.0 1.5" ./eval_out/run_concept_steer_generate.sh
#
# writes eval_out/concept_steer_generate_db14_clusterdir.json / _judge_db14_clusterdir.json.
# 14 concepts x 5 handles x 9 alphas = 630 generate() calls (no bypass-encode pass -- label_z
# is not requested) -> budget ~20-25 min GPU. Judge: ~14 x 20 x (5x9 + 1) = ~12,880 unique
# calls -- but 7 of these 9 alphas (0.1,0.2,0.3,0.5,0.75,1.0,1.5) match the earlier full db14
# judge run's grid, and label/base/bypass generate the SAME continuations given the same
# checkpoints/seed, so those (concept, prompt, alpha, handle) prompts for label/base/bypass
# are already in the disk cache and cost nothing; the real new API spend is mostly
# base_dir/bypass_dir (2 new handles) plus the 2 new alpha values (0.4, 0.6) for every handle.
#
#   ./eval_out/run_concept_steer_generate.sh              # gen (~30-45 min GPU) then judge (API)
#   ./eval_out/run_concept_steer_generate.sh gen
#   ./eval_out/run_concept_steer_generate.sh judge         # needs OPENROUTER_API_KEY (from .env)
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE
PY=.venv/bin/python
BASE=e2e/checkpoints/general/llama3.2-3B/layer27/balanced_kmeans_k2000_dpc_sampled.npz
BYPASS=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt
# DATASETS=db14 (or biasbios) runs one dataset into its own files, e.g. concept_steer_generate_db14.json
DATASETS=${DATASETS:-db14,biasbios}
HANDLES=${HANDLES:-label,base,bypass}
ALPHAS=${ALPHAS:-}
TAG=${TAG:-}
SUF=$([ "$DATASETS" = db14,biasbios ] || echo "_${DATASETS//,/_}")
[ -n "$TAG" ] && SUF="${SUF}_${TAG}"
GEN=eval_out/concept_steer_generate$SUF.json
JUDGE=eval_out/concept_steer_judge$SUF.json

GEN_ARGS=(--datasets "$DATASETS" --handles "$HANDLES")
[ -n "$ALPHAS" ] && GEN_ARGS+=(--alphas $ALPHAS)

STEPS=("$@"); [ $# -eq 0 ] && STEPS=(gen judge)
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
have() { [ -z "$FORCE" ] && [ -f "$1" ] && echo "[skip] $1 exists" && return 0; return 1; }

if want gen && ! have $GEN; then
    $PY -u -m geoae.interp.concept_steer_generate --base $BASE --bypass $BYPASS "${GEN_ARGS[@]}" \
        --out $GEN 2>&1 | tee logs/concept_steer_generate$SUF.log || exit 1
fi
if want judge && ! have $JUDGE; then
    [ -f $GEN ] || { echo "[abort] run the gen step first"; exit 1; }
    set -a; [ -f .env ] && source .env; set +a
    [ -n "$OPENROUTER_API_KEY" ] || { echo "[abort] OPENROUTER_API_KEY not set (.env)"; exit 1; }
    $PY -u -m geoae.interp.concept_steer_judge --gen $GEN --provider openrouter \
        --model google/gemini-2.5-flash-lite --out $JUDGE 2>&1 | tee logs/concept_steer_judge$SUF.log || exit 1
fi
