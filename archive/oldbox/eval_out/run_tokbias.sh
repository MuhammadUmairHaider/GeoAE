#!/usr/bin/env bash
# DESIGN B — TOKEN-BYPASS AE at L27, d6144, new sampled dump.
#
# The encoder sees x - b[current token], the reconstruction is decoder(z) + b[current
# token], b = shrunk per-token mean (geoae/token_bias.py). Parent / control: d6144_new
# (llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled), identical except model.token_bias.
#
#   ./eval_out/run_tokbias.sh table cache      # ~15 min, independent of training — run first
#   ./eval_out/run_tokbias.sh train            # ~9 h (10.8 min/epoch x 50); resumable, rerun after a crash
#   ./eval_out/run_tokbias.sh evals summary    # ~40 min after epoch 50
#
# Steps
#   table    per-token bias table over the train split          -> $TABLE (~300 MB)
#   cache    sequence probe caches rebuilt WITH last_token_id    -> cache_ids/ (the 09-01 caches
#            cannot be reproduced, so every arm is re-scored here); token rungs symlinked
#   train    the bypass AE                                       -> $CKDIR (41 ckpts x ~0.8 GB)
#   evals    token structure, chance-corrected probe, MMLU splice, clustering quality
#   summary  eval_out/summarize_tokbias.py
# Finished steps are skipped; FORCE=1 redoes one.
set -o pipefail
# Delta: env.sh resolves GEOAE_ROOT from its own location, exports HF_HOME /
# UV_PROJECT_ENVIRONMENT / SBATCH_ACCOUNT, and cds to the repo root.
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/delta/env.sh"
PY=scripts/delta/py
E=e2e/checkpoints/general/llama3.2-3B/layer27
CFG=configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias.yaml
TABLE=$E/token_bias_sampled.npz
PARENT=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled/step_0014450.pt
CKDIR=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias
TB=$CKDIR/step_0014450.pt                      # epoch 50: 289 steps/epoch, as for the parent
KM_NEW=$E/balanced_kmeans_k2000_dpc_sampled.npz
KM_TOK=$E/balanced_kmeans_k2000_dpc_sampled_tok64.npz
SEQ=sentiment,sentiment_long,subjectivity,language,topic4,topic14,topic20,formality,domain
STEPS=("$@"); [ $# -eq 0 ] && { echo "usage: $0 table cache | train | evals summary"; exit 2; }
want() { for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
have() { [ -z "$FORCE" ] && [ -e "$1" ] && echo "[skip] $1 exists" && return 0; return 1; }

if want table && ! have $TABLE; then
    $PY -u -m geoae.token_bias --activations_dir activations_sampled_10M --out $TABLE \
        2>&1 | tee logs/token_bias_sampled.log || exit 1
fi

if want cache; then
    mkdir -p cache_ids
    for f in pos.npz ner.npz ner_coarse.npy ioi.npz ravel.npz; do     # already carry token_id
        [ -e cache_ids/$f ] || ln -s ../cache/$f cache_ids/$f
    done
    $PY -u -m geoae.interp.concept_suite --checkpoint $PARENT --out cache_ids --only $SEQ \
        2>&1 | tee logs/concept_suite_cache_ids.log || exit 1
fi

if want train; then
    [ -f $TABLE ] || { echo "[abort] run the table step first"; exit 1; }
    resume=(); ls $CKDIR/step_*.pt >/dev/null 2>&1 && resume=(--resume latest)
    $PY -u -m geoae.train --config $CFG "${resume[@]}" 2>&1 \
        | tee -a logs/llama_l27_d6144_dpc_sampled_tokbias.log || exit 1
fi

if want evals; then
    [ -f $TB ] || { echo "[abort] $TB missing — training not at epoch 50"; exit 1; }
    have eval_out/token_structure_tokbias.json || $PY -u eval_out/token_structure.py \
        --arms "d6144_new=$PARENT,tokbias=$TB,km_new=$KM_NEW,km_tok64=$KM_TOK" \
        --out eval_out/token_structure_tokbias.json 2>&1 | tee logs/token_structure_tokbias.log
    have eval_out/probe_tokbias.json || $PY -u eval_out/probe_chance_corrected.py --cache cache_ids \
        --models "d6144_new=$PARENT,tokbias=$TB" --baselines "km_new=$KM_NEW,km_tok64=$KM_TOK" \
        --out eval_out/probe_tokbias.json 2>&1 | tee logs/probe_tokbias.log
    have eval_out/mmlu_d6144_sampled_tokbias.json || $PY -u -m geoae.interp.mmlu_splice \
        --checkpoint $TB --out eval_out/mmlu_d6144_sampled_tokbias.json 2>&1 | tee logs/mmlu_tokbias.log
    have eval_out/cq_tokbias.json || $PY -u -m geoae.interp.clustering_quality --baseline $KM_NEW \
        --checkpoints $PARENT $TB --names km_new d6144_new tokbias \
        --activations_dir activations_sampled_10M --layer 27 --n_sample 1000000 --seed 0 \
        --out eval_out/cq_tokbias.json 2>&1 | tee logs/cq_tokbias.log
fi

want summary && $PY eval_out/summarize_tokbias.py | tee logs/summary_tokbias.log
