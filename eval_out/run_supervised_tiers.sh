#!/usr/bin/env bash
# Supervised-recovery ladder on the finished d12288 dpc arm, cheapest first.
#
#   tier0  no training: re-fit the codebook on the FROZEN latent, supervised
#          (labelled class means) and unsupervised (dpc) — asks whether a better
#          partition of the same representation is all that was missing.
#   tier1  5-epoch fine-tune, decoder frozen: can labels improve the ENCODER
#          while it stays decodable by the parent's decoder?
#   tier2  5-epoch fine-tune, everything trainable.
#
# Each tier ends with its own eval so you can stop after any of them.
# Usage:  ./eval_out/run_supervised_tiers.sh tier0
#         ./eval_out/run_supervised_tiers.sh tier1 tier1_eval
#         FORCE=1 ./eval_out/run_supervised_tiers.sh tier0_probe
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE

B=checkpoints/llama3.2-3B/layer27
E=e2e/checkpoints/general/llama3.2-3B/layer27
PARENT=$B/k2000_bnh_b32k_lam1_d12288_dpc/step_0014200.pt      # epoch 50
FT_ENC=$B/d12288_dpc_ft_sup_enc/step_0015620.pt               # epoch 55 = 14200 + 5*284
FT_FULL=$B/d12288_dpc_ft_sup_full/step_0015620.pt
ACTS=activations_diverse_10M/layer_27.npy
BAL=$E/balanced_kmeans_k2000.npz                              # encoder-free control
LAT_SEED=$E/latent_seeded_k2000_d12288.npz                    # tier0, supervised codebook
LAT_DPC=$E/latent_dpc_k2000_d12288.npz                        # tier0, unsupervised codebook
FAILED=()

ALL=(tier0 tier0_probe tier1 tier1_eval tier2 tier2_eval)
REQ=("$@"); [ $# -eq 0 ] && REQ=("${ALL[@]}")
for r in "${REQ[@]}"; do case " ${ALL[*]} " in *" $r "*) ;; *) echo "[abort] unknown step '$r' (have: ${ALL[*]})"; exit 2;; esac; done
want() { for s in "${REQ[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
step() { want "$1" || return; if [ -z "$FORCE" ] && eval "$2"; then echo "[skip] $1 — done"; return; fi
         echo -e "\n[run ] $1 ($(date +%H:%M))"; "$3" || { echo "[FAIL] $1"; FAILED+=("$1"); }; }

# --- TIER 0 -----------------------------------------------------------------
# ~45 min. The activation read dominates (random rows over a 115 GB dump).
# GPU memory: the latent sample is n_sample x 12288 x 4 bytes = 14.7 GB at 300k,
# plus ~6 GB transient, so it fits a 40 GB card with room to spare. The density
# fill runs on a --density_pool (32,768) subsample, NOT on the full sample; that
# is what the first version got wrong and it OOM'd asking for 3.7 GB chunks.
# If anything else is on the card, drop --n_sample to 200000 (9.8 GB).
tier0() {
    uv run python -u -m geoae.interp.fit_balanced_kmeans \
        --checkpoint $PARENT --activations $ACTS --space latent \
        --init seeded --fill_mode peaks --anchor_cache cache \
        --anchor_per_class 25 --anchor_min_examples 5 \
        --atlas_last cache/atlas8k_last.npz --atlas_min_examples 25 \
        --n_clusters 2000 --n_sample 300000 --batch_size 4096 --epochs 8 \
        --out $LAT_SEED | tee logs/fit_latent_seeded_d12288.log || return 1
    # Unsupervised refit in the SAME space: without it, a tier0 win cannot be
    # attributed to the labels rather than to simply refitting the codebook.
    uv run python -u -m geoae.interp.fit_balanced_kmeans \
        --checkpoint $PARENT --activations $ACTS --space latent \
        --init dpc --reinit peaks \
        --n_clusters 2000 --n_sample 300000 --batch_size 4096 --epochs 8 \
        --out $LAT_DPC | tee logs/fit_latent_dpc_d12288.log
}
tier0_probe() {   # ~6 min. --exclude_anchors is REQUIRED: the seeded codebook is built from cache rows.
    uv run python -u -m geoae.interp.concept_probe --cache cache \
        --models "d12288=$PARENT" \
        --baselines "latent_seeded=$LAT_SEED,latent_dpc=$LAT_DPC,balanced_kmeans=$BAL" \
        --exclude_anchors --anchor_seed 42 --anchor_per_class 25 --anchor_min_examples 5 \
        --atlas_last cache/atlas8k_last.npz --atlas_min_examples 25 \
        --out eval_out/probe_tier0_latent.json | tee logs/probe_tier0_latent.log
}

# --- TIER 1 (encoder only) --------------------------------------------------
tier1() {          # ~1.7 h (5 x ~20 min)
    uv run python -u -m geoae.train \
        --config configs/base/llama3.2-3b_l27_d12288_dpc_ft_sup_enc.yaml \
        --resume $PARENT | tee logs/ft_sup_enc_d12288.log
}
tier1_eval() { ft_eval "$FT_ENC" ft_enc; }

# --- TIER 2 (full) ----------------------------------------------------------
tier2() {          # ~1.7 h
    uv run python -u -m geoae.train \
        --config configs/base/llama3.2-3b_l27_d12288_dpc_ft_sup_full.yaml \
        --resume $PARENT | tee logs/ft_sup_full_d12288.log
}
tier2_eval() { ft_eval "$FT_FULL" ft_full; }

# Shared fine-tune eval: probe (transfer), MMLU (drift), geometry (did width's win survive).
# Read the SUPERVISED rungs (surface, pos_*, ner_*, ioi_*) as "did it fit" — they are
# training data. The semantic rungs are the transfer result. checkpoints_dir/sup_manifest.json
# lists the 38 classes held out entirely, for a stricter within-rung read.
ft_eval() {
    local ckpt=$1 tag=$2
    [ -f "$ckpt" ] || { echo "[abort] missing $ckpt — did the fine-tune finish?"; return 1; }
    uv run python -u -m geoae.interp.concept_probe --cache cache \
        --models "$tag=$ckpt,d12288=$PARENT" --baselines "balanced_kmeans=$BAL" \
        --exclude_anchors --anchor_seed 42 --anchor_per_class 25 --anchor_min_examples 5 \
        --atlas_last cache/atlas8k_last.npz --atlas_min_examples 25 \
        --out eval_out/probe_$tag.json | tee logs/probe_$tag.log || return 1
    uv run python -u -m geoae.interp.causal_concept_compare --checkpoint "$ckpt" \
        --layer 27 --mmlu 2000 --seed 42 | tee logs/mmlu_$tag.log || return 1
    mv results_ccc_mmlu.json eval_out/mmlu_$tag.json
    uv run python -u -m geoae.interp.clustering_quality \
        --checkpoints "$ckpt" $PARENT --names $tag d12288 \
        --activations_dir activations_diverse_10M --layer 27 --n_sample 1000000 --seed 0 \
        --out eval_out/cq_$tag.json | tee logs/cq_$tag.log
}

step "tier0"       "[ -f $LAT_SEED ] && [ -f $LAT_DPC ]"        tier0
step "tier0_probe" "[ -f eval_out/probe_tier0_latent.json ]"    tier0_probe
step "tier1"       "[ -f $FT_ENC ]"                             tier1
step "tier1_eval"  "[ -f eval_out/cq_ft_enc.json ]"             tier1_eval
step "tier2"       "[ -f $FT_FULL ]"                            tier2
step "tier2_eval"  "[ -f eval_out/cq_ft_full.json ]"            tier2_eval

echo
if [ ${#FAILED[@]} -eq 0 ]; then echo "[done] complete ($(date +%H:%M))"; else echo "[done] FAILED: ${FAILED[*]}"; exit 1; fi
