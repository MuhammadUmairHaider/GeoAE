#!/bin/bash
cd /home/exouser/RepresentationAE/GeoAE
B=checkpoints/llama3.2-3B/layer27
N=2000
for arm in k2000_bnh_b32k_lam1_d6144:kmeanspp \
           k2000_bnh_b32k_lam1_d6144_seeded_atlas:seeded_atlas \
           k2000_bnh_b32k_lam1_d6144_dpc:dpc; do
  d="${arm%%:*}"; name="${arm##*:}"
  echo "############ MMLU arm=$name"
  uv run python -u -m geoae.interp.causal_concept_compare \
    --checkpoint $B/$d/best_val.pt --layer 27 --mmlu $N --seed 42
  # --out is IGNORED on the mmlu path (hardcoded results_ccc_mmlu.json), so rename
  mv results_ccc_mmlu.json eval_out/mmlu_$name.json
done
