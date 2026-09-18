#!/bin/bash
cd /home/exouser/RepresentationAE/GeoAE
B=checkpoints/llama3.2-3B/layer27
E=e2e/checkpoints/general/llama3.2-3B/layer27
uv run python -u -m geoae.interp.concept_probe \
  --cache cache \
  --models "kmeanspp=$B/k2000_bnh_b32k_lam1_d6144/best_val.pt,seeded_atlas=$B/k2000_bnh_b32k_lam1_d6144_seeded_atlas/best_val.pt,dpc=$B/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt" \
  --baselines "balanced_kmeans=$E/balanced_kmeans_k2000.npz,plain_kmeans=$E/baseline_kmeans_k2000_refit.npz" \
  --exclude_anchors --anchor_seed 42 --anchor_per_class 25 --anchor_min_examples 5 \
  --atlas_last cache/atlas8k_last.npz --atlas_min_examples 25 \
  --out eval_out/probe_init_arms.json
