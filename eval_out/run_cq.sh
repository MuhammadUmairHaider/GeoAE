#!/bin/bash
cd /home/exouser/RepresentationAE/GeoAE
B=checkpoints/llama3.2-3B/layer27
uv run python -u -m geoae.interp.clustering_quality \
  --baseline e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu/baseline_kmeans_k2000.npz \
  --checkpoints \
    $B/k2000_bnh_b32k_lam1_d6144/best_val.pt \
    $B/k2000_bnh_b32k_lam1_d6144_seeded_atlas/best_val.pt \
    $B/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt \
  --names raw_kmeans kmeanspp seeded_atlas dpc \
  --activations_dir activations_diverse_10M --layer 27 \
  --n_sample 1000000 --seed 0 \
  --out eval_out/cq_init_arms.json
