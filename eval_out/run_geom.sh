#!/bin/bash
cd /home/exouser/RepresentationAE/GeoAE
B=checkpoints/llama3.2-3B/layer27
E=e2e/checkpoints/general/llama3.2-3B/layer27
uv run python -u -m geoae.interp.concept_geometry \
  --cache cache \
  --rungs pos_coarse,pos_fine,ravel_country,topic14,language \
  --models "kmeanspp=$B/k2000_bnh_b32k_lam1_d6144/best_val.pt,dpc=$B/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt" \
  --baselines "balanced_kmeans=$E/balanced_kmeans_k2000.npz" \
  --projections tsne,pca \
  --n_points 6000 --perplexity 30 --max_classes 8 \
  --outdir figures/init_arms
