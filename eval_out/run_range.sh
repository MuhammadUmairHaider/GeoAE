#!/usr/bin/env bash
# Range-based removal + steering, base residual (h) vs AE latent (z), DBpedia-14, L27.
# Each run is ~60-80 min on a shared GPU (448 interventions). Run the abs one first.
#
# Sanity check after run 1: st_global_a1.0 must reproduce results/steer_db14_dpc.json
# with the sign flipped (this harness: higher = better). Expect h ~ +0.574, z ~ +0.609.
cd /home/exouser/RepresentationAE/GeoAE
B=checkpoints/llama3.2-3B/layer27

# 1. NeuronLens-faithful saliency (mean |a|), dpc AE
uv run python -u -m geoae.interp.range_intervention_compare \
  --checkpoint $B/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt \
  --dataset db14 --correct_json dbpedia/joint_correct_db14_l27_dpc.json \
  --saliency abs | tee logs/range_db14_dpc_abs.log

# 2. Discriminative saliency (d'), same AE, same docs -> paired against run 1
uv run python -u -m geoae.interp.range_intervention_compare \
  --checkpoint $B/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt \
  --dataset db14 --correct_json dbpedia/joint_correct_db14_l27_dpc.json \
  --saliency dprime | tee logs/range_db14_dpc_dprime.log

# 3. (optional) k-means++ parent: is any z advantage the recipe, not the init?
#    First run builds its own joint-correct set (~10 min extra).
# uv run python -u -m geoae.interp.range_intervention_compare \
#   --checkpoint $B/k2000_bnh_b32k_lam1_d6144/step_0014200.pt \
#   --dataset db14 --correct_json dbpedia/joint_correct_db14_l27_kmeanspp.json \
#   --saliency abs | tee logs/range_db14_kmeanspp_abs.log
