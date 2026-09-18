#!/usr/bin/env bash
# Discriminative dim selection (d' = |mu_c - mu_rest| / pooled sd) instead of the
# NeuronLens mean-|a| rule. tao stays 2.0, so the ranges are the same +-2 sigma.
# Same checkpoints and the same joint-correct doc sets as the abs runs, so every
# number pairs directly against results/range_intervention_*_abs.json.
# ~45 min each on a free GPU.
set -o pipefail
cd /home/exouser/RepresentationAE/GeoAE
B=checkpoints/llama3.2-3B/layer27

# GELU arm (dpc best_val, ep47)
uv run python -u -m geoae.interp.range_intervention_compare \
  --checkpoint $B/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt --dataset db14 \
  --correct_json dbpedia/joint_correct_db14_l27_dpc.json \
  --saliency dprime --tao 2.0 \
  --out results/range_intervention_db14_b32k_dpc_dprime.json \
  | tee logs/range_db14_dpc_dprime.log

# tanh arm (ep50)
uv run python -u -m geoae.interp.range_intervention_compare \
  --checkpoint $B/k2000_bnh_b32k_lam1_d6144_dpc_tanh/step_0014200.pt --dataset db14 \
  --correct_json dbpedia/joint_correct_db14_l27_dpc_tanh50.json \
  --saliency dprime --tao 2.0 \
  --out results/range_intervention_db14_dpc_tanh50_dprime.json \
  | tee logs/range_db14_tanh50_dprime.log
