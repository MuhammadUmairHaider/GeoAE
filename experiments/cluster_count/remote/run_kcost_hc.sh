#!/bin/bash
# Reporting-rule rerun of the K = 2,000 / 4,000 / 8,000 Gemma comparison: every K with plain residual, plain token-removed
# and GeoAE; per-row labels saved, then homogeneity / completeness / raw NMI on rows in clusters of >5 in every codebook.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
cd "$GEOAE_ROOT"
source configs/sites/local.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.
PY=.venv/bin/python
until grep -q TOKMEAN_DONE logs/kcost_tokmean.log; do sleep 30; done
E=e2e/checkpoints/general/gemma3-4B/layer33/delta; B=checkpoints/gemma3-4B/layer33; O=results/delta/g4b_l33_d5120/evals/kcost
T2=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
T4=$B/k4000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
T8=$B/k8000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
P="plain=$E/plain_kmeans_k2000_dpc_sampled_struct5.npz,plain4k=$E/plain_kmeans_k4000_dpc_sampled_struct5.npz,plain8k=$E/plain_kmeans_k8000_dpc_sampled_struct5.npz"
PT="plain_tokmean=$E/plain_kmeans_k2000_dpc_sampled_struct5_tokmean.npz,plain_tokmean4k=$E/plain_kmeans_k4000_dpc_sampled_struct5_tokmean.npz,plain_tokmean8k=$E/plain_kmeans_k8000_dpc_sampled_struct5_tokmean.npz"
G="K2000=plain,plain_tokmean,bypass;K4000=plain4k,plain_tokmean4k,bypass4k;K8000=plain8k,plain_tokmean8k,bypass8k"
step() { local name=$1; shift; echo "[hc $(date +%H:%M:%S)] start $name"; "$@" > $O/$name.log 2>&1; echo "[hc $(date +%H:%M:%S)] done $name rc=$?"; }
step token_structure_all $PY -u geoae/evaluation/token_structure.py --arms "$P,$PT,bypass=$T2,bypass4k=$T4,bypass8k=$T8" --activations_dir activations_gemma3_4b_sampled_struct5_10M --layer 33 --out $O/token_structure_all.json --save_labels $O/token_structure_labels.npz
step probe_all $PY -u geoae/evaluation/chance_corrected_probe.py --cache cache/gemma3-4b/l33 --models "bypass=$T2,bypass4k=$T4,bypass8k=$T8" --baselines "$P,$PT" --atlas_last "" --out $O/probe_all.json --save_labels $O/probe_labels.npz
step hc_tokstruct $PY -u geoae/evaluation/filtered_probe.py --labels $O/token_structure_labels.npz --mode tokstruct --groups "$G" --n_boot 100 --out $O/hc_tokstruct.json
step hc_probe $PY -u geoae/evaluation/filtered_probe.py --labels $O/probe_labels.npz --mode probe --groups "$G" --n_boot 200 --out $O/hc_probe.json
echo "[hc $(date +%H:%M:%S)] HC_DONE"
