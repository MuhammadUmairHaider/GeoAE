#!/bin/bash
# Qwen2.5-3B layer 35: the full four-space probe table (the paired-table recipe of eval_out/latex_pair_tables.py).
# Spaces: plain residual k-means, plain token-removed k-means (Lloyd, density peaks, seed 42), bypass AE, no-bypass AE.
#   probes  probe_dense_subsets.py, anchor exclusion as the Llama/Gemma files, min10 + pair_fp10 for both pairs
#   atlas   no-bypass AE labels on the last-word cache, then atlas_nmi_filtered.py --arms plain42,nobypass --min_rows 10
# Needs scripts/argo/qwen_seq_table.sbatch first (plain codebooks, toxicity cache, Atlas cache, bypass labels).
#   sbatch scripts/argo/qwen_full_table.sbatch
#SBATCH --job-name=qwen-fulltab
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=argo76
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=.
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
LAYER=35
E=e2e/checkpoints/general/qwen2.5-3B/layer$LAYER/delta
PKM=$E/plain_kmeans_k2000_dpc_struct5_train10M_seed42.npz; PKMT=$E/plain_kmeans_k2000_dpc_struct5_train10M_tokmean_seed42.npz
AE=checkpoints/qwen2.5-3B/layer$LAYER/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt
NB=checkpoints/qwen2.5-3B/layer$LAYER/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M/step_0015250.pt
CACHE=cache/qwen2.5-3B/layer$LAYER
ATL=/scratch/dogar/geoae/atlas_qwen25_l35_lastword
META=/scratch/dogar/geoae/atlas_concept_meta.json
OUT=results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer$LAYER/seq_table
for f in "$PKM" "$PKMT" "$AE" "$NB" "$CACHE/toxicity.npz" "$ATL/chunks.npz" "$OUT/atlas_lastword/assign.npz" "$META"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
echo "[qwen-fulltab] host $(hostname -f) $(date)"

"$PY" -u eval_out/probe_dense_subsets.py --cache "$CACHE" \
    --arms "res_plain=km:$PKM,tok_plain=km:$PKMT,geoae=ae:$AE,nobypass=ae:$NB" --fp_min 6 \
    --pairs res_plain:nobypass,tok_plain:geoae --pair_min 10 --out "$OUT/probe_dense_subsets_fourway.json" \
    2>&1 | tee "$OUT/probe_dense_subsets_fourway.log"
"$PY" -u eval_out/atlas_assign_extra.py --cache "$ATL" --models "nobypass=$NB" \
    --out "$OUT/atlas_lastword/assign_nobypass.npz" 2>&1 | tee "$OUT/atlas_lastword/assign_nobypass.log"
OMP_NUM_THREADS=8 "$PY" -u eval_out/atlas_nmi_filtered.py \
    --assign "$OUT/atlas_lastword/assign.npz,$OUT/atlas_lastword/assign_nobypass.npz" \
    --chunks "$ATL/chunks.npz" --concepts "$META" --arms plain42,nobypass --ref nobypass --min_rows 10 \
    --n_boot 300 --seed 0 --out "$OUT/atlas_lastword/nmi_pair10_nobypass.json" 2>&1 | tee "$OUT/atlas_lastword/nmi_pair10_nobypass.log"
echo "[qwen-fulltab] done $(date)"
