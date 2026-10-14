#!/bin/bash
# SAE arms (4 data-matched TopK + Gemma Scope 2 16k/262k) on the representation-level evals, beside GeoAE.
# One job: probe (base, base_tokmean, bypass AND the six SAEs in ONE json), token_structure, mmlu per SAE.
#   sbatch scripts/argo/sae_suite.sbatch
#   sbatch --gres=gpu:l40:1 --nodelist=argo76 scripts/argo/sae_suite.sbatch      # argo73 draining
# (command-line sbatch flags override the #SBATCH defaults below.)
# Outputs: results/delta/g4b_l33_d5120/evals/sae_suite/{probe.json,token_structure.json,mmlu_<name>.json}
# Paths follow eval_out/run_delta_evals.sh, FAM=g4b branch (LAYER=33, K=2000, DIM=5120).
#SBATCH --job-name=sae-suite
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

ARM=g4b_l33_d5120; LAYER=33; K=2000
E=e2e/checkpoints/general/gemma3-4B/layer$LAYER
B=checkpoints/gemma3-4B/layer$LAYER
TB=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
KM=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5.npz
KMT=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
CACHE_IDS=cache/gemma3-4b/l$LAYER
# token_structure reads only the contiguous val tail (<=500k rows) of layer_33.npy + rows_tok.npy: no staging needed.
ACT="${ACT_DIR:-activations_gemma3_4b_sampled_struct5_10M}"
OUT=results/delta/$ARM/evals/sae_suite
TRAIN=results/delta/$ARM/evals/gemma_scope_compare/matched_topk/training
mkdir -p "$OUT"

SAES="bt10k=matched:$TRAIN/batch_topk_10240/last.pt,bt20k=matched:$TRAIN/batch_topk_20480/last.pt"
SAES="$SAES,mat10k=matched:$TRAIN/matryoshka_10240/last.pt,mat20k=matched:$TRAIN/matryoshka_20480/last.pt"
SAES="$SAES,gs16k=scope:16k,gs262k=scope:262k"
NAMES="bt10k bt20k mat10k mat20k gs16k gs262k"

for f in "$TB" "$KM" "$KMT" "$ACT/layer_$LAYER.npy" "$CACHE_IDS" "$TRAIN"/{batch_topk,matryoshka}_{10240,20480}/last.pt; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
RC=0
run() { local log="$1"; shift; echo "[sae-suite $(date +%H:%M:%S)] $*"; "$@" 2>&1 | tee "logs/$log.log"; local r=${PIPESTATUS[0]}
        [ "$r" -eq 0 ] || { echo "[sae-suite] FAILED rc=$r: $log"; RC=1; }; }

run sae_suite_probe "$PY" -u eval_out/probe_chance_corrected.py --cache "$CACHE_IDS" \
    --models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT" --saes "$SAES" \
    --atlas_last "" --out "$OUT/probe.json"
run sae_suite_tokstruct "$PY" -u eval_out/token_structure.py \
    --arms "base=$KM,base_tokmean=$KMT,bypass=$TB" --saes "$SAES" \
    --activations_dir "$ACT" --layer "$LAYER" --out "$OUT/token_structure.json"
for n in $NAMES; do
    spec=$(tr ',' '\n' <<<"$SAES" | sed -n "s/^$n=//p")
    run "sae_suite_mmlu_$n" "$PY" -u -m geoae.interp.mmlu_splice --sae "$spec" --model google/gemma-3-4b-pt \
        --layer "$LAYER" --out "$OUT/mmlu_$n.json"
done
[ "$RC" -eq 0 ] && echo "SAE SUITE COMPLETE" || { echo "SAE SUITE INCOMPLETE"; exit 1; }
