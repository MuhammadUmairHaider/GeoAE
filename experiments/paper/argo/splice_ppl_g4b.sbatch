#!/bin/bash
# WikiText-2 perplexity under the splice (geoae.evaluation.perplexity) for the Gemma 3 4B L33 bypass AE and every SAE
# (data-matched TopK/Matryoshka + Gemma Scope 2 16k/262k), one LM load, one shared clean pass.
#   sbatch --gres=gpu:l40:1 --nodelist=argo76 experiments/paper/argo/splice_ppl_g4b.sbatch
# Outputs: results/delta/g4b_l33_d5120/evals/splice_ppl.json, .../evals/sae_suite/splice_ppl_<name>.json
#SBATCH --job-name=splice-ppl-g4b
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

ARM=g4b_l33_d5120
TB=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
OUT=results/delta/$ARM/evals
TRAIN=$OUT/gemma_scope_compare/matched_topk/training
for f in "$TB" "$TRAIN"/{batch_topk,matryoshka}_{10240,20480}/last.pt "$TRAIN/batch_topk_20480_k100/last.pt"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
"$PY" -u -m geoae.evaluation.perplexity --model google/gemma-3-4b-pt --layer 33 \
    --ae "$OUT/splice_ppl.json=$TB" \
    --sae "$OUT/sae_suite/splice_ppl_gs16k.json=scope:16k" \
    --sae "$OUT/sae_suite/splice_ppl_gs262k.json=scope:262k" \
    --sae "$OUT/sae_suite/splice_ppl_bt10k.json=matched:$TRAIN/batch_topk_10240/last.pt" \
    --sae "$OUT/sae_suite/splice_ppl_bt20k.json=matched:$TRAIN/batch_topk_20480/last.pt" \
    --sae "$OUT/sae_suite/splice_ppl_mat10k.json=matched:$TRAIN/matryoshka_10240/last.pt" \
    --sae "$OUT/sae_suite/splice_ppl_mat20k.json=matched:$TRAIN/matryoshka_20480/last.pt" \
    --sae "$OUT/sae_suite/splice_ppl_bt20k_k100.json=matched:$TRAIN/batch_topk_20480_k100/last.pt" \
    2>&1 | tee logs/splice_ppl_g4b.log
exit "${PIPESTATUS[0]}"
