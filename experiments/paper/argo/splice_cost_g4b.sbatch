#!/bin/bash
# Cost of splicing the AE vs the SAE into Gemma 3 4B at layer 33, standardized: both reconstructions scored in the SAME
# run, on the same data, against one shared clean pass (bypass GeoAE K=2000 vs Gemma Scope 16k).
#   1 MMLU        geoae.evaluation.mmlu --checkpoint + --sae : 2,000 test questions (seed 42), zero-shot
#   2 perplexity  geoae.evaluation.perplexity  --ae + --sae         : WikiText-2 raw test, every token scored once
#   sbatch experiments/paper/argo/splice_cost_g4b.sbatch                   (one L40 on argo76, ~20 min)
# Output: results/delta/g4b_l33_d5120/evals/splice_cost/{mmlu_ae_sae.json,splice_ppl_ae.json,splice_ppl_gs16k.json}
#SBATCH --job-name=splice-cost
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=argo76
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
TB=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
OUT=results/delta/g4b_l33_d5120/evals/splice_cost; mkdir -p "$OUT" logs
echo "[splice-cost] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -u -m geoae.evaluation.mmlu --checkpoint "$TB" --sae scope:16k --model google/gemma-3-4b-pt --layer 33 \
    --out "$OUT/mmlu_ae_sae.json" 2>&1 | tee "$OUT/mmlu_ae_sae.log"
"$PY" -u -m geoae.evaluation.perplexity --model google/gemma-3-4b-pt --layer 33 \
    --ae "$OUT/splice_ppl_ae.json=$TB" --sae "$OUT/splice_ppl_gs16k.json=scope:16k" 2>&1 | tee "$OUT/splice_ppl.log"
echo "[splice-cost] done $(date)"
