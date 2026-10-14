#!/bin/bash
# WikiText-2 perplexity under the splice (geoae.interp.splice_ppl) for the Qwen2.5-3B L35 bypass AE. Qwen2.5 has no BOS
# token: each chunk is prefixed with the text token before it (see make_chunks).
#   sbatch scripts/argo/splice_ppl_qwen.sbatch            (one L40 on argo76, ~15 min)
# Output: results/delta/qwen25_l35_d4096/evals/splice_ppl.json
#SBATCH --job-name=splice-ppl-qwen
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
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
TB=checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt
OUT=results/delta/qwen25_l35_d4096/evals; mkdir -p "$OUT" logs
echo "[splice-ppl-qwen] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -u -m geoae.interp.splice_ppl --ae "$OUT/splice_ppl.json=$TB" 2>&1 | tee "$OUT/splice_ppl.log"
exit "${PIPESTATUS[0]}"
