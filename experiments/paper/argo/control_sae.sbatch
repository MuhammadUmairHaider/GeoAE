#!/bin/bash
# Control edits (confidence setting: range steering + plain steering at alpha 0.5, range removal) in Gemma Scope 16k's code
# space (substrate s: encode -> edit codes -> decode, SAE reconstruction spliced in, BOS kept) next to the base residual
# (substrate h) on the same documents; Gemma 3 4B L33. Operators and doc pools as experiments/paper/remote/run_control_conf.sh.
#   sbatch experiments/paper/argo/control_sae.sbatch <db14|biasbios|ag_news|emotion6|news20> [--smoke]
# Output: results/delta/g4b_l33_d5120/interventions/sae/range_intervention_<ds>_dprime_gs16k.json
#SBATCH --job-name=control-sae
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=argo76
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0      # wikimedia/wikipedia (perplexity texts) is streamed from the Hub
DS="${1:?dataset}"; shift || true
ARM=g4b_l33_d5120
CK=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
EXTRA=(); TAG=""
[ "$DS" = biasbios ] && EXTRA=(--concepts 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27)
[ "${1:-}" = --smoke ] && { EXTRA=(--smoke); TAG=_smoke; }
OUT=results/delta/$ARM/interventions/sae; mkdir -p "$OUT"
"$UV_PROJECT_ENVIRONMENT/bin/python" -u -m geoae.evaluation.control --checkpoint "$CK" --dataset "$DS" "${EXTRA[@]}" \
    --saliency dprime --tao 2.0 --removal_modes range --replace comp --steer_modes global,range --alphas 0.5 \
    --sae scope:16k --substrates h,s --correct_json results/delta/joint_correct/joint_correct_${DS}_l33_$ARM.json \
    --out "$OUT/range_intervention_${DS}_dprime_gs16k$TAG.json" 2>&1 | tee "$OUT/range_${DS}_dprime_gs16k$TAG.log"
