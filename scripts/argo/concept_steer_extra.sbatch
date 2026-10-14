#!/bin/bash
# Long-generation concept steering (Gemma 3 4B L33) with the SAE and the no-bypass AE added as direction handles next to
# label / base_dir / bypass_dir of concept_steer_generate.py: 20 neutral prompts, 32 new tokens, every handle a constant
# direction of the same size added to the UNEDITED model at every position. geoae/interp/concept_steer_extra.py.
# Judge afterwards (OpenRouter, on Delta where the judge cache lives): geoae.interp.concept_steer_judge --gen <out>.
#   sbatch scripts/argo/concept_steer_extra.sbatch [db14|biasbios|atlas_s0|atlas_s1|atlas_s2] [--smoke]   (atlas_sN = FineWeb-Atlas topic draws)
# Output: results/delta/g4b_l33_d5120/concept/concept_steer_generate_sae_<ds>.json
#SBATCH --job-name=concept-steer-extra
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=argo73
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0
DS="${1:-db14}"; shift || true
TAG=""; EXTRA=()
[ "${1:-}" = --smoke ] && { EXTRA=(--smoke); TAG=_smoke; }
ARM=g4b_l33_d5120
C=results/delta/$ARM/concept
"$UV_PROJECT_ENVIRONMENT/bin/python" -u -m geoae.interp.concept_steer_extra --gen $C/concept_steer_generate.json \
    --activations_dir activations_gemma3_4b_sampled_struct5_10M --datasets "$DS" "${EXTRA[@]}" --sae scope:16k \
    --sae_kmeans results/delta/$ARM/evals/sae_suite/controls/plain_kmeans_gs16k.npz \
    --extra_ae nobypass=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_train10M/step_0015250.pt \
    --out $C/concept_steer_generate_sae_${DS}$TAG.json 2>&1 | tee $C/concept_steer_generate_sae_${DS}$TAG.log
