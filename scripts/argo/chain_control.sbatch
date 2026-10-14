#!/bin/bash
# Control edits on ONE shared document set with one or both reconstructions spliced in (Gemma 3 4B L33; AE = bypass GeoAE,
# SAE = Gemma Scope 16k): orders ae, sae, ae_sae (h -> AE -> SAE) and sae_ae (h -> SAE -> AE); every arm of an order starts
# from the same unedited model and edits one space (z, s, s_top, or the residual before / after the chain); range removal,
# range steering, plain steering at alpha 0.5, plus random-coordinate and shuffled-label controls. Documents: correct under
# the base model and every unedited order, exactly 50 evaluation documents per class (classes that cannot supply them are
# dropped and listed in the log). geoae/interp/chain_control.py; tests tests/test_chain_control.py;
# table eval_out/paper_chain_control_table.py. Rerunning the same line resumes after the last finished concept.
#   sbatch scripts/argo/chain_control.sbatch <ag_news|db14|biasbios|emotion6|news20|atlas_s0|atlas_s1|atlas_s2> [--smoke] [chain_control.py options]
#   sbatch --gres=gpu:l40:1 --nodelist=argo76 scripts/argo/chain_control.sbatch emotion6
#   all datasets: bash scripts/argo/submit_chain_control.sh
# Output: results/delta/g4b_l33_d5120/interventions/chain/chain_control_<ds>.json (+ .log)
#SBATCH --job-name=chain-control
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=argo73
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-14:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0      # wikimedia/wikipedia (perplexity texts) is streamed from the Hub
DS="${1:?dataset}"; shift || true
ARM=g4b_l33_d5120
CK=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
EXTRA=(); TAG=""
case "$DS" in
    ag_news)  EXTRA+=(--extra_split train) ;;           # Sci/Tech runs out of jointly-correct test documents
    biasbios) EXTRA+=(--extra_split train --concepts 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27) ;;
    atlas_s*) EXTRA+=(--max_try 600) ;;                 # frozen FineWeb-Atlas draw, document-level fit / test splits
esac
for a in "$@"; do                                       # --smoke, plus any chain_control.py option passed through
    [ "$a" = --smoke ] && TAG=_smoke
    EXTRA+=("$a")
done
OUT=results/delta/$ARM/interventions/chain; mkdir -p "$OUT" logs
echo "[chain-control $DS] host $(hostname -f) $(date) job ${SLURM_JOB_ID:-none}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$UV_PROJECT_ENVIRONMENT/bin/python" -u -m geoae.interp.chain_control --checkpoint "$CK" --sae scope:16k --dataset "$DS" \
    "${EXTRA[@]}" --correct_json results/delta/joint_correct/joint_correct_${DS}_l33_${ARM}_chain4_gs16k$TAG.json \
    --out "$OUT/chain_control_${DS}$TAG.json" 2>&1 | tee -a "$OUT/chain_control_${DS}$TAG.log"
echo "[chain-control $DS] done $(date)"
