#!/bin/bash
# Submit the chain-splice control runs (experiments/paper/argo/chain_control.sbatch) for every control dataset and one
# FineWeb-Atlas draw. Run on Argo from the repo root:
#     bash experiments/paper/argo/submit_chain_control.sh                 # all six
#     bash experiments/paper/argo/submit_chain_control.sh db14 news20     # a subset
# Every job asks for one L40 on argo76 (three run at a time, the rest wait); CHAIN_GPU=h100 sends them to argo73 instead.
# A job that is cancelled or times out resumes after its last finished concept when the same line is submitted again.
# Afterwards, on Delta:
#     rsync -a argo:<repo>/results/delta/g4b_l33_d5120/interventions/chain/ results/delta/g4b_l33_d5120/interventions/chain/
#     scripts/py experiments/paper/analysis/paper_chain_control_table.py results/delta/g4b_l33_d5120/interventions/chain/chain_control_{ag_news,db14,biasbios,emotion6,news20,atlas_s0}.json
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "$GEOAE_ROOT"
SETS=("$@"); [ ${#SETS[@]} -eq 0 ] && SETS=(biasbios db14 atlas_s0 news20 emotion6 ag_news)
for ds in "${SETS[@]}"; do
    case "${CHAIN_GPU:-l40}" in
        h100) where=(--gres=gpu:h100:1 --nodelist=argo73) ;;
        *)    where=(--gres=gpu:l40:1 --nodelist=argo76) ;;
    esac
    id=$(sbatch --parsable --job-name="chain-$ds" "${where[@]}" experiments/paper/argo/chain_control.sbatch "$ds")
    echo "$ds -> job $id (${where[0]#--gres=gpu:})"
done
squeue -u "$USER" -o "%.9i %.16j %.9T %.10M %R" | grep -E "JOBID|chain-" || true
