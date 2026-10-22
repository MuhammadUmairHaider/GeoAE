#!/bin/bash
# Gemma 3 4B arm g4b_l33_d5120 eval groups on ONE Argo H100, after training (the box runner's eval stage):
#   sbatch --dependency=afterok:<train job> experiments/paper/argo/evals_g4b.sbatch
# Prep (g4b_km, g4b_cache) was done on the box and copied over; geometry/probes are skipped as on the box.
# Two chains share the 95 GB H100: `interventions` | `evals steering concept` (the box's PAR=1 split; on the
# 40 GB A100 that risked OOM, here it has room). A failing group does not stop the other chain.
#SBATCH --job-name=g4b-evals
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=260G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
ARM=g4b_l33_d5120
mkdir -p logs
# The dump on /scratch is GPFS (no Linux page cache), so random-row readers (cq, tokstruct) would crawl:
# stage it into tmpfs and point the sheets at it via ACT_DIR (see scripts/slurm/argo_train.sbatch).
SHM="/dev/shm/geoae_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$SHM"' EXIT
trap 'exit 143' TERM INT
export ACT_DIR="$SHM/activations_gemma3_4b_sampled_struct5_10M"
mkdir -p "$ACT_DIR"
echo "[stage] dump -> $ACT_DIR"
time cp -rL activations_gemma3_4b_sampled_struct5_10M/. "$ACT_DIR/"
ev() {   # ev <group...>
    local tag="${*// /_}"
    echo "[g4b-evals $(date +%H:%M:%S)] start: $*"
    bash experiments/paper/scripts/run_delta_evals.sh "$ARM" "$@" > "logs/evals_${ARM}_${tag}.log" 2>&1
    local rc=$?
    echo "[g4b-evals $(date +%H:%M:%S)] done: $* rc=$rc"; tail -3 "logs/evals_${ARM}_${tag}.log"
    return $rc
}
( ev interventions ) & P1=$!
sleep 120   # let the first chain load its model before the second starts
( ev evals; r1=$?; ev steering; r2=$?; ev concept; r3=$?; exit $(( r1 || r2 || r3 )) ) & P2=$!
wait "$P1"; R1=$?
wait "$P2"; R2=$?
echo "[g4b-evals] interventions rc=$R1, evals/steering/concept rc=$R2"
[ "$R1" -eq 0 ] && [ "$R2" -eq 0 ] && echo "G4B EVALS COMPLETE" || { echo "G4B EVALS INCOMPLETE"; exit 1; }
