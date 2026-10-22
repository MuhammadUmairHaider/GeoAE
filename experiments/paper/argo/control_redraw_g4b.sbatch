#!/bin/bash
# One document redraw of a Hub control dataset on Gemma 3 4B L33, as the paper's rows were run: the three-way
# true-positive control run (TP3=1 experiments/paper/argo/control_three_way.sbatch: base, GeoAE, Gemma Scope 16k) and its SAE top-1
# arm on the same documents, with the test split shuffled by <seed> instead of 42.
#   sbatch experiments/paper/argo/control_redraw_g4b.sbatch <db14|biasbios|ag_news> <seed>
# Output: results/delta/g4b_l33_d5120/interventions/sae/range_intervention_<dataset>_s<seed>_dprime_{hzs,s1}_tp3.json.
# A finished stage is skipped, so the same line can be resubmitted. Normally started by experiments/paper/argo/control_new_draws.sh.
#SBATCH --job-name=control-redraw-g4b
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1
NAME="${1:?dataset}"; S="${2:?seed}"; DS=${NAME}_s$S
OUT=results/delta/g4b_l33_d5120/interventions/sae
mkdir -p "$OUT" logs
echo "[redraw-g4b] $DS host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
done_ok() { [ -f "$1" ] && grep -q '"summary"' "$1"; }
rc=0
for SUBS in h,z,s s1; do
  f=$OUT/range_intervention_${DS}_dprime_${SUBS//,/}_tp3.json
  if done_ok "$f"; then echo "[redraw-g4b] skip $SUBS (done)"; continue; fi
  rm -f "$f"
  echo "[redraw-g4b $(date +%H:%M:%S)] start $SUBS"
  TP3=1 SUBS=$SUBS bash experiments/paper/argo/control_three_way.sbatch "$DS" > "logs/control_redraw_g4b_${DS}_${SUBS//,/}.log" 2>&1
  r=$?; [ "$r" -ne 0 ] && rc=$r
  echo "[redraw-g4b $(date +%H:%M:%S)] $SUBS exit $r $(done_ok "$f" && echo ok || echo INCOMPLETE)"
done
echo "[redraw-g4b] finished rc=$rc $(date)"
exit "$rc"
