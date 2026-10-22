#!/bin/bash
# One control dataset with confidence records, same settings as experiments/paper/remote/run_control_conf.sh:
# range removal (other-class mean), range steering, plain steering; alphas 0.5 1 2; d-prime top 30%, tao 2;
# n_fit 80 / n_eval 50 / n_comp 80, seed 42; Bias in Bios without "teacher".
#   sbatch experiments/paper/delta/control_conf_one.sbatch <arm> <checkpoint> <layer> <db14|biasbios|ag_news>
#   e.g. sbatch experiments/paper/delta/control_conf_one.sbatch qwen25_l35_d4096 \
#          checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt 35 db14
#   Optional env (pass with sbatch --export=ALL,...): ALPHAS="0.1 0.2 0.3 0.4 0.5"  REMOVAL="" (skip range removal)
#   TAG=conf_small  -> results/delta/<arm>/interventions/range_intervention_<ds>_dprime_<TAG>.json
#SBATCH --job-name=control-conf
#SBATCH --account=bbyl-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-control-conf-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/env.sh
ARM=${1:?arm}; TB=${2:?checkpoint}; LY=${3:?layer}; ds=${4:?dataset}
PY=scripts/py
ALPHAS=${ALPHAS:-0.5 1 2}; REMOVAL=${REMOVAL-range}; TAG=${TAG:-conf}
I=results/delta/$ARM/interventions; L=results/delta/$ARM/logs/interventions; JCD=results/delta/joint_correct
mkdir -p "$I" "$L" "$JCD"
[ -f "$TB" ] || { echo "[abort] missing $TB"; exit 1; }
extra=(); [ "$ds" = biasbios ] && extra=(--concepts 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27)
out=$I/range_intervention_${ds}_dprime_${TAG}.json; rm -f "$out"
echo "[control-conf] $ARM $ds alphas=[$ALPHAS] removal=[$REMOVAL] tag=$TAG host $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
"$PY" -u -m geoae.evaluation.control --checkpoint "$TB" --dataset "$ds" "${extra[@]}" \
  --saliency dprime --tao 2.0 --removal_modes "$REMOVAL" --replace comp --steer_modes global,range --alphas $ALPHAS \
  --correct_json "$JCD/joint_correct_${ds}_l${LY}_$ARM.json" --out "$out" > "$L/range_${ds}_dprime_${TAG}.log" 2>&1
rc=$?
echo "[control-conf] $ARM $ds exit $rc $( grep -q '"summary"' "$out" 2>/dev/null && echo ok || echo INCOMPLETE) $(date)"
