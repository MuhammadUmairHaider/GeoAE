#!/bin/bash
# §7 control tasks for the Gemma 3 4B layer-33 token-bypass AE trained with K = 4,000 or 8,000 prototypes, with the exact
# settings of the K = 2,000 files in results/delta/g4b_l33_d5120/interventions/ (run_delta_evals.sh steps number, steer,
# range_db14; range_bios at alpha 0.5 = range_intervention_biasbios_dprime_a0p5.json). Joint-correct lists are rebuilt for
# this AE (cached under results/delta/joint_correct/, fingerprinted by the AE).
#   sbatch experiments/cluster_count/argo/controls_g4b_k.sbatch 4000      (and 8000)
#SBATCH --job-name=g4b-controls-k
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
export PYTHONUNBUFFERED=1   # NOT offline: the steering / removal tools load wikimedia/wikipedia through datasets, which fails offline
K="${1:?usage: sbatch experiments/cluster_count/argo/controls_g4b_k.sbatch <4000|8000>}"
PY=scripts/py
ARM=g4b_l33_d5120k$K
TB=checkpoints/gemma3-4B/layer33/k${K}_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
I=results/delta/$ARM/interventions; L=results/delta/$ARM/logs/interventions; F=results/delta/$ARM/figures
JCD=results/delta/joint_correct; JC_DB14=$JCD/joint_correct_db14_l33_$ARM.json; JC_BIOS=$JCD/joint_correct_biasbios_l33_$ARM.json
BIOS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27
mkdir -p "$I" "$L" "$F" "$JCD"
[ -f "$TB" ] || { echo "[abort] missing $TB"; exit 1; }
echo "[controls K=$K] host $(hostname) $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
run() { local name=$1 out=$2; shift 2; [ -s "$out" ] && { echo "[controls K=$K] skip $name ($out exists)"; return 0; }
        echo "[controls K=$K $(date +%H:%M:%S)] start $name"; "$@" > "$L/$name.log" 2>&1; local rc=$?
        if [ -s "$out" ]; then echo "[controls K=$K $(date +%H:%M:%S)] done $name rc=$rc"; else echo "[controls K=$K $(date +%H:%M:%S)] FAILED $name rc=$rc (no $out)"; tail -3 "$L/$name.log"; fi; }
run number_dprime "$I/range_number_dprime.json" env NUMBER_CHECKPOINT="$TB" NUMBER_OUTPUT="$I/range_number_dprime.json" bash experiments/number_control/scripts/run_number_control.sh --saliency dprime
run number_review "$I/number_review.json" "$PY" experiments/paper/analysis/review_number_control.py --source "$I/range_number_dprime.json" --review_out "$I/number_review.json" --figure_prefix "$F/number_review"
run steer_db14 "$I/steer_db14.json" "$PY" -u -m geoae.evaluation.steering --checkpoint "$TB" --dataset db14 --correct_json "$JC_DB14" --out "$I/steer_db14.json"
run range_db14_dprime "$I/range_intervention_db14_dprime.json" "$PY" -u -m geoae.evaluation.control --checkpoint "$TB" --dataset db14 --correct_json "$JC_DB14" --saliency dprime --tao 2.0 --out "$I/range_intervention_db14_dprime.json"
run range_biasbios_dprime_a0p5 "$I/range_intervention_biasbios_dprime_a0p5.json" "$PY" -u -m geoae.evaluation.control --checkpoint "$TB" --dataset biasbios --concepts "$BIOS" --correct_json "$JC_BIOS" --saliency dprime --tao 2.0 --alphas 0.5 --out "$I/range_intervention_biasbios_dprime_a0p5.json"
echo "[controls K=$K $(date +%H:%M:%S)] CONTROLS_DONE"
