#!/bin/bash
# Control tasks with confidence records for the Qwen2.5-3B layer-35 token-bypass AE (2x width, K = 2,000, epoch 50),
# with the exact settings of experiments/paper/remote/run_control_conf.sh (Llama / Gemma): DBpedia-14, Bias in Bios (no "teacher"),
# AG News; range removal (other-class mean), range steering, plain steering; alphas 0.5 1 2; d-prime top 30%, tao 2;
# n_fit 80 / n_eval 50 / n_comp 80, seed 42. Joint-correct lists are built for this AE under results/delta/joint_correct/.
#   sbatch experiments/model_transfer/argo/control_conf_qwen.sbatch
#SBATCH --job-name=qwen-control-conf
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1   # NOT offline: the removal tool streams wikimedia/wikipedia for perplexity texts
PY=scripts/py
ARM=qwen25_l35_d4096
TB=checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt
I=results/delta/$ARM/interventions; L=results/delta/$ARM/logs/interventions; JCD=results/delta/joint_correct
BIOS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27   # no "teacher" (0% few-shot)
mkdir -p "$I" "$L" "$JCD"
[ -f "$TB" ] || { echo "[abort] missing $TB"; exit 1; }
echo "[qwen-control] host $(hostname) $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
for ds in db14 biasbios ag_news; do
  out=$I/range_intervention_${ds}_dprime_conf.json
  if [ -f "$out" ] && python3 -c "import json,sys; sys.exit(0 if 'summary' in json.load(open('$out')) else 1)"; then
    echo "[qwen-control] skip $ds (done)"; continue; fi
  rm -f "$out"
  extra=(); [ "$ds" = biasbios ] && extra=(--concepts "$BIOS")
  echo "[qwen-control $(date +%H:%M:%S)] start $ds"
  "$PY" -u -m geoae.evaluation.control --checkpoint "$TB" --dataset "$ds" "${extra[@]}" \
    --saliency dprime --tao 2.0 --removal_modes range --replace comp --steer_modes global,range \
    --correct_json "$JCD/joint_correct_${ds}_l35_$ARM.json" --out "$out" > "$L/range_${ds}_dprime_conf.log" 2>&1
  echo "[qwen-control $(date +%H:%M:%S)] $ds exit $? $( [ -f "$out" ] && grep -q '"summary"' "$out" && echo ok || echo INCOMPLETE)"
done
echo "[qwen-control] all done $(date)"
