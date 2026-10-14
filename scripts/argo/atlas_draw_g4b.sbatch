#!/bin/bash
# One additional FineWeb-Atlas draw on Gemma 3 4B L33, as the paper's Atlas rows were run: the model's kept classes
# (>= 40 documents right under the base model and the AE splice), then the three-way true-positive control run
# (TP3=1 scripts/argo/control_three_way.sbatch: base, GeoAE, Gemma Scope 16k) and its SAE top-1 arm on the same documents.
#   sbatch scripts/argo/atlas_draw_g4b.sbatch <seed>
# Output: results/delta/g4b_l33_d5120/interventions/sae/range_intervention_atlas_s<seed>_dprime_{hzs,s1}_tp3.json.
# A finished stage is skipped, so the same line can be resubmitted. Normally started by scripts/argo/atlas_new_draws.sh.
#SBATCH --job-name=atlas-draw-g4b
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-10:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
S="${1:?seed}"; DS=atlas_s$S
CK=checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
SD=results/delta/d6144/interventions/atlas_classes/s$S
OUT=results/delta/g4b_l33_d5120/interventions/sae
[ -f "$SD/cfg.json" ] || { echo "[draw-g4b] $SD/cfg.json missing: run scripts/argo/atlas_draw_prep.sbatch $S first"; exit 1; }
mkdir -p "$SD/logs" "$OUT"
echo "[draw-g4b] $DS host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f "$SD/jc_report_g4b.json" ]; then
  echo "[draw-g4b $(date +%H:%M:%S)] building the kept-class list"
  HF_HUB_OFFLINE=1 "$PY" -u eval_out/atlas_classes_joint_correct.py --seed "$S" --checkpoint "$CK" --n_fit 80 --n_eval 50 \
      --max_try 600 --min_test 40 --classes all --tag _g4b > "$SD/logs/joint_correct_g4b.log" 2>&1 || { echo "[draw-g4b] kept-class list failed"; exit 1; }
fi
done_ok() { [ -f "$1" ] && grep -q '"summary"' "$1"; }
rc=0
for SUBS in h,z,s s1; do
  f=$OUT/range_intervention_${DS}_dprime_${SUBS//,/}_tp3.json
  if done_ok "$f"; then echo "[draw-g4b] skip $SUBS (done)"; continue; fi
  rm -f "$f"
  echo "[draw-g4b $(date +%H:%M:%S)] start $SUBS"
  TP3=1 SUBS=$SUBS bash scripts/argo/control_three_way.sbatch "$DS" > "logs/atlas_draw_g4b_${DS}_${SUBS//,/}.log" 2>&1
  r=$?; [ "$r" -ne 0 ] && rc=$r
  echo "[draw-g4b $(date +%H:%M:%S)] $SUBS exit $r $(done_ok "$f" && echo ok || echo INCOMPLETE)"
done
echo "[draw-g4b] finished rc=$rc $(date)"
exit "$rc"
