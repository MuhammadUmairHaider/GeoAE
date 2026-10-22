#!/bin/bash
# Is the TOKEN-REMOVED residual stream a different base for plain steering? One model, one dataset.
# Adds substrate hb (range_intervention_compare.py --hb_table: the edit is applied to (h - mean)/std - b[tok] and b[tok] is
# added back, with the bypass AE's own token table, no encoder) next to h (raw residual) and z (GeoAE latent), on the SAME
# two-way true-positive documents as experiments/paper/argo/control_tp2_plain.sbatch (same pool file, seed and sizes), plain steering
# only, alpha 0.5, no perplexity pass.
#   sbatch experiments/paper/argo/control_tokrm_base.sbatch <d6144|qwen25_l35_d4096> <db14|biasbios|ag_news|emotion6|news20>
# Output: results/delta/<arm>/interventions/tp2/range_intervention_<ds>_dprime_hzhb_tp2.json (+ .log)
#SBATCH --job-name=ctrl-tokrm
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --nodelist=argo76
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0      # wikimedia/wikipedia (reference texts) is streamed from the Hub
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
ARM="${1:?arm}"; DS="${2:?dataset}"
case "$ARM" in
  d6144)            CK=checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt; LY=27
                    TB=e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz ;;
  qwen25_l35_d4096) CK=checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt; LY=35
                    TB=e2e/checkpoints/general/qwen2.5-3B/layer35/token_bias_struct5_train10M.npz ;;
  *) echo "[tokrm] unknown arm $ARM"; exit 2 ;;
esac
for f in "$CK" "$TB"; do [ -f "$f" ] || { echo "[tokrm] missing $f"; exit 1; }; done
OUT=results/delta/$ARM/interventions/tp2; mkdir -p "$OUT" logs
JC=results/delta/joint_correct/joint_correct_${DS}_l${LY}_${ARM}_tp2.json
[ -f "$JC" ] || { echo "[tokrm] missing pool $JC (run control_tp2_plain.sbatch first)"; exit 1; }
EXTRA=(--max_try "${MAX_TRY:-3000}")
[ "$DS" = biasbios ] && EXTRA+=(--concepts 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27)
[ "$DS" = ag_news ] && EXTRA+=(--extra_split train)
out=$OUT/range_intervention_${DS}_dprime_hzhb_tp2.json; rm -f "$out"
echo "[tokrm] $ARM $DS host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -u -m geoae.evaluation.control --checkpoint "$CK" --dataset "$DS" "${EXTRA[@]}" \
    --n_eval "${N_EVAL:-100}" --min_eval "${MIN_EVAL:-20}" --saliency dprime --tao 2.0 --removal_modes "" \
    --steer_modes global --alphas 0.5 --substrates h,z,hb --hb_table "$TB" --edit_ppl 0 --correct_json "$JC" \
    --out "$out" > "$OUT/range_${DS}_dprime_hzhb_tp2.log" 2>&1
r=$?
echo "[tokrm] $ARM $DS exit $r $( [ -f "$out" ] && grep -q '"summary"' "$out" && echo ok || echo INCOMPLETE) $(date)"
exit "$r"
