#!/bin/bash
# Direction-projection faithfulness test (Gemma 3 4B L33): per concept, ONE assigned direction per method is removed from the
# UNEDITED base model's residual by orthogonal projection, h - (h.d)d, and the change in gold-label confidence is read on the
# concept's own documents and on other-class documents. All methods share the model, the documents and the starting
# confidence (no reconstruction is spliced in). geoae/evaluation/control.py --proj.
# Methods: label (supervised mean difference), km / kmt (plain k-means on the residual / token-removed residual, seed 42),
#          geoae (bypass AE), nobypass (no-bypass AE), sae_km (plain k-means on Gemma Scope 16k codes), sae_f1 / sae_f5
#          (Gemma Scope's top-1 / top-5 features), h_n1/5, geoae_z1/5, nobypass_z1/5 (top residual / AE-latent dimensions).
#          Records h_pj_<m> (plain projection) and h_pjc_<m> (centred on the class-balanced answer-position mean).
#          Documents: the joint-correct pools of the control tables.
#   sbatch experiments/paper/argo/control_proj.sbatch <db14|biasbios|ag_news|emotion6|news20|atlas_s0|atlas_s1|atlas_s2> [--smoke]
#   sbatch --gres=gpu:l40:1 --nodelist=argo76 experiments/paper/argo/control_proj.sbatch ag_news
# Output: results/delta/g4b_l33_d5120/interventions/proj/direction_projection_<ds>.json
#SBATCH --job-name=control-proj
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --nodelist=argo73
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0      # wikimedia/wikipedia (perplexity texts) is streamed from the Hub
DS="${1:?dataset}"; shift || true
ARM=g4b_l33_d5120
B=checkpoints/gemma3-4B/layer33; E=e2e/checkpoints/general/gemma3-4B/layer33/delta
CK=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
NB=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_train10M/step_0015250.pt
EXTRA=(); TAG=""
[ "$DS" = biasbios ] && EXTRA=(--concepts 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27)
JC=results/delta/joint_correct/joint_correct_${DS}_l33_$ARM.json
case "$DS" in atlas_s*)        # FineWeb-Atlas draw: frozen 14-topic classification set, Gemma pool of the Atlas control rows
    AD=results/delta/d6144/interventions/atlas_classes/s${DS#atlas_s}; JC=$AD/joint_correct_kept_g4b.json
    EXTRA=(--concepts "$("$UV_PROJECT_ENVIRONMENT/bin/python" -c "import json;print(json.load(open('$AD/jc_report_g4b.json'))['concepts_arg'])")" --max_try 600) ;;
esac
[ "${1:-}" = --smoke ] && { EXTRA=(--smoke); TAG=_smoke; }
OUT=results/delta/$ARM/interventions/proj; mkdir -p "$OUT"
"$UV_PROJECT_ENVIRONMENT/bin/python" -u -m geoae.evaluation.control --checkpoint "$CK" --dataset "$DS" "${EXTRA[@]}" \
    --saliency dprime --tao 2.0 --removal_modes "" --steer_modes "" --substrates h \
    --proj label,km,kmt,geoae,nobypass,sae_km,sae_f1,sae_f5,h_n1,h_n5,geoae_z1,geoae_z5,nobypass_z1,nobypass_z5 --proj_ae "nobypass=$NB" \
    --km_plain $E/plain_kmeans_k2000_dpc_sampled_struct5_seed42.npz --km_tok $E/plain_kmeans_k2000_dpc_sampled_struct5_tokmean_seed42.npz \
    --sae scope:16k --sae_kmeans results/delta/$ARM/evals/sae_suite/controls/plain_kmeans_gs16k.npz \
    --correct_json "$JC" \
    --out "$OUT/direction_projection_${DS}$TAG.json" 2>&1 | tee "$OUT/direction_projection_${DS}$TAG.log"
