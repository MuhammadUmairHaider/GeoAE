#!/bin/bash
# Last-word FineWeb-Atlas alignment of the Gemma 3 4B L33 arms added on 11 Oct: the K = 256 bypass AE and the K = 2,000
# component ablations (nopull, nosep, novicreg, kmppinit, nostage; nosinkhorn = the 7 Oct no-balancing run A). Same tools and rows as atlas_lastword_k4000.sbatch /
# atlas_lastword_gs16k_plain_kscale.sbatch: codebook labels on the 100,000 last-word rows, then homogeneity /
# completeness / NMI on the rows whose cluster holds > 5 rows in every compared space (eval_out/atlas_nmi_filtered.py).
#   sbatch scripts/argo/atlas_lastword_ablations.sbatch k256 nopull nosep novicreg        (any subset; CPU only, ~10 min)
# Labels   OUT/assign_<arm>.npz           (k256: bypass256 + plain256 + plain_tokmean256, the plain k-means fits at K = 256)
# Scores   OUT/nmi_filtered_<arm>.json    ablation arm: plain, plain_tokmean, <arm>, bypass (ref bypass = the full recipe)
#          OUT/nmi_filtered_k256.json     plain256, plain_tokmean256, bypass256 (ref bypass256)
#          OUT/nmi_filtered_joint.json    every ablation arm labelled so far + plain, plain_tokmean, bypass on ONE row set
# OUT = results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/ablations
#SBATCH --job-name=atlas-abl
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-01:30:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
CACHE=/scratch/dogar/geoae/atlas_g4b_l33_lastword
META=/scratch/dogar/geoae/atlas_concept_meta.json
BASE=results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword
OUT=$BASE/ablations; mkdir -p "$OUT" logs
B=checkpoints/gemma3-4B/layer33
E=e2e/checkpoints/general/gemma3-4B/layer33/delta
echo "[atlas-abl] arms: $* host $(hostname) $(date)"
rc=0
for a in "$@"; do
  [ -f "$OUT/assign_$a.npz" ] && { echo "[atlas-abl] labels of $a exist"; continue; }
  if [ "$a" = k256 ]; then
    M="bypass256=$B/k256_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt"
    BL="plain256=$E/plain_kmeans_k256_dpc_sampled_struct5.npz,plain_tokmean256=$E/plain_kmeans_k256_dpc_sampled_struct5_tokmean.npz"
  else
    M="$a=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias_$a/step_0014450.pt"; BL=""
  fi
  "$PY" -u eval_out/atlas_assign_extra.py --cache "$CACHE" --models "$M" --baselines "$BL" --out "$OUT/assign_$a.npz" \
      2>&1 | tee "$OUT/assign_$a.log" || rc=1
done
score() {      # name arms ref assign-files
  "$PY" -u eval_out/atlas_nmi_filtered.py --assign "$4" --chunks "$CACHE/chunks.npz" --concepts "$META" \
      --arms "$2" --ref "$3" --out "$OUT/nmi_filtered_$1.json" 2>&1 | tee "$OUT/nmi_filtered_$1.log" || rc=1
}
JOINT=""; JF="$BASE/assign.npz"
for a in nosinkhorn nosinkhorn_lloyd_noreinit nopull nosep novicreg kmppinit nostage; do
  [ -f "$OUT/assign_$a.npz" ] || continue
  JOINT="$JOINT,$a"; JF="$JF,$OUT/assign_$a.npz"
  case " $* " in *" $a "*) score "$a" "plain,plain_tokmean,$a,bypass" bypass "$BASE/assign.npz,$OUT/assign_$a.npz" ;; esac
done
case " $* " in *" k256 "*) [ -f "$OUT/assign_k256.npz" ] && score k256 "plain256,plain_tokmean256,bypass256" bypass256 "$OUT/assign_k256.npz" ;; esac
[ -n "$JOINT" ] && score joint "plain,plain_tokmean${JOINT},bypass" bypass "$JF"
echo "[atlas-abl] finished rc=$rc $(date)"
exit "$rc"
