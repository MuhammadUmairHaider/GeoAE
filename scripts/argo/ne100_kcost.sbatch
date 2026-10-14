#!/bin/bash
# K rows of the 12 Oct no-balancing / 100-epoch Gemma 3 4B L33 bypass models (K = 4,000 and 8,000; K = 2,000 only for
# the Atlas K-scaling file), with the commands of the box sheet logs/run_kcost_g4b.sh and of
# scripts/argo/atlas_lastword_gs16k_plain_kscale.sbatch; baselines = plain k-means at the same K (already fitted).
#   sbatch --mem=260G scripts/argo/ne100_kcost.sbatch <4000|8000>      (cq, token structure, MMLU, perplexity, Atlas)
#   sbatch --mem=32G  scripts/argo/ne100_kcost.sbatch 2000             (Atlas only; needs ne100_evals.sbatch gemma bypass first)
# Output: results/delta/g4b_l33_d5120k<K>_ne100/evals/{cq.json,token_structure.json,mmlu.json,splice_ppl.json,
#         atlas_lastword/assign.npz,atlas_lastword/nmi_filtered4.json}; K = 2,000: results/delta/g4b_l33_d5120_ne100/evals/atlas_lastword/nmi_filtered4.json
#SBATCH --job-name=ne100-kcost
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=260G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTHONPATH=. OMP_NUM_THREADS=8
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
K="${1:?K}"
LAYER=33; ACT=activations_gemma3_4b_sampled_struct5_10M
E=e2e/checkpoints/general/gemma3-4B/layer33/delta
ATL=/scratch/dogar/geoae/atlas_g4b_l33_lastword; META=/scratch/dogar/geoae/atlas_concept_meta.json
BASE=results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword
rc=0
step() { local name=$1 out=$2; shift 2
  if [ -s "$out" ]; then echo "[kcost $(date +%H:%M:%S)] skip $name"; return 0; fi
  echo "[kcost $(date +%H:%M:%S)] start $name"; "$@" > "$R/logs/$name.log" 2>&1; local r=$?
  echo "[kcost $(date +%H:%M:%S)] done $name rc=$r"; [ $r -ne 0 ] && { rc=1; tail -4 "$R/logs/$name.log"; }; return 0; }
if [ "$K" = 2000 ]; then
  R=results/delta/g4b_l33_d5120_ne100; O=$R/evals; mkdir -p "$O/atlas_lastword" "$R/logs"
  [ -s "$O/atlas_lastword/assign_bypass.npz" ] || { echo "[abort] run ne100_evals.sbatch gemma bypass first"; exit 1; }
  step atlas_filtered4 "$O/atlas_lastword/nmi_filtered4.json" "$PY" -u eval_out/atlas_nmi_filtered.py \
      --assign "$BASE/assign.npz,$BASE/assign_gs16k_plain.npz,$O/atlas_lastword/assign_bypass.npz" --chunks "$ATL/chunks.npz" --concepts "$META" \
      --arms "plain,plain_tokmean,gs16k_plain,bypass_ne100" --ref bypass_ne100 --out "$O/atlas_lastword/nmi_filtered4.json"
  echo "[kcost 2000] finished rc=$rc"; exit "$rc"
fi
S=$((K / 1000))k
CK=checkpoints/gemma3-4B/layer33/k${K}_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias_nobal_e100/step_0028900.pt
P=$E/plain_kmeans_k${K}_dpc_sampled_struct5.npz; PT=$E/plain_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
R=results/delta/g4b_l33_d5120k${K}_ne100; O=$R/evals; mkdir -p "$O/atlas_lastword" "$R/logs" logs
for f in "$CK" "$P" "$PT" "$BASE/k8000/assign_plainK.npz" "$BASE/k8000/assign_plainKt.npz" "$BASE/kscale/assign_gs16k_plainK.npz"; do
  [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
echo "[kcost K=$K] $CK host $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SHM="/dev/shm/geoae_${SLURM_JOB_ID:-$$}"; trap 'rm -rf "$SHM"' EXIT; trap 'exit 143' TERM INT
A="$SHM/$ACT"; mkdir -p "$A"; echo "[stage] $ACT -> $A"; time cp -rL "$ACT/." "$A/"
step cq "$O/cq.json" "$PY" -u -m geoae.interp.clustering_quality --baseline "$P" --checkpoints "$CK" --names "plain$S" "bypass$S" \
    --activations_dir "$A" --layer $LAYER --n_sample 1000000 --seed 0 --out "$O/cq.json"
step token_structure "$O/token_structure.json" "$PY" -u eval_out/token_structure.py --arms "plain$S=$P,plain_tokmean$S=$PT,bypass$S=$CK" \
    --activations_dir "$A" --layer $LAYER --out "$O/token_structure.json"
rm -rf "$SHM"
step mmlu "$O/mmlu.json" "$PY" -u -m geoae.interp.mmlu_splice --checkpoint "$CK" --out "$O/mmlu.json"
step splice_ppl "$O/splice_ppl.json" "$PY" -u -m geoae.interp.splice_ppl --ae "$O/splice_ppl.json=$CK"
step atlas_assign "$O/atlas_lastword/assign.npz" "$PY" -u eval_out/atlas_assign_extra.py --cache "$ATL" --models "bypass${S}_ne100=$CK" \
    --out "$O/atlas_lastword/assign.npz"
step atlas_filtered4 "$O/atlas_lastword/nmi_filtered4.json" "$PY" -u eval_out/atlas_nmi_filtered.py \
    --assign "$BASE/k8000/assign_plainK.npz,$BASE/k8000/assign_plainKt.npz,$BASE/kscale/assign_gs16k_plainK.npz,$O/atlas_lastword/assign.npz" \
    --chunks "$ATL/chunks.npz" --concepts "$META" --arms "plain$S,plain_tokmean$S,gs16k_plain$S,bypass${S}_ne100" --ref "bypass${S}_ne100" \
    --out "$O/atlas_lastword/nmi_filtered4.json"
echo "[kcost K=$K] finished rc=$rc $(date)"
exit "$rc"
