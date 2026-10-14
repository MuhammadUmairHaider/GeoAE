#!/bin/bash
# Symmetric k-means readout on the FineWeb-Atlas cache (Gemma 3 4B layer 33). Five spaces -- base, base_tokmean, bypass z,
# Matryoshka-10k codes, Gemma Scope 16k codes -- each get the SAME balanced k-means (K = 2000 / 4000 / 8000; rows and batch
# scale with K: 100 rows per centroid, batch 4096 x K / 2000, 40 epochs) fitted on the SAME training-dump rows, plus the
# "unit readout" (argmax over the N most-used coordinates, N = 500 / 1000 / 1500 / 2000 / 3000 / all); then every Atlas
# cache row is labelled in every space. eval_out/atlas_symmetric_kmeans.py (read its docstring for the exact procedure).
#   sbatch scripts/argo/atlas_symmetric_kmeans.sbatch                     (H100 only: gs16k at K=8000 needs ~66 GB of GPU memory)
#   sbatch scripts/argo/atlas_symmetric_kmeans.sbatch --ks 2000 --spaces base,bypass    (extra arguments go to the script)
# Resubmitting the same command RESUMES: codebooks already in $OUT/codebooks that match the requested fit are skipped (a
# mismatching one aborts; add --refit to overwrite). ACT_DIR=<dump dir> overrides the training dump.
# Output: results/delta/g4b_l33_d5120/evals/sae_suite/atlas_symmetric/{assign.npz,meta.json,codebooks/}   (~2.1 GB of codebooks)
# STAGING: /scratch is GPFS WITHOUT a page cache, and the fits read 200k / 400k / 800k scattered rows of the dump. So
# layer_33.npy (102 GB fp32) + rows_tok.npy of the training dump and the Atlas row arrays (~9 GB) are copied to
# /dev/shm/geoae_atlassymkm_<jobid>/ (tmpfs, charged to the job's cgroup: --mem=240G) and deleted on exit.
# NO_STAGE_DUMP=1 reads the dump straight from /scratch instead (as sae_controls.sbatch did; fine for a small smoke run,
# slow for 800k scattered rows). ONE GPU only: the node is shared.
#SBATCH --job-name=atlas-symkm
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
mkdir -p logs

ARM=g4b_l33_d5120; LAYER=33; K=2000
ATLAS_CACHE=/scratch/dogar/geoae/atlas_g4b_l33
OUT=results/delta/$ARM/evals/sae_suite/atlas_symmetric
E=e2e/checkpoints/general/gemma3-4B/layer$LAYER
B=checkpoints/gemma3-4B/layer$LAYER
TB=$B/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/step_0014450.pt
KM=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5.npz
KMT=$E/delta/balanced_kmeans_k${K}_dpc_sampled_struct5_tokmean.npz
TRAIN=results/delta/$ARM/evals/gemma_scope_compare/matched_topk/training
MAT=$TRAIN/matryoshka_10240/last.pt
CTL=results/delta/$ARM/evals/sae_suite/controls            # the existing code k-means (agreement check): kmeans_{mat10k,gs16k}.npz
# training dump: the repo link made by scripts/argo/link_data.sh, else $DELTA_DATA/activations (as matched_topk.sbatch)
ACT_SRC="${ACT_DIR:-activations_gemma3_4b_sampled_struct5_10M}"
[ -e "$ACT_SRC/layer_$LAYER.npy" ] || ACT_SRC="$DELTA_DATA/activations/activations_gemma3_4b_sampled_struct5_10M"

for f in "$TB" "$KM" "$KMT" "$MAT" "$CTL/kmeans_mat10k.npz" "$CTL/kmeans_gs16k.npz" \
         "$ACT_SRC/layer_$LAYER.npy" "$ACT_SRC/rows_tok.npy" \
         "$ATLAS_CACHE/rows.npy" "$ATLAS_CACHE/rows_tok.npy" "$ATLAS_CACHE/rows_nxt.npy" "$ATLAS_CACHE/rows_chunk.npy" \
         "$ATLAS_CACHE/rows_pos.npy" "$ATLAS_CACHE/chunks.npz"; do
    [ -e "$f" ] || { echo "[abort] missing $f"; exit 1; }
done
ls -d "$HF_HOME"/hub/models--google--gemma-scope-2-4b-pt* >/dev/null 2>&1 \
    || echo "[warn] no google/gemma-scope-2-4b-pt in $HF_HOME/hub: scope:16k cannot load offline (HF_HUB_OFFLINE=1) if it is not cached elsewhere"
mkdir -p "$OUT"
echo "[atlas-symkm] host $(hostname -f) $(date)  job ${SLURM_JOB_ID:-none}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[atlas-symkm] dump $ACT_SRC -> out $OUT"

# ---- stage the training dump (layer_33.npy + rows_tok.npy only) and the Atlas row arrays in tmpfs
SHM="/dev/shm/geoae_atlassymkm_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$SHM"' EXIT; trap 'exit 143' TERM INT
files=("$ATLAS_CACHE"/rows{,_tok,_nxt,_chunk,_pos}.npy)
[ -n "${NO_STAGE_DUMP:-}" ] || files+=("$ACT_SRC/layer_$LAYER.npy" "$ACT_SRC/rows_tok.npy")
need=0
for f in "${files[@]}"; do need=$(( need + $(stat -L -c %s "$f") )); done
free=$(df --output=avail -B1 /dev/shm | tail -1 | tr -d ' ')
echo "[stage] need $(( need / 2**30 )) GiB in /dev/shm, free $(( free / 2**30 )) GiB"
[ "$free" -gt $(( need + 8 * 2**30 )) ] || { echo "[abort] /dev/shm has too little free space (the node's other jobs hold dumps too)"; exit 1; }
mkdir -p "$SHM/dump" "$SHM/atlas"
time cp "$ATLAS_CACHE"/rows.npy "$ATLAS_CACHE"/rows_tok.npy "$ATLAS_CACHE"/rows_nxt.npy "$ATLAS_CACHE"/rows_chunk.npy \
        "$ATLAS_CACHE"/rows_pos.npy "$SHM/atlas/" || { echo "[abort] staging the Atlas rows failed"; exit 1; }
if [ -n "${NO_STAGE_DUMP:-}" ]; then
    DUMP_DIR="$ACT_SRC"
else
    time cp "$ACT_SRC/layer_$LAYER.npy" "$ACT_SRC/rows_tok.npy" "$SHM/dump/" || { echo "[abort] staging the dump failed"; exit 1; }
    for f in layer_$LAYER.npy rows_tok.npy; do
        [ "$(stat -L -c %s "$ACT_SRC/$f")" = "$(stat -c %s "$SHM/dump/$f")" ] || { echo "[abort] short copy of $f"; exit 1; }
    done
    DUMP_DIR="$SHM/dump"
fi
df -h /dev/shm | tail -1

"$PY" -u eval_out/atlas_symmetric_kmeans.py --cache "$ATLAS_CACHE" --rows_dir "$SHM/atlas" \
    --activations_dir "$DUMP_DIR" --layer "$LAYER" \
    --models "bypass=$TB" --baselines "base=$KM,base_tokmean=$KMT" \
    --saes "mat10k=matched:$MAT,gs16k=scope:16k" \
    --controls_dir "$CTL" --out_dir "$OUT" "$@" 2>&1 | tee -a logs/atlas_symmetric_kmeans.log
exit "${PIPESTATUS[0]}"
