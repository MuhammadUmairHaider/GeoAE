#!/bin/bash
# PRELIMINARY read of the Gemma 3 4B arm from an intermediate checkpoint, on a spare H100, while training runs:
#   sbatch scripts/argo/preview_g4b.sbatch <step_XXXX.pt> <tag>
# Runs the fast `evals` group (mmlu, tokstruct, probe, cq) with exactly the commands run_delta_evals.sh uses,
# but on the given checkpoint and into results/delta/g4b_l33_d5120_preview_<tag>/ — never the real arm dir,
# so the final evals (which skip outputs that already exist) are unaffected.
#SBATCH --job-name=g4b-preview
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
CK="${1:?checkpoint}"; TAG="${2:?tag}"
PY=scripts/delta/py
E=e2e/checkpoints/general/gemma3-4B/layer33/delta
KM=$E/balanced_kmeans_k2000_dpc_sampled_struct5.npz
KMT=$E/balanced_kmeans_k2000_dpc_sampled_struct5_tokmean.npz
OUT=results/delta/g4b_l33_d5120_preview_$TAG
mkdir -p "$OUT/evals" "$OUT/logs" logs
SHM="/dev/shm/geoae_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$SHM"' EXIT
trap 'exit 143' TERM INT
ACT="$SHM/activations_gemma3_4b_sampled_struct5_10M"
mkdir -p "$ACT"; echo "[stage] dump -> $ACT"; time cp -rL activations_gemma3_4b_sampled_struct5_10M/. "$ACT/"
run() { local n="$1"; shift; echo "[preview $(date +%H:%M:%S)] start $n"; "$@" > "$OUT/logs/$n.log" 2>&1; local rc=$?
        echo "[preview $(date +%H:%M:%S)] done $n rc=$rc"; tail -2 "$OUT/logs/$n.log"; return $rc; }
( run mmlu "$PY" -u -m geoae.interp.mmlu_splice --checkpoint "$CK" --out "$OUT/evals/mmlu.json" ) & P1=$!
( run tokstruct "$PY" -u eval_out/token_structure.py --arms "base=$KM,base_tokmean=$KMT,bypass=$CK" \
      --activations_dir "$ACT" --layer 33 --out "$OUT/evals/token_structure.json"
  run probe "$PY" -u eval_out/probe_chance_corrected.py --cache cache/gemma3-4b/l33 --models "bypass=$CK" \
      --baselines "base=$KM,base_tokmean=$KMT" --atlas_last "" --out "$OUT/evals/probe.json"
  run cq "$PY" -u -m geoae.interp.clustering_quality --baseline "$KM" --checkpoints "$CK" --names base bypass \
      --activations_dir "$ACT" --layer 33 --n_sample 1000000 --seed 0 --out "$OUT/evals/cq.json" ) & P2=$!
wait $P1; wait $P2
echo "[preview] done -> $OUT/evals"; ls -l "$OUT/evals"
