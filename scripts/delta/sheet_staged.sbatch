#!/bin/bash
# Run one eval_out command sheet on one GPU, with activations_sampled_10M staged to
# node-local /tmp first. Same header/behaviour as scripts/delta/sheet.sbatch — use
# THIS one instead of sheet.sbatch for any sheet/group that reads the activation
# dump (tokstruct, cq, csteer, cgen, cgen_concept in eval_out/run_delta_evals.sh;
# km in eval_out/run_delta_prep.sh). Groups that never touch the dump (probe, mmlu,
# number*, steer, range_*, cjudge*) gain nothing from staging — see
# scripts/delta/submit_delta_evals.sh, which picks the right one per group.
#
#   sbatch scripts/delta/sheet_staged.sbatch eval_out/run_delta_evals.sh d6144 evals
#   sbatch scripts/delta/sheet_staged.sbatch eval_out/run_delta_prep.sh km
#
# Staging mirrors scripts/delta/train.sbatch exactly: resolve the
# activations_sampled_10M symlink (it points into /work/hdd), content-check for
# layer_*.npy, `time rsync -a` the WHOLE directory (activations + every sidecar —
# rows_tok.npy, rows_doc.npy, rows_pos.npy, rows_src.npy, rows_norm.npy, docs.npy,
# meta.json, norm_params_layer*.npz) to /tmp/$SLURM_JOB_ID/activations_sampled_10M,
# then exports ACT_DIR to that path so the sheet's own ACT="${ACT_DIR:-...}" picks
# it up. Per-job /tmp is private node-local NVMe and is wiped at job end — nothing
# here persists, and nothing here writes back to the Lustre copy (unlike
# train.sbatch, which does write norm_params back; this dump's norm_params already
# exists from training and is read-only from every eval sheet's point of view).
#
# Outside Slurm (no SLURM_JOB_ID — e.g. sourced or run directly on a login node)
# this does NOT stage anything and leaves ACT_DIR unset, so the sheet falls back to
# reading the Lustre symlink directly (slow, but correct) — see the `bash -x` dry
# local-execution test in this session's report.
#SBATCH --job-name=geoae-sheet-staged
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
set -euo pipefail
SHEET="${1:?usage: sbatch sheet_staged.sbatch <eval_out/run_*.sh> [steps...]}"; shift
source scripts/delta/env.sh
[ -f "$SHEET" ] || { echo "no such sheet: $SHEET"; exit 1; }
mkdir -p logs

ACT_NAME=activations_sampled_10M
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SRC="$(readlink -f "$GEOAE_ROOT/$ACT_NAME")"
    if [ ! -d "$SRC" ] || [ -z "$(ls "$SRC"/layer_*.npy 2>/dev/null)" ]; then
        echo "no extracted dump at $SRC (no layer_*.npy)."
        echo "Run the extraction first:  sbatch scripts/delta/extract_sampled.sbatch <extract config>"
        exit 1
    fi
    STAGE="/tmp/${SLURM_JOB_ID}/${ACT_NAME}"
    mkdir -p "$STAGE"
    echo "[stage] $SRC -> $STAGE"
    time rsync -a "$SRC/" "$STAGE/"
    export ACT_DIR="$STAGE"
    echo "[stage] ACT_DIR=$ACT_DIR"
else
    echo "[stage] no SLURM_JOB_ID — not staging; ACT_DIR left unset (sheet falls back to $ACT_NAME)"
fi

echo "[sheet] $SHEET $*"
bash "$SHEET" "$@"
