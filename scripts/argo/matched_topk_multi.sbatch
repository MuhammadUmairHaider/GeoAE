#!/usr/bin/env bash
# Usage: sbatch scripts/argo/matched_topk_multi.sbatch [--resume]
# One job, two GPUs, one staged dump copy. Slurm gives every job a private
# /dev/shm, so separate per-width jobs each copied 96 GB and ran out of tmpfs
# (285316/285317). GPU 0 trains 20480; GPU 1 trains 10240, then 5120.
#SBATCH --job-name=matched-topk-train
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=300G
#SBATCH --time=30:00:00
#SBATCH --output=results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk/logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1
resume=("$@")
out="$GEOAE_ROOT/results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk"
job="$out/provenance/jobs/${SLURM_JOB_ID}"
mkdir -p "$job" "$out/logs"
snapshot="$job/source"; mkdir -p "$snapshot"
tar --exclude=__pycache__ --exclude='*.pyc' -cf "$job/source.tar" geoae scripts/argo/matched_topk_multi.sbatch
tar -xf "$job/source.tar" -C "$snapshot"
sha256sum "$job/source.tar" > "$job/source.sha256"
export PYTHONPATH="$snapshot${PYTHONPATH:+:$PYTHONPATH}"
py="$UV_PROJECT_ENVIRONMENT/bin/python"
cd "$snapshot"
stage_dir="/dev/shm/geoae_matched_topk_${UID}_${SLURM_JOB_ID}"
trap 'rm -rf -- "$stage_dir"' EXIT
"$py" -u -m geoae.matched_topk.stage \
    --source "$DELTA_DATA/activations/activations_gemma3_4b_sampled_struct5_10M" --target "$stage_dir"
ckpt="$GEOAE_ROOT/checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/best_val.pt"
train() {  # gpu width...
    local gpu=$1; shift
    for w in "$@"; do
        CUDA_VISIBLE_DEVICES=$gpu "$py" -u -m geoae.matched_topk.train --dump "$stage_dir" --out "$out" \
            --geoae-checkpoint "$ckpt" --widths "$w" "${resume[@]}" > "$out/logs/train_${w}_${SLURM_JOB_ID}.log" 2>&1
    done
}
train 0 20480 & p0=$!
train 1 10240 5120 & p1=$!
status=0
wait $p0 || status=1
wait $p1 || status=1
exit $status
