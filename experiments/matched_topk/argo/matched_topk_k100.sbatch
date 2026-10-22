#!/usr/bin/env bash
# Usage: sbatch matched_topk_k100.sbatch train STAGE_DIR
#        sbatch --dependency=afterok:TRAIN matched_topk_k100.sbatch eval STAGE_DIR TRAIN
# One new BatchTopK arm; existing evaluations use their own source snapshots.
#SBATCH --job-name=matched-topk-k100
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodelist=argo76
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk/logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export PYTHONUNBUFFERED=1
[[ "$(hostname -s)" == argo76 ]] || { echo "This amendment requires argo76" >&2; exit 2; }
mode="${1:?train|eval}"; stage_dir="${2:?Exact experiment stage directory required}"
[[ "$stage_dir" == /dev/shm/geoae_matched_topk_${UID}_k100_* ]] || { echo "Invalid stage directory" >&2; exit 2; }
out="$GEOAE_ROOT/results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk"
job="$out/provenance/jobs/${SLURM_JOB_ID}"
mkdir -p "$job" "$out/logs"
py="$UV_PROJECT_ENVIRONMENT/bin/python"
ckpt="$GEOAE_ROOT/checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/best_val.pt"
if [[ "$mode" == train ]]; then
    snapshot="$job/source"; mkdir -p "$snapshot"
    tar --exclude=__pycache__ --exclude='*.pyc' -cf "$job/source.tar" geoae experiments/matched_topk/argo/matched_topk_k100.sbatch \
        results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk/analysis_plan.md
    tar -xf "$job/source.tar" -C "$snapshot"
    sha256sum "$job/source.tar" > "$job/source.sha256"
    export PYTHONPATH="$snapshot${PYTHONPATH:+:$PYTHONPATH}"
    cd "$snapshot"
    printf '%s\n' "phase: stage immutable dump on argo76"
    "$py" -u -m geoae.baselines.matched_topk.stage \
        --source "$DELTA_DATA/activations/activations_gemma3_4b_sampled_struct5_10M" --target "$stage_dir"
    printf '%s\n' "phase: disposable k100 throughput benchmark"
    "$py" -u -m geoae.baselines.matched_topk.benchmark --out "$out" --widths 20480 --methods batch_topk --k 100 --steps 32
    args=(--dump "$stage_dir" --out "$out" --geoae-checkpoint "$ckpt" --widths 20480 --methods batch_topk --k 100 \
        --analysis-plan "$snapshot/results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk/analysis_plan.md")
    printf '%s\n' "phase: real-data k100 preprocessing/checkpoint smoke (separate weights)"
    "$py" -u -m geoae.baselines.matched_topk.train "${args[@]}" --smoke-steps 24 --log-every 4
    printf '%s\n' "phase: fresh fixed-budget 500M-visit k100 training"
    "$py" -u -m geoae.baselines.matched_topk.train "${args[@]}"
elif [[ "$mode" == eval ]]; then
    train_job="${3:?Training job ID required for immutable evaluation source}"
    [[ "$train_job" =~ ^[0-9]+$ ]] || exit 2
    snapshot="$out/provenance/jobs/$train_job/source"
    [[ -f "$snapshot/geoae/baselines/matched_topk/evaluate.py" ]] || exit 2
    cp "$out/provenance/jobs/$train_job/source.sha256" "$job/source.sha256"
    printf '%s\n' "$snapshot" > "$job/source_path.txt"
    export PYTHONPATH="$snapshot${PYTHONPATH:+:$PYTHONPATH}"
    cd "$snapshot"
    "$py" -u -m geoae.baselines.matched_topk.evaluate --out "$out" --cache-dir "$DELTA_DATA/cache/gemma_scope_compare" \
        --geoae-checkpoint "$ckpt" --widths 20480 --methods batch_topk --k 100 --scope-sanity
else
    echo "Unknown mode: $mode" >&2; exit 2
fi
