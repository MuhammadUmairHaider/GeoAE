#!/usr/bin/env bash
# Usage: sbatch scripts/argo/matched_topk.sbatch smoke|train|eval|benchmark [width]
# Independent of gemma_scope.sbatch and all existing jobs.
#SBATCH --job-name=matched-topk
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=130G
#SBATCH --time=24:00:00
#SBATCH --output=results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk/logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
export PYTHONUNBUFFERED=1
mode="${1:?smoke|train|eval}"; width="${2:-5120}"
out="$GEOAE_ROOT/results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk"
mkdir -p "$out/provenance/jobs/${SLURM_JOB_ID}" logs
snapshot="$out/provenance/jobs/${SLURM_JOB_ID}/source"
mkdir -p "$snapshot"
tar --exclude=__pycache__ --exclude='*.pyc' -cf "$out/provenance/jobs/${SLURM_JOB_ID}/source.tar" geoae scripts/argo/matched_topk.sbatch
tar -xf "$out/provenance/jobs/${SLURM_JOB_ID}/source.tar" -C "$snapshot"
sha256sum "$out/provenance/jobs/${SLURM_JOB_ID}/source.tar" > "$out/provenance/jobs/${SLURM_JOB_ID}/source.sha256"
export PYTHONPATH="$snapshot${PYTHONPATH:+:$PYTHONPATH}"
python_bin="$UV_PROJECT_ENVIRONMENT/bin/python"
cd "$snapshot"
if [[ "$mode" == smoke || "$mode" == train ]]; then
    stage_dir="/dev/shm/geoae_matched_topk_${UID}_20261004"
    "$python_bin" -u -m geoae.matched_topk.stage \
        --source "$DELTA_DATA/activations/activations_gemma3_4b_sampled_struct5_10M" --target "$stage_dir"
    train_args=(--dump "$stage_dir" --out "$out" --geoae-checkpoint "$GEOAE_ROOT/checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/best_val.pt")
    if [[ "$mode" == smoke ]]; then
        "$python_bin" -u -m geoae.matched_topk.train "${train_args[@]}" --widths 5120 10240 20480 --smoke-steps 24 --log-every 4
        "$python_bin" -u -m geoae.matched_topk.evaluate --out "$out" --cache-dir "$DELTA_DATA/cache/gemma_scope_compare" \
            --geoae-checkpoint "${train_args[5]}" --smoke --widths 5120 10240 20480
    else
        "$python_bin" -u -m geoae.matched_topk.train "${train_args[@]}" --widths "$width"
    fi
elif [[ "$mode" == eval ]]; then
    extra_eval=()
    if [[ "$width" == 5120 ]]; then extra_eval=(--scope-sanity); fi
    "$python_bin" -u -m geoae.matched_topk.evaluate --out "$out" --cache-dir "$DELTA_DATA/cache/gemma_scope_compare" \
        --geoae-checkpoint "$GEOAE_ROOT/checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/best_val.pt" --widths "$width" "${extra_eval[@]}"
elif [[ "$mode" == benchmark ]]; then
    "$python_bin" -u -m geoae.matched_topk.benchmark --out "$out"
else
    echo "unknown mode: $mode" >&2; exit 2
fi
