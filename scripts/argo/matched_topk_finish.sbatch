#!/usr/bin/env bash
# CPU-only report and cleanup after this experiment's eval jobs have finished.
#SBATCH --job-name=matched-topk-report
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodelist=argo76
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk/logs/slurm-%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/argo/env.sh
[[ "$(hostname -s)" == argo76 ]] || { echo "This amendment requires argo76" >&2; exit 2; }
stage_dir="${1:?Exact argo76 k100 stage directory required}"
[[ "$stage_dir" == /dev/shm/geoae_matched_topk_${UID}_k100_* ]] || { echo "Invalid cleanup directory" >&2; exit 2; }
out="$GEOAE_ROOT/results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk"
job="$out/provenance/jobs/${SLURM_JOB_ID}"
snapshot="$job/source"; mkdir -p "$snapshot"
tar --exclude=__pycache__ --exclude='*.pyc' -cf "$job/source.tar" geoae scripts/argo/matched_topk_finish.sbatch
tar -xf "$job/source.tar" -C "$snapshot"
sha256sum "$job/source.tar" > "$job/source.sha256"
export PYTHONPATH="$snapshot${PYTHONPATH:+:$PYTHONPATH}"
cd "$snapshot"
"$UV_PROJECT_ENVIRONMENT/bin/python" -m geoae.matched_topk.report --out "$out" \
    --widths 10240 20480 --extra-models batch_topk_20480_k100
"$UV_PROJECT_ENVIRONMENT/bin/python" - "$out/summary.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report["status"] != "complete":
    raise SystemExit("Report is incomplete; preserving staged data")
PY
# Only this experiment's own tmpfs directory. No sessions or other staging are touched.
if [[ -d "$stage_dir" ]]; then
    (
        flock -x 9
        rm -f -- "$stage_dir/layer_33.npy" "$stage_dir/rows_tok.npy" "$stage_dir/norm_params_layer33.npz" "$stage_dir/meta.json"
        rm -f -- "$stage_dir/layer_33.npy.receipt.json" "$stage_dir/rows_tok.npy.receipt.json" "$stage_dir/norm_params_layer33.npz.receipt.json" "$stage_dir/meta.json.receipt.json"
    ) 9>"$stage_dir/stage.lock"
    rm -f -- "$stage_dir/stage.lock"
    rmdir -- "$stage_dir"
fi
