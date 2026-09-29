#!/usr/bin/env bash
# Submit eval_out/run_delta_evals.sh groups as Slurm jobs, one job per group.
#
#   scripts/delta/submit_delta_evals.sh <arm> [groups...]   # default groups below
#   scripts/delta/submit_delta_evals.sh prep                 # bench+cache prep job (staged)
#   scripts/delta/submit_delta_evals.sh km                   # the 4 balanced-k-means fits (staged)
#
#   scripts/delta/submit_delta_evals.sh d6144
#   scripts/delta/submit_delta_evals.sh d6144 evals steering
#   DEP=afterok:12345 scripts/delta/submit_delta_evals.sh d6144 interventions
#   DRY=1 scripts/delta/submit_delta_evals.sh d6144            # print sbatch lines only
#
# arm: d768 | d3072 | d6144 | d12288 | d6144k4000 | d12288k4000
# Default groups (if none given): evals interventions steering concept — NOT judge
# (the LLM-judge steps call an external API, are cheap, non-GPU, and must run
# SERIALLY across arms from the login node — geoae/interp/llm_judge.py's disk
# cache write is not atomic and base prompts are shared between arms; see
# eval_out/run_delta_evals.sh's docstring. Submit `judge` yourself, one arm at a
# time, via `bash eval_out/run_delta_evals.sh <arm> judge` directly — not through
# this submitter).
#
# STAGING. tokstruct/cq (in evals) and csteer/cgen (in steering) and cgen_concept
# (in concept) read activations_sampled_10M/layer_27.npy — scattered random-row
# access over 61 GB on /work/hdd Lustre is very slow — so those three groups, plus
# `prep` (its `km`/`bench`/`cache` steps) and the standalone `km` target, submit
# through scripts/delta/sheet_staged.sbatch, which rsyncs the whole dump to
# node-local /tmp first and exports ACT_DIR. `interventions` never touches the
# dump (number/steer/range_*/number_hb all do live LM forward passes on small
# datasets, not the pre-extracted dump — verified by grepping every tool for
# activations_dir/layer_*.npy/rows_*.npy reads), so it stays on plain
# scripts/delta/sheet.sbatch — staging it would rsync 61 GB for nothing.
#
# Each group becomes its own job:
#   scripts/delta/sheet[_staged].sbatch eval_out/run_delta_evals.sh <arm> <group>
# with -p gpuA100x4,gpuA40x4 --gpus-per-node=1 --cpus-per-task=8 --mem=64g and a
# per-group time limit (evals 6h, interventions 10h, steering 4h, concept 8h).
# arm=d12288's evals group gets --mem=128g (clustering_quality holds
# n_sample x latent_dim float32 latents in RAM: 1M x 12288 = 49 GB).
#
# DEP, if set, is passed to every submitted job as --dependency=$DEP.
# DRY=1 prints the resolved `sbatch` command lines instead of submitting anything.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ARM="${1:-}"; shift || true
SEL_GROUPS=("$@")

declare -A TIME=( [evals]=6:00:00 [interventions]=10:00:00 [steering]=4:00:00 [concept]=8:00:00 )
# Which sbatch wrapper each group needs: 'staged' rsyncs the dump to node-local
# /tmp first (scripts/delta/sheet_staged.sbatch); 'plain' skips that
# (scripts/delta/sheet.sbatch) because none of that group's tools read the dump.
declare -A SHEET_FOR=( [evals]=staged [interventions]=plain [steering]=staged [concept]=staged )
COMMON=(-p gpuA100x4,gpuA40x4 --gpus-per-node=1 --cpus-per-task=8)

submit() {   # submit <job-name> <time> <mem, default 64g> <sbatch-script> -- <sheet> [args...]
    local name="$1" time="$2" mem="${3:-64g}" sbatch_script="$4"; shift 4
    [ "$1" = "--" ] && shift
    local args=(--job-name="$name" --time="$time" "${COMMON[@]}" --mem="$mem")
    [ -n "${DEP:-}" ] && args+=(--dependency="$DEP")
    if [ -n "${DRY:-}" ]; then
        echo "[dry ] sbatch ${args[*]} $sbatch_script $*"
        return 0
    fi
    local out
    out=$(sbatch "${args[@]}" "$sbatch_script" "$@")
    echo "$out"
    echo "$out" | grep -oE '[0-9]+$'
}

# prep: bench+cache (staged, per spec — see the header note; bench/cache do not
# themselves read the dump, only km does, but a staged prep job is what was asked).
if [ "$ARM" = "prep" ]; then
    submit "ev-prep" "4:00:00" "" scripts/delta/sheet_staged.sbatch -- eval_out/run_delta_prep.sh bench cache
    exit 0
fi
# km: the 4 balanced-k-means fits, one staged job (so a future refit is cheap —
# it rsyncs the dump once and fits all four codebooks in the same job).
if [ "$ARM" = "km" ]; then
    submit "ev-km" "4:00:00" "" scripts/delta/sheet_staged.sbatch -- eval_out/run_delta_prep.sh km
    exit 0
fi

case "$ARM" in
    d768|d3072|d6144|d12288|d6144k4000|d12288k4000) ;;
    *)
        echo "usage: $0 <d768|d3072|d6144|d12288|d6144k4000|d12288k4000|prep|km> [groups...]" >&2
        echo "  groups: evals interventions steering concept  (default: all four, not judge)" >&2
        exit 2
        ;;
esac
[ ${#SEL_GROUPS[@]} -eq 0 ] && SEL_GROUPS=(evals interventions steering concept)

for g in "${SEL_GROUPS[@]}"; do
    t="${TIME[$g]:-}"
    if [ -z "$t" ]; then
        echo "[abort] unknown group '$g' (have: evals interventions steering concept)" >&2
        exit 2
    fi
    mem=""
    case "$ARM" in d12288|d12288k4000) [ "$g" = evals ] && mem="128g" ;; esac
    sbatch_script=scripts/delta/sheet.sbatch
    [ "${SHEET_FOR[$g]}" = staged ] && sbatch_script=scripts/delta/sheet_staged.sbatch
    submit "ev-$ARM-$g" "$t" "$mem" "$sbatch_script" -- eval_out/run_delta_evals.sh "$ARM" "$g"
done
