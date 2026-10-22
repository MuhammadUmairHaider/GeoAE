#!/usr/bin/env bash
# Submit experiments/paper/scripts/run_delta_evals.sh groups as Slurm jobs, one job per group.
#
#   experiments/paper/delta/submit_delta_evals.sh <arm> [groups...]   # default groups below
#   experiments/paper/delta/submit_delta_evals.sh prep                 # bench+cache prep job (staged)
#   experiments/paper/delta/submit_delta_evals.sh km                   # the 4 balanced-k-means fits (staged)
#   experiments/paper/delta/submit_delta_evals.sh probe_acts           # last-token acts + token ids for `probes` (plain)
#   experiments/paper/delta/submit_delta_evals.sh g1b_prep             # Gemma 3 1B: run_delta_prep.sh g1b_km g1b_cache (staged, both layers)
#   experiments/paper/delta/submit_delta_evals.sh g1b_probe_acts       # Gemma 3 1B probe activations, both layers (plain)
#   experiments/paper/delta/submit_delta_evals.sh g1b_l25_d2304 [groups...]   # a Gemma arm, same groups as the Llama arms
#
#   experiments/paper/delta/submit_delta_evals.sh d6144
#   experiments/paper/delta/submit_delta_evals.sh d6144 evals steering
#   DEP=afterok:12345 experiments/paper/delta/submit_delta_evals.sh d6144 interventions
#   DRY=1 experiments/paper/delta/submit_delta_evals.sh d6144            # print sbatch lines only
#
# arm: d768 | d3072 | d6144 | d12288 | d6144k4000 | d12288k4000 | g1b_l{13,25}_d{1152,2304,4608}
# Gemma arms (google/gemma-3-1b-pt): staged jobs export ACT_NAME=activations_gemma3_1b_sampled_struct5_10M and
# STAGE_LAYERS=<the arm's layer>, so only that layer's 46 GB file (plus the sidecars) is copied to node-local /tmp.
# Layer-13 arms get NO steering/concept jobs: every step in those groups needs the last block (layer 25) and is
# skipped by run_delta_evals.sh, so the job would only stage 46 GB to do nothing.
# Default groups (if none given): evals interventions steering concept — NOT judge
# (the LLM-judge steps call an external API, are cheap, non-GPU, and must run
# SERIALLY across arms from the login node — geoae/analysis/llm_judge.py's disk
# cache write is not atomic and base prompts are shared between arms; see
# experiments/paper/scripts/run_delta_evals.sh's docstring. Submit `judge` yourself, one arm at a
# time, via `bash experiments/paper/scripts/run_delta_evals.sh <arm> judge` directly — not through
# this submitter).
#
# `geometry` (PCA/t-SNE/UMAP figures -> results/delta/<arm>/figures/geometry/) is NOT in the
# default list: it is mostly CPU-bound t-SNE, so it is a 3h job you ask for explicitly:
#   experiments/paper/delta/submit_delta_evals.sh d6144 geometry
# It reads only the small cache_ids/ rung caches (never the dump), so it is `plain`
# (unstaged), 64g for every arm (peak is the ~2 GB pos/ner token-rung H plus <0.3 GB of
# 6000 x 12288 latents, so d12288 does not need 128g).
#
# `probes` (biasprobe + tpp -> results/delta/<arm>/probes/) is likewise NOT in the default list. It
# needs the act dirs from the one-off `probe_acts` prep job first (run_delta_prep.sh probe_acts:
# ~1-1.5 h on one GPU, live Llama forward of 70k bias_in_bios + 60k DBpedia documents), then is a
# short GPU job (--device cuda: encoding + hundreds of small matmul sweeps; ~30-60 min, 2h limit),
# `plain` (never reads the dump), 64g for every arm (largest array: 50k x 12288 fp32 = 2.5 GB
# per space):
#   experiments/paper/delta/submit_delta_evals.sh probe_acts
#   DEP=afterok:<id> experiments/paper/delta/submit_delta_evals.sh d6144 probes
#
# STAGING. tokstruct/cq (in evals) and csteer/cgen (in steering) and cgen_concept
# (in concept) read activations_sampled_10M/layer_27.npy — scattered random-row
# access over 61 GB on /work/hdd Lustre is very slow — so those three groups, plus
# `prep` (its `km`/`bench`/`cache` steps) and the standalone `km` target, submit
# through scripts/slurm/delta_sheet_staged.sbatch, which rsyncs the whole dump to
# node-local /tmp first and exports ACT_DIR. `interventions` never touches the
# dump (number/steer/range_*/number_hb all do live LM forward passes on small
# datasets, not the pre-extracted dump — verified by grepping every tool for
# activations_dir/layer_*.npy/rows_*.npy reads), so it stays on plain
# scripts/slurm/delta_sheet.sbatch — staging it would rsync 61 GB for nothing.
#
# Each group becomes its own job:
#   scripts/delta/sheet[_staged].sbatch experiments/paper/scripts/run_delta_evals.sh <arm> <group>
# with -p gpuA100x4,gpuA40x4 --gpus-per-node=1 --cpus-per-task=8 --mem=64g and a
# per-group time limit (evals 6h, interventions 10h, steering 4h, concept 8h).
# arm=d12288's evals group gets --mem=128g (clustering_quality holds
# n_sample x latent_dim float32 latents in RAM: 1M x 12288 = 49 GB).
#
# DEP, if set, is passed to every submitted job as --dependency=$DEP.
# DRY=1 prints the resolved `sbatch` command lines instead of submitting anything.
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -euo pipefail
source "$GEOAE_ROOT/configs/sites/local.sh"

ARM="${1:-}"; shift || true
SEL_GROUPS=("$@")

declare -A TIME=( [evals]=6:00:00 [interventions]=10:00:00 [steering]=4:00:00 [concept]=8:00:00 [geometry]=3:00:00 [probes]=2:00:00 )
# Which sbatch wrapper each group needs: 'staged' rsyncs the dump to node-local
# /tmp first (scripts/slurm/delta_sheet_staged.sbatch); 'plain' skips that
# (scripts/slurm/delta_sheet.sbatch) because none of that group's tools read the dump.
declare -A SHEET_FOR=( [evals]=staged [interventions]=plain [steering]=staged [concept]=staged [geometry]=plain [probes]=plain )
COMMON=(-p gpuA100x4,gpuA40x4 --gpus-per-node=1 --cpus-per-task=8)

submit() {   # submit <job-name> <time> <mem, default 64g> <sbatch-script> -- <sheet> [args...]
    #            optional env EXPORT="VAR=val,VAR2=val" is passed as --export=ALL,$EXPORT
    local name="$1" time="$2" mem="${3:-64g}" sbatch_script="$4"; shift 4
    [ "$1" = "--" ] && shift
    local args=(--job-name="$name" --time="$time" "${COMMON[@]}" --mem="$mem")
    [ -n "${DEP:-}" ] && args+=(--dependency="$DEP")
    [ -n "${EXPORT:-}" ] && args+=(--export="ALL,$EXPORT")
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
    submit "ev-prep" "4:00:00" "" scripts/slurm/delta_sheet_staged.sbatch -- experiments/paper/scripts/run_delta_prep.sh bench cache
    exit 0
fi
# km: the 4 balanced-k-means fits, one staged job (so a future refit is cheap —
# it rsyncs the dump once and fits all four codebooks in the same job).
if [ "$ARM" = "km" ]; then
    submit "ev-km" "4:00:00" "" scripts/slurm/delta_sheet_staged.sbatch -- experiments/paper/scripts/run_delta_prep.sh km
    exit 0
fi

G1B_DUMP=activations_gemma3_1b_sampled_struct5_10M
# g1b_prep: Gemma 3 1B base arms (balanced k-means, both layers) + probe caches (both layers), one staged
# job. Both layers are staged (no STAGE_LAYERS) because g1b_km fits layers 13 AND 25.
if [ "$ARM" = "g1b_prep" ]; then
    EXPORT="ACT_NAME=$G1B_DUMP" submit "ev-g1b-prep" "4:00:00" "" scripts/slurm/delta_sheet_staged.sbatch -- experiments/paper/scripts/run_delta_prep.sh g1b_km g1b_cache
    exit 0
fi
# g1b_probe_acts: extraction for the Gemma probes group (live LM forward; reads no dump -> plain).
if [ "$ARM" = "g1b_probe_acts" ]; then
    submit "ev-g1b-probe-acts" "3:00:00" "" scripts/slurm/delta_sheet.sbatch -- experiments/paper/scripts/run_delta_prep.sh g1b_probe_acts
    exit 0
fi

# probe_acts: extraction for the probes group (GPU live LM forward; reads no dump -> plain).
if [ "$ARM" = "probe_acts" ]; then
    submit "ev-probe-acts" "3:00:00" "" scripts/slurm/delta_sheet.sbatch -- experiments/paper/scripts/run_delta_prep.sh probe_acts
    exit 0
fi

case "$ARM" in
    d768|d3072|d6144|d12288|d6144k4000|d12288k4000) ;;
    g1b_l13_d1152|g1b_l13_d2304|g1b_l13_d4608|g1b_l25_d1152|g1b_l25_d2304|g1b_l25_d4608) ;;
    *)
        echo "usage: $0 <d768|d3072|d6144|d12288|d6144k4000|d12288k4000|g1b_l{13,25}_d{1152,2304,4608}|prep|km|probe_acts|g1b_prep|g1b_probe_acts> [groups...]" >&2
        echo "  groups: evals interventions steering concept geometry probes  (default: first four, not judge/geometry/probes)" >&2
        exit 2
        ;;
esac
[ ${#SEL_GROUPS[@]} -eq 0 ] && SEL_GROUPS=(evals interventions steering concept)

for g in "${SEL_GROUPS[@]}"; do
    t="${TIME[$g]:-}"
    if [ -z "$t" ]; then
        echo "[abort] unknown group '$g' (have: evals interventions steering concept geometry probes)" >&2
        exit 2
    fi
    mem=""
    case "$ARM" in d12288|d12288k4000) [ "$g" = evals ] && mem="128g" ;; esac
    sbatch_script=scripts/slurm/delta_sheet.sbatch
    [ "${SHEET_FOR[$g]}" = staged ] && sbatch_script=scripts/slurm/delta_sheet_staged.sbatch
    export_env=""
    case "$ARM" in
        g1b_l*)
            gl="${ARM#g1b_l}"; gl="${gl%%_*}"
            if [ "$gl" != 25 ] && { [ "$g" = steering ] || [ "$g" = concept ]; }; then
                echo "[skip] $ARM $g — needs the last block (layer 25)"; continue
            fi
            [ "${SHEET_FOR[$g]}" = staged ] && export_env="ACT_NAME=$G1B_DUMP,STAGE_LAYERS=$gl"
            ;;
    esac
    EXPORT="$export_env" submit "ev-$ARM-$g" "$t" "$mem" "$sbatch_script" -- experiments/paper/scripts/run_delta_evals.sh "$ARM" "$g"
done
