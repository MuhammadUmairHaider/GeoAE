#!/usr/bin/env bash
# Idempotent migration: move every delta eval/intervention/steering/concept output
# from its OLD scattered location (eval_out/delta/<arm>/, results/delta/<arm>/*.json
# at top level, logs/delta/<arm>/, figures/delta/<arm>/, dbpedia/delta/) into the
# single browsable tree results/delta/<arm>/{evals,interventions,steering,concept,
# figures,logs}/ and results/delta/joint_correct/ — see results/delta/README.md.
#
#   scripts/delta/organize_delta_results.sh              # migrate in place
#   scripts/delta/organize_delta_results.sh --dry-run     # print every action, touch nothing
#   ROOT=/some/copy scripts/delta/organize_delta_results.sh   # migrate a COPY of the tree
#
# SAFE TO RE-RUN. For every (old_path, new_path) pair:
#   - old_path is already a symlink (a previous run migrated it) -> verified, skipped.
#   - old_path doesn't exist -> nothing to do, skipped.
#   - old_path is a real file modified in the last 10 minutes -> LEFT ALONE and
#     reported (RECENT): it may be a job mid-write; the next run will pick it up
#     once it's quiesced. This also covers a still-running OLD-version job that
#     resurrects a real file at an old path after an earlier migration replaced it
#     with a symlink — writing through/over a symlink typically preserves it (an
#     in-place tool write does), but a tool that does an atomic temp+rename WILL
#     leave a fresh real file at the old path; either way, once it's >10 min old
#     the next run migrates it again, same as the first time.
#   - old_path is a real file, new_path does not exist -> mv old -> new, then a
#     RELATIVE symlink is left at old_path pointing at new_path.
#   - BOTH old_path and new_path are real files (a genuine conflict — e.g. an old
#     real file plus a fresh write from the new-layout sheet) -> the OLDER of the
#     two is renamed to '<path>.stale-<epoch-mtime>' (never deleted), the newer
#     ends up at new_path, and old_path becomes a symlink to it. Reported (CONFLICT).
#   - Second run over an already-migrated tree: every pair falls into the first two
#     cases -> no filesystem writes at all (verified by this session's test).
#
# Never deletes data. Never touches results/delta/summary.json or
# eval_out/delta/summary.json — eval_out/summarize_delta.py owns those.
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
cd "$ROOT" || { echo "[abort] no such ROOT: $ROOT" >&2; exit 1; }

ARMS=(d768 d3072 d6144 d12288 d6144k4000 d12288k4000)
NOW=$(date +%s)
FRESH_SECS=600   # 10 minutes

MOVED=() SYMLINKED_ONLY=() RECENT=() CONFLICTS=() MISSING_BOTH=0

act() {   # act <description of the mv/ln that would run> -- <cmd...>
    local desc="$1"; shift
    [ "$1" = "--" ] && shift
    if [ "$DRY" -eq 1 ]; then
        echo "[dry ] $desc"
    else
        "$@"
    fi
}

age_secs() {   # age_secs <path> -> seconds since last mtime
    local m
    m=$(stat -c %Y "$1" 2>/dev/null) || { echo 999999; return; }
    echo $(( NOW - m ))
}

relsymlink() {   # relsymlink <old_path> <new_path>  -- ln -sfn <relative(new, dirname(old))> old_path
    local old="$1" new="$2" rel
    rel=$(realpath -m --relative-to="$(dirname "$old")" "$new")
    act "ln -sfn $rel $old" -- ln -sfn "$rel" "$old"
}

migrate() {   # migrate <old_path> <new_path>
    local old="$1" new="$2"

    if [ -L "$old" ]; then
        # Already migrated (or a stale/broken link from elsewhere) — verify and leave.
        local resolved want
        resolved=$(realpath -m "$old" 2>/dev/null)
        want=$(realpath -m "$new" 2>/dev/null)
        if [ "$resolved" != "$want" ]; then
            echo "[warn] $old is a symlink but does not resolve to $new (-> $(readlink "$old")); leaving it — investigate by hand"
        fi
        return
    fi

    if [ ! -e "$old" ]; then
        [ -e "$new" ] || return   # nothing at either path — nothing to do, nothing to report
        return                    # new-layout file already exists on its own — nothing to migrate
    fi

    local age
    age=$(age_secs "$old")
    if [ "$age" -lt "$FRESH_SECS" ]; then
        echo "[skip] $old modified ${age}s ago (< ${FRESH_SECS}s) — possibly mid-write; left for next run"
        RECENT+=("$old")
        return
    fi

    act "mkdir -p $(dirname "$new")" -- mkdir -p "$(dirname "$new")"

    if [ -e "$new" ] && [ ! -L "$new" ]; then
        # Genuine conflict: both are real files. Keep the newer at $new, verbatim;
        # rename the older one aside (never delete) and still symlink old_path -> new.
        local old_m new_m
        old_m=$(stat -c %Y "$old"); new_m=$(stat -c %Y "$new")
        if [ "$old_m" -gt "$new_m" ]; then
            local backup="${new}.stale-${new_m}"
            echo "[CONFLICT] $old (mtime $old_m) is newer than $new (mtime $new_m) — backing up $new -> $backup, promoting $old -> $new"
            act "mv -n $new $backup" -- mv -n "$new" "$backup"
            act "mv $old $new" -- mv "$old" "$new"
        else
            local backup="${old}.stale-${old_m}"
            echo "[CONFLICT] $new (mtime $new_m) is newer than or same as $old (mtime $old_m) — keeping $new, backing up $old -> $backup"
            act "mv -n $old $backup" -- mv -n "$old" "$backup"
        fi
        CONFLICTS+=("$old")
    else
        act "mv $old $new" -- mv "$old" "$new"
        MOVED+=("$old -> $new")
    fi
    relsymlink "$old" "$new"
}

for ARM in "${ARMS[@]}"; do
    OLDOUT=eval_out/delta/$ARM
    OLDRES=results/delta/$ARM
    OLDLOG=logs/delta/$ARM
    OLDFIG=figures/delta/$ARM
    NEWARM=results/delta/$ARM
    EVALSD=$NEWARM/evals
    INTERVD=$NEWARM/interventions
    STEERD=$NEWARM/steering
    CONCEPTD=$NEWARM/concept
    FIGD=$NEWARM/figures
    LOGD=$NEWARM/logs

    # evals (were under eval_out/delta/<arm>/)
    migrate "$OLDOUT/token_structure.json" "$EVALSD/token_structure.json"
    migrate "$OLDOUT/probe.json"            "$EVALSD/probe.json"
    migrate "$OLDOUT/mmlu.json"             "$EVALSD/mmlu.json"
    migrate "$OLDOUT/cq.json"               "$EVALSD/cq.json"

    # interventions (results.json were under results/delta/<arm>/ at top level;
    # number_review.json was under eval_out/delta/<arm>/)
    migrate "$OLDRES/range_number_dprime.json"              "$INTERVD/range_number_dprime.json"
    migrate "$OLDRES/range_number_dprime_hb.json"           "$INTERVD/range_number_dprime_hb.json"
    migrate "$OLDRES/range_intervention_db14_dprime.json"   "$INTERVD/range_intervention_db14_dprime.json"
    migrate "$OLDRES/range_intervention_biasbios_dprime.json" "$INTERVD/range_intervention_biasbios_dprime.json"
    migrate "$OLDRES/steer_db14.json"                       "$INTERVD/steer_db14.json"
    migrate "$OLDOUT/number_review.json"                    "$INTERVD/number_review.json"

    # steering (+ .rows.npz side files) — were under eval_out/delta/<arm>/
    migrate "$OLDOUT/cluster_steering.json"           "$STEERD/cluster_steering.json"
    migrate "$OLDOUT/cluster_steering.rows.npz"        "$STEERD/cluster_steering.rows.npz"
    migrate "$OLDOUT/cluster_steering_allhubs.json"    "$STEERD/cluster_steering_allhubs.json"
    migrate "$OLDOUT/cluster_steering_allhubs.rows.npz" "$STEERD/cluster_steering_allhubs.rows.npz"
    migrate "$OLDOUT/cluster_steer_generate.json"      "$STEERD/cluster_steer_generate.json"
    migrate "$OLDOUT/cluster_steer_judge.json"         "$STEERD/cluster_steer_judge.json"

    # concept — were under eval_out/delta/<arm>/
    migrate "$OLDOUT/concept_steer_generate.json" "$CONCEPTD/concept_steer_generate.json"
    migrate "$OLDOUT/concept_steer_judge.json"    "$CONCEPTD/concept_steer_judge.json"

    # figures (number-control review plots) — were under figures/delta/<arm>/
    migrate "$OLDFIG/number_review.png" "$FIGD/number_review.png"
    migrate "$OLDFIG/number_review.pdf" "$FIGD/number_review.pdf"

    # logs — flat under logs/delta/<arm>/, grouped by run type in the new tree
    declare -A LOG_GROUP=(
        [token_structure.log]=evals [probe.log]=evals [mmlu.log]=evals [cq.log]=evals
        [number_dprime.log]=interventions [number_dprime_hb.log]=interventions
        [steer_db14.log]=interventions [range_db14_dprime.log]=interventions
        [range_biasbios_dprime.log]=interventions
        [cluster_steering.log]=steering [cluster_steering_allhubs.log]=steering
        [cluster_steer_generate.log]=steering [cluster_steer_judge.log]=steering
        [concept_steer_generate.log]=concept [concept_steer_judge.log]=concept
    )
    for logname in "${!LOG_GROUP[@]}"; do
        grp="${LOG_GROUP[$logname]}"
        migrate "$OLDLOG/$logname" "$LOGD/$grp/$logname"
    done

    # joint-correct (were under dbpedia/delta/) -> shared results/delta/joint_correct/
    migrate "dbpedia/delta/joint_correct_db14_l27_$ARM.json"     "results/delta/joint_correct/joint_correct_db14_l27_$ARM.json"
    migrate "dbpedia/delta/joint_correct_biasbios_l27_$ARM.json" "results/delta/joint_correct/joint_correct_biasbios_l27_$ARM.json"
done

echo
echo "[organize] done. moved=${#MOVED[@]} conflicts=${#CONFLICTS[@]} left-recent=${#RECENT[@]}"
[ ${#CONFLICTS[@]} -gt 0 ] && { echo "[organize] CONFLICTS (backed up, never deleted):"; printf '  %s\n' "${CONFLICTS[@]}"; }
[ ${#RECENT[@]} -gt 0 ] && { echo "[organize] left in place (modified < 10 min ago, re-run later):"; printf '  %s\n' "${RECENT[@]}"; }
exit 0
