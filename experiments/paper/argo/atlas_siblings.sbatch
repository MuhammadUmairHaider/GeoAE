#!/bin/bash
# Atlas sibling test of the ranges hypothesis on ONE Argo H100 (group partition c-hsajjad). Stages, in order:
#   pairs     Step 1  freeze sibling + null pairs (labels/names only; writes pairs.json, never touches activations)
#   extract   Step 2  layer-27 residuals of the first N_CHUNKS Atlas chunks -> $CACHE (also pair_counts.json)
#   geometry  Step 3  phase 1: share of range pairs, siblings vs nulls, base h / token-centred h / GeoAE z
#   removal   Step 4  phase 2: remove A by its range on the shared direction vs the whole direction; is sibling B spared?
#   tests     the new unit tests (tests/unit/test_atlas_siblings.py, tests/unit/test_atlas_geometry.py)
#   suite     the whole pytest suite (junit xml; the pytest summary line can be suppressed)
#
#   sbatch experiments/paper/argo/atlas_siblings.sbatch <N_CHUNKS> <stages,comma,separated> [key=value ...]
#   full run:    sbatch experiments/paper/argo/atlas_siblings.sbatch 300000 extract,geometry,removal
#   smoke:       sbatch --mem=48G --time=01:00:00 experiments/paper/argo/atlas_siblings.sbatch 20000 extract,geometry,removal min_excl=8 max_pairs=10 n_boot=40
# key=value options: min_excl (default 100) | max_pairs (removal; default all) | n_boot (document bootstrap replicates, 200)
#   | min_rows (removal, 30) | n_ctrl_fit (24000) | pairs (pairs file, default the frozen pairs.json) | force_pairs=1 (pairs stage may overwrite)
#   | tag (smoke: suffix of the output directory; N=300000: a sub-directory of atlas_siblings/, e.g. tag=extended with
#     pairs=results/delta/d6144/evals/atlas_siblings/pairs_extended.json for the one-way pair list; geometry/removal reuse the cache)
#
# Outputs: N=300000 -> results/delta/d6144/evals/atlas_siblings/{geometry,removal}.json, cache /scratch/dogar/geoae/atlas_l27
#          (about 15 min extraction + 10-20 min geometry + 1.5-2.5 h removal on one H100);
#          any other N is a smoke run -> .../atlas_siblings/smoke_n<N>/ and /scratch/dogar/geoae/atlas_l27_n<N>.
# /scratch is GPFS WITHOUT a page cache: the row arrays (29.5 GB at N=300k, 16 rows/chunk x 3072 x fp16) are copied into
# /dev/shm/geoae_<jobid>/ before geometry/removal read them, and deleted on exit; --mem covers them (tmpfs is charged
# to the job's cgroup). ONE GPU only: the node is shared.
#SBATCH --job-name=atlas-siblings
#SBATCH --account=def-hsajjad
#SBATCH --partition=c-hsajjad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm-%x-%j.out
export GEOAE_ROOT="${GEOAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}}"
cd "$GEOAE_ROOT"
set -uo pipefail
N_CHUNKS="${1:?usage: sbatch experiments/paper/argo/atlas_siblings.sbatch <N_CHUNKS> <stages> [key=value ...]}"
STAGES="${2:?stages, e.g. extract,geometry,removal}"
shift 2
declare -A OPT=([min_excl]=100 [max_pairs]=0 [n_boot]=200 [min_rows]=30 [n_ctrl_fit]=24000 [force_pairs]=0 [pairs]="" [tag]="")
for kv in "$@"; do OPT[${kv%%=*}]="${kv#*=}"; done
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source configs/sites/argo.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
PY="$UV_PROJECT_ENVIRONMENT/bin/python"
ATLAS=/scratch/dogar/geoae/atlas
EV=results/delta/d6144/evals/atlas_siblings
PAIRS="${OPT[pairs]:-$EV/pairs.json}"
if [ "$N_CHUNKS" = "300000" ]; then CACHE=/scratch/dogar/geoae/atlas_l27; OUT=$EV${OPT[tag]:+/${OPT[tag]}}; else CACHE=/scratch/dogar/geoae/atlas_l27_n$N_CHUNKS; OUT=$EV/smoke_n${N_CHUNKS}${OPT[tag]}; fi
mkdir -p "$OUT/logs" logs
has() { [[ ",$STAGES," == *",$1,"* ]]; }
SHM="/dev/shm/geoae_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$SHM"' EXIT
trap 'exit 143' TERM INT
echo "[atlas] N_CHUNKS=$N_CHUNKS stages=$STAGES cache=$CACHE out=$OUT pairs=$PAIRS opts: $(for k in "${!OPT[@]}"; do printf '%s=%s ' "$k" "${OPT[$k]}"; done)"
echo "[atlas] host $(hostname -f) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

rc=0
run() { local name="$1"; shift; echo "[atlas $(date +%H:%M:%S)] start $name"; local t0=$SECONDS
        "$@" 2>&1 | tee "$OUT/logs/$name.log"; local r=${PIPESTATUS[0]}
        echo "[atlas $(date +%H:%M:%S)] done $name rc=$r in $((SECONDS - t0))s"; [ "$r" -ne 0 ] && rc=$r; return "$r"; }

if has pairs; then
  extra=(); [ "${OPT[force_pairs]}" = "1" ] && extra=(--overwrite)
  run pairs "$PY" -u geoae/extraction/atlas_siblings_pairs.py --atlas_dir "$ATLAS" --out_dir "$EV" ${extra[@]+"${extra[@]}"} || exit "$rc"
fi
if has extract; then
  run extract "$PY" -u geoae/extraction/atlas_siblings_extract.py --n_chunks "$N_CHUNKS" --out_dir "$CACHE" --atlas_dir "$ATLAS" \
      --pairs "$PAIRS" --min_excl "${OPT[min_excl]}" || exit "$rc"
fi
if has geometry || has removal; then
  mkdir -p "$SHM"
  echo "[stage] row arrays -> $SHM (tmpfs)"; time cp "$CACHE"/rows.npy "$CACHE"/rows_tok.npy "$CACHE"/rows_nxt.npy "$CACHE"/rows_chunk.npy "$CACHE"/rows_pos.npy "$SHM"/
fi
if has geometry; then
  run geometry "$PY" -u experiments/paper/analysis/paper_atlas_sibling_ranges.py --cache "$CACHE" --rows_dir "$SHM" --pairs "$PAIRS" \
      --min_excl "${OPT[min_excl]}" --n_boot "${OPT[n_boot]}" --out "$OUT/geometry.json" || true
fi
if has removal; then
  run removal "$PY" -u experiments/paper/analysis/paper_atlas_sibling_removal.py --cache "$CACHE" --rows_dir "$SHM" --pairs "$PAIRS" \
      --min_excl "${OPT[min_excl]}" --max_pairs "${OPT[max_pairs]}" --min_rows "${OPT[min_rows]}" \
      --n_ctrl_fit "${OPT[n_ctrl_fit]}" --geometry "$OUT/geometry.json" --out "$OUT/removal.json" || true
fi
if has tests; then
  run tests "$PY" -m pytest -q -p no:cacheprovider -rf tests/unit/test_atlas_siblings.py tests/unit/test_atlas_geometry.py || true
fi
if has suite; then
  mkdir -p "$ARGO_SCRATCH/geoae/logs"; JX="$ARGO_SCRATCH/geoae/logs/pytest-atlas-${SLURM_JOB_ID:-local}.xml"
  timeout 3000 "$PY" -m pytest -q -p no:cacheprovider -rf --junitxml="$JX" 2>&1 | tail -40
  echo "pytest exit code: ${PIPESTATUS[0]}"
  "$PY" - "$JX" <<'PYEOF'
import sys, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot(); s = r if r.tag == "testsuite" else r.find("testsuite")
print({k: s.get(k) for k in ("tests", "failures", "errors", "skipped", "time")})
for tc in s.iter("testcase"):
    if tc.find("failure") is not None or tc.find("error") is not None:
        print("FAILED:", tc.get("classname"), tc.get("name"))
PYEOF
fi
echo "[atlas] finished rc=$rc $(date)"
exit "$rc"
