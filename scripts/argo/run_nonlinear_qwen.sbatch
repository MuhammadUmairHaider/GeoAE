#!/usr/bin/env bash
# Resources and absolute output paths are supplied by submit_nonlinear_qwen.py.
# One arm, one GPU. Always start from scratch in an isolated output directory.
set -euo pipefail
RUN_DIR="${1:?usage: run_nonlinear_qwen.sbatch <run-dir> <arm>}"
ARM="${2:?missing arm}"
case "$ARM" in mlp|residual_mlp) ;; *) echo "Invalid arm: $ARM" >&2; exit 2 ;; esac
SOURCE="$RUN_DIR/source"
PYTHON="/scratch/dogar/venvs/geoae/bin/python"
[ -x "$PYTHON" ] || { echo "Python is not executable: $PYTHON" >&2; exit 1; }
cd "$SOURCE"
export PYTHONPATH="$SOURCE"
export GEOAE_ROOT="$SOURCE"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export GEOAE_TRAIN_WORKERS=2 GEOAE_VAL_WORKERS=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# --export=NONE otherwise propagates to any subsequent srun invocation.
export SLURM_EXPORT_ENV=ALL
for option in ${!SBATCH_@}; do unset "$option"; done
exec 9>"$RUN_DIR/locks/$ARM.lock"
flock -n 9 || { echo "An active job already owns arm $ARM" >&2; exit 1; }
if find "$RUN_DIR/checkpoints/$ARM" -mindepth 1 -maxdepth 1 -print -quit | read -r _; then
    echo "Arm already has checkpoint outputs; refusing to overwrite: $ARM" >&2
    exit 1
fi
PROVENANCE="$RUN_DIR/provenance/${ARM}_${SLURM_JOB_ID:-local}.json"
"$PYTHON" scripts/argo/validate_nonlinear_qwen.py --run-dir "$RUN_DIR" \
    --require-cuda --runtime-output "$PROVENANCE"
read -r DATA_DIR < <("$PYTHON" - "$RUN_DIR/manifest.json" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
print(manifest["dataset"]["data_dir"])
PY
)
# GPFS random mmap reads bypass ordinary page caching. Stage this one ~41 GiB
# dump into tmpfs; cgroup memory includes these pages, hence the 160G request.
SHM="/dev/shm/geoae_nonlinear_qwen_${SLURM_JOB_ID:-$$}"
[ ! -e "$SHM" ] || { echo "Stage directory already exists: $SHM" >&2; exit 1; }
mkdir -m 700 "$SHM"
trap 'rm -rf -- "$SHM"' EXIT
trap 'exit 143' TERM INT
export GEOAE_TRAIN_ACTIVATIONS_DIR="$SHM/activations"
mkdir "$GEOAE_TRAIN_ACTIVATIONS_DIR"
"$PYTHON" - "$DATA_DIR" "$GEOAE_TRAIN_ACTIVATIONS_DIR" "$RUN_DIR/manifest.json" <<'PY'
import json, shutil, sys
from pathlib import Path
from scripts.argo.nonlinear_qwen_common import DATA_FILES, inspect_dataset
source, destination = map(Path, sys.argv[1:3])
manifest = json.load(open(sys.argv[3]))
needed = sum((source / name).stat().st_size for name in DATA_FILES)
if shutil.disk_usage(destination).free < needed + 1024**3:
    raise SystemExit("Insufficient /dev/shm capacity to stage the Qwen dump")
for name in DATA_FILES:
    shutil.copy2(source / name, destination / name)
staged = inspect_dataset(destination, Path(manifest["dataset"]["token_table"]))
for name in DATA_FILES:
    expected = dict(manifest["dataset"]["files"][name])
    actual = dict(staged["files"][name])
    expected.pop("path")
    actual.pop("path")
    if actual != expected:
        raise SystemExit(f"Staged activation content identity mismatch: {name}")
print(f"Staged {needed / 1024**3:.2f} GiB at {destination}", flush=True)
PY
export GEOAE_DATASET_PROVENANCE
GEOAE_DATASET_PROVENANCE="$("$PYTHON" - "$RUN_DIR/manifest.json" "$PROVENANCE" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
print(json.dumps({"source": manifest["dataset"], "runtime_provenance": sys.argv[2]}, separators=(",", ":")))
PY
)"
"$PYTHON" -u -m geoae.train --config "$RUN_DIR/configs/$ARM.yaml" --no_wandb \
    2>&1 | tee "$RUN_DIR/logs/${ARM}_${SLURM_JOB_ID:-local}.log"
