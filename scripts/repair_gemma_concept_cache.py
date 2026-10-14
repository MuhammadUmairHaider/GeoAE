"""Rebuild seven sequence caches after fixing their label-sorted sampling.

Run with the same Python interpreter as the original pipeline. This script
performs inference only when explicitly invoked; it never launches evaluations.
Old cache files, metadata, and a repair audit remain under the run's repairs/.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

# Permit `scripts/delta/py scripts/repair_gemma_concept_cache.py` from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoae.config import Config
from geoae.gemma_dpc import (
    DEFAULT_CONFIG, Pipeline, completed, digest, manifest, output_signature,
    validate_cache, write_json,
)
from geoae.paths import PACKAGE_ROOT

SOURCE = "geoae/interp/concept_suite.py"
OLD_SHA256 = "6805c1264801572feaa16c059f6b67fc225190788aafe39d17b4425c89710de5"
NAMES = ("sentiment", "sentiment_long", "subjectivity", "language",
         "topic4", "topic14", "topic20")
FILES = tuple(f"{name}.npz" for name in NAMES)
REPAIR = "concept_sequence_shuffle_v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_copy(source: Path, target: Path) -> None:
    """Never overwrite an earlier backup with different contents."""
    if target.exists():
        if digest(target) != digest(source):
            raise RuntimeError(f"Backup does not match its source: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def hardlink(source: Path, target: Path) -> None:
    if target.exists():
        if not os.path.samefile(source, target):
            raise RuntimeError(f"Unexpected existing staging file: {target}")
    else:
        os.link(source, target)


def verify_completed(p: Pipeline) -> None:
    for marker in sorted((p.root / "state").glob("*.json")):
        if marker.stem not in p.stages:
            raise RuntimeError(f"Unknown completion marker: {marker}")
        completed(p, p.stages[marker.stem])
    for name in ("train", "concept_cache", "ravel_cache", "ioi_cache", "atlas_cache"):
        if not completed(p, p.stages[name]):
            raise RuntimeError(f"Repair requires completed {name}")


def forbid_dependents(p: Pipeline) -> None:
    for name in ("validate_cache", "probe", "geometry", "summary"):
        paths = (p.root / "state" / f"{name}.json", *p.stages[name].outputs)
        for path in paths:
            if path.exists():
                raise RuntimeError(f"Dependent result already exists; refusing repair: {path}")
    figures = p.root / "figures"
    if figures.exists() and any(figures.iterdir()):
        raise RuntimeError(f"Potential partial geometry results exist: {figures}")


def replace_link(source: Path, destination: Path, temporary: Path) -> None:
    """Install a hard link atomically, keeping the staged/backup source intact."""
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, destination)


def repair(p: Pipeline, *, check_only: bool = False) -> None:
    manifest_path = p.root / "manifest.json"
    marker_path = p.root / "state" / "concept_cache.json"
    work = p.root / "repairs" / REPAIR
    backup = work / "backup"
    staged = work / "cache"
    audit_path = work / "audit.json"
    current = manifest(p)
    saved = read_json(manifest_path)

    # Successful repeat invocations are read-only, including after evaluations.
    if audit_path.exists() and read_json(audit_path).get("status") == "complete":
        audit = read_json(audit_path)
        if saved != current or audit["new_manifest"] != current:
            raise RuntimeError("Source/config changed after this repair")
        verify_completed(p)
        if read_json(marker_path) != audit["new_concept_marker"]:
            raise RuntimeError("Repaired concept completion marker changed")
        for name, sha in audit["new_cache_sha256"].items():
            if digest(p.cache / name) != sha:
                raise RuntimeError(f"Repaired cache changed: {name}")
        print(f"Repair already complete and verified. Audit: {audit_path}", flush=True)
        return

    if saved.get("source_sha256", {}).get(SOURCE) != OLD_SHA256:
        raise RuntimeError("Run does not have the expected original concept-suite source hash")
    expected = deepcopy(saved)
    expected["source_sha256"][SOURCE] = current["source_sha256"][SOURCE]
    if expected != current or current["source_sha256"][SOURCE] == OLD_SHA256:
        raise RuntimeError("Only the reviewed concept_suite.py sampling fix may differ from the run manifest")
    if Config.from_yaml(p.effective_config).to_dict() != p.cfg.to_dict():
        raise RuntimeError("Effective training config changed")
    verify_completed(p)
    forbid_dependents(p)
    if check_only:
        print("Repair precheck passed: only concept-suite source changed; completed outputs verified.",
              flush=True)
        print(f"Would rebuild: {', '.join(NAMES)}. No inference or cache changes performed.", flush=True)
        return
    old_marker = read_json(marker_path)
    plan = {"version": 1, "repair": REPAIR, "old_manifest": saved,
            "new_manifest": current, "old_concept_marker": old_marker,
            "affected_files": list(FILES)}
    plan_path = work / "plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError(f"Existing repair staging belongs to different inputs: {work}")
    if not plan_path.exists():
        if work.exists() and any(work.iterdir()):
            raise RuntimeError(f"Unidentified repair staging exists: {work}")
        write_json(plan_path, plan)
    backup.mkdir(exist_ok=True)
    staged.mkdir(exist_ok=True)
    save_copy(manifest_path, backup / "manifest.json")
    save_copy(marker_path, backup / "concept_cache.json")
    save_copy(PACKAGE_ROOT / SOURCE, work / "concept_suite.new.py")

    # The old source may still be available in Git. Never label a different
    # revision as the source that produced the original caches.
    old_source = backup / "concept_suite.old.py"
    if not old_source.exists():
        result = subprocess.run(["git", "show", f"HEAD:{SOURCE}"], cwd=PACKAGE_ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == OLD_SHA256:
            old_source.write_bytes(result.stdout)
    if old_source.exists() and digest(old_source) != OLD_SHA256:
        raise RuntimeError(f"Original source backup hash mismatch: {old_source}")

    untouched = []
    for source in sorted(p.cache.iterdir()):
        if source.name in FILES:
            continue
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"Unexpected cache entry: {source}")
        hardlink(source, staged / source.name)
        untouched.append(source)
    untouched_signature = output_signature(tuple(untouched))
    cmd = [sys.executable, "-u", "-m", "geoae.interp.concept_suite",
           "--checkpoint", str(p.final), "--out", str(staged),
           "--only", ",".join(NAMES)]
    log = work / "concept_cache.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for key, value in {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "MPLBACKEND": "Agg", "WANDB_MODE": "disabled"}.items():
        env.setdefault(key, value)
    from geoae.hf_auth import ensure_hf_login
    ensure_hf_login(verbose=False)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if key in os.environ:
            env[key] = os.environ[key]
    started = time.time()
    print(f"Rebuilding seven sequence caches; log: {log}", flush=True)
    print(f"Follow with: tail -f {shlex.quote(str(log))}", flush=True)
    with log.open("a") as output:
        output.write(f"\n# {time.ctime()} {shlex.join(cmd)}\n")
        output.flush()
        result = subprocess.run(cmd, cwd=PACKAGE_ROOT, env=env,
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Cache generation failed ({result.returncode}); active cache untouched. "
                           f"See {log}; rerunning reuses completed staged sources.")

    active_cache = p.cache
    try:
        p.cache = staged
        validation = validate_cache(p)
    except Exception as exc:
        raise RuntimeError(f"Staged cache validation failed: {exc}. Active cache untouched. "
                           f"Preserve/move the offending staged file out of {staged} before retrying; "
                           "the generator skips existing staged files.") from exc
    finally:
        p.cache = active_cache
    # Fail closed if code, metadata, or any completed artifacts changed while
    # the subprocess was running, even if another writer ignored our run lock.
    if manifest(p) != current or read_json(manifest_path) != saved:
        raise RuntimeError("Recipe/source changed during repair; active cache untouched")
    if read_json(marker_path) != old_marker:
        raise RuntimeError("Concept completion marker changed during repair")
    verify_completed(p)
    forbid_dependents(p)
    if output_signature(tuple(untouched)) != untouched_signature:
        raise RuntimeError("Untouched cache files changed during repair")
    old_hashes, new_hashes = {}, {}
    for name in FILES:
        old_hashes[name] = digest(active_cache / name)
        new_hashes[name] = digest(staged / name)
        hardlink(active_cache / name, backup / name)
    audit = {**plan, "status": "installing", "old_source_sha256": OLD_SHA256,
             "new_source_sha256": current["source_sha256"][SOURCE],
             "old_source_backup": str(old_source) if old_source.exists() else None,
             "old_cache_sha256": old_hashes, "new_cache_sha256": new_hashes,
             "validation": validation, "command": cmd, "log": str(log),
             "started_at": started, "generation_seconds": time.time() - started}
    write_json(audit_path, audit)
    try:
        for name in FILES:
            replace_link(staged / name, active_cache / name, work / f"install-{name}")
        new_marker = {"outputs": output_signature(p.stages["concept_cache"].outputs),
                      "completed_at": time.time(), "seconds": time.time() - started,
                      "command": cmd,
                      "provenance": {"repair": REPAIR, "audit": str(audit_path),
                                     "retained_outputs": [str(x) for x in p.stages["concept_cache"].outputs
                                                          if x.name not in FILES],
                                     "previous_marker": str(backup / "concept_cache.json")}}
        write_json(marker_path, new_marker)
        write_json(manifest_path, current)
        audit.update(status="complete", new_concept_marker=new_marker,
                     completed_at=time.time())
        write_json(audit_path, audit)
    except BaseException:
        # Ordinary failures/interruptions restore originals. A hard kill during
        # installation fails closed on the next run; all versions remain here.
        for name in FILES:
            replace_link(backup / name, active_cache / name, work / f"rollback-{name}")
        write_json(marker_path, old_marker)
        write_json(manifest_path, saved)
        audit["status"] = "rolled_back"
        write_json(audit_path, audit)
        raise
    print(f"Repair complete: all concept caches validated. Audit: {audit_path}", flush=True)
    print("Evaluations were not launched. Resume with:", flush=True)
    print(shlex.join([sys.executable, "-u", "-m", "geoae.gemma_dpc", "run",
                      "--config", str(p.config_path), "--run-dir", str(p.root)]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--check-only", action="store_true",
                        help="Verify repair prerequisites without inference or cache changes")
    args = parser.parse_args()
    p = Pipeline(args.config, args.run_dir)
    if not (p.root / "manifest.json").is_file():
        raise RuntimeError("Repair requires an existing pipeline run with a manifest")
    with (p.root / ".pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pipeline/repair process owns this run directory") from None
        repair(p, check_only=args.check_only)


if __name__ == "__main__":
    main()
