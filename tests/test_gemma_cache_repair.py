"""CPU-only transaction checks for the one-off Gemma cache repair."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess

import pytest
import yaml

from geoae.gemma_dpc import Pipeline, manifest, output_signature, write_json
from geoae.paths import PACKAGE_ROOT

spec = importlib.util.spec_from_file_location(
    "gemma_cache_repair", PACKAGE_ROOT / "scripts/repair_gemma_concept_cache.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


@pytest.fixture
def run(tmp_path, monkeypatch):
    p = Pipeline(run_dir=tmp_path / "run")
    for name in ("train", "concept_cache", "ravel_cache", "ioi_cache", "atlas_cache"):
        stage = p.stages[name]
        for path in stage.outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if path.suffix == ".json" else f"old:{path.name}")
        write_json(p.root / "state" / f"{name}.json",
                   {"outputs": output_signature(stage.outputs), "command": ["original"]})
    p.effective_config.write_text(yaml.safe_dump(p.cfg.to_dict()))
    saved = deepcopy(manifest(p))
    saved["source_sha256"][repair.SOURCE] = repair.OLD_SHA256
    write_json(p.root / "manifest.json", saved)
    monkeypatch.setattr("geoae.hf_auth.ensure_hf_login", lambda **kwargs: None)
    monkeypatch.setattr(repair, "validate_cache", lambda pipeline: {
        "rungs": {name: {"classes": 2} for name in repair.NAMES}})
    return p


def snapshots(p):
    paths = [p.root / "manifest.json", p.root / "state" / "concept_cache.json",
             *p.cache.iterdir()]
    return {path: path.read_bytes() for path in paths}


def fake_child(monkeypatch, *, failure=False):
    commands = []

    def child(cmd, **kwargs):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 1, stdout=b"")
        commands.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        for name in repair.FILES[:1] if failure else repair.FILES:
            (out / name).write_text(f"new:{name}")
        return subprocess.CompletedProcess(cmd, 1 if failure else 0)

    monkeypatch.setattr(repair.subprocess, "run", child)
    return commands


def test_failed_child_preserves_active_artifacts(run, monkeypatch):
    before = snapshots(run)
    calls = fake_child(monkeypatch, failure=True)
    with pytest.raises(RuntimeError, match="active cache untouched"):
        repair.repair(run)
    assert len(calls) == 1
    assert snapshots(run) == before
    work = run.root / "repairs" / repair.REPAIR
    assert (work / "concept_cache.log").exists()
    assert (work / "cache" / repair.FILES[0]).read_text().startswith("new:")


def test_validation_failure_preserves_active_artifacts(run, monkeypatch):
    before = snapshots(run)
    fake_child(monkeypatch)

    def invalid(pipeline):
        raise ValueError("topic14.npz is invalid")

    monkeypatch.setattr(repair, "validate_cache", invalid)
    with pytest.raises(RuntimeError, match="Preserve/move the offending staged file"):
        repair.repair(run)
    assert snapshots(run) == before


def test_success_preserves_backups_and_is_idempotent(run, monkeypatch):
    before = snapshots(run)
    untouched = tuple(path for path in run.cache.iterdir() if path.name not in repair.FILES)
    original_signatures = output_signature(untouched)
    calls = fake_child(monkeypatch)
    repair.repair(run)
    work = run.root / "repairs" / repair.REPAIR
    for name in repair.FILES:
        assert (run.cache / name).read_text() == f"new:{name}"
        assert (work / "backup" / name).read_bytes() == before[run.cache / name]
    assert output_signature(untouched) == original_signatures
    assert (work / "backup" / "manifest.json").read_bytes() == before[run.root / "manifest.json"]
    assert (work / "backup" / "concept_cache.json").read_bytes() == before[
        run.root / "state" / "concept_cache.json"]
    assert repair.read_json(run.root / "manifest.json") == manifest(run)
    assert repair.read_json(work / "audit.json")["status"] == "complete"
    after = snapshots(run)
    repair.repair(run)
    assert len(calls) == 1
    assert snapshots(run) == after


@pytest.mark.parametrize("drift", ["source", "config"])
def test_other_drift_is_rejected_before_inference(run, monkeypatch, drift):
    path = run.root / "manifest.json"
    saved = repair.read_json(path)
    if drift == "source":
        saved["source_sha256"]["geoae/gemma_dpc.py"] = "unexpected"
    else:
        saved["config"]["train"]["n_epochs"] += 1
    write_json(path, saved)
    before = snapshots(run)
    calls = fake_child(monkeypatch)
    with pytest.raises(RuntimeError, match="Only the reviewed"):
        repair.repair(run)
    assert calls == []
    assert snapshots(run) == before
