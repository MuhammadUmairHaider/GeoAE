"""Portable scratch selection and snapshot integrity, without research assets."""
import hashlib
import importlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
import sys

import pytest

from geoae.workflow import execute, snapshot


def test_scratch_uses_platform_temp_when_shared_memory_is_unavailable(tmp_path, monkeypatch):
    missing = tmp_path / "missing-shared-memory"
    monkeypatch.setattr(execute, "staging_roots", lambda: (missing, tmp_path))
    monkeypatch.setattr(execute.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * execute.GIB))
    with execute.temporary_staging_directory(1, prefix="portable_") as directory:
        assert directory.parent == tmp_path
        (directory / "activation.npy").write_bytes(b"fixture")
    assert not directory.exists()


@pytest.mark.parametrize("memory_mb,expected", [(1024, "disk"), (128 * 1024, "memory")])
def test_shared_memory_respects_slurm_working_memory_reserve(tmp_path, monkeypatch, memory_mb, expected):
    memory = Path("/dev/shm")
    monkeypatch.setattr(execute, "staging_roots", lambda: (memory, tmp_path))
    monkeypatch.setattr(Path, "is_dir", lambda path: True)
    monkeypatch.setattr(execute.os, "access", lambda *args: True)
    monkeypatch.setattr(execute.shutil, "disk_usage", lambda path: SimpleNamespace(free=200 * execute.GIB))
    monkeypatch.setenv("SLURM_MEM_PER_NODE", str(memory_mb))
    selected = execute.select_staging_root(40 * execute.GIB)
    assert selected == (memory if expected == "memory" else tmp_path)


def test_scratch_refuses_insufficient_capacity_before_creating_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(execute, "staging_roots", lambda: (tmp_path,))
    monkeypatch.setattr(execute.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="sufficient capacity"):
        with execute.temporary_staging_directory(2, prefix="insufficient_"):
            pytest.fail("Insufficient scratch must not be used")
    assert not list(tmp_path.iterdir())


def test_streaming_hash_and_atomic_json_keep_artifact_identity(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.bin"
    payload = b"research artifact" * 1000
    artifact.write_bytes(payload)
    monkeypatch.setattr(Path, "read_bytes", lambda path: pytest.fail("Hash must stream files"))
    assert snapshot.sha256(artifact) == hashlib.sha256(payload).hexdigest()
    manifest = tmp_path / "manifest.json"
    snapshot.atomic_json(manifest, {"sha256": snapshot.sha256(artifact)})
    original = manifest.read_text()
    with pytest.raises(ValueError):
        snapshot.atomic_json(manifest, {"metric": float("nan")})
    assert manifest.read_text() == original
    assert json.loads(original)["sha256"] == hashlib.sha256(payload).hexdigest()


def test_new_evaluation_snapshot_includes_workflow_dependencies(tmp_path):
    old_source = tmp_path / "source"
    model = old_source / "geoae/model.py"
    model.parent.mkdir(parents=True)
    model.write_text("# Older paper model remains untouched.\n")
    copied = snapshot.copy_workflow_sources(old_source)
    assert "geoae/workflow/execute.py" in copied
    assert "geoae/workflow/snapshot.py" in copied
    assert all(snapshot.sha256(old_source / name) == digest for name, digest in copied.items())
    assert model.read_text() == "# Older paper model remains untouched.\n"


def test_workflow_dispatch_preserves_explicit_submission_semantics(monkeypatch):
    from experiments.nonlinear_decoders import plan

    calls = []
    previous = sys.argv
    monkeypatch.setattr(plan.importlib, "import_module", lambda name:
                        SimpleNamespace(main=lambda: calls.append((name, list(sys.argv)))))
    plan.main(["submit", "campaign", "--campaign", "/prepared/campaign"])
    assert calls[0][0] == "nonlinear_paper_campaign"
    assert calls[0][1][1:] == ["submit", "--campaign", "/prepared/campaign"]
    assert "--submit" not in calls[0][1]
    plan.main(["submit", "campaign", "--campaign", "/prepared/campaign", "--submit"])
    assert calls[1][1][-1] == "--submit"
    assert sys.argv is previous


def test_shared_qwen_plan_preserves_every_frozen_command_and_output(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "experiments/nonlinear_decoders/deltaai"))
    plan = importlib.import_module("qwen_paper_plan")
    assets = {key: "/assets/" + key for key in plan.REQUIRED_ASSETS}
    assets["atlas_control_classes"] = {"atlas_s0": [0, 2], "atlas_s1": "1,3", "atlas_s2": [4, 5]}
    steps = plan.build_steps(assets, Path("/checkpoint/final.pt"), Path("/output"), "/python")
    assert len(steps) == 18
    serialized = json.dumps(steps, sort_keys=True)
    # Normalize documented source moves only. The digest was captured from the
    # independent Qwen implementation before replacing it with the shared plan.
    moves = json.loads((root / "docs/history/audits/source_paths.json").read_text())
    for old, new in sorted(moves["paths"].items(), key=lambda pair: -len(pair[1])):
        serialized = serialized.replace(new, old)
    for old, new in sorted(moves["modules"].items(), key=lambda pair: -len(pair[1])):
        serialized = re.sub(r"(?<![A-Za-z0-9_.])" + re.escape(new) + r"(?![A-Za-z0-9_])", old, serialized)
    assert hashlib.sha256(serialized.encode()).hexdigest() == "dfd1c087afbbd92408099b43fc12ab5e055104193012cbb990785e964b102daa"


def test_evaluation_snapshot_bundles_recipe_aliases_and_required_source_resources(tmp_path):
    origin, destination = tmp_path / "origin", tmp_path / "snapshot"
    resources = {
        "README.md": "# Source checkout\n",
        "LICENSE": "MIT License\n",
        "scripts/py": "#!/bin/sh\nexec python \"$@\"\n",
        "geoae/model.py": "# model\n",
        "experiments/paper/configs/llama.yaml": "model: {hidden_size: 16}\n",
        "experiments/config_paths.json": json.dumps({"paths": {"configs/base/old.yaml": "experiments/paper/configs/llama.yaml"}}),
        "geoae/baselines/matched_topk/_reference/SOURCE.json": "{}\n",
        "configs/examples/default.yaml": "{}\n",
        "configs/sites/local.sh": "# site\n",
        "experiments/width/status/panel_template.html": "<html>template</html>\n",
        "experiments/nonlinear_decoders/deltaai/run.sbatch": "# scheduler template\n",
        "docs/paper/acl.sty": "% style\n",
        "docs/paper/acl_natbib.bst": "% bibliography style\n",
    }
    for name, content in resources.items():
        path = origin / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    saved_result = origin / "experiments/nonlinear_decoders/results/table.json"
    saved_result.parent.mkdir(parents=True)
    saved_result.write_text('{"scientific_result": 1}')
    hashes = snapshot.copy_evaluation_sources(origin, destination)
    assert set(hashes) == set(resources)
    assert all(snapshot.sha256(destination / name) == digest for name, digest in hashes.items())
    assert not (destination / saved_result.relative_to(origin)).exists()
    from geoae.paths import config_path
    assert config_path("configs/base/old.yaml", base=destination) == destination / "experiments/paper/configs/llama.yaml"
