"""Consolidated allocations retain frozen evaluator behavior and failure state."""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def bundle(monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        "qwen_paper_bundle", repository / "experiments/nonlinear_decoders/deltaai/qwen_paper_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def suite(bundle, tmp_path):
    root = tmp_path / "paper_suite"
    source = root / "source"
    launchers = source / "experiments/nonlinear_decoders/deltaai"
    launchers.mkdir(parents=True)
    original = Path(__file__).resolve().parents[2] / "experiments/nonlinear_decoders/deltaai/qwen_paper_evals.py"
    shutil.copy2(original, launchers / original.name)
    for name in ("locks", "state", "logs", "provenance"):
        (root / name).mkdir()
    activations = root / "activations"
    activations.mkdir()
    (activations / "layer.npy").write_bytes(b"activation fixture")
    manifest = {
        "groups": list(bundle.GROUPS), "source_root": str(source),
        "source_sha256": {str(path.relative_to(source)): bundle.sha256(path)
                          for path in source.rglob("*.py")},
        "asset_identities": {}, "assets": {"activations_dir": str(activations)},
        "training_jobs": {"mlp": "3410182", "residual_mlp": "3410183"},
        "slurm": {"account": "bbyl-dtai-gh", "partition": "ghx4"},
        "plans": {arm: [{"name": f"original-{index}", "command": ["unchanged", str(index)]}
                        for index in range(18)] for arm in bundle.ARMS},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "preflight.json").write_text(json.dumps({"state": "passed"}))
    original_manifest = (root / "manifest.json").read_bytes()
    bundle.prepare(root)
    assert (root / "manifest.json").read_bytes() == original_manifest
    return root, manifest


def test_preparation_pins_existing_plans_and_is_repeatable(bundle, suite):
    root, manifest = suite
    frozen_before = (root / "manifest.json").read_bytes()
    bundle_before = (root / "orchestration/manifest.json").read_bytes()
    bundle.prepare(root)
    assert (root / "manifest.json").read_bytes() == frozen_before
    assert (root / "orchestration/manifest.json").read_bytes() == bundle_before
    data = bundle.validate(root)
    assert data["steps_per_arm"] == {arm: 18 for arm in bundle.ARMS}
    assert data["evaluation_plans_unchanged"] is True
    assert data["evaluation_manifest_sha256"] == bundle.sha256(root / "manifest.json")
    assert json.loads(frozen_before)["plans"] == manifest["plans"]


def test_every_group_runs_sequentially_with_one_persistent_activation_copy(bundle, suite, monkeypatch):
    root, manifest = suite
    groups, staged_paths, original_stage_calls = [], [], []

    def original_stage(_manifest, directory):
        original_stage_calls.append(directory)
        output = directory / "activations"
        output.mkdir()
        (output / "verified.npy").write_bytes(b"exact-original-data")
        return output

    fake = SimpleNamespace(stage_activations=original_stage)

    def run_group(args):
        groups.append(args.group)
        if args.group in ("core", "ranges"):
            # Match the frozen runner: its own temporary directory is discarded
            # after each group, while the bundle's activation copy must survive.
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                staged = fake.stage_activations(manifest, Path(temporary))
                assert (staged / "verified.npy").read_bytes() == b"exact-original-data"
                staged_paths.append(staged)
        if args.group == "alignment":
            assert staged_paths[0].is_dir()

    fake.run_group = run_group
    monkeypatch.setattr(bundle, "load_runner", lambda _root: (fake, manifest))
    bundle.run_all(root, "mlp")
    assert groups == list(bundle.GROUPS)
    assert len(original_stage_calls) == 1
    assert staged_paths[0] == staged_paths[1]
    assert not staged_paths[0].exists()
    assert fake.stage_activations is original_stage
    report = json.loads((root / "state/mlp_bundle.json").read_text())
    assert report["state"] == "complete" and report["failed_groups"] == []
    assert all(item["state"] == "complete" for item in report["groups"].values())


def test_group_failure_attempts_all_remaining_groups_before_reporting_failure(bundle, suite, monkeypatch):
    root, manifest = suite
    calls = []

    def run_group(args):
        calls.append(args.group)
        if args.group == "alignment":
            raise RuntimeError("one original evaluator failed")

    original_stage = lambda *_args: None
    fake = SimpleNamespace(run_group=run_group, stage_activations=original_stage)
    monkeypatch.setattr(bundle, "load_runner", lambda _root: (fake, manifest))
    with pytest.raises(RuntimeError, match="All groups attempted.*alignment"):
        bundle.run_all(root, "residual_mlp")
    assert calls == list(bundle.GROUPS)
    assert fake.stage_activations is original_stage
    report = json.loads((root / "state/residual_mlp_bundle.json").read_text())
    assert report["failed_groups"] == ["alignment"] and report["state"] == "failed"
    assert report["groups"]["alignment"]["state"] == "failed"
    assert report["groups"]["ranges"]["state"] == "complete"


def test_partial_activation_staging_is_cleaned_before_later_group_retries(bundle, suite, monkeypatch):
    root, manifest = suite
    stage_calls, groups = [], []

    def original_stage(_manifest, directory):
        stage_calls.append(directory)
        output = directory / "activations"
        assert not output.exists(), "Partial copy must be removed before retry"
        output.mkdir()
        (output / "layer.npy").write_bytes(b"partial" if len(stage_calls) == 1 else b"complete")
        if len(stage_calls) == 1:
            raise OSError("interrupted copy")
        return output

    fake = SimpleNamespace(stage_activations=original_stage)

    def run_group(args):
        groups.append(args.group)
        if args.group in ("core", "ranges"):
            staged = fake.stage_activations(manifest, root / args.group)
            assert (staged / "layer.npy").read_bytes() == b"complete"

    fake.run_group = run_group
    monkeypatch.setattr(bundle, "load_runner", lambda _root: (fake, manifest))
    with pytest.raises(RuntimeError, match="All groups attempted.*core"):
        bundle.run_all(root, "mlp")
    assert groups == list(bundle.GROUPS)
    assert len(stage_calls) == 2 and stage_calls[0] == stage_calls[1]
    report = json.loads((root / "state/mlp_bundle.json").read_text())
    assert report["groups"]["core"]["state"] == "failed"
    assert report["groups"]["ranges"]["state"] == "complete"


@pytest.mark.parametrize("arm,training", [("mlp", "3410182"), ("residual_mlp", "3410183")])
def test_each_arm_requests_one_gpu_48_hours_and_its_own_successful_training(bundle, suite, arm, training):
    root, manifest = suite
    command = bundle.build_command(root, manifest, arm, submit=True, hold=True)
    assert command[0:2] == ["sbatch", "--parsable"]
    for option in ("--gpus-per-node=1", "--time=48:00:00", "--cpus-per-task=12", "--mem=160G",
                   "--account=bbyl-dtai-gh", "--partition=ghx4", "--export=NONE", "--hold",
                   f"--dependency=afterok:{training}", "--kill-on-invalid-dep=yes"):
        assert option in command
    assert command[-2:] == [str(root), arm]
    assert not any("afterany:" in option for option in command)


def test_two_submissions_are_durable_and_retry_does_not_duplicate(bundle, suite, monkeypatch):
    root, _manifest = suite
    monkeypatch.setenv("SBATCH_PARTITION", "inherited-wrong-partition")
    calls = []

    def submit(command, **kwargs):
        assert not any(name.startswith("SBATCH_") for name in kwargs["env"])
        calls.append(command)
        return SimpleNamespace(stdout=f"{9100 + len(calls)};deltaai\n", stderr="")

    monkeypatch.setattr(bundle.subprocess, "run", submit)
    bundle.submit(root, actual=True, hold=True)
    bundle.submit(root, actual=True, hold=True)
    assert len(calls) == 2
    records = [json.loads(line) for line in (root / "bundle_submissions.jsonl").read_text().splitlines()]
    assert [(record["arm"], record["group"]) for record in records] == [
        (arm, "all") for arm in bundle.ARMS]
    assert all(record["initially_held"] for record in records)
    assert all(record["evaluation_manifest_sha256"] == bundle.sha256(root / "manifest.json")
               for record in records)
    assert all(record["bundle_manifest_sha256"] == bundle.sha256(root / "orchestration/manifest.json")
               for record in records)


def test_scheduler_validation_does_not_submit_or_create_receipts(bundle, suite, monkeypatch):
    root, _manifest = suite
    calls = []
    monkeypatch.setattr(bundle.subprocess, "run", lambda command, **kwargs:
                        (calls.append(command) or SimpleNamespace(stdout="", stderr="test successful")))
    bundle.submit(root)
    assert len(calls) == 2 and all("--test-only" in command for command in calls)
    assert all("--parsable" not in command for command in calls)
    assert not (root / "bundle_submissions.jsonl").exists()


@pytest.mark.parametrize("changed", ["original_manifest", "launcher", "bundle_manifest", "source"])
def test_frozen_source_and_orchestration_drift_are_rejected(bundle, suite, changed):
    root, manifest = suite
    if changed == "original_manifest":
        (root / "manifest.json").write_text(json.dumps(dict(manifest, altered=True)))
    elif changed == "launcher":
        with (root / "orchestration/qwen_paper_bundle.py").open("a") as handle:
            handle.write("\n# changed\n")
    elif changed == "bundle_manifest":
        path = root / "orchestration/manifest.json"
        data = json.loads(path.read_text())
        data["evaluation_manifest_sha256"] = "changed"
        path.write_text(json.dumps(data))
    else:
        with (Path(manifest["source_root"]) / "experiments/nonlinear_decoders/deltaai/qwen_paper_evals.py").open("a") as handle:
            handle.write("\n# changed\n")
    with pytest.raises(ValueError, match="changed"):
        bundle.validate(root)
        bundle.load_runner(root)
