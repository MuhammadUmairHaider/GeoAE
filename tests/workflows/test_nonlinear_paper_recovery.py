"""Recovery preserves completed science and spends only three GPU allocations."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "experiments/nonlinear_decoders/deltaai"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("paper_recovery", SCRIPTS / "nonlinear_paper_recovery.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
base = runner.base


@pytest.fixture(params=runner.SCRIPT_DIRECTORIES)
def recovery_inputs(tmp_path, request):
    layout = request.param
    parent, root = tmp_path / "parent", tmp_path / "recovery"
    parent.mkdir(); root.mkdir()
    for model in base.MODELS:
        run = parent / model
        source = run / "evaluation/source"
        (source / "geoae").mkdir(parents=True)
        (source / layout).mkdir(parents=True)
        (source / "geoae/checkpoint.py").write_text("# original loader\n")
        (source / layout / "nonlinear_paper_evals.py").write_text("# original evaluator\n")
        training_source = run / "source"
        (training_source / layout).mkdir(parents=True)
        (training_source / layout / "nonlinear_paper_training.py").write_text("# original trainer\n")
        (run / "evaluation/results").mkdir()
        (run / "evaluation/results/completed.json").write_text('{"old": true}\n')
        (run / "evaluation/state").mkdir()
        data = run / "cache"; data.mkdir()
        (data / "layer.npy").write_bytes(b"activation bytes")
        ready = run / "assets.json"
        base.atomic_json(ready, {"state": "ready", "hf_model": {
            "snapshot": str(run / "hf_snapshot"),
            "files": {"model.safetensors": {"size": 16, "sha256": "a" * 64}}}})
        profile = {"model": model, "model_name": model + "-model", "ready_manifest": str(ready)}
        runtime = {"python": "/env/bin/python", "modules": ["torch"]}
        training = {"profile": profile, "arms": list(base.ARMS), "runtime": runtime,
                    "source_root": str(training_source), "source_sha256": runner.hashes(training_source),
                    "dataset": {"data_dir": str(data),
                    "files": {"layer.npy": {"size": 16, "sha256": base.digest(data / "layer.npy")}}}}
        plan = [{"name": "passed_" + str(i)} for i in range(15)] + [{"name": "control_atlas_s5"}]
        evaluation = {"source_root": str(source), "source_sha256": runner.hashes(source),
                      "runtime": runtime, "profile": profile,
                      "assets": {"hf_home": "/cache/hf", "hf_datasets_cache": "/cache/datasets"},
                      "plans": {arm: plan for arm in base.ARMS}}
        base.atomic_json(run / "training_manifest.json", training)
        base.atomic_json(run / "evaluation/manifest.json", evaluation)
        base.atomic_json(run / "evaluation/cpu_preflight.json", {"state": "passed"})
        base.atomic_json(run / "evaluation/state/residual_mlp.json", {"arm": "residual_mlp", "steps": {
            step["name"]: {"state": "failed" if step["name"] == "control_atlas_s5" else "complete"}
            for step in plan}})
    base.prepare(parent)
    (root / "orchestration").mkdir()
    for name in runner.LAUNCHERS:
        (root / "orchestration" / name).write_text("# launcher " + name + "\n")
    for name in ("geoae/checkpoint.py", str(Path(layout) / "nonlinear_paper_evals.py")):
        path = root / "overlays" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# recovery overlay " + name + "\n")
    return root, parent


@pytest.fixture
def recovery(recovery_inputs):
    root, parent = recovery_inputs
    runner.prepare(root, parent)
    for model in base.MODELS:
        base.atomic_json(root / model / "evaluation/cpu_preflight.json", {"state": "passed"})
    base.atomic_json(root / "native_validation.json", {"counts": {"tests": 12, "failures": 0, "errors": 0, "skipped": 0}})
    return root, parent


def test_recovery_rejects_overlay_requiring_absent_refactor_helpers(recovery_inputs):
    root, parent = recovery_inputs
    original = base.read(parent / "llama/evaluation/manifest.json")
    name = runner.snapshot_script(original, "nonlinear_paper_evals.py")
    (root / "overlays" / name).write_text("from geoae.workflow.execute import temporary_staging_directory\n")
    before = runner.hashes(Path(original["source_root"]))
    with pytest.raises(ValueError, match="unavailable snapshot module: geoae.workflow.execute"):
        runner.prepare(root, parent)
    assert not (root / "llama/evaluation").exists()
    assert runner.hashes(Path(original["source_root"])) == before


def test_prepare_exact_three_tasks_preserves_originals_and_reuses_fifteen(recovery):
    root, parent = recovery
    manifest = runner.verify(root)
    assert set(manifest["tasks"]) == set(runner.TASKS)
    assert manifest["tasks"]["llama_residual_mlp"]["stages"] == ["evaluations"]
    for task in ("gemma_mlp", "gemma_residual_mlp"):
        assert manifest["tasks"][task]["stages"] == ["gpu_preflight", "training", "evaluations"]
    for model in base.MODELS:
        original = parent / model / "evaluation"
        fresh = root / model / "evaluation"
        assert (original / "source/geoae/checkpoint.py").read_text() == "# original loader\n"
        assert (fresh / "source/geoae/checkpoint.py").read_text().startswith("# recovery overlay")
        assert (fresh / "results").is_symlink()
        assert (fresh / "results").resolve() == (original / "results").resolve()
        assert (fresh / "results/completed.json").read_text() == '{"old": true}\n'
        assert (fresh / "logs").resolve() != (original / "logs").resolve()
        previous = base.read(original / "manifest.json")
        current = base.read(fresh / "manifest.json")
        assert runner.hashes(original / "source") == previous["source_sha256"]
        name = runner.snapshot_script(previous, "nonlinear_paper_evals.py")
        assert name in current["recovery"]["overlays"]
        assert current["recovery"]["overlays"][name] == {
            "old": previous["source_sha256"][name], "new": base.digest(fresh / "source" / name)}
    old = base.read(parent / "llama/evaluation/state/residual_mlp.json")
    copied = base.read(root / "llama/evaluation/state/residual_mlp.json")
    assert copied == old
    assert sum(x["state"] == "complete" for x in copied["steps"].values()) == 15
    before = base.digest(root / "manifest.json")
    runner.prepare(root, parent)
    assert base.digest(root / "manifest.json") == before


def test_resource_limits_one_gpu_two_hour_llama_twelve_hour_gemma(recovery):
    root, _ = recovery
    manifest = runner.verify(root)
    commands = {task: runner.build_command(root, manifest, task, True) for task in runner.TASKS}
    assert len(commands) == 3
    for task, command in commands.items():
        assert "--gpus-per-node=1" in command and "--export=NONE" in command
        assert "--hold" in command and "--parsable" in command
        assert ("--time=02:00:00" if task.startswith("llama") else "--time=12:00:00") in command
        assert ("--mem=64G" if task.startswith("llama") else "--mem=160G") in command
        assert command[-1] == task
    accounts = [next(x for x in c if x.startswith("--account=")) for c in commands.values()]
    assert set(accounts) == {"--account=bbyl-dtai-gh", "--account=bimc-dtai-gh"}


def test_dry_run_has_no_records_and_clears_inherited_sbatch(recovery, monkeypatch):
    root, _ = recovery
    monkeypatch.setenv("SBATCH_ACCOUNT", "wrong")
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        assert not any(k.startswith("SBATCH_") for k in kwargs["env"])
        return SimpleNamespace(stdout="test allocation accepted\n", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", fake)
    runner.submit(root, False)
    assert len(calls) == 3 and all("--test-only" in c for c in calls)
    assert not (root / "submissions.jsonl").exists()


def test_held_partial_submission_is_durable_and_retry_only_submits_missing(recovery, monkeypatch):
    root, _ = recovery
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=str(800 + len(calls)) + ";cluster\n", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", fake)
    with pytest.raises(subprocess.CalledProcessError):
        runner.submit(root, True)
    first = base.submission_records(root)
    assert len(first) == 1 and first[0]["initially_held"] is True
    runner.submit(root, True)
    records = base.submission_records(root)
    assert len(records) == 3 and len({r["task"] for r in records}) == 3
    assert records[0] == first[0]
    count = len(calls)
    runner.submit(root, True)
    assert len(calls) == count


def test_ambiguous_submission_is_not_recorded(recovery, monkeypatch):
    root, _ = recovery
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="ambiguous", stderr=""))
    with pytest.raises(ValueError, match="ambiguous submission"):
        runner.submit(root, True)
    assert base.submission_records(root) == []


def record_jobs(root):
    records = [{"task": task, "job_id": str(900+i)} for i, task in enumerate(runner.TASKS)]
    (root / "submissions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def test_partial_release_can_resume_after_first_job_started(recovery, monkeypatch):
    root, _ = recovery
    records = record_jobs(root); manifest = runner.verify(root)
    states = {r["job_id"]: "PENDING" for r in records}
    calls = []
    def inspect(job):
        record = next(r for r in records if r["job_id"] == job)
        resources = manifest["tasks"][record["task"]]["resources"]
        return {"Account": resources["account"], "QOS": resources["qos"], "TimeLimit": resources["walltime"],
                "JobState": states[job], "Reason": "JobHeldUser" if states[job] == "PENDING" else "None"}
    def release(command, **kwargs):
        job = command[-1]; calls.append(job)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command)
        states[job] = "RUNNING"
    monkeypatch.setattr(base, "inspect_job", inspect)
    monkeypatch.setattr(runner.subprocess, "run", release)
    with pytest.raises(subprocess.CalledProcessError):
        runner.release(root)
    assert base.read(root / "release_receipt.json")["released_jobs"] == ["900"]
    runner.release(root)
    assert calls == ["900", "901", "901", "902"]
    assert base.read(root / "release_receipt.json")["state"] == "complete"


@pytest.mark.parametrize("which", ["cpu", "native"])
def test_failed_validation_blocks_scheduler(recovery, monkeypatch, which):
    root, _ = recovery
    if which == "cpu":
        base.atomic_json(root / "gemma/evaluation/cpu_preflight.json", {"state": "failed"})
    else:
        base.atomic_json(root / "native_validation.json", {"counts": {"tests": 12, "failures": 1, "errors": 0, "skipped": 0}})
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("Must validate before Slurm"))
    with pytest.raises(ValueError):
        runner.submit(root, True)


def mock_model_stage(monkeypatch, root):
    import nonlinear_model_assets
    stage = root / "model_stage"; stage.mkdir()
    monkeypatch.setattr(runner, "model_stage_parent", lambda needed: stage)
    calls = []
    def materialize(snapshot, destination, expected=None):
        assert isinstance(snapshot, (str, Path)), "Pass the snapshot path, not its receipt"
        assert expected is not None, "Pin copied model bytes to the original receipt"
        destination.mkdir()
        calls.append((snapshot, destination, expected))
        return {"state": "ready", "snapshot": str(destination)}
    monkeypatch.setattr(nonlinear_model_assets, "materialize_snapshot", materialize)
    return calls


def test_llama_runs_evaluation_only_without_activation_staging(recovery, monkeypatch):
    root, parent = recovery
    staged = mock_model_stage(monkeypatch, root)
    monkeypatch.setattr(base, "choose_stage", lambda *a: pytest.fail("Llama must reuse completed activation-dependent evaluations"))
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        assert Path(command[2]).is_file(), "Launch the evaluator inside its own recorded snapshot layout"
        assert "--activations-dir" not in command
        assert "--evaluation-dir" in command
        assert kwargs["cwd"] == str(root / "llama/evaluation/source")
        override = json.loads(kwargs["env"]["GEOAE_LM_SNAPSHOTS_JSON"])
        assert Path(override["llama-model"]).is_dir()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", fake)
    runner.run(root, "llama_residual_mlp")
    assert len(calls) == 1 and calls[0][3] == "run"
    assert calls[0][2].endswith("nonlinear_paper_evals.py")
    assert base.read(root / "state/llama_residual_mlp.json")["state"] == "complete"
    assert not staged[0][1].exists()
    assert base.read(parent / "llama/evaluation/state/residual_mlp.json")["steps"]["control_atlas_s5"]["state"] == "failed"


def test_gemma_preflight_training_evals_share_verified_staging(recovery, monkeypatch):
    root, _ = recovery
    mock_model_stage(monkeypatch, root)
    stage = root / "activation_stage"; stage.mkdir()
    monkeypatch.setattr(base, "choose_stage", lambda dataset: stage)
    calls, datasets = [], []
    def fake(command, **kwargs):
        calls.append(command)
        assert Path(command[2]).is_file(), "Launch each stage from its own recorded snapshot layout"
        if "--activations-dir" in command:
            data = Path(command[command.index("--activations-dir") + 1])
            assert (data / "layer.npy").read_bytes() == b"activation bytes"
            datasets.append(data)
        if len(calls) == 2:
            assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
            assert "--evaluation-dir" not in command
        else:
            assert kwargs["env"]["OMP_NUM_THREADS"] == "8"
            assert "--evaluation-dir" in command
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", fake)
    runner.run(root, "gemma_mlp")
    assert [c[3] for c in calls] == ["gpu-preflight", "run", "run"]
    assert calls[1][2].endswith("nonlinear_paper_training.py")
    assert calls[2][2].endswith("nonlinear_paper_evals.py")
    assert datasets[0] == datasets[1] and not datasets[0].exists()
    report = base.read(root / "state/gemma_mlp.json")
    assert report["state"] == "complete"
    assert list(report["stages"]) == ["gpu_preflight", "training", "evaluations"]


def test_training_failure_blocks_evaluation_and_records_failure(recovery, monkeypatch):
    root, _ = recovery
    mock_model_stage(monkeypatch, root)
    monkeypatch.setattr(base, "choose_stage", lambda dataset: None)
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=4 if len(calls) == 2 else 0)
    monkeypatch.setattr(runner.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="training failed"):
        runner.run(root, "gemma_residual_mlp")
    report = base.read(root / "state/gemma_residual_mlp.json")
    assert len(calls) == 2 and report["state"] == "failed"
    assert "evaluations" not in report["stages"]


def test_mutated_recovery_source_or_parent_manifests_rejected(recovery):
    root, parent = recovery
    source = root / "gemma/evaluation/source/geoae/checkpoint.py"
    original = source.read_text(); source.write_text(original + "# changed\n")
    with pytest.raises(ValueError, match="Recovery source changed"):
        runner.verify(root)
    source.write_text(original)
    manifest = parent / "llama/training_manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="Original training manifest changed"):
        runner.verify(root)


@pytest.mark.parametrize("resource", ["experiments/config_paths.json", "geoae/baselines/matched_topk/_reference/SOURCE.json"])
def test_recovery_resource_hashes_preserve_historical_schema_and_pin_new_resources(tmp_path, resource):
    source = tmp_path / "source"
    source.mkdir()
    (source / "evaluator.py").write_text("# frozen source\n")
    historical = runner.hashes(source)
    additions = (resource, "README.md", "LICENSE", "scripts/py", "experiments/template.html",
                 "docs/paper/acl.sty", "docs/paper/acl_natbib.bst", "configs/sites/site.yml")
    for name in additions:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"version": 1}\n')
    assert runner.hashes(source, expected=historical) == historical
    current = runner.hashes(source)
    assert set(additions).issubset(current)
    for name in additions:
        path = source / name
        path.write_text('{"version": 2}\n')
        assert runner.hashes(source, expected=current) != current
        path.write_text('{"version": 1}\n')


def test_launcher_is_python36_compatible_and_shell_is_valid():
    ast.parse((SCRIPTS / "nonlinear_paper_recovery.py").read_text(), feature_version=(3, 6))
    subprocess.run(["bash", "-n", str(SCRIPTS / "nonlinear_paper_recovery.sbatch")], check=True)


def test_evaluator_recovery_skips_fifteen_verified_outputs_and_only_runs_atlas(recovery, monkeypatch):
    """Exercise the actual resume loop, including command/checkpoint/output hashes."""
    import nonlinear_paper_evals as evaluator
    import nonlinear_paper_summary as summary
    import torch
    root, parent = recovery
    fresh = root / "llama/evaluation"
    source = fresh / "source"
    steps, records = [], {}
    for index in range(15):
        output = parent / "llama/evaluation/results/residual_mlp" / ("passed_" + str(index) + ".json")
        output.parent.mkdir(exist_ok=True)
        output.write_text('{"valid": true}\n')
        step = {"name": "passed_" + str(index), "command": ["python", "already-completed", str(index)],
                "outputs": [str(output)], "dependencies": [], "needs_activations": True, "group": "core"}
        steps.append(step)
        records[step["name"]] = {"state": "complete", "checkpoint_sha256": "A",
                                  "command_sha256": evaluator.command_sha256(step),
                                  "output_sha256": {str(output): evaluator.digest(output)}}
    output = parent / "llama/evaluation/results/residual_mlp/atlas.json"
    atlas = {"name": "control_atlas_s5", "command": ["python", "atlas-only", "--substrates", "h,z"],
             "outputs": [str(output)], "dependencies": [], "needs_activations": False, "group": "controls"}
    steps.append(atlas); records[atlas["name"]] = {"state": "failed"}
    base.atomic_json(fresh / "state/residual_mlp.json", {"arm": "residual_mlp", "steps": records})
    original_state = (parent / "llama/evaluation/state/residual_mlp.json").read_bytes()
    manifest = {"source_root": str(source), "training_run_dir": str(parent / "llama"),
                "profile": {"final_step": 14450}, "plans": {"residual_mlp": steps},
                "assets": {"summary_control_datasets": ["atlas_s5"]}}
    monkeypatch.setattr(evaluator, "load_manifest", lambda directory: manifest)
    monkeypatch.setattr(evaluator, "verify_checkpoint", lambda *a: {"sha256": "A", "epoch": 50})
    monkeypatch.setattr(evaluator, "runtime_environment", lambda *a: {})
    monkeypatch.setattr(evaluator, "gpu_preflight", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "mock GH200")
    monkeypatch.setattr(evaluator.shutil, "disk_usage", lambda *a: pytest.fail("Completed core evals must not stage activations"))
    summary_calls = []
    monkeypatch.setattr(summary, "write_control_summary", lambda directory, **kwargs: summary_calls.append(directory))
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        assert command == atlas["command"]
        assert kwargs["cwd"] == source
        concept = {"ref": {"h": {}, "z": {}},
                   **{space + "_st_global_a" + alpha: {} for space in ("h", "z") for alpha in ("0.4", "0.5")}}
        output.write_text(json.dumps({"meta": {}, "summary": {"complete": True}, "concepts": {"0": concept}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(evaluator.subprocess, "run", execute)
    evaluator.run(SimpleNamespace(run_dir=parent / "llama", evaluation_dir=fresh, arm="residual_mlp", activations_dir=None))
    state = base.read(fresh / "state/residual_mlp.json")
    assert len(calls) == 1 and len(summary_calls) == 1
    assert state["state"] == "complete" and state["failed_steps"] == []
    assert all(record["state"] == "complete" for record in state["steps"].values())
    assert (parent / "llama/evaluation/state/residual_mlp.json").read_bytes() == original_state
    assert not (parent / "llama/evaluation/logs/residual_mlp_control_atlas_s5.log").exists()
