"""One allocation per arm, durable held submissions, and sequential reuse."""
import ast
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/nonlinear_decoders/deltaai/nonlinear_paper_campaign.py"
SPEC = importlib.util.spec_from_file_location("paper_campaign", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def campaign(tmp_path):
    for model in runner.MODELS:
        run = tmp_path / model
        (run / "evaluation/source").mkdir(parents=True)
        (run / "source").mkdir()
        data = run / "cache"
        data.mkdir()
        (data / "layer.npy").write_bytes(b"activation bytes")
        training = {"profile": {"model": model, "model_name": model + "-model", "assets": {}},
                    "arms": list(runner.ARMS), "runtime": {"python": "/env/bin/python", "modules": ["torch"]},
                    "source_root": str(run / "source"), "dataset": {"data_dir": str(data),
                    "files": {"layer.npy": {"size": len(b"activation bytes"), "sha256": runner.digest(data / "layer.npy")}}}}
        evaluation = {"source_root": str(run / "evaluation/source"), "runtime": training["runtime"],
                      "plans": {arm: [{"name": "geometry"}, {"name": "mmlu"}] for arm in runner.ARMS}}
        runner.atomic_json(run / "training_manifest.json", training)
        runner.atomic_json(run / "evaluation/manifest.json", evaluation)
        runner.atomic_json(run / "evaluation/cpu_preflight.json", {"state": "passed"})
    runner.prepare(tmp_path)
    return tmp_path


def test_four_jobs_one_gpu_short_walltime_and_both_accounts(campaign):
    manifest = runner.verify(campaign)
    commands = [runner.build_command(campaign, manifest, model, arm, actual=True)
                for model in runner.MODELS for arm in runner.ARMS]
    assert manifest["job_count"] == len(commands) == 4
    accounts = []
    for command in commands:
        assert "--gpus-per-node=1" in command
        assert "--cpus-per-task=12" in command
        assert "--time=12:00:00" in command
        assert "--hold" in command
        assert "--export=NONE" in command
        accounts.append(next(x for x in command if x.startswith("--account=")))
        model = command[-2]
        assert ("--mem=128G" if model == "llama" else "--mem=160G") in command
    assert accounts.count("--account=bbyl-dtai-gh") == 2
    assert accounts.count("--account=bimc-dtai-gh") == 2
    original = runner.digest(campaign / "campaign_manifest.json")
    runner.prepare(campaign)
    assert runner.digest(campaign / "campaign_manifest.json") == original


def test_test_only_does_not_submit_or_create_receipt(campaign, monkeypatch):
    calls = []
    monkeypatch.setenv("SBATCH_ACCOUNT", "inherited-wrong-account")
    monkeypatch.setenv("SBATCH_GPUS", "4")
    def fake(command, **kwargs):
        calls.append(command)
        assert not any(k.startswith("SBATCH_") for k in kwargs["env"])
        assert kwargs["universal_newlines"] is True
        return SimpleNamespace(stdout="estimated schedule\n", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", fake)
    runner.submit(campaign, actual=False)
    assert len(calls) == 4
    assert all("--test-only" in command and "--hold" not in command for command in calls)
    assert not (campaign / "submissions.jsonl").exists()


def test_partial_submission_persists_held_job_and_retry_only_submits_missing(campaign, monkeypatch):
    calls = []
    should_fail = [True]
    def fake(command, **kwargs):
        calls.append(command)
        assert "--hold" in command and "--parsable" in command
        if len(calls) == 2 and should_fail[0]:
            raise subprocess.CalledProcessError(1, command, stderr="temporary scheduler rejection")
        return SimpleNamespace(stdout=str(800 + len(calls)) + ";cluster\n", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", fake)
    with pytest.raises(subprocess.CalledProcessError):
        runner.submit(campaign, actual=True)
    first = runner.submission_records(campaign)
    assert len(first) == 1 and first[0]["initially_held"] is True
    assert (first[0]["model"], first[0]["arm"]) == ("llama", "mlp")
    should_fail[0] = False
    runner.submit(campaign, actual=True)
    records = runner.submission_records(campaign)
    assert len(records) == 4
    assert len({(x["model"], x["arm"]) for x in records}) == 4
    assert records[0] == first[0]
    assert all(x["initially_held"] for x in records)
    count = len(calls)
    runner.submit(campaign, actual=True)
    assert len(calls) == count


def test_submission_refuses_ambiguous_job_id(campaign, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="ambiguous", stderr=""))
    with pytest.raises(ValueError, match="Inspect Slurm"):
        runner.submit(campaign, actual=True)
    assert runner.submission_records(campaign) == []


def test_partial_release_can_resume_after_earlier_jobs_start(campaign, monkeypatch):
    manifest = runner.verify(campaign)
    records = [{"model": model, "arm": arm, "job_id": str(900 + index)}
               for index, (model, arm) in enumerate((m, a) for m in runner.MODELS for a in runner.ARMS)]
    (campaign / "submissions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    states = {r["job_id"]: "PENDING" for r in records}
    calls = []
    def inspect(job):
        record = next(r for r in records if r["job_id"] == job)
        resources = manifest["models"][record["model"]]["resources"][record["arm"]]
        return {"Account": resources["account"], "QOS": resources["qos"],
                "TimeLimit": "12:00:00", "JobState": states[job],
                "Reason": "JobHeldUser" if states[job] == "PENDING" else "None"}
    def release(command, **kwargs):
        calls.append(command[-1])
        if len(calls) == 3:
            raise subprocess.CalledProcessError(1, command)
        states[command[-1]] = "RUNNING"
    monkeypatch.setattr(runner, "inspect_job", inspect)
    monkeypatch.setattr(runner.subprocess, "run", release)
    with pytest.raises(subprocess.CalledProcessError):
        runner.release(campaign)
    assert runner.read(campaign / "release_receipt.json")["released_jobs"] == ["900", "901"]
    runner.release(campaign)
    assert calls == ["900", "901", "902", "902", "903"]
    assert runner.read(campaign / "release_receipt.json")["state"] == "complete"


def test_failed_cpu_preflight_blocks_release(campaign, monkeypatch):
    records = [{"model": model, "arm": arm, "job_id": str(900 + index)}
               for index, (model, arm) in enumerate((m, a) for m in runner.MODELS for a in runner.ARMS)]
    (campaign / "submissions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    runner.atomic_json(campaign / "gemma/evaluation/cpu_preflight.json", {"state": "failed"})
    monkeypatch.setattr(runner, "inspect_job", lambda job: pytest.fail("Must reject before touching scheduler"))
    with pytest.raises(ValueError, match="CPU preflight"):
        runner.release(campaign)


def test_training_then_eval_share_persistent_directory_until_both_finish(campaign, tmp_path, monkeypatch):
    parent = tmp_path / "stage"
    parent.mkdir()
    monkeypatch.setattr(runner, "choose_stage", lambda dataset: parent)
    calls, directories = [], []
    def fake(command, **kwargs):
        calls.append(command)
        if "--activations-dir" in command:
            data = Path(command[command.index("--activations-dir") + 1])
            directories.append(data)
            assert data.is_dir() and (data / "layer.npy").read_bytes() == b"activation bytes"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", fake)
    runner.run(campaign, "llama", "mlp")
    assert len(calls) == 3
    assert calls[0][2].endswith("nonlinear_paper_evals.py") and calls[0][3] == "gpu-preflight"
    assert calls[1][2].endswith("nonlinear_paper_training.py")
    assert calls[2][2].endswith("nonlinear_paper_evals.py") and calls[2][3] == "run"
    assert directories[0] == directories[1]
    assert not directories[0].exists()
    state = runner.read(campaign / "state/llama_mlp.json")
    assert state["state"] == "complete"
    assert state["staging"]["mode"] == "persistent_node_local_copy"
    assert list(state["stages"]) == ["gpu_preflight", "training", "evaluations"]


@pytest.mark.parametrize("codes,failed_stage,calls_expected", [
    ([4], "gpu_preflight", 1), ([0, 2], "training", 2), ([0, 0, 3], "evaluations", 3)])
def test_failure_records_stage_and_training_failure_blocks_evaluations(campaign, monkeypatch, codes, failed_stage, calls_expected):
    monkeypatch.setattr(runner, "choose_stage", lambda dataset: None)
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=codes[len(calls)-1])
    monkeypatch.setattr(runner.subprocess, "run", fake)
    with pytest.raises(RuntimeError, match=failed_stage + " failed"):
        runner.run(campaign, "gemma", "residual_mlp")
    assert len(calls) == calls_expected
    state = runner.read(campaign / "state/gemma_residual_mlp.json")
    assert state["state"] == "failed"
    assert state["stage"] == failed_stage
    assert state["stages"][failed_stage]["state"] == "failed"
    if failed_stage in ("gpu_preflight", "training"):
        assert "evaluations" not in state["stages"]
    else:
        assert state["stages"]["training"]["state"] == "complete"
    assert state["staging"]["mode"] == "verified_source_mmap"


def test_staging_capacity_falls_back_to_tmp_then_source(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1024")
    dataset = {"files": {"layer.npy": {"size": 100}}}
    # Even with ample /dev/shm space, respect allocation RAM headroom.
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: SimpleNamespace(free=200 * 1024**3))
    assert runner.choose_stage(dataset) == tmp_path
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    assert runner.choose_stage(dataset) is None


def test_snapshot_manifest_and_launcher_mutation_rejected(campaign):
    source = campaign / "llama/training_manifest.json"
    saved = source.read_text()
    source.write_text(saved + " ")
    with pytest.raises(ValueError, match="manifest changed"):
        runner.verify(campaign)
    source.write_text(saved)
    launcher = campaign / "orchestration/nonlinear_paper_campaign.py"
    launcher.write_text(launcher.read_text() + "\n# mutation\n")
    with pytest.raises(ValueError, match="launcher changed"):
        runner.verify(campaign)


def test_collector_is_python36_compatible_and_preserves_scalar_digest_pins(campaign, monkeypatch):
    ast.parse(SCRIPT.read_text(), feature_version=(3, 6))
    manifest = runner.verify(campaign)
    for info in manifest["models"].values():
        assert isinstance(info["training_manifest_sha256"], str)
        assert isinstance(info["evaluation_manifest_sha256"], str)
        assert len(info["training_manifest_sha256"]) == len(info["evaluation_manifest_sha256"]) == 64
    records = [{"job_id": "123", "model": "llama", "arm": "mlp"}]
    (campaign / "submissions.jsonl").write_text("\n".join(json.dumps(x) for x in records) + "\n")
    def fake(command, **kwargs):
        assert kwargs.get("universal_newlines") is True
        if command[0] == "squeue":
            return "123|PENDING|0:00||Priority|bbyl-dtai-gh|bbyl-dtai-gh|12:00:00|2026-10-20T12:00:00\n"
        return ""
    monkeypatch.setattr(runner.subprocess, "check_output", fake)
    result = runner.status(campaign)
    arm = result["jobs"]["llama"]["mlp"]
    assert arm["state"] == "PENDING"
    assert arm["estimated_start_utc"] == "2026-10-20T12:00:00+00:00"
