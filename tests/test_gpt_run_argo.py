"""Verify the Argo campaign's dependencies, one-GPU scheduling and retries."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.gpt2 import run_gpt_run as runner
from scripts.gpt2 import submit_argo as submit


def test_submit_dry_run_requires_dependency_and_creates_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(submit, "ROOT", tmp_path)
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **k: pytest.fail("Dry run queried/submitted a job"))
    assert submit.main(["--after-gemma", "284383", "--dry-run", "--manifest", str(tmp_path / "campaign.json")]) == 0
    output = capsys.readouterr().out
    assert "--dependency=afterok:284383" in output
    assert "--job-name=gpt_run" in output
    assert "--gres=gpu:h100:1" in output
    assert "--train-concurrency 12" in output
    assert "--eval-concurrency 12" in output
    assert not list(tmp_path.iterdir())
    with pytest.raises(SystemExit):
        submit.main(["--dry-run"])


@pytest.mark.parametrize("state,exit_code,valid", [("COMPLETED", "0:0", True),
    ("RUNNING", "0:0", True), ("FAILED", "1:0", False), ("COMPLETED", "1:0", False)])
def test_dependency_guard_rejects_unsuccessful_gemma_jobs(monkeypatch, state, exit_code, valid):
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=f"284383|g4b-evals|{state}|{exit_code}|\n"))
    if valid:
        assert submit.verify_dependencies(["284383"])[0]["state"] == state
    else:
        with pytest.raises(ValueError, match="afterok cannot succeed"):
            submit.verify_dependencies(["284383"])


def test_submission_preserves_old_delta_manifest_and_archives_resubmission(tmp_path, monkeypatch):
    monkeypatch.setattr(submit, "ROOT", tmp_path)
    old_delta = tmp_path / "runs/gpt2_2x/campaign.json"
    old_delta.parent.mkdir(parents=True)
    old_delta.write_text('{"jobs":{"train":"22639322"}}\n')
    path = tmp_path / "runs/gpt_run/campaign.json"
    path.parent.mkdir(parents=True)
    old_argo = '{"jobs":{"pipeline":"300"}}\n'
    path.write_text(old_argo)
    calls = []
    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "sacct":
            return SimpleNamespace(stdout="284383|g4b-evals|COMPLETED|0:0|\n")
        if argv[0] == "squeue":
            return SimpleNamespace(stdout="")
        # Record the intended recipe before sbatch can create a job.
        assert json.loads(path.read_text())["jobs"] == {}
        return SimpleNamespace(stdout="301;argo\n")
    monkeypatch.setattr(submit.subprocess, "run", command)
    assert submit.main(["--after-gemma", "284383", "--manifest", str(path), "--resubmit"]) == 0
    result = json.loads(path.read_text())
    assert result["jobs"] == {"pipeline": "301"}
    assert result["run_name"] == "gpt_run"
    assert result["gpu_count"] == 1
    assert result["dependency"] == "afterok:284383"
    assert result["applied_dependency"] is None
    assert result["gemma_ordering"] == "verified_completed"
    submitted = next(command for command in calls if command[0] == "sbatch")
    assert not any(argument.startswith("--dependency=") for argument in submitted)
    assert len(result["training_configs"]) == 12
    assert old_delta.read_text() == '{"jobs":{"train":"22639322"}}\n'
    assert next(path.parent.glob("campaign.*.json")).read_text() == old_argo


def test_active_gemma_dependency_retains_afterok_and_completed_job_is_recorded(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(submit, "ROOT", tmp_path)
    path = tmp_path / "campaign.json"
    calls = []
    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "sacct":
            return SimpleNamespace(stdout="284382|geoae-train|COMPLETED|0:0|\n284383|g4b-evals|RUNNING|0:0|\n")
        return SimpleNamespace(stdout="301\n")
    monkeypatch.setattr(submit.subprocess, "run", command)
    assert submit.main(["--after-gemma", "284382,284383", "--manifest", str(path)]) == 0
    result = json.loads(path.read_text())
    assert result["dependency"] == "afterok:284382:284383"
    assert result["applied_dependency"] == "afterok:284383"
    assert result["gemma_ordering"] == "waiting_afterok"
    assert [row["state"] for row in result["dependency_jobs"]] == ["COMPLETED", "RUNNING"]
    assert "--dependency=afterok:284383" in calls[-1]
    assert "--dependency=afterok:284382:284383" not in calls[-1]
    assert "--dependency=afterok:284383" in capsys.readouterr().out


def test_resubmit_refuses_duplicate_running_job(tmp_path, monkeypatch):
    monkeypatch.setattr(submit, "ROOT", tmp_path)
    path = tmp_path / "campaign.json"
    path.write_text('{"jobs":{"pipeline":"300"}}')
    def command(argv, **kwargs):
        if argv[0] == "sacct":
            return SimpleNamespace(stdout="284383|g4b-evals|COMPLETED|0:0|\n")
        if argv[0] == "squeue":
            return SimpleNamespace(stdout="300 RUNNING\n")
        pytest.fail("Duplicate job submitted")
    monkeypatch.setattr(submit.subprocess, "run", command)
    assert submit.main(["--after-gemma", "284383", "--manifest", str(path), "--resubmit"]) == 1
    assert json.loads(path.read_text()) == {"jobs": {"pipeline": "300"}}


def test_runner_dry_run_plans_all_layers_without_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("Dry run executed a process"))
    assert runner.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "gpt_run" in output and "blocks 0–11" in output
    assert "requested 12" in output
    assert "epoch-50 completion marker" in output
    assert "--steps gpu" in output and "--steps api" in output
    assert not list(tmp_path.iterdir())


def test_training_resume_is_selected_from_existing_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    folder = tmp_path / "checkpoints/gpt2/layer4" / runner.CKDIR
    folder.mkdir(parents=True)
    (folder / "step_0001000.pt").touch()
    monkeypatch.setattr(runner, "verified", lambda command: False)
    calls = []
    monkeypatch.setattr(runner, "run", lambda command, *args: calls.append(command))
    runner.train_layer(4, tmp_path / "stage")
    training = next(cmd for cmd in calls if "geoae.train" in cmd)
    assert training[-2:] == ["--resume", "latest"]
    assert training[training.index("--activations_dir") + 1] == str(tmp_path / "stage")
    assert calls[-1][-3:] == ["complete", "--layer", "4"]


def test_gpu_only_campaign_does_not_plan_external_judges(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    assert runner.main(["--dry-run", "--judges", "skip"]) == 0
    output = capsys.readouterr().out
    assert "--steps gpu" in output
    assert "--steps api" not in output


def test_argo_environment_links_campaign_data_and_refuses_foreign_paths(tmp_path):
    checkout, bulk = tmp_path / "checkout", tmp_path / "bulk"
    checkout.mkdir()
    bulk.mkdir()
    sentinel = bulk / "existing_gemma_run"
    sentinel.write_text("keep")
    environment = dict(os.environ, GEOAE_ROOT=str(checkout), DELTA_DATA=str(bulk),
                       ARGO_SCRATCH=str(tmp_path / "scratch"), GEOAE_PYTHON=sys.executable,
                       UV_CACHE_DIR=str(tmp_path / "uv_cache"),
                       UV_PYTHON_INSTALL_DIR=str(tmp_path / "uv_python"),
                       WANDB_DIR=str(tmp_path / "wandb"), TRITON_CACHE_DIR=str(tmp_path / "triton"))
    script = runner.ROOT / "scripts/gpt2/argo_env.sh"
    command = ["bash", "-c", 'set -e; source "$1"', "gpt_run_env_test", str(script)]
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (checkout / "checkpoints/gpt2").resolve() == bulk / "checkpoints/gpt2"
    assert (checkout / "results/gpt_run").resolve() == bulk / "results/gpt_run"
    assert sentinel.read_text() == "keep"
    link = checkout / "cache/gpt2"
    link.unlink()
    link.mkdir()
    conflict = subprocess.run(command, env=environment, text=True, capture_output=True)
    assert conflict.returncode != 0
    assert "resolves outside DELTA_DATA" in conflict.stderr
    assert link.is_dir() and not link.is_symlink()


def test_scheduler_retries_only_oom_layer_and_keeps_completed_layer(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    state_path = tmp_path / "state.json"
    state = {}
    launches = {}
    script = ("import pathlib,sys; p=pathlib.Path(sys.argv[1]); first=not p.exists(); "
              "p.touch(); print('CUDA out of memory' if first else 'ok'); sys.exit(1 if first else 0)")
    def command(layer):
        launches[layer] = launches.get(layer, 0) + 1
        return [sys.executable, "-c", script, str(tmp_path / "oom_once")] if layer == 0 else [sys.executable, "-c", "print('ok')"]
    assert runner.run_layers("train", [0, 1], 2, command, state, state_path, logs, poll_seconds=0.01)
    assert launches == {0: 2, 1: 1}
    assert state["stages"]["train"]["concurrency"] == 1
    assert state["stages"]["train"]["oom_fallback"] is True
    assert all(row["status"] == "complete" for row in state["stages"]["train"]["layers"].values())
    assert len(list(logs.glob("train_layer0_attempt*.log"))) == 2


def test_scheduler_reports_persistent_failure_after_single_layer_oom(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    state = {}
    command = lambda layer: [sys.executable, "-c", "print('CUDA out of memory'); raise SystemExit(1)"]
    assert not runner.run_layers("eval_gpu", [0], 1, command, state, tmp_path / "state.json", tmp_path, poll_seconds=0.01)
    assert state["stages"]["eval_gpu"]["status"] == "failed"
    assert state["stages"]["eval_gpu"]["failed_layers"] == [0]
