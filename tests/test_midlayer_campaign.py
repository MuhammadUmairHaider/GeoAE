"""Two-H100 campaign submission stays durable across interruptions and retries."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts.midlayer import campaign


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setattr(campaign, "validate_campaign_recipe", lambda: {
        "source_mix": {"structured": 5}, "source_fingerprint": "test", "n_train_rows": 10_000_000})
    return tmp_path


def test_dry_run_does_not_query_or_write(sandbox, monkeypatch, capsys):
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("Dry run called a command"))
    assert campaign.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("--gres=gpu:h100:1") == 2
    assert "--dependency=afterok:LLAMA_JOB:GEMMA4B_JOB" in out
    assert "midlayer_judges.sbatch" in out
    assert "--judges required" in out
    assert not list(sandbox.iterdir())


def test_corpus_dependency_applies_only_to_llama(sandbox):
    args = campaign.parser().parse_args(["--dry-run", "--corpus-job", "123"])
    commands = campaign.submit_commands(args)
    assert "--dependency=afterok:123" in commands["llama"]
    assert not any(arg.startswith("--dependency") for arg in commands["gemma4b"])
    assert all(command[command.index("--judges") + 1] == "skip" for command in commands.values())


def test_submission_records_each_job_before_next(sandbox, monkeypatch):
    path = sandbox / "runs" / campaign.DEFAULT_RUN / "campaign.json"
    calls = []
    def run(command, **kwargs):
        if command[0] == "squeue":
            return SimpleNamespace(stdout="")
        calls.append(command)
        if len(calls) == 2:
            assert json.loads(path.read_text())["jobs"] == {"llama": "101"}
        if len(calls) == 3:
            assert json.loads(path.read_text())["jobs"] == {"llama": "101", "gemma4b": "102"}
        return SimpleNamespace(stdout=f"{100 + len(calls)};argo\n")
    monkeypatch.setattr(campaign.subprocess, "run", run)
    assert campaign.main(["--submit"]) == 0
    saved = json.loads(path.read_text())
    assert saved["jobs"] == {"llama": "101", "gemma4b": "102", "judges": "103"}
    assert saved["new_training_count"] == 6
    assert saved["gpu_count"] == 2
    assert saved["status"] == "submitted"
    assert "--dependency=afterok:101:102" in calls[2]
    assert not any(argument.startswith("--gres") for argument in calls[2])


@pytest.mark.parametrize("failed_call", [2, 3])
def test_failure_preserves_ids_and_resume_only_submits_missing(sandbox, monkeypatch, failed_call):
    path = sandbox / "runs" / campaign.DEFAULT_RUN / "campaign.json"
    calls = []
    def fail(command, **kwargs):
        if command[0] == "squeue":
            return SimpleNamespace(stdout="")
        calls.append(command)
        if len(calls) == failed_call:
            raise subprocess.CalledProcessError(1, command, stderr="temporary Slurm rejection")
        return SimpleNamespace(stdout=f"{100 + len(calls)}\n")
    monkeypatch.setattr(campaign.subprocess, "run", fail)
    assert campaign.main(["--submit"]) == 1
    previous = json.loads(path.read_text())
    assert len(previous["jobs"]) == failed_call - 1
    assert previous["status"] == "submission_failed"
    resumed = []
    def resume(command, **kwargs):
        if command[0] == "squeue":
            return SimpleNamespace(stdout="")
        resumed.append(command)
        return SimpleNamespace(stdout=f"{200 + len(resumed)}\n")
    monkeypatch.setattr(campaign.subprocess, "run", resume)
    assert campaign.main(["--submit", "--resume-submission"]) == 0
    saved = json.loads(path.read_text())
    for lane, identifier in previous["jobs"].items():
        assert saved["jobs"][lane] == identifier
    assert len(resumed) == 4 - failed_call
    assert "submission_error" not in saved


def test_active_duplicate_without_manifest_is_refused(sandbox, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"555|{campaign.DEFAULT_RUN}_llama|RUNNING\n")
    monkeypatch.setattr(campaign.subprocess, "run", run)
    assert campaign.main(["--submit"]) == 1
    assert len(calls) == 1
    assert not (sandbox / "runs" / campaign.DEFAULT_RUN / "campaign.json").exists()


def test_wrong_recipe_is_refused_before_querying_slurm(sandbox, monkeypatch):
    def wrong():
        raise ValueError("Exactly ten sources and exactly 10M training rows required")
    monkeypatch.setattr(campaign, "validate_campaign_recipe", wrong)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("Invalid recipe reached Slurm"))
    assert campaign.main(["--submit"]) == 1
    assert not list(sandbox.iterdir())


def test_lane_has_exactly_three_trainings_and_no_legacy_recipe(sandbox, monkeypatch, capsys):
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("Dry lane ran a command"))
    assert campaign.main(["--lane", "llama", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("--steps all") == 3
    assert out.count("--judges skip") == 3
    assert "--arm llama_mid_bypass" in out
    assert "--arm llama_mid_plain" in out
    assert "--arm llama_last_plain" in out
    assert "--arm llama_last_bypass" not in out
    assert not list(sandbox.iterdir())


def test_unowned_staging_directory_is_preserved(sandbox, monkeypatch):
    staging = sandbox / "staging"
    staging.mkdir()
    important = staging / "unrelated"
    important.write_text("keep")
    assert campaign.main(["--lane", "llama", "--stage-dump-root", str(staging)]) == 1
    assert important.read_text() == "keep"


def test_absent_final_reference_is_pending_not_trained(sandbox, monkeypatch):
    from scripts.gpt2 import evaluate
    def missing(*a, **kw):
        raise FileNotFoundError("No new-dataset checkpoint")
    monkeypatch.setattr(evaluate, "checkpoint_for", missing)
    result = campaign.reference_status("llama")
    assert result == {"arm": "llama_last_bypass", "status": "pending", "reason": "No new-dataset checkpoint"}


def test_cpu_judges_are_serial_and_cover_all_six(sandbox, monkeypatch):
    directory = sandbox / "runs" / campaign.DEFAULT_RUN
    directory.mkdir(parents=True)
    for lane in campaign.LANES:
        (directory / f"lane_{lane}.json").write_text(json.dumps({"status": "complete", "reference": {"status": "pending"}}))
    monkeypatch.setattr(campaign, "evaluation_status", lambda *a: "complete")
    commands = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    assert campaign.main(["--run-judges"]) == 0
    evaluated = [cmd[cmd.index("--arm") + 1] for cmd in commands[:-1]]
    assert evaluated == [arm for arms in campaign.LANES.values() for arm in arms]
    assert all(cmd[cmd.index("--eval-steps") + 1] == "api" for cmd in commands[:-1])
    assert json.loads((directory / "judges_state.json").read_text())["status"] == "complete"


def test_cpu_judges_refuse_unfinished_gpu_lane(sandbox, monkeypatch):
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: pytest.fail("Judge ran before GPU completion"))
    assert campaign.main(["--run-judges"]) == 1
    path = sandbox / "runs" / campaign.DEFAULT_RUN / "judges_state.json"
    assert json.loads(path.read_text())["status"] == "failed"
