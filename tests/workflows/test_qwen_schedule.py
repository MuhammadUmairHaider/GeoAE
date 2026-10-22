"""Qwen submission dependencies, durable retries and owned staging."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.model_transfer.qwen import submit


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(submit, "ROOT", tmp_path)
    monkeypatch.setattr(submit, "validate_recipe", lambda: {"model": submit.MODEL, "n_train_rows": 10_000_000})
    path = tmp_path / "runs" / submit.GEMMA_RUN / "campaign.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run_name": submit.GEMMA_RUN, "jobs": {"gemma4b": "292220"},
                               "lanes": {"gemma4b": ["gemma4b_mid_bypass", "gemma4b_mid_plain", "gemma4b_last_plain"]}}))
    return tmp_path


class Slurm:
    def __init__(self, root, *, state="RUNNING", exit_code="0:0", fail_at=None, active="", name=None):
        self.root, self.state, self.exit_code = root, state, exit_code
        self.fail_at, self.active = fail_at, active
        self.name = name or submit.job_name(submit.GEMMA_RUN, "gemma4b")
        self.commands, self.environments, self.attempts = [], [], 0
        self.path = root / "runs" / submit.DEFAULT_RUN / "campaign.json"

    def run(self, command, **kwargs):
        if command[0] == "squeue":
            return SimpleNamespace(stdout=self.active)
        if command[0] == "sacct":
            return SimpleNamespace(stdout=f"292220|{self.name}|{self.state}|{self.exit_code}|\n")
        assert command[0] == "sbatch"
        manifest = json.loads(self.path.read_text())
        # Every acknowledgement is present on disk before another submission.
        assert len(manifest["jobs"]) == len(self.commands)
        self.attempts += 1
        if self.attempts == self.fail_at:
            raise subprocess.CalledProcessError(1, command, stderr="Slurm unavailable")
        self.commands.append(command)
        self.environments.append(kwargs["env"])
        return SimpleNamespace(stdout=f"{100 + len(self.commands)};argo\n")


def test_dry_run_has_three_one_h100_jobs_and_cpu_dependency_without_io(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(submit, "ROOT", tmp_path)
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **k: pytest.fail("Dry run queried Slurm"))
    monkeypatch.setattr(submit, "validate_recipe", lambda: pytest.fail("Dry run read recipe"))
    for method in ("open", "read_text", "write_text", "mkdir"):
        monkeypatch.setattr(Path, method, lambda *a, **k: pytest.fail("Dry run touched filesystem"))
    assert submit.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert output.count("--gres=gpu:h100:1") == 3
    assert output.count("--mem=160G") == 3
    assert "--dependency=afterok:PREP_JOB:292220" in output
    judges = next(line for line in output.splitlines() if line.startswith("[judges]"))
    assert "--dependency=afterok:BYPASS_JOB:PLAIN_JOB" in judges
    assert "--cpus-per-task=4" in judges and "--mem=32G" in judges
    assert "--gres" not in judges


def test_fresh_dry_run_imports_only_standard_library():
    root = Path(__file__).absolute().parents[2]
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'numpy', 'yaml', 'geoae'}:
            raise AssertionError('Dry run imported ' + fullname)
sys.meta_path.insert(0, Block())
from experiments.model_transfer.qwen import submit
assert submit.main(['--dry-run']) == 0
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "COMPLETING", "COMPLETED"])
def test_submission_dependencies_and_incremental_manifest(sandbox, monkeypatch, state):
    slurm = Slurm(sandbox, state=state)
    monkeypatch.setattr(submit.subprocess, "run", slurm.run)
    monkeypatch.setenv("SBATCH_GRES", "gpu:4")
    monkeypatch.setenv("SBATCH_ACCOUNT", "wrong-account")
    monkeypatch.delenv("GEOAE_ENV_FILE", raising=False)
    assert submit.main(["--submit"]) == 0
    saved = json.loads(slurm.path.read_text())
    assert saved["jobs"] == dict(zip(submit.PHASES, ("101", "102", "103", "104")))
    assert saved["status"] == "submitted" and saved["new_training_count"] == 2
    assert saved["gemma_dependency"]["model_validated"] is True
    assert saved["gemma_dependency"]["state"] == state
    assert not any(arg.startswith("--dependency") for arg in slurm.commands[0])
    assert "--dependency=afterok:101" in slurm.commands[1]
    expected = "--dependency=afterok:101" + ("" if state == "COMPLETED" else ":292220")
    assert expected in slurm.commands[2]
    assert "--dependency=afterok:102:103" in slurm.commands[3]
    assert not any(arg.startswith("--gres") for arg in slurm.commands[3])
    assert all(not any(key.startswith("SBATCH_") for key in env) for env in slurm.environments)
    assert all(env["GEOAE_ENV_FILE"] == ".env.midlast" for env in slurm.environments)


@pytest.mark.parametrize("failed_call", [1, 2, 3, 4])
def test_failure_preserves_each_id_and_resume_only_submits_missing(sandbox, monkeypatch, failed_call):
    slurm = Slurm(sandbox, fail_at=failed_call)
    monkeypatch.setattr(submit.subprocess, "run", slurm.run)
    assert submit.main(["--submit"]) == 1
    previous = json.loads(slurm.path.read_text())
    assert len(previous["jobs"]) == failed_call - 1
    assert previous["status"] == "submission_failed"
    slurm.fail_at = None
    slurm.state = "COMPLETED"
    assert submit.main(["--submit", "--resume-submission"]) == 0
    saved = json.loads(slurm.path.read_text())
    assert saved["status"] == "submitted" and len(slurm.commands) == 4
    assert "submission_error" not in saved
    assert all(saved["jobs"][phase] == identifier for phase, identifier in previous["jobs"].items())
    assert len(saved["gemma_state_history"]) == 2
    assert saved["gemma_state_history"][0]["state"] == "RUNNING"
    assert saved["gemma_state_history"][1]["state"] == "COMPLETED"


def test_existing_manifest_never_overwritten_or_resubmitted(sandbox, monkeypatch):
    slurm = Slurm(sandbox)
    monkeypatch.setattr(submit.subprocess, "run", slurm.run)
    assert submit.main(["--submit"]) == 0
    original = slurm.path.read_bytes()
    assert submit.main(["--submit"]) == 1
    assert submit.main(["--submit", "--resume-submission"]) == 1
    assert slurm.path.read_bytes() == original and len(slurm.commands) == 4


def test_active_job_without_manifest_refuses_duplicate(sandbox, monkeypatch):
    slurm = Slurm(sandbox, active=f"555|{submit.job_name(submit.DEFAULT_RUN, 'prep')}|RUNNING\n")
    monkeypatch.setattr(submit.subprocess, "run", slurm.run)
    assert submit.main(["--submit"]) == 1
    assert not slurm.path.exists() and not slurm.commands


def test_active_unrecorded_job_blocks_partial_resume(sandbox, monkeypatch):
    slurm = Slurm(sandbox, fail_at=2)
    monkeypatch.setattr(submit.subprocess, "run", slurm.run)
    assert submit.main(["--submit"]) == 1
    previous = slurm.path.read_bytes()
    slurm.active = f"666|{submit.job_name(submit.DEFAULT_RUN, 'bypass')}|RUNNING\n"
    slurm.fail_at = None
    assert submit.main(["--submit", "--resume-submission"]) == 1
    assert len(slurm.commands) == 1 and slurm.path.read_bytes() == previous


def test_submitter_lock_blocks_second_submitter(sandbox, monkeypatch):
    path = sandbox / "runs" / submit.DEFAULT_RUN / "campaign.json"
    path.parent.mkdir(parents=True)
    slurm = Slurm(sandbox)
    monkeypatch.setattr(submit.subprocess, "run", slurm.run)
    with path.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert submit.main(["--submit"]) == 1
    assert not slurm.commands and not path.exists()


@pytest.mark.parametrize("state,exit_code", [("FAILED", "1:0"), ("CANCELLED", "0:15"),
                                             ("TIMEOUT", "0:0"), ("COMPLETED", "1:0")])
def test_unsuccessful_gemma_requires_manual_resolution(sandbox, monkeypatch, state, exit_code, capsys):
    slurm = Slurm(sandbox, state=state, exit_code=exit_code)
    monkeypatch.setattr(submit.subprocess, "run", slurm.run)
    assert submit.main(["--submit"]) == 1
    assert "manual resolution required" in capsys.readouterr().err
    assert not slurm.commands and not slurm.path.exists()


def test_gemma_wrong_job_name_or_model_refused(sandbox, monkeypatch):
    slurm = Slurm(sandbox, name="some_other_job")
    monkeypatch.setattr(submit.subprocess, "run", slurm.run)
    assert submit.main(["--submit"]) == 1
    path = sandbox / "runs" / submit.GEMMA_RUN / "campaign.json"
    saved = json.loads(path.read_text())
    saved["lanes"]["gemma4b"] = ["llama_mid_bypass"]
    path.write_text(json.dumps(saved))
    assert submit.main(["--submit"]) == 1
    assert not slurm.commands and not slurm.path.exists()


def test_wrong_recipe_is_refused_before_slurm_or_new_files(sandbox, monkeypatch):
    def wrong():
        raise ValueError("Qwen must have exactly 10M training rows")
    monkeypatch.setattr(submit, "validate_recipe", wrong)
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **k: pytest.fail("Wrong recipe queried Slurm"))
    assert submit.main(["--submit"]) == 1
    assert not (sandbox / "runs" / submit.DEFAULT_RUN).exists()


def test_worker_commands_use_explicit_pair_and_split_gpu_api_phases(sandbox):
    args = submit.parser().parse_args(["--phase", "prep"])
    prep = submit.worker_commands(args, sandbox / "ram")[0]
    assert prep[prep.index("--steps") + 1] == "corpus,extract,table"
    for phase, arm in zip(("bypass", "plain"), submit.ARMS):
        args.phase = phase
        command = submit.worker_commands(args, sandbox / "ram")[0]
        assert command[command.index("--arm") + 1] == arm
        assert command[command.index("--steps") + 1] == "train,evaluate,report"
        assert command[command.index("--eval-steps") + 1] == "gpu"
        assert command[command.index("--judges") + 1] == "skip"
    args.phase = "judges"
    commands = submit.worker_commands(args)
    assert [command[3] for command in commands[:2]] == list(submit.ARMS)
    assert all(command[command.index("--steps") + 1] == "api" for command in commands[:2])
    assert all(command[command.index("--judges") + 1] == "required" for command in commands[:2])
    assert [command[command.index("--arm") + 1] for command in commands[2:]] == list(submit.ARMS)
    assert all("all" not in command and "new" not in command for command in commands)


@pytest.mark.parametrize("fails", [False, True])
def test_owned_gpu_staging_is_removed_even_on_failure(sandbox, monkeypatch, fails):
    staging = sandbox / "ram"
    source = sandbox / "durable"
    source.mkdir()
    (source / "sentinel").write_text("keep")
    def run(command, **kwargs):
        assert (staging / ".qwen_owner.json").is_file()
        (staging / "copied_dump").mkdir()
        if fails:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(submit.subprocess, "run", run)
    result = submit.main(["--phase", "bypass", "--stage-dump-root", str(staging)])
    assert result == (1 if fails else 0)
    assert not staging.exists() and (source / "sentinel").read_text() == "keep"
    state = json.loads((sandbox / "runs" / submit.DEFAULT_RUN / "phase_bypass.json").read_text())
    assert state["status"] == ("failed" if fails else "complete")


def test_unowned_or_symlink_staging_is_preserved(sandbox, monkeypatch):
    staging = sandbox / "ram"
    staging.mkdir()
    (staging / "unrelated").write_text("keep")
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **k: pytest.fail("Unowned staging ran training"))
    assert submit.main(["--phase", "bypass", "--stage-dump-root", str(staging)]) == 1
    assert (staging / "unrelated").read_text() == "keep"
    link = sandbox / "link"
    link.symlink_to(staging, target_is_directory=True)
    assert submit.main(["--phase", "bypass", "--stage-dump-root", str(link)]) == 1
    assert (staging / "unrelated").read_text() == "keep"


def test_cpu_judges_require_both_gpu_phases_then_run_serially(sandbox, monkeypatch):
    commands = []
    monkeypatch.setattr(submit.subprocess, "run", lambda command, **kwargs: commands.append(command))
    assert submit.main(["--phase", "judges"]) == 1
    assert not commands
    directory = sandbox / "runs" / submit.DEFAULT_RUN
    for phase in ("bypass", "plain"):
        (directory / f"phase_{phase}.json").write_text('{"status": "complete"}')
    assert submit.main(["--phase", "judges"]) == 0
    assert len(commands) == 4
    assert [command[3] for command in commands[:2]] == list(submit.ARMS)
    saved = json.loads((directory / "phase_judges.json").read_text())
    assert saved["status"] == "complete" and saved["stage_dump_root"] is None


def test_cpu_judge_failure_is_durable_and_stops_final_report(sandbox, monkeypatch):
    directory = sandbox / "runs" / submit.DEFAULT_RUN
    directory.mkdir(parents=True)
    for phase in ("bypass", "plain"):
        (directory / f"phase_{phase}.json").write_text('{"status": "complete"}')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(submit.subprocess, "run", run)
    assert submit.main(["--phase", "judges"]) == 1
    saved = json.loads((directory / "phase_judges.json").read_text())
    assert saved["status"] == "failed" and len(calls) == 1


def test_approved_recipe_validates():
    # The real validation may import the repository config package; dry runs
    # above deliberately skip it and require only the standard library.
    root = Path(__file__).absolute().parents[2]
    code = """
import sys
from experiments.model_transfer.qwen import submit
recipe = submit.validate_recipe()
assert recipe['model'] == 'Qwen/Qwen2.5-3B'
assert recipe['n_train_rows'] == 10_000_000
assert len(recipe['config_sha256']) == 3
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
