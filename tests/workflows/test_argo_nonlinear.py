"""Preparing Qwen runs must not accidentally allocate GPUs or rerun the baseline."""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("submit", [False, True])
def test_qwen_submission_defaults_to_validation_and_only_two_new_arms(tmp_path, monkeypatch, submit):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "experiments" / "nonlinear_decoders" / "argo"))
    runner = importlib.import_module("submit_nonlinear_qwen")
    source = tmp_path / "source"
    manifest = {"source_root": str(source), "slurm": {"account": "group", "partition": "group",
                "cpus": 12, "mem": "160G", "time": "3-00:00:00"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "load_and_verify", lambda path: manifest)
    monkeypatch.setenv("SBATCH_ACCOUNT", "wrong-inherited-account")
    calls = []

    def fake_run(command, **kwargs):
        assert "SBATCH_ACCOUNT" not in kwargs["env"]
        calls.append(command)
        return SimpleNamespace(stdout=str(123 + len(calls)), stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    args = ["submit_nonlinear_qwen.py", "--run-dir", str(tmp_path)]
    if submit:
        args.append("--submit")
    monkeypatch.setattr(sys, "argv", args)
    runner.main()
    assert [command[-1] for command in calls] == ["mlp", "residual_mlp"]
    assert all(("--parsable" if submit else "--test-only") in command for command in calls)
    assert all("--gres=gpu:l40:1" in command for command in calls)
    record = tmp_path / "submissions.jsonl"
    if submit:
        assert [json.loads(line)["arm"] for line in record.read_text().splitlines()] == ["mlp", "residual_mlp"]
    else:
        assert not record.exists()
