"""Campaign checks cover dependency handling and durable completion guards."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(f"gpt2_{name}_test", ROOT / "scripts/gpt2" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_dry_run_does_not_submit_or_write(tmp_path, monkeypatch, capsys):
    submit = module("submit")
    monkeypatch.setattr(submit, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["submit.py", "--dry-run", "--manifest", str(tmp_path / "campaign.json")])
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **k: pytest.fail("Dry run submitted a command"))
    assert submit.main() == 0
    output = capsys.readouterr().out
    assert "--dependency=afterok:CORPUS_JOB" in output
    assert "--dependency=afterok:EXTRACT_JOB" in output
    assert "--array=0-11%2" in output
    assert not list(tmp_path.iterdir())


def test_dependency_ids_are_recorded_incrementally(tmp_path, monkeypatch):
    submit = module("submit")
    monkeypatch.setattr(submit, "ROOT", tmp_path)
    path = tmp_path / "campaign.json"
    monkeypatch.setattr(sys, "argv", ["submit.py", "--manifest", str(path), "--concurrency", "3"])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            assert json.loads(path.read_text())["jobs"] == {"corpus": "101"}
        return SimpleNamespace(stdout=f"{100 + len(calls)};delta\n")
    monkeypatch.setattr(submit.subprocess, "run", run)
    assert submit.main() == 0
    assert json.loads(path.read_text())["jobs"] == {"corpus": "101", "extract": "102", "train": "103"}
    assert "--dependency=afterok:101" in calls[1]
    assert "--dependency=afterok:102" in calls[2]
    assert "--array=0-11%3" in calls[2]


def test_submission_failure_preserves_already_submitted_job(tmp_path, monkeypatch):
    submit = module("submit")
    monkeypatch.setattr(submit, "ROOT", tmp_path)
    path = tmp_path / "campaign.json"
    monkeypatch.setattr(sys, "argv", ["submit.py", "--manifest", str(path)])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) > 1:
            raise subprocess.CalledProcessError(1, command, stderr="Account limit")
        return SimpleNamespace(stdout="101\n")
    monkeypatch.setattr(submit.subprocess, "run", run)
    assert submit.main() == 1
    saved = json.loads(path.read_text())
    assert saved["jobs"] == {"corpus": "101"}
    assert "submission_error" in saved
    assert len(calls) == 2


@pytest.mark.parametrize("epoch,layer,table,valid", [
    (49, 0, 0, False), (50, 1, 0, False), (50, 0, 1, False), (50, 0, 0, True)])
def test_completion_requires_final_epoch_and_matching_layer(tmp_path, monkeypatch, epoch, layer, table, valid):
    lifecycle = module("lifecycle")
    monkeypatch.setattr(lifecycle, "ROOT", tmp_path)
    directory = lifecycle.checkpoint_dir(0)
    directory.mkdir(parents=True)
    (directory / "step_0001000.pt").touch()
    config = dict(model=dict(hidden_size=768, latent_dim=1536, n_clusters=2000,
                             token_bias_mode="bypass", token_bias=f"e2e/checkpoints/general/gpt2/layer{table}/token_bias_sampled_struct5.npz"),
                  data=dict(target_layer=layer), train=dict(seed=42, n_epochs=50), extraction=dict(model_name="gpt2"))
    checkpoint = dict(config=config, epoch=epoch, step=1000)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **k: checkpoint))
    if valid:
        assert lifecycle.write_completion(0)["epoch"] == 50
        marker = json.loads((directory / "training_complete.json").read_text())
        assert marker["checkpoint"].endswith("step_0001000.pt")
        assert not list(directory.glob("*.tmp"))
    else:
        with pytest.raises(ValueError):
            lifecycle.write_completion(0)
        assert not (directory / "training_complete.json").exists()


def test_table_check_rejects_mismatched_normalization(tmp_path, monkeypatch):
    lifecycle = module("lifecycle")
    monkeypatch.setattr(lifecycle, "ROOT", tmp_path)
    act = tmp_path / "act"
    act.mkdir()
    monkeypatch.setattr(lifecycle, "ACT", act)
    path = tmp_path / "e2e/checkpoints/general/gpt2/layer0/token_bias_sampled_struct5.npz"
    path.parent.mkdir(parents=True)
    mean, std = np.zeros(768, np.float32), np.ones(768, np.float32)
    np.savez(act / "norm_params_layer0.npz", mean=mean, std=std)
    def save(table_mean):
        np.savez(path, layer=0, model_name="gpt2", table=np.zeros((1, 768), np.float16),
                 token_ids=[1], vocab_size=50257, shrink_k=30.0, min_count=10,
                 norm_mean=table_mean, norm_std=std)
    save(mean)
    assert lifecycle.check_table(0) == path
    save(mean + 1)
    with pytest.raises(ValueError, match="mean disagree"):
        lifecycle.check_table(0)
