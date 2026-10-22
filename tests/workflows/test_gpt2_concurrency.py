"""Admission planning and opt-in DPC coordination for one-GPU layer jobs."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from geoae.workflow.resources import serialized_density_peaks

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gpt2_concurrency_test", ROOT / "experiments/gpt2/concurrency.py")
concurrency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(concurrency)
GIB = concurrency.GIB


def test_all_twelve_fit_when_measured_peak_allows():
    count, reservation = concurrency.concurrency_limit(95 * GIB, 4 * GIB)
    assert count == 12
    assert count * reservation + 4 * GIB <= 95 * GIB


def test_large_eval_stages_reduce_concurrency_without_changing_recipe():
    count, reservation = concurrency.concurrency_limit(95 * GIB, 9 * GIB)
    assert count < 12
    assert count * reservation + 4 * GIB <= 95 * GIB
    assert (count + 1) * reservation + 4 * GIB > 95 * GIB


def test_profile_never_silently_admits_worker_that_does_not_fit():
    with pytest.raises(RuntimeError, match="One worker needs"):
        concurrency.concurrency_limit(8 * GIB, 7 * GIB)
    with pytest.raises(ValueError):
        concurrency.concurrency_limit(95 * GIB, 0)


def test_no_cuda_profile_writes_no_success_file(tmp_path, monkeypatch):
    output = tmp_path / "profile.json"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["concurrency.py", "profile", "--config", "unused.yaml", "--out", str(output)])
    with pytest.raises(SystemExit) as error:
        concurrency.main()
    assert error.value.code == 2
    assert not output.exists()


def test_cpu_density_call_keeps_normal_behavior_and_creates_no_lock(tmp_path, monkeypatch):
    path = tmp_path / "init.lock"
    monkeypatch.setenv("GEOAE_GPU_INIT_LOCK", str(path))
    original = torch.arange(3)
    function = serialized_density_peaks(lambda z, count: (z + count, {"count": count}))
    value, metrics = function(original, 4)
    assert torch.equal(value, original + 4)
    assert metrics == {"count": 4}
    assert not path.exists()


def test_cuda_density_work_is_serialized_and_releases_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOAE_GPU_INIT_LOCK", str(tmp_path / "init.lock"))
    first_entered, release_first, second_attempted = (threading.Event() for _ in range(3))
    calls = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    z = SimpleNamespace(device=SimpleNamespace(type="cuda"))

    @serialized_density_peaks
    def function(z, worker):
        calls.append(f"start{worker}")
        if worker == 1:
            first_entered.set()
            assert release_first.wait(5)
        calls.append(f"end{worker}")
        return worker

    def second_worker():
        second_attempted.set()
        assert function(z, 2) == 2
    first = threading.Thread(target=lambda: function(z, 1))
    second = threading.Thread(target=second_worker)
    first.start()
    assert first_entered.wait(5)
    second.start()
    assert second_attempted.wait(5)
    assert "start2" not in calls
    release_first.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert calls == ["start1", "end1", "synchronize", "empty_cache",
                     "start2", "end2", "synchronize", "empty_cache"]


def test_failed_density_call_releases_shared_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOAE_GPU_INIT_LOCK", str(tmp_path / "init.lock"))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    z = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    @serialized_density_peaks
    def failed(z):
        raise ValueError("original failure")
    with pytest.raises(ValueError, match="original failure"):
        failed(z)
    assert serialized_density_peaks(lambda z: 7)(z) == 7
