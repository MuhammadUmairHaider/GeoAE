"""Checkpoint writes must survive a disk-full crash. CPU only.

Regression for 2026-09-18: torch.save hit a full disk mid-write and left an 8 MB
step_0008804.pt among 251 MB siblings, which `--resume latest` would have loaded.
"""
from pathlib import Path

import pytest
import torch

from geoae import train_common as tc


def _write_truncated(path: Path, src: Path, nbytes: int) -> None:
    path.write_bytes(src.read_bytes()[:nbytes])


def test_atomic_save_leaves_nothing_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "step_0000002.pt"
    real_save = torch.save

    def failing_save(obj, f):          # write part of the file, then die like a full disk
        Path(f).write_bytes(b"PK\x03\x04partial")
        raise RuntimeError("basic_ios::clear: iostream error")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(RuntimeError):
        tc.atomic_torch_save({"a": 1}, target)
    monkeypatch.setattr(torch, "save", real_save)
    assert not target.exists(), "no file may appear under the real name"
    assert not list(tmp_path.glob("*.tmp")), "the temp file must be cleaned up"


def test_atomic_save_replaces_previous_only_on_success(tmp_path):
    target = tmp_path / "best_val.pt"
    tc.atomic_torch_save({"v": 1}, target)
    tc.atomic_torch_save({"v": 2}, target)
    assert torch.load(target, weights_only=False)["v"] == 2


def test_atomic_copy_protects_destination(tmp_path, monkeypatch):
    src, dst = tmp_path / "step_0000009.pt", tmp_path / "best_val.pt"
    tc.atomic_torch_save({"v": "new"}, src)
    tc.atomic_torch_save({"v": "old-best"}, dst)

    def failing_copy(a, b):
        Path(b).write_bytes(b"half")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tc.shutil, "copy2", failing_copy)
    with pytest.raises(OSError):
        tc.atomic_copy(src, dst)
    assert torch.load(dst, weights_only=False)["v"] == "old-best", "best_val.pt must survive"
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_latest_skips_truncated_newest(tmp_path, capsys):
    good = tmp_path / "step_0008520.pt"
    tc.atomic_torch_save({"model_state": {}, "step": 8520}, good)
    _write_truncated(tmp_path / "step_0008804.pt", good, 64)
    assert tc.resolve_resume("latest", tmp_path) == good
    assert "skipping step_0008804.pt" in capsys.readouterr().out


def test_resume_latest_all_corrupt_raises(tmp_path):
    (tmp_path / "step_0000001.pt").write_bytes(b"not a zip")
    with pytest.raises(FileNotFoundError, match="no readable"):
        tc.resolve_resume("latest", tmp_path)


def test_ensure_free_space_message(tmp_path):
    with pytest.raises(OSError, match="Nothing was written"):
        tc.ensure_free_space(tmp_path, 10**18, "step_0000001.pt")
    tc.ensure_free_space(tmp_path, 1, "tiny")   # enough room: no error
