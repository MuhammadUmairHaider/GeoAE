"""Offline snapshot recovery must preserve model bytes and reject stale copies."""
import ast
import fcntl
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/deltaai/nonlinear_model_assets.py"
SPEC = importlib.util.spec_from_file_location("nonlinear_model_assets", SCRIPT)
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


@pytest.fixture
def snapshot(tmp_path):
    source = tmp_path / "cache/snapshots/commit"
    source.mkdir(parents=True)
    blob_dir = tmp_path / "cache/blobs"
    blob_dir.mkdir()
    (source / "config.json").write_text('{"model_type":"llama"}')
    (source / "tokenizer_config.json").write_text("{}")
    (source / "tokenizer.json").write_text("{}")
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}}))
    data = tmp_path / "shard"
    data.write_bytes(b"model tensor bytes")
    blob = blob_dir / assets.digest(data)
    data.rename(blob)
    (source / "model-00001-of-00001.safetensors").symlink_to(blob)
    return source


def test_original_receipt_and_lfs_blob_are_fully_verified(snapshot):
    receipt = assets.verify_snapshot(snapshot)
    assert receipt["state"] == "verified"
    assert receipt["shards"] == ["model-00001-of-00001.safetensors"]
    original = {"hf_model": {"files": receipt["files"]}}
    assert assets.verify_snapshot(snapshot, original)["files"] == receipt["files"]


def test_materialized_copy_is_independent_atomic_ready_and_idempotent(snapshot, tmp_path):
    destination = tmp_path / "stage/model"
    receipt = assets.materialize_snapshot(snapshot, destination)
    assert receipt["state"] == "ready"
    assert receipt["snapshot"] == str(destination)
    assert (destination / assets.READY).is_file()
    assert all(not entry.is_symlink() for entry in destination.iterdir())
    assert assets.materialize_snapshot(snapshot, destination) == receipt
    assert assets.verify_snapshot(destination, receipt)["files"] == receipt["files"]


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json", "model-00001-of-00001.safetensors"])
def test_expected_receipt_rejects_changed_bytes(snapshot, name):
    receipt = assets.verify_snapshot(snapshot)
    path = snapshot / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256 mismatch|differs from expected"):
        assets.verify_snapshot(snapshot, receipt)


def test_model_shard_symlink_cannot_be_broken(snapshot):
    (snapshot / "model-00001-of-00001.safetensors").resolve().unlink()
    with pytest.raises(FileNotFoundError):
        assets.verify_snapshot(snapshot)


@pytest.mark.parametrize("name", ["../escape.safetensors", "/absolute.safetensors", "geoae_snapshot_ready.json"])
def test_index_cannot_reference_files_outside_snapshot(snapshot, name):
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": name}}))
    with pytest.raises(ValueError, match="Unsafe"):
        assets.verify_snapshot(snapshot)


def test_receipt_must_cover_every_shard_and_offline_tokenizer(snapshot):
    receipt = assets.verify_snapshot(snapshot)
    del receipt["files"]["model-00001-of-00001.safetensors"]
    with pytest.raises(ValueError, match="omits required"):
        assets.verify_snapshot(snapshot, receipt)
    receipt = assets.verify_snapshot(snapshot)
    del receipt["files"]["tokenizer.json"]
    with pytest.raises(ValueError, match="omits the offline tokenizer"):
        assets.verify_snapshot(snapshot, receipt)


def test_stale_staged_bytes_are_not_accepted(snapshot, tmp_path):
    destination = tmp_path / "stage"
    assets.materialize_snapshot(snapshot, destination)
    (destination / "tokenizer.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="differs from expected"):
        assets.materialize_snapshot(snapshot, destination)


def test_incomplete_destination_is_never_overwritten(snapshot, tmp_path):
    destination = tmp_path / "partial"
    destination.mkdir()
    (destination / "keep").write_text("preexisting")
    with pytest.raises(ValueError, match="not ready"):
        assets.materialize_snapshot(snapshot, destination)
    assert (destination / "keep").read_text() == "preexisting"


def test_copy_failure_never_publishes_ready_receipt(snapshot, tmp_path, monkeypatch):
    destination = tmp_path / "stage"
    def fail(*args):
        raise OSError("copy failure")
    monkeypatch.setattr(assets.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="copy failure"):
        assets.materialize_snapshot(snapshot, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".stage.*"))


def test_concurrent_destination_materialization_fails_before_mutation(snapshot, tmp_path):
    destination = tmp_path / "stage"
    with (tmp_path / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            assets.materialize_snapshot(snapshot, destination)
    assert not destination.exists()


def test_helper_runs_on_native_login_python36():
    ast.parse(SCRIPT.read_text(), feature_version=(3, 6))
