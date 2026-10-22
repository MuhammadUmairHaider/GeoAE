#!/usr/bin/env python3
"""Verify and materialize an immutable, offline Hugging Face model snapshot.

The caller chooses the destination and its storage budget. This helper does not
download files, allocate GPUs, or modify the shared Hugging Face cache.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


READY = "geoae_snapshot_ready.json"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _name(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value in (".", "..", READY):
        raise ValueError("Unsafe model snapshot filename: {!r}".format(value))
    return value


def _expected_files(expected):
    if expected is None:
        return None
    if isinstance(expected, (str, Path)):
        expected = json.loads(Path(expected).read_text())
    if "hf_model" in expected:
        expected = expected["hf_model"]
    records = expected["files"]
    if not isinstance(records, dict) or not records:
        raise ValueError("Expected snapshot receipt has no files")
    for name, record in records.items():
        _name(name)
        if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            raise ValueError("Expected snapshot receipt needs full SHA256: " + name)
        size = record.get("size", record.get("bytes"))
        if not isinstance(size, int) or size < 1:
            raise ValueError("Expected snapshot receipt has an invalid size: " + name)
    return records


def verify_snapshot(snapshot, expected=None):
    """Fully read every declared file and check index, sizes, and all hashes.

    ``expected`` accepts the original campaign verified-assets receipt, its
    ``hf_model`` member, or this helper's own verification receipt. HF LFS blob
    names are also checked independently against the computed SHA256.
    """
    snapshot = Path(snapshot).absolute()
    if not snapshot.is_dir():
        raise ValueError("Model snapshot directory is missing: " + str(snapshot))
    wanted = _expected_files(expected)
    index_name = "model.safetensors.index.json"
    config_name = "config.json"
    config = json.loads((snapshot / config_name).read_text())
    if not isinstance(config, dict) or not isinstance(config.get("model_type"), str):
        raise ValueError("Model config has no model_type")
    index = json.loads((snapshot / index_name).read_text())
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights or not all(isinstance(k, str) for k in weights):
        raise ValueError("Model shard index has no valid weight_map")
    shards = sorted({_name(value) for value in weights.values()})
    if any(not name.endswith(".safetensors") for name in shards):
        raise ValueError("Model index references a non-safetensors shard")
    required = {config_name, index_name, "tokenizer_config.json"} | set(shards)
    if not any((snapshot / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise ValueError("Model snapshot has no offline tokenizer")
    names = set(wanted) if wanted is not None else {
        p.name for p in snapshot.iterdir() if (p.is_file() or p.is_symlink())
        and p.name != READY and not p.name.startswith(".")}
    if not required <= names:
        raise ValueError("Snapshot receipt omits required files: " + repr(sorted(required - names)))
    if not {"tokenizer.json", "tokenizer.model"} & names:
        raise ValueError("Snapshot receipt omits the offline tokenizer")
    files = {}
    for name in sorted(names):
        _name(name)
        path = snapshot / name
        resolved = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("Snapshot entry is not a regular file: " + str(path))
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Snapshot file changed during verification: " + name)
        if re.fullmatch(r"[0-9a-f]{64}", resolved.name) and sha != resolved.name:
            raise ValueError("HF LFS blob SHA256 mismatch: " + name)
        if wanted is not None:
            record = wanted[name]
            if sha != record["sha256"] or before.st_size != record.get("size", record.get("bytes")):
                raise ValueError("Snapshot differs from expected receipt: " + name)
        files[name] = {"size": before.st_size, "sha256": sha}
    return {"schema": 1, "state": "verified", "snapshot": str(snapshot),
            "model_type": config["model_type"], "shards": shards, "files": files,
            "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def materialize_snapshot(snapshot, destination, expected=None):
    """Copy snapshot bytes with dereferenced symlinks and publish atomically.

    A sibling file lock protects concurrent callers. Existing ready copies are
    accepted only after full revalidation; incomplete destinations are rejected.
    The returned snapshot path can populate ``GEOAE_LM_SNAPSHOTS_JSON``.
    """
    snapshot, destination = Path(snapshot).absolute(), Path(destination).absolute()
    if snapshot.resolve() == destination.resolve():
        raise ValueError("Materialization destination must differ from source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (destination.parent / (destination.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_receipt = verify_snapshot(snapshot, expected)
        if destination.exists():
            receipt_path = destination / READY
            if not receipt_path.is_file():
                raise ValueError("Existing snapshot destination is not ready: " + str(destination))
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("source_snapshot") != str(snapshot) or receipt.get("state") != "ready":
                raise ValueError("Existing snapshot destination belongs to another source")
            verified = verify_snapshot(destination, source_receipt)
            if receipt.get("files") != verified["files"]:
                raise ValueError("Existing snapshot ready receipt differs from bytes")
            return receipt
        required = sum(record["size"] for record in source_receipt["files"].values())
        if shutil.disk_usage(destination.parent).free < required + 64 * 1024 * 1024:
            raise ValueError("Insufficient space for the verified model snapshot")
        temporary = Path(tempfile.mkdtemp(prefix="." + destination.name + ".", dir=str(destination.parent)))
        try:
            for name in source_receipt["files"]:
                # copyfile follows symlinks; the resulting snapshot is independent
                # of shared-cache aliases and symlink/blob metadata lookups.
                shutil.copyfile(str(snapshot / name), str(temporary / name))
            verified = verify_snapshot(temporary, source_receipt)
            verified.update(state="ready", snapshot=str(destination), source_snapshot=str(snapshot))
            with (temporary / READY).open("w") as handle:
                json.dump(verified, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.rename(str(temporary), str(destination))
            return verified
        finally:
            if temporary.exists():
                shutil.rmtree(str(temporary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "materialize"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.action == "materialize":
        if args.destination is None:
            parser.error("materialize requires --destination")
        result = materialize_snapshot(args.snapshot, args.destination, args.expected)
    else:
        result = verify_snapshot(args.snapshot, args.expected)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
