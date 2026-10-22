#!/usr/bin/env python3
"""Prepare two Qwen decoder runs and reuse the existing linear baseline. No jobs are submitted."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from nonlinear_qwen_common import (
    ARMS, BASELINE_NAME, DATA_NAME, TABLE_NAME, TEMPLATES, DEFAULT_PYTHON, DEFAULT_REFERENCE_ROOT,
    DEFAULT_RUN_DIR, assert_snapshot_imports, inspect_baseline, inspect_dataset, load_and_verify, sha256, source_hashes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source_root = Path(__file__).resolve().parents[2]
    if source_root != run_dir / "source":
        raise SystemExit(f"Stage reviewed code into {run_dir / 'source'} and execute this script there first")
    if (run_dir / "manifest.json").exists():
        load_and_verify(run_dir)
        print(f"Existing preparation verified: {run_dir}; no jobs submitted")
        return
    if not os.access(DEFAULT_PYTHON, os.X_OK):
        raise SystemExit(f"Python is not executable: {DEFAULT_PYTHON}")
    assert_snapshot_imports(source_root)
    from geoae.config import Config

    data_dir = (args.reference_root / DATA_NAME).resolve(strict=True)
    table_path = (args.reference_root / TABLE_NAME).resolve(strict=True)
    dataset = inspect_dataset(data_dir, table_path)
    baseline = inspect_baseline((args.reference_root / BASELINE_NAME).resolve(strict=True), source_root, data_dir)
    for name in ("configs", "checkpoints", "runs", "logs", "provenance", "locks"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    configs = {}
    for arm in ARMS:
        for parent in ("checkpoints", "runs"):
            output = run_dir / parent / arm
            if output.exists() and any(output.iterdir()):
                raise SystemExit(f"Refusing a populated output directory: {output}")
            output.mkdir(exist_ok=True)
        cfg = yaml.safe_load((source_root / TEMPLATES / f"{arm}.yaml").read_text())
        cfg["data"]["activations_dir"] = str(data_dir)
        cfg["model"]["token_bias"] = str(table_path)
        cfg["train"]["checkpoints_dir"] = str(run_dir / "checkpoints" / arm)
        cfg["train"]["runs_dir"] = str(run_dir / "runs" / arm)
        path = run_dir / "configs" / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        Config.from_yaml(path)
        configs[arm] = sha256(path)
    manifest = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root), "python": str(DEFAULT_PYTHON),
        "arms": list(ARMS), "source_sha256": source_hashes(source_root), "config_sha256": configs,
        "dataset": dataset,
        "existing_linear_baseline": baseline,
        "slurm": {"account": "def-hsajjad", "partition": "c-hsajjad", "gpu": "l40", "gpus": 1,
                  "cpus": 12, "mem": "160G", "time": "3-00:00:00"},
        "initialization": "The two new arms train from scratch with seed 42; no checkpoint is resumed.",
        "comparison_limitations": [
            "The MLP adds hidden/output biases and parameters as well as nonlinearity.",
            "Existing linear decoder columns are renormalized; MLP branches are not renormalized.",
            "The residual decoder retains the reference linear-skip column renormalization.",
        ],
        "submission_status": "prepared_only",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_and_verify(run_dir)
    print(f"Prepared {', '.join(ARMS)} at {run_dir}; no jobs submitted")
    print("Each arm: 50 epochs, 305 steps/epoch, 15,250 steps, batch 32,768, seed 42")


if __name__ == "__main__":
    main()
