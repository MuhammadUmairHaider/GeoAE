#!/usr/bin/env python3
"""Verify a prepared Qwen snapshot, recipes and dataset without training."""
from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from nonlinear_qwen_common import ARMS, DEFAULT_RUN_DIR, load_and_verify, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--build-model", action="store_true", help="Read back architecture through the real model factory")
    parser.add_argument("--runtime-output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = load_and_verify(run_dir)
    report = {
        "validated_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID"), "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "manifest_sha256": sha256(run_dir / "manifest.json"), "source_root": manifest["source_root"],
        "data_dir": manifest["dataset"]["data_dir"], "models": {},
    }
    if args.require_cuda or args.build_model:
        import torch
        from geoae.config import Config
        from geoae.training.mse import build_model
        import geoae.model

        source_root = Path(manifest["source_root"])
        if not Path(geoae.model.__file__).resolve().is_relative_to(source_root):
            raise SystemExit(f"Model imported outside snapshot: {geoae.model.__file__}")
        report["model_module"] = str(Path(geoae.model.__file__).resolve())
        report["torch"] = torch.__version__
        report["cuda_runtime"] = torch.version.cuda
        if args.require_cuda and not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable; refusing CPU training fallback")
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        if args.build_model:
            for arm in ARMS:
                cfg = Config.from_yaml(run_dir / "configs" / f"{arm}.yaml")
                # Shape/count readback needs no token table or GPU allocation.
                cfg.model.token_bias = ""
                model = build_model(cfg, torch.device("cpu"))
                report["models"][arm] = {
                    "decoder": repr(model.decoder),
                    "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                }
                del model
    if args.runtime_output:
        args.runtime_output.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Qwen run preparation verified; no training performed")


if __name__ == "__main__":
    main()
