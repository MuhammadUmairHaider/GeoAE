#!/usr/bin/env python3
"""Read Slurm and per-layer GPT-2 training/evaluation status without tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/gpt_run/campaign.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.manifest.exists():
        print(f"No submitted campaign manifest: {args.manifest}")
        return 1
    manifest = json.loads(args.manifest.read_text())
    ids = [str(v) for v in manifest.get("jobs", {}).values() if v]
    states: dict = {"jobs": manifest.get("jobs", {}), "queue": [], "layers": {}}
    if ids:
        try:
            result = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T|%R"],
                                    capture_output=True, text=True, check=True)
            states["queue"] = [dict(zip(("id", "state", "reason"), line.split("|", 2)))
                               for line in result.stdout.splitlines()]
        except (OSError, subprocess.CalledProcessError) as error:
            states["queue_error"] = str(error)
    for layer in manifest["layers"]:
        directory = ROOT / manifest["checkpoint_dirs"][str(layer)]
        marker = directory / "training_complete.json"
        checkpoints = sorted(directory.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        evaluation = ROOT / "results" / manifest.get("run_name", "gpt2") / f"layer{layer}" / "pipeline_state.json"
        states["layers"][str(layer)] = dict(
            training=json.loads(marker.read_text()) if marker.exists() else {"complete": False},
            latest_checkpoint=str(checkpoints[-1].relative_to(ROOT)) if checkpoints else None,
            evaluation=json.loads(evaluation.read_text()) if evaluation.exists() else None)
    if args.json:
        print(json.dumps(states, indent=2))
    else:
        print("Jobs: " + ", ".join(f"{key}={value}" for key, value in states["jobs"].items()))
        for job in states["queue"]:
            print(f"  {job['id']} {job['state']}: {job['reason']}")
        for layer, row in states["layers"].items():
            training = row["training"]
            train = f"complete (epoch {training['epoch']})" if "epoch" in training else "waiting/in progress"
            evaluation = row["evaluation"]
            print(f"Layer {int(layer):2d}: training {train}; eval {evaluation.get('status', 'in progress') if evaluation else 'waiting'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
