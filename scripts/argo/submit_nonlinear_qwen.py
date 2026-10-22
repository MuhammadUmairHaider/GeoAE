#!/usr/bin/env python3
"""Validate Slurm requests by default; only --submit creates real jobs."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from nonlinear_qwen_common import ARMS, DEFAULT_RUN_DIR, load_and_verify, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--gpu", choices=("l40", "h100"), default="l40")
    parser.add_argument("--submit", action="store_true", help="Submit actual training jobs (default: sbatch --test-only)")
    parser.add_argument("--print-only", action="store_true", help="Print commands without contacting Slurm")
    args = parser.parse_args()
    if args.submit and args.print_only:
        parser.error("--submit and --print-only are mutually exclusive")
    run_dir = args.run_dir.resolve()
    manifest = load_and_verify(run_dir)
    source_root = Path(manifest["source_root"])
    resources = manifest["slurm"]
    env = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    for arm in dict.fromkeys(args.arms):
        command = [
            "sbatch", "--parsable" if args.submit else "--test-only",
            f"--account={resources['account']}", f"--partition={resources['partition']}",
            f"--gres=gpu:{args.gpu}:1", "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={resources['cpus']}", f"--mem={resources['mem']}", f"--time={resources['time']}",
            f"--chdir={source_root}", f"--job-name=qnld-{arm}",
            f"--output={run_dir / 'logs'}/%x-%j.out", "--export=NONE",
            str(source_root / "scripts/argo/run_nonlinear_qwen.sbatch"), str(run_dir), arm,
        ]
        print(shlex.join(command), flush=True)
        if args.print_only:
            continue
        result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
        if args.submit:
            record = {"submitted_utc": datetime.now(timezone.utc).isoformat(), "arm": arm,
                      "job_id": result.stdout.strip(), "manifest_sha256": sha256(run_dir / "manifest.json"),
                      "command": command}
            with (run_dir / "submissions.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
    print("Training jobs submitted" if args.submit else "No jobs submitted")


if __name__ == "__main__":
    main()
