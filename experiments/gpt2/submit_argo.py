#!/usr/bin/env python3
"""Queue gpt_run on one Argo H100 after explicit, verified Gemma job IDs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CKDIR = "k2000_bnh_b32k_lam1_d1536_dpc_sampled_struct5_tokbias"
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING"}


def job_ids(value: str) -> list[str]:
    ids = list(dict.fromkeys(value.replace(":", ",").split(",")))
    if not ids or any(not v.isdigit() or int(v) < 1 for v in ids):
        raise argparse.ArgumentTypeError("Use comma-separated positive Slurm job IDs")
    return ids


def verify_dependencies(ids: list[str]) -> list[dict]:
    """Require known active jobs or successful completed jobs; reject failures."""
    result = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
                             "--format=JobIDRaw,JobName,State,ExitCode"],
                            capture_output=True, text=True, check=True)
    records = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[0] in ids:
            state = fields[2].split()[0].rstrip("+")
            records[fields[0]] = dict(id=fields[0], name=fields[1], state=state, exit_code=fields[3])
    missing = [v for v in ids if v not in records]
    if missing:
        # Accounting can lag behind a fresh sbatch; the queue is authoritative.
        queued = subprocess.run(["squeue", "-h", "-j", ",".join(missing), "-o", "%i|%j|%T"],
                                capture_output=True, text=True, check=True)
        for line in queued.stdout.splitlines():
            identifier, name, state = line.split("|", 2)
            if identifier in missing:
                records[identifier] = dict(id=identifier, name=name, state=state, exit_code=None)
    for identifier in ids:
        row = records.get(identifier)
        if row is None:
            raise ValueError(f"Gemma dependency {identifier} was not found on this cluster")
        if row["state"] not in ACTIVE_STATES and not (
                row["state"] == "COMPLETED" and row["exit_code"] == "0:0"):
            raise ValueError(f"Gemma dependency {identifier} is {row['state']} ({row['exit_code']}); afterok cannot succeed")
    return [records[v] for v in ids]


def write_manifest(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-gemma", required=True, type=job_ids,
                        help="Gemma training/evaluation IDs that must all finish successfully")
    parser.add_argument("--dry-run", action="store_true", help="Print only; query no cluster and create no files")
    parser.add_argument("--account", default=os.environ.get("ARGO_ACCOUNT", "def-hsajjad"))
    parser.add_argument("--partition", default=os.environ.get("ARGO_PARTITION", "c-hsajjad"))
    parser.add_argument("--cpus", type=int, choices=(12, 24), default=24)
    parser.add_argument("--memory", default="450G")
    parser.add_argument("--time", default="7-00:00:00")
    parser.add_argument("--train-concurrency", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--eval-concurrency", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--judges", choices=("auto", "required", "skip"), default="auto")
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/gpt_run/campaign.json")
    parser.add_argument("--resubmit", action="store_true", help="Archive a previous manifest after its job leaves the queue")
    args = parser.parse_args(argv)
    path = args.manifest.absolute()
    command = ["sbatch", "--parsable", f"--chdir={ROOT}", "--job-name=gpt_run",
               f"--account={args.account}", f"--partition={args.partition}", "--gres=gpu:h100:1",
               f"--cpus-per-task={args.cpus}", f"--mem={args.memory}", f"--time={args.time}",
               "--dependency=afterok:" + ":".join(args.after_gemma), "--kill-on-invalid-dep=yes",
               "experiments/gpt2/argo.sbatch", "--train-concurrency", str(args.train_concurrency),
               "--eval-concurrency", str(args.eval_concurrency), "--judges", args.judges]
    if args.dry_run:
        print(shlex.join(command), flush=True)
        print(f"Manifest on submission: {path}")
        return 0
    try:
        dependencies = verify_dependencies(args.after_gemma)
        # Completed jobs can age out of the Slurm controller while remaining
        # in accounting. Their successful exit already satisfies the ordering;
        # afterok is needed only for dependencies still active in the queue.
        waiting = [row["id"] for row in dependencies if row["state"] in ACTIVE_STATES]
        applied_dependency = "afterok:" + ":".join(waiting) if waiting else None
        command = [argument for argument in command
                   if not argument.startswith("--dependency=") and argument != "--kill-on-invalid-dep=yes"]
        if applied_dependency:
            position = command.index("experiments/gpt2/argo.sbatch")
            command[position:position] = [f"--dependency={applied_dependency}", "--kill-on-invalid-dep=yes"]
        print(shlex.join(command), flush=True)
        if path.exists():
            if not args.resubmit:
                raise ValueError(f"Campaign exists at {path}; inspect it before --resubmit")
            old = json.loads(path.read_text())
            ids = [str(v) for v in old.get("jobs", {}).values() if v]
            if ids:
                queued = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T"],
                                        capture_output=True, text=True, check=True).stdout.strip()
                if queued:
                    raise ValueError(f"Previous gpt_run job is still queued/running: {queued}")
            archived = path.with_name(f"campaign.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json")
            archived.write_text(path.read_text())
        (ROOT / "logs/gpt_run").mkdir(parents=True, exist_ok=True)
        manifest = dict(version=2, run_name="gpt_run", site="argo", created_at=datetime.now(timezone.utc).isoformat(),
                        root=str(ROOT), model="gpt2", layers=list(range(12)), hidden_size=768,
                        latent_dim=1536, n_clusters=2000, seed=42, n_epochs=50, n_tokens=10_000_000,
                        context_len=1024, requested_train_concurrency=args.train_concurrency,
                        requested_eval_concurrency=args.eval_concurrency, gpu_count=1,
                        dependency_jobs=dependencies, dependency="afterok:" + ":".join(args.after_gemma),
                        applied_dependency=applied_dependency,
                        gemma_ordering="waiting_afterok" if waiting else "verified_completed",
                        extraction_config="experiments/gpt2/configs/extract_all.yaml",
                        training_configs={str(v): f"experiments/gpt2/configs/layer{v}_2x.yaml" for v in range(12)},
                        checkpoint_dirs={str(v): f"checkpoints/gpt2/layer{v}/{CKDIR}" for v in range(12)},
                        evaluation=dict(steps="gpu" if args.judges == "skip" else "whole",
                                        host="argoh100", concurrency="profiled", api_concurrency=1,
                                        judges=args.judges, output_root="results/gpt_run"),
                        pipeline="experiments/gpt2/run_gpt_run.py", jobs={}, commands={"pipeline": command})
        write_manifest(path, manifest)
        try:
            response = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
            identifier = response.split(";")[0]
            if not identifier.isdigit():
                raise RuntimeError(f"Unexpected sbatch response: {response!r}")
            manifest["jobs"]["pipeline"] = identifier
            write_manifest(path, manifest)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            manifest["submission_error"] = getattr(error, "stderr", None) or str(error)
            write_manifest(path, manifest)
            raise
        ordering = f"waiting for Gemma {','.join(waiting)}" if waiting else "after successfully completed Gemma"
        print(f"Submitted gpt_run {identifier} on one H100, {ordering}; {path}")
        return 0
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        print(f"Submission stopped: {getattr(error, 'stderr', None) or str(error)}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
