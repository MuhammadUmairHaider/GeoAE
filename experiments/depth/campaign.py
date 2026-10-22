#!/usr/bin/env python3
"""Submit two Argo H100 lanes, each running three matched AE experiments serially.

Use --dry-run to inspect commands, --submit to queue the two lanes, and
--resume-submission only to finish a submission interrupted between lanes.
The Slurm wrappers call --lane and --run-judges internally. API evaluations
run on a CPU allocation after both GPU lanes. Existing final-layer bypass
AEs are evaluated only after their ten-source provenance is verified, and
are never trained by this six-training campaign.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_RUN = "midlast_ablation_20261006"
LANES = {
    "llama": ("llama_mid_bypass", "llama_mid_plain", "llama_last_plain"),
    "gemma4b": ("gemma4b_mid_bypass", "gemma4b_mid_plain", "gemma4b_last_plain"),
}
TRAIN_ROWS = 10_000_000
TOTAL_ROWS = 10_526_316


def write_manifest(path: Path, value: dict) -> None:
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def job_name(run_name: str, lane: str) -> str:
    return f"{run_name}_{lane}"


def submit_commands(args) -> dict[str, list[str]]:
    commands = {}
    for lane in LANES:
        commands[lane] = [
            "sbatch", "--parsable", f"--chdir={ROOT}",
            f"--job-name={job_name(args.run_name, lane)}",
            f"--account={args.account}", f"--partition={args.partition}",
            "--nodes=1", "--ntasks=1", "--gres=gpu:h100:1",
            f"--cpus-per-task={args.cpus}", f"--mem={args.memory}",
            f"--time={args.time}",
            f"--output=logs/{args.run_name}/slurm-%x-%j.out",
            "experiments/depth/argo/midlayer_lane.sbatch", "--lane", lane,
            "--run-name", args.run_name, "--judges", "skip",
            "--eval-steps", args.eval_steps,
        ]
        if lane == "llama" and args.corpus_job:
            position = commands[lane].index("experiments/depth/argo/midlayer_lane.sbatch")
            commands[lane][position:position] = [f"--dependency=afterok:{args.corpus_job}", "--kill-on-invalid-dep=yes"]
    return commands


def judge_command(args, jobs: dict[str, str]) -> list[str]:
    dependency = ":".join(jobs[lane] for lane in LANES)
    return [
        "sbatch", "--parsable", f"--chdir={ROOT}",
        f"--job-name={job_name(args.run_name, 'judges')}",
        f"--account={args.account}", f"--partition={args.partition}",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=32G",
        "--time=7-00:00:00", f"--dependency=afterok:{dependency}",
        "--kill-on-invalid-dep=yes",
        f"--output=logs/{args.run_name}/slurm-%x-%j.out",
        "experiments/depth/argo/midlayer_judges.sbatch", "--run-judges",
        "--run-name", args.run_name, "--judges", args.judges,
    ]


def validate_campaign_recipe() -> dict:
    """Reject every nine-source recipe before creating any Slurm jobs."""
    from experiments.depth.dataset import SOURCE_FINGERPRINT, USER_MIX, validate_extraction_config
    from experiments.depth.evaluate import PROFILES
    from geoae.config import Config
    for arms in LANES.values():
        for arm in arms:
            profile = PROFILES[arm]
            validate_extraction_config(profile["extract_config"])
            extraction = Config.from_yaml(ROOT / profile["extract_config"]).extraction
            training = Config.from_yaml(ROOT / profile["train_config"])
            if extraction.n_tokens != TOTAL_ROWS or training.data.val_frac != 0.05:
                raise ValueError(f"{arm} must extract {TOTAL_ROWS:,} rows with val_frac=0.05")
            if int(extraction.n_tokens * (1 - training.data.val_frac)) != TRAIN_ROWS:
                raise ValueError(f"{arm} does not have exactly {TRAIN_ROWS:,} training rows")
    return {"source_mix": dict(USER_MIX), "source_fingerprint": SOURCE_FINGERPRINT,
            "n_train_rows": TRAIN_ROWS, "n_validation_rows": TOTAL_ROWS - TRAIN_ROWS,
            "n_extraction_rows": TOTAL_ROWS}


def active_campaign_jobs(run_name: str) -> dict[str, str]:
    """Find duplicates even if an earlier submitter lost its manifest."""
    result = subprocess.run(
        ["squeue", "--noheader", "--user", os.environ.get("USER", getpass.getuser()),
         "--format=%i|%j|%T"], capture_output=True, text=True, check=True,
    )
    names = {job_name(run_name, lane) for lane in (*LANES, "judges")}
    found = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|", 2)
        if len(fields) == 3 and fields[1] in names:
            if fields[1] in found:
                raise RuntimeError(f"Multiple active jobs already use {fields[1]}")
            found[fields[1]] = fields[0]
    return found


def submit(args) -> int:
    commands = submit_commands(args)
    manifest_path = (args.manifest or ROOT / "runs" / args.run_name / "campaign.json").absolute()
    for lane, command in commands.items():
        print(f"[{lane}] {shlex.join(command)}", flush=True)
    if args.dry_run:
        print(f"[judges] {shlex.join(judge_command(args, {lane: lane.upper() + '_JOB' for lane in LANES}))}")
        print(f"Manifest on submission: {manifest_path}")
        return 0
    provenance = validate_campaign_recipe()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.with_suffix(".lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another submitter holds {manifest_path}") from exc
        active = active_campaign_jobs(args.run_name)
        if manifest_path.exists():
            if not args.resume_submission:
                raise ValueError(f"Campaign manifest already exists: {manifest_path}; inspect its jobs before retrying")
            manifest = json.loads(manifest_path.read_text())
            saved_gpu_commands = {lane: manifest.get("commands", {}).get(lane) for lane in LANES}
            if (saved_gpu_commands != commands
                    or manifest.get("lanes") != {k: list(v) for k, v in LANES.items()}
                    or manifest.get("api_judges") != args.judges
                    or manifest.get("dataset_provenance") != provenance):
                raise ValueError("Previous manifest has a different campaign recipe; refusing to overwrite it")
            if set(manifest.get("jobs", {})) == {*LANES, "judges"}:
                raise ValueError("All three jobs were already submitted; --resume-submission creates no replacement jobs")
        else:
            if args.resume_submission:
                raise ValueError("--resume-submission requires an existing manifest")
            manifest = {
                "version": 2, "run_name": args.run_name, "site": "argo", "root": str(ROOT),
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "submitting",
                "lanes": {k: list(v) for k, v in LANES.items()}, "new_training_count": 6,
                "gpu_count": 2, "gpus_per_lane": 1, "concurrent_trainings_per_lane": 1,
                "cpus_per_lane": args.cpus, "memory_per_lane": args.memory,
                "walltime": args.time, "evaluation": {"gpu_steps": args.eval_steps, "api_steps": "api",
                                                        "gpu_judges": "skip", "api_judges": args.judges},
                "api_judges": args.judges, "dataset_provenance": provenance,
                "commands": commands.copy(), "jobs": {},
            }
        for lane in (*LANES, "judges"):
            queued = active.get(job_name(args.run_name, lane))
            if queued and queued != manifest["jobs"].get(lane):
                raise ValueError(f"Campaign lane {lane} already has active job {queued}; refusing duplicate submission")
        manifest["status"] = "submitting"
        manifest.pop("submission_error", None)
        write_manifest(manifest_path, manifest)
        (ROOT / "logs" / args.run_name).mkdir(parents=True, exist_ok=True)
        try:
            for lane, command in commands.items():
                if lane in manifest["jobs"]:
                    print(f"[reuse] {lane}: submitted job {manifest['jobs'][lane]}", flush=True)
                    continue
                response = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
                identifier = response.split(";", 1)[0]
                if not identifier.isdigit() or int(identifier) < 1:
                    raise RuntimeError(f"Unexpected sbatch response for {lane}: {response!r}")
                manifest["jobs"][lane] = identifier
                write_manifest(manifest_path, manifest)
                print(f"[submitted] {lane}: {identifier}", flush=True)
            if "judges" not in manifest["jobs"]:
                command = judge_command(args, manifest["jobs"])
                manifest["commands"]["judges"] = command
                write_manifest(manifest_path, manifest)
                print(f"[judges] {shlex.join(command)}", flush=True)
                response = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
                identifier = response.split(";", 1)[0]
                if not identifier.isdigit() or int(identifier) < 1:
                    raise RuntimeError(f"Unexpected sbatch response for judges: {response!r}")
                manifest["jobs"]["judges"] = identifier
                write_manifest(manifest_path, manifest)
                print(f"[submitted] judges: {identifier}", flush=True)
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            manifest["status"] = "submission_failed"
            manifest["submission_error"] = getattr(exc, "stderr", None) or str(exc)
            write_manifest(manifest_path, manifest)
            raise
        manifest["status"] = "submitted"
        write_manifest(manifest_path, manifest)
        print(f"Six trainings queued on two H100s; CPU judgments follow both lanes; {manifest_path}", flush=True)
    return 0


def stage_root(args) -> Path:
    if args.stage_dump_root:
        return args.stage_dump_root.absolute()
    identifier = os.environ.get("SLURM_JOB_ID")
    if not identifier and not args.dry_run:
        raise ValueError("A lane requires a Slurm allocation or explicit --stage-dump-root")
    return Path("/dev/shm") / f"geoae-{getpass.getuser()}-{identifier or 'DRY_JOB'}-{args.lane}"


def reference_status(lane: str) -> dict:
    """Verify a final bypass reference without searching legacy recipe paths."""
    from experiments.depth.dataset import SOURCE_FINGERPRINT, check_dump
    from experiments.depth.evaluate import PROFILES
    from experiments.gpt2.evaluate import checkpoint_for, guard_checkpoint
    arm = f"{lane}_last_bypass"
    profile = PROFILES[arm]
    try:
        checkpoint = checkpoint_for(profile["layer"], profile=profile)
        provenance = check_dump(profile["extract_config"])["dataset_provenance"]
        guard_checkpoint(checkpoint, profile["layer"], profile)
        import torch
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["config"]["data"]["activations_dir"] != profile["activations_dir"]:
            raise ValueError("Reference checkpoint points to another activation recipe")
        recorded = saved.get("dataset_provenance")
        if recorded is not None and recorded.get("source_fingerprint") != SOURCE_FINGERPRINT:
            raise ValueError("Reference checkpoint records a different source fingerprint")
        return {"arm": arm, "status": "ready", "checkpoint": str(checkpoint),
                "dataset_provenance": provenance}
    except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as exc:
        return {"arm": arm, "status": "pending", "reason": str(exc)}


def evaluation_status(arm: str, run_name: str) -> str:
    from experiments.depth.evaluate import PROFILES
    from experiments.gpt2.evaluate import profile_paths
    profile = PROFILES[arm]
    out = profile_paths(profile["layer"], argparse.Namespace(profile=profile, run_name=run_name))["out"]
    path = ROOT / out / "pipeline_state.json"
    return json.loads(path.read_text()).get("status", "unknown") if path.exists() else "unknown"


def run_lane(args) -> int:
    """Keep both variants of one layer in RAM, then free it before the next."""
    staging = stage_root(args)
    commands = []
    for arm in LANES[args.lane]:
        command = [sys.executable, "-u", "experiments/depth/run.py", "--arm", arm,
                   "--run-name", args.run_name, "--steps", "all", "--eval-steps", args.eval_steps,
                   "--judges", "skip", "--stage-dump-root", str(staging)]
        if os.environ.get("DELTA_DATA"):
            command += ["--data-root", os.environ["DELTA_DATA"]]
        commands.append((arm, command))
    if args.dry_run:
        for arm, command in commands:
            print(f"[{arm}] {shlex.join(command)}")
        print(f"[reference] {args.lane}_last_bypass: evaluate only if its checkpoint and ten-source dump are verified")
        return 0
    os.chdir(ROOT)
    validate_campaign_recipe()
    staging.mkdir(parents=True, exist_ok=True)
    ownership = staging / ".midlayer_lane_owner.json"
    owner = {"lane": args.lane, "run_name": args.run_name, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    if ownership.exists() and json.loads(ownership.read_text()) != owner:
        raise ValueError(f"Staging directory belongs to another lane: {staging}")
    if any(staging.iterdir()) and not ownership.exists():
        raise ValueError(f"Refusing to use or clean an unowned staging directory: {staging}")
    ownership.write_text(json.dumps(owner) + "\n")
    state_path = ROOT / "runs" / args.run_name / f"lane_{args.lane}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {**owner, "status": "running", "stage_dump_root": str(staging), "arms": {}}
    write_manifest(state_path, state)
    try:
        for index, (arm, command) in enumerate(commands):
            if index == 2:
                # Only the contents of this lane's verified job directory are removed.
                for child in staging.iterdir():
                    if child == ownership:
                        continue
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            print(f"[lane] {arm}: {shlex.join(command)}", flush=True)
            state["arms"][arm] = {"status": "running", "command": command}
            write_manifest(state_path, state)
            subprocess.run(command, cwd=ROOT, check=True)
            state["arms"][arm]["status"] = evaluation_status(arm, args.run_name)
            write_manifest(state_path, state)
        reference = reference_status(args.lane)
        state["reference"] = reference
        write_manifest(state_path, state)
        if reference["status"] == "ready":
            arm = reference["arm"]
            command = [sys.executable, "-u", "experiments/depth/run.py", "--arm", arm,
                       "--run-name", args.run_name, "--steps", "evaluate,report", "--eval-steps", args.eval_steps,
                       "--judges", "skip", "--stage-dump-root", str(staging)]
            print(f"[reference] {arm}: {shlex.join(command)}", flush=True)
            try:
                subprocess.run(command, cwd=ROOT, check=True)
                reference["status"] = evaluation_status(arm, args.run_name)
            except (OSError, subprocess.CalledProcessError) as exc:
                reference.update(status="failed", reason=str(exc))
            write_manifest(state_path, state)
        else:
            print(f"[pending reference] {reference['arm']}: {reference['reason']}", flush=True)
        state["status"] = "complete"
        write_manifest(state_path, state)
        return 0
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        write_manifest(state_path, state)
        raise
    finally:
        if ownership.exists() and json.loads(ownership.read_text()) == owner:
            shutil.rmtree(staging)


def run_judges(args) -> int:
    """Run all API evaluations serially on CPUs after the GPU stage succeeds."""
    arms = [arm for values in LANES.values() for arm in values]
    py = [sys.executable, "-u"]
    if args.dry_run:
        for arm in arms:
            print(shlex.join([*py, "experiments/depth/run.py", "--arm", arm,
                             "--run-name", args.run_name, "--steps", "evaluate,report",
                             "--eval-steps", "api", "--judges", args.judges]))
        print(shlex.join([*py, "experiments/depth/report.py", "--arm", "all", "--run-name", args.run_name]))
        return 0
    os.chdir(ROOT)
    validate_campaign_recipe()
    path = ROOT / "runs" / args.run_name / "judges_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "judges": args.judges, "arms": {}, "references": {}}
    write_manifest(path, state)
    for lane in LANES:
        lane_path = path.parent / f"lane_{lane}.json"
        if not lane_path.exists() or json.loads(lane_path.read_text()).get("status") != "complete":
            state.update(status="failed", error=f"GPU lane {lane} has not completed")
            write_manifest(path, state)
            return 1
        reference = json.loads(lane_path.read_text()).get("reference", {"status": "pending"})
        state["references"][lane] = reference
        if reference.get("status") in ("complete", "partial"):
            arms.append(reference["arm"])
    failed = False
    partial = False
    for arm in arms:
        command = [*py, "experiments/depth/run.py", "--arm", arm,
                   "--run-name", args.run_name, "--steps", "evaluate,report",
                   "--eval-steps", "api", "--judges", args.judges]
        state["arms"][arm] = {"status": "running", "command": command}
        write_manifest(path, state)
        print(f"[judges] {arm}: {shlex.join(command)}", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            status = evaluation_status(arm, args.run_name)
            state["arms"][arm]["status"] = status
            partial |= status != "complete"
        except (OSError, subprocess.CalledProcessError) as exc:
            failed = True
            state["arms"][arm].update(status="failed", error=str(exc))
        write_manifest(path, state)
    report = [*py, "experiments/depth/report.py", "--arm", "all", "--run-name", args.run_name]
    try:
        subprocess.run(report, cwd=ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        failed = True
        state["report_error"] = str(exc)
    state["status"] = "failed" if failed else "partial" if partial else "complete"
    write_manifest(path, state)
    return 1 if failed else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=DEFAULT_RUN)
    p.add_argument("--lane", choices=tuple(LANES), help=argparse.SUPPRESS)
    p.add_argument("--run-judges", action="store_true", help=argparse.SUPPRESS)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Print commands; do not query Slurm or write files")
    action.add_argument("--submit", action="store_true", help="Submit two independent one-H100 jobs")
    p.add_argument("--resume-submission", action="store_true", help="With --submit, submit only lanes missing from a prior manifest")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--account", default=os.environ.get("ARGO_ACCOUNT", "def-hsajjad"))
    p.add_argument("--partition", default=os.environ.get("ARGO_PARTITION", "c-hsajjad"))
    p.add_argument("--cpus", type=int, default=12)
    p.add_argument("--memory", default="240G")
    p.add_argument("--time", default="14-00:00:00")
    p.add_argument("--corpus-job", help="Successful Llama corpus preparation required before its GPU lane")
    p.add_argument("--eval-steps", default="gpu")
    p.add_argument("--judges", choices=("auto", "required", "skip"), default="required")
    p.add_argument("--stage-dump-root", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        p.error("--run-name must be one directory name")
    if args.cpus < 1:
        p.error("--cpus must be positive")
    if args.corpus_job and (not args.corpus_job.isdigit() or int(args.corpus_job) < 1):
        p.error("--corpus-job must be a positive Slurm job ID")
    if not args.lane and not args.run_judges and not (args.submit or args.dry_run):
        p.error("Choose --dry-run or --submit")
    if args.resume_submission and not args.submit:
        p.error("--resume-submission requires --submit")
    if (args.lane or args.run_judges) and args.submit:
        p.error("Internal worker modes cannot submit more jobs")
    if args.lane and args.run_judges:
        p.error("Choose one internal worker mode")
    try:
        return run_lane(args) if args.lane else run_judges(args) if args.run_judges else submit(args)
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
        print(f"[abort] {getattr(exc, 'stderr', None) or str(exc)}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
