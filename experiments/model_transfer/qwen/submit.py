#!/usr/bin/env python3
"""Submit the approved Qwen final-block bypass/plain pair on Argo.

Preparation precedes both one-H100 training/evaluation jobs. The plain job
also waits for the existing Gemma GPU evaluation; CPU API judgments follow
both Qwen jobs. --dry-run uses only the standard library and performs no
commands, downloads, or filesystem reads/writes. Resume finishes only a
partially submitted manifest and never replaces an existing job.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[3]
sys.path.insert(0, str(ROOT))
DEFAULT_RUN = "qwen25_3b_l35_2x_k2000_20261008"
GEMMA_RUN = "midlast_ablation_20261006"
MODEL = "Qwen/Qwen2.5-3B"
ARMS = ("qwen_last_bypass", "qwen_last_plain")
PHASES = ("prep", "bypass", "plain", "judges")
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
                 "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "STAGE_OUT"}


def write_manifest(path: Path, value: dict) -> None:
    """Make each received job ID durable before the next sbatch call."""
    value["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def job_name(run_name: str, phase: str) -> str:
    return f"{run_name}_{phase}"


def submission_command(args, phase: str, jobs: dict[str, str], gemma: dict) -> list[str]:
    gpu = phase != "judges"
    command = ["sbatch", "--parsable", f"--chdir={ROOT}",
               f"--job-name={job_name(args.run_name, phase)}",
               f"--account={args.account}", f"--partition={args.partition}",
               "--export=ALL", "--nodes=1", "--ntasks=1",
               f"--cpus-per-task={12 if gpu else 4}",
               f"--mem={'160G' if gpu else '32G'}",
               f"--time={'14-00:00:00' if gpu else '7-00:00:00'}",
               f"--output=logs/{args.run_name}/slurm-%x-%j.out"]
    if gpu:
        command.append("--gres=gpu:h100:1")
    dependencies = []
    if phase in ("bypass", "plain"):
        dependencies.append(jobs["prep"])
    if phase == "plain" and gemma["requires_dependency"]:
        dependencies.append(args.gemma_job)
    if phase == "judges":
        dependencies.extend(jobs[name] for name in ("bypass", "plain"))
    if dependencies:
        command += ["--dependency=afterok:" + ":".join(dependencies), "--kill-on-invalid-dep=yes"]
    return command + ["experiments/model_transfer/argo/qwen_run.sbatch", "--phase", phase, "--run-name", args.run_name]


def worker_commands(args, staging: Path | None = None) -> list[list[str]]:
    py = [sys.executable, "-u"]
    if args.phase == "judges":
        return [[*py, "experiments/depth/evaluate.py", arm, "--run-name", args.run_name,
                 "--steps", "api", "--judges", "required"] for arm in ARMS] + [
            [*py, "experiments/depth/report.py", "--arm", arm, "--run-name", args.run_name] for arm in ARMS]
    arm = ARMS[1] if args.phase == "plain" else ARMS[0]
    command = [*py, "experiments/depth/run.py", "--arm", arm, "--run-name", args.run_name,
               "--steps", "corpus,extract,table" if args.phase == "prep" else "train,evaluate,report",
               "--eval-steps", "gpu", "--judges", "skip", "--stage-dump-root", str(staging)]
    if os.environ.get("DELTA_DATA"):
        command += ["--data-root", os.environ["DELTA_DATA"]]
    return [command]


def validate_recipe() -> dict:
    """Validate the model, immutable ten-source recipe and paired DPC configs."""
    from experiments.depth.config import CampaignConfig
    from geoae.config import Config
    from experiments.depth.dataset import USER_MIX, SOURCE_FINGERPRINT, validate_extraction_config
    from experiments.depth.evaluate import PROFILES
    hashes, recipes = {}, []
    for arm in ARMS:
        profile = PROFILES[arm]
        expected = {"model_name": MODEL, "layer": 35, "hidden_size": 2048,
                    "latent_dim": 4096, "n_clusters": 2000, "n_epochs": 50}
        if any(profile.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Wrong approved Qwen profile for {arm}")
        extraction_path = ROOT / profile["extract_config"]
        training_path = ROOT / profile["train_config"]
        validate_extraction_config(extraction_path)
        extraction = Config.from_yaml(extraction_path).extraction
        training = CampaignConfig.from_yaml(training_path)
        if (extraction.model_name != MODEL or extraction.layers != [35]
                or extraction.target_layer != 35 or extraction.hidden_size != 2048
                or extraction.n_tokens != 10_526_316 or extraction.dtype != "float16"
                or extraction.activations_dir != profile["activations_dir"]
                or extraction.corpus_dir != profile["corpus_dir"]):
            raise ValueError(f"Wrong approved Qwen extraction recipe for {arm}")
        if (training.extraction.model_name != MODEL or training.data.target_layer != 35
                or training.data.activations_dir != profile["activations_dir"]
                or training.data.val_frac != .05 or training.data.batch_size != 32768
                or int(extraction.n_tokens * (1 - training.data.val_frac)) != 10_000_000
                or training.model.hidden_size != 2048 or training.model.latent_dim != 4096
                or training.model.n_clusters != 2000 or training.train.n_epochs != 50
                or training.train.seed != 42 or training.train.centroid_init != "dpc"
                or training.train.freeze != "none" or training.train.sup_frac != 0):
            raise ValueError(f"Wrong approved Qwen training recipe for {arm}")
        expected_bias = (f"e2e/checkpoints/general/{profile['family']}/layer35/"
                         f"{profile['token_bias_filename']}" if arm == ARMS[0] else "")
        expected_checkpoint = f"checkpoints/{profile['family']}/layer35/{profile['ckdir']}"
        if (training.model.token_bias != expected_bias or training.model.token_bias_mode != "bypass"
                or training.train.checkpoints_dir != expected_checkpoint):
            raise ValueError(f"Wrong Qwen bypass/checkpoint path for {arm}")
        recipe = training.to_dict()
        recipe["model"].pop("token_bias")
        recipe["train"].pop("checkpoints_dir")
        recipes.append(recipe)
        for path in (extraction_path, training_path):
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if recipes[0] != recipes[1]:
        raise ValueError("Qwen bypass/plain recipes differ beyond the token bypass and checkpoint directory")
    return {"model": MODEL, "layer": 35, "hidden_size": 2048, "latent_dim": 4096,
            "n_clusters": 2000, "n_train_rows": 10_000_000, "n_validation_rows": 526_316,
            "n_extraction_rows": 10_526_316, "source_mix": dict(USER_MIX),
            "source_fingerprint": SOURCE_FINGERPRINT, "config_sha256": hashes}


def gemma_job_status(args) -> dict:
    """Fail closed if the existing Gemma model/job association is unverified."""
    from experiments.depth.evaluate import PROFILES
    path = args.gemma_manifest or ROOT / "runs" / GEMMA_RUN / "campaign.json"
    previous = json.loads(path.read_text())
    arms = previous.get("lanes", {}).get("gemma4b", [])
    if (previous.get("jobs", {}).get("gemma4b") != args.gemma_job
            or not arms or any(PROFILES.get(arm, {}).get("model_name") != "google/gemma-3-4b-pt" for arm in arms)):
        raise ValueError(f"Cannot validate Gemma job {args.gemma_job} against {path}")
    expected_name = job_name(previous["run_name"], "gemma4b")
    command = ["sacct", "--noheader", "--allocations", "--jobs", args.gemma_job,
               "--format=JobIDRaw,JobName%128,State,ExitCode", "--parsable2"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    records = [line.strip().split("|") for line in result.stdout.splitlines()]
    matching = [record for record in records if len(record) >= 4 and record[0] == args.gemma_job]
    if len(matching) != 1:
        raise ValueError(f"Expected one accounting record for Gemma job {args.gemma_job}")
    _, name, state, exit_code, *_ = matching[0]
    state = state.split()[0].rstrip("+")
    if name != expected_name:
        raise ValueError(f"Gemma job {args.gemma_job} has unexpected name {name!r}; expected {expected_name!r}")
    completed = state == "COMPLETED" and exit_code == "0:0"
    if not completed and state not in ACTIVE_STATES:
        raise ValueError(f"Gemma job {args.gemma_job} is {state} ({exit_code}); explicit manual resolution required")
    return {"job_id": args.gemma_job, "job_name": name, "state": state, "exit_code": exit_code,
            "model": "google/gemma-3-4b-pt", "model_validated": True, "arms": arms,
            "campaign_manifest": str(path), "requires_dependency": not completed,
            "captured_at": datetime.now(timezone.utc).isoformat()}


def active_campaign_jobs(run_name: str) -> dict[str, str]:
    result = subprocess.run(["squeue", "--noheader", "--user", os.environ.get("USER", getpass.getuser()),
                             "--format=%i|%j|%T"], capture_output=True, text=True, check=True)
    names = {job_name(run_name, phase) for phase in PHASES}
    found = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|", 2)
        if len(fields) == 3 and fields[1] in names:
            if fields[1] in found:
                raise ValueError(f"Multiple active jobs already use {fields[1]}")
            found[fields[1]] = fields[0]
    return found


def submission_environment() -> dict[str, str]:
    # sbatch environment flags can silently add GPU requests to the CPU job.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    environment.setdefault("GEOAE_ENV_FILE", ".env.midlast")
    return environment


def submit(args) -> int:
    path = (args.manifest or ROOT / "runs" / args.run_name / "campaign.json").absolute()
    if args.dry_run:
        jobs = {phase: phase.upper() + "_JOB" for phase in PHASES}
        for phase in PHASES:
            print(f"[{phase}] {shlex.join(submission_command(args, phase, jobs, {'requires_dependency': True}))}")
        print(f"Gemma {args.gemma_job}: state/model validated at submission; successful completed jobs need no dependency.")
        print(f"Manifest on submission: {path}")
        return 0
    recipe = validate_recipe()
    # Validation precedes any submission, including the existing Gemma state.
    gemma = gemma_job_status(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another Qwen submitter holds {path}") from exc
        active = active_campaign_jobs(args.run_name)
        spec = {"run_name": args.run_name, "root": str(ROOT), "arms": list(ARMS),
                "recipe": recipe, "account": args.account, "partition": args.partition,
                "gemma_job": args.gemma_job, "gpu_memory": "160G", "gpu_cpus": 12,
                "gpu_walltime": "14-00:00:00", "api_judges": "required"}
        if path.exists():
            if not args.resume_submission:
                raise ValueError(f"Qwen manifest already exists: {path}; refusing to overwrite submitted jobs")
            manifest = json.loads(path.read_text())
            if manifest.get("spec") != spec:
                raise ValueError("Previous Qwen manifest has a different recipe; refusing to overwrite it")
            jobs = manifest.get("jobs", {})
            if not isinstance(jobs, dict) or not set(jobs).issubset(PHASES):
                raise ValueError("Previous Qwen manifest has invalid jobs")
            if any(not isinstance(identifier, str) or not identifier.isdigit() or int(identifier) < 1 for identifier in jobs.values()):
                raise ValueError("Previous Qwen manifest has invalid job IDs")
            if set(jobs) == set(PHASES):
                raise ValueError("All four Qwen jobs already submitted; resume creates no replacement jobs")
        else:
            if args.resume_submission:
                raise ValueError("--resume-submission requires an existing manifest")
            manifest = {"version": 1, "site": "argo", "created_at": datetime.now(timezone.utc).isoformat(),
                        "spec": spec, "jobs": {}, "commands": {}, "new_training_count": 2,
                        "gpu_jobs": 3, "gpus_per_job": 1, "evaluation": {"gpu_steps": "gpu", "api_steps": "api"}}
        for phase in PHASES:
            identifier = active.get(job_name(args.run_name, phase))
            if identifier and identifier != manifest["jobs"].get(phase):
                raise ValueError(f"Qwen {phase} already has active job {identifier}; refusing duplicate submission")
        manifest["gemma_dependency"] = gemma
        manifest.setdefault("gemma_state_history", []).append(gemma)
        manifest["status"] = "submitting"
        manifest.pop("submission_error", None)
        write_manifest(path, manifest)
        (ROOT / "logs" / args.run_name).mkdir(parents=True, exist_ok=True)
        try:
            for phase in PHASES:
                if phase in manifest["jobs"]:
                    continue
                command = submission_command(args, phase, manifest["jobs"], gemma)
                manifest["commands"][phase] = command
                write_manifest(path, manifest)
                print(f"[{phase}] {shlex.join(command)}", flush=True)
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True,
                                        env=submission_environment())
                response = result.stdout.strip()
                identifier = response.split(";", 1)[0]
                if not identifier.isdigit() or int(identifier) < 1:
                    raise ValueError(f"Unexpected sbatch response for {phase}: {response!r}")
                manifest["jobs"][phase] = identifier
                write_manifest(path, manifest)
                print(f"[submitted] {phase}: {identifier}", flush=True)
        except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
            manifest["status"] = "submission_failed"
            manifest["submission_error"] = getattr(exc, "stderr", None) or str(exc)
            write_manifest(path, manifest)
            raise
        manifest["status"] = "submitted"
        write_manifest(path, manifest)
        print(f"Qwen preparation, two trainings and CPU judgments queued; {path}", flush=True)
    return 0


def stage_root(args) -> Path:
    if args.stage_dump_root:
        return args.stage_dump_root.absolute()
    identifier = os.environ.get("SLURM_JOB_ID")
    if not identifier and not args.dry_run:
        raise ValueError("GPU phases require a Slurm allocation or explicit --stage-dump-root")
    return Path("/dev/shm") / f"geoae-{getpass.getuser()}-{args.run_name}-{identifier or 'DRY_JOB'}-{args.phase}"


def run_phase(args) -> int:
    staging = None if args.phase == "judges" else stage_root(args)
    commands = worker_commands(args, staging)
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return 0
    validate_recipe()
    state_path = ROOT / "runs" / args.run_name / f"phase_{args.phase}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"run_name": args.run_name, "phase": args.phase, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    ownership = None
    if staging is not None:
        if staging.is_symlink():
            raise ValueError(f"Refusing a symlink staging directory: {staging}")
        staging.mkdir(parents=True, exist_ok=True)
        ownership = staging / ".qwen_owner.json"
        if ownership.exists():
            if json.loads(ownership.read_text()) != owner:
                raise ValueError(f"Staging directory belongs to another Qwen job: {staging}")
        elif any(staging.iterdir()):
            raise ValueError(f"Refusing to use or clean an unowned staging directory: {staging}")
        ownership.write_text(json.dumps(owner) + "\n")
    state = {**owner, "status": "running", "commands": commands, "stage_dump_root": str(staging) if staging else None,
             "completed_commands": []}
    try:
        write_manifest(state_path, state)
        if args.phase == "judges":
            for phase in ("bypass", "plain"):
                path = state_path.parent / f"phase_{phase}.json"
                if not path.is_file() or json.loads(path.read_text()).get("status") != "complete":
                    raise ValueError(f"Qwen GPU phase {phase} has not completed")
        for command in commands:
            print(f"[{args.phase}] {shlex.join(command)}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            state["completed_commands"].append(command)
            write_manifest(state_path, state)
        state["status"] = "complete"
        write_manifest(state_path, state)
        return 0
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
        state.update(status="failed", error=getattr(exc, "stderr", None) or str(exc))
        write_manifest(state_path, state)
        raise
    finally:
        if (ownership is not None and not staging.is_symlink() and ownership.is_file()
                and json.loads(ownership.read_text()) == owner):
            shutil.rmtree(staging)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=DEFAULT_RUN)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    p.add_argument("--resume-submission", action="store_true")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--gemma-job", default="292220")
    p.add_argument("--gemma-manifest", type=Path, help="Manifest validating the existing Gemma lane/model")
    p.add_argument("--account", default="def-hsajjad")
    p.add_argument("--partition", default="c-hsajjad")
    p.add_argument("--phase", choices=PHASES, help=argparse.SUPPRESS)
    p.add_argument("--stage-dump-root", type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        p.error("--run-name must be one directory name")
    if not args.gemma_job.isdigit() or int(args.gemma_job) < 1:
        p.error("--gemma-job must be a positive Slurm job ID")
    if args.resume_submission and not args.submit:
        p.error("--resume-submission requires --submit")
    if args.phase and args.submit:
        p.error("Worker phases cannot submit jobs")
    if not args.phase and not (args.submit or args.dry_run):
        p.error("Choose --dry-run or --submit")
    try:
        return run_phase(args) if args.phase else submit(args)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"[abort] {getattr(exc, 'stderr', None) or str(exc)}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
