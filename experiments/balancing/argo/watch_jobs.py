#!/usr/bin/env python3
"""Live, dependency-free Slurm/GeoAE dashboard. Run on an Argo login node."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import select
import shlex
import shutil
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JOBS = {"291620": "nosk-reinit", "291621": "nosk-lloyd", "291622": "nosk-frozen"}
FAILURES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED"}
ERROR_PATTERN = re.compile(r"Traceback \(most recent call last\)|CUDA.*out of memory|\b\w*Error:|slurmstepd:.*(?:error|killed|oom)|\b(?:loss|recon|val_mse)[ =]+(?:nan|inf)\b", re.I)


def run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"{argv[0]} exited {result.returncode}")
    return result.stdout


def job_metadata(job_id):
    """Get this submission's own log and config; avoid appended training logs."""
    metadata = {}
    try:
        text = run(["scontrol", "show", "job", "-o", job_id])
        fields = dict(re.findall(r"(?:^|\s)(\w+)=(.*?)(?=\s+\w+=|$)", text.strip()))
        metadata["name"] = fields.get("JobName", DEFAULT_JOBS.get(job_id, job_id))
        if fields.get("StdOut"):
            metadata["log"] = Path(fields["StdOut"])
        workdir = Path(fields.get("WorkDir", str(ROOT)))
        args = shlex.split(fields.get("SubmitLine", ""))
        config = next((arg for arg in args if arg.endswith((".yaml", ".yml"))), None)
        if config:
            path = Path(config) if Path(config).is_absolute() else workdir / config
            # Only the integer scalars needed here; no YAML or training imports.
            section = None
            for line in path.read_text().splitlines():
                header = re.match(r"^(data|train):\s*(?:#.*)?$", line)
                if header:
                    section = header.group(1)
                elif re.match(r"^\S", line) and not line.startswith("#"):
                    section = None
                match = re.match(r"^  (batch_size|n_epochs|geometry_start_epoch|clustering_start_epoch):\s*(\d+)", line)
                if match and section in {"data", "train"}:
                    metadata[match.group(1)] = int(match.group(2))
        for index, arg in enumerate(args):
            if arg == "--batch_size" and index + 1 < len(args):
                metadata["batch_size"] = int(args[index + 1])
            elif arg.startswith("--batch_size="):
                metadata["batch_size"] = int(arg.split("=", 1)[1])
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
        pass  # Logs still provide progress when completed jobs leave scontrol.
    return metadata


def slurm_snapshot(job_ids):
    jobs = {jid: {"name": DEFAULT_JOBS.get(jid, jid), "state": "UNKNOWN"} for jid in job_ids}
    warnings = []
    try:
        output = run(["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%j|%T|%M|%R"])
        for line in output.splitlines():
            fields = [field.strip() for field in line.split("|", 4)]
            if len(fields) == 5 and fields[0] in jobs:
                jid, name, state, elapsed, location = fields
                jobs[jid].update(name=name, state=state, elapsed=elapsed, location=location)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        warnings.append(f"squeue unavailable: {exc}")
    missing = [jid for jid, job in jobs.items() if job["state"] == "UNKNOWN"]
    if missing:
        try:
            output = run(["sacct", "-n", "-X", "-P", "-j", ",".join(missing), "-o", "JobIDRaw,JobName%100,State%30,Elapsed,NodeList%100,ExitCode"])
            for line in output.splitlines():
                fields = [field.strip() for field in line.split("|")]
                if len(fields) >= 6 and fields[0] in missing:
                    jid, name, state, elapsed, location, exit_code = fields[:6]
                    jobs[jid].update(name=name, state=state.split()[0].rstrip("+"), elapsed=elapsed, location=location, exit_code=exit_code)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"sacct unavailable: {exc}")
    return jobs, warnings


def parse_log(text, metadata=None, steps_per_epoch=None):
    metadata = metadata or {}
    headers = list(re.finditer(r"\[train\] Epoch (\d+)/(\d+)", text))
    epoch, epochs = map(int, headers[-1].groups()) if headers else (0, metadata.get("n_epochs", 0))
    steps = [int(value) for value in re.findall(r"^\s*step\s+(\d+)\s*\|", text, re.M)]
    checkpoints = list(re.finditer(r"Saved checkpoint: step_(\d+)\.pt", text))
    completed = [int(value) for value in re.findall(r"\[train\] Epoch (\d+) done", text)]
    if not steps_per_epoch and checkpoints:
        checkpoint = checkpoints[-1]
        preceding = [header for header in headers if header.start() < checkpoint.start()]
        if preceding:
            saved_epoch = int(preceding[-1].group(1))
            saved_step = int(checkpoint.group(1))
            if saved_step and saved_step % saved_epoch == 0:
                steps_per_epoch = saved_step // saved_epoch
    if not steps_per_epoch and metadata.get("batch_size"):
        tokens = re.findall(r"Train tokens: ([\d,]+)", text)
        if tokens:
            steps_per_epoch = int(tokens[-1].replace(",", "")) // metadata["batch_size"]
    candidates = steps + [int(match.group(1)) for match in checkpoints]
    # A resumed submission may have no diagnostic line yet.
    candidates += [int(value) for value in re.findall(r"\[resume\][^\n]*\bstep[ =]+(\d+)", text)]
    if steps_per_epoch:
        candidates += [(epoch - 1) * steps_per_epoch, max(completed, default=0) * steps_per_epoch]
    step = max(candidates, default=0)
    total = epochs * steps_per_epoch if steps_per_epoch and epochs else None
    done = "[train] Training complete." in text
    if done and total:
        step = total
    diagnostic = re.findall(r"^\s*step\s+\d+\s*\|(.+)$", text, re.M)
    metrics = {}
    if diagnostic:
        for name in ("loss", "val_mse", "fve", "eff_K", "dying"):
            match = re.search(r"(?:^|\|)\s*" + name + r"\s+(\S+)", diagnostic[-1])
            if match:
                metrics[name] = match.group(1)
    geometry = metadata.get("geometry_start_epoch", 6)
    clustering = metadata.get("clustering_start_epoch", 11)
    if done:
        phase = "training finished"
    elif completed and completed[-1] == clustering - 1 and epoch < clustering:
        phase = "centroid initialization / transition"
    elif epoch >= clustering:
        phase = "clustering"
    elif epoch >= geometry:
        phase = "reconstruction + geometry"
    elif epoch:
        phase = "reconstruction warmup"
    elif "[stage]" in text:
        phase = "staging / startup"
    else:
        phase = "waiting for training output"
    errors = [line.strip() for line in text.splitlines() if ERROR_PATTERN.search(line)]
    return dict(epoch=epoch, epochs=epochs, step=step, total=total, done=done, metrics=metrics, phase=phase, error=errors[-1] if errors else None)


def collect(job_ids, metadata, log_dir, steps_per_epoch=None):
    jobs, warnings = slurm_snapshot(job_ids)
    for jid, job in jobs.items():
        details = metadata[jid]
        if job["name"] == jid:
            job["name"] = details.get("name", jid)
        path = details.get("log", log_dir / f"slurm-{job['name']}-{jid}.out")
        job["log"] = str(path)
        text = ""
        try:
            text = path.read_text(errors="replace")
            job["log_age"] = max(0, int(time.time() - path.stat().st_mtime))
        except FileNotFoundError:
            pass
        except OSError as exc:
            warnings.append(f"{jid} cannot read log: {exc}")
        job.update(parse_log(text, details, steps_per_epoch))
    return jobs, warnings


def progress_bar(step, total, width=34):
    if not total:
        return "[" + " " * width + "]   --.-%"
    fraction = min(1.0, max(0.0, step / total))
    filled = int(fraction * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {fraction * 100:6.2f}%"


def render(jobs, warnings=(), width=100, color=False, refresh=30):
    def paint(text, state):
        code = "31" if state in FAILURES else "33" if state == "PENDING" else "36" if state == "COMPLETED" else "32" if state == "RUNNING" else "37"
        return f"\033[{code}m{text}\033[0m" if color else text

    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    counts = {state: sum(job["state"] == state for job in jobs.values()) for state in ("RUNNING", "PENDING", "COMPLETED")}
    lines = [f"GeoAE / Argo   {now}", f"{counts['RUNNING']} running   {counts['PENDING']} pending   {counts['COMPLETED']} completed   {sum(job['state'] in FAILURES for job in jobs.values())} failed", ""]
    bar_width = max(8, min(44, width - 24))
    for jid, job in jobs.items():
        state = job["state"]
        title = f"{jid}  {job['name']}  {state}  |  elapsed {job.get('elapsed', '--')}"
        if job.get("exit_code"):
            title += f"  |  exit {job['exit_code']}"
        lines.append(paint(title[:width], state))
        pending = state == "PENDING"
        if pending:
            lines.append("  " + paint("[" + "." * bar_width + "]  queued", state))
            lines.append("  Waiting: " + job.get("location", "scheduler"))
        else:
            lines.append("  " + paint(progress_bar(job['step'], job['total'], bar_width), state))
            step = f"{job['step']:,}" + (f" / {job['total']:,}" if job["total"] else " logged steps")
            epoch = f"{job['epoch']} / {job['epochs']}" if job["epochs"] else "--"
            lines.append(f"  Epoch {epoch}  |  Step {step}  |  {job.get('location', '--')}")
            lines.append("  Phase: " + job["phase"])
            if job["metrics"]:
                labels = [("loss", "loss"), ("val_mse", "val MSE"), ("fve", "FVE")]
                if job["epoch"] >= metadata_clustering_epoch(job):
                    labels += [("eff_K", "effective K"), ("dying", "dying")]
                lines.append("  " + "  |  ".join(label + " " + job["metrics"][key] for key, label in labels if key in job["metrics"]))
            if "log_age" in job and state not in FAILURES | {"COMPLETED"}:
                lines.append(f"  Log updated {job['log_age']}s ago")
        if job.get("error"):
            lines.append(paint("  Log error: " + job["error"], "FAILED"))
        lines.append("")
    lines.extend("Warning: " + warning for warning in warnings)
    lines += [f"Refresh every {refresh:g}s   |   q / Ctrl+C: quit", "Progress follows logged steps; queue wait time is unknown."]
    return "\n".join(line[:width] if "\033" not in line else line for line in lines)


def metadata_clustering_epoch(job):
    return job.get("clustering_start_epoch", 11)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="*", help="Slurm job IDs; default: 291620 291621 291622")
    parser.add_argument("--refresh", type=float, default=30, help="Refresh seconds (minimum 5; default 30)")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs", help="Fallback Slurm log directory")
    parser.add_argument("--steps-per-epoch", type=int, help="Override auto-detected training step count")
    args = parser.parse_args(argv)
    job_ids = list(dict.fromkeys(jid for arg in args.jobs for jid in arg.split(","))) or list(DEFAULT_JOBS)
    if any(not re.fullmatch(r"\d+(?:_\d+)?", jid) for jid in job_ids):
        parser.error("Job IDs must be numeric (array task IDs such as 123_4 are accepted)")
    if args.refresh < 5:
        parser.error("--refresh must be at least 5 seconds")
    if args.steps_per_epoch is not None and args.steps_per_epoch <= 0:
        parser.error("--steps-per-epoch must be positive")
    if not shutil.which("squeue"):
        print("Run this dashboard on Argo, where squeue is available.", file=sys.stderr)
        return 1
    metadata = {jid: job_metadata(jid) for jid in job_ids}
    interactive = sys.stdout.isatty() and sys.stdin.isatty() and not args.once
    color = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    settings = None
    try:
        if interactive:
            settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            print("\033[?1049h\033[?25l", end="", flush=True)
        while True:
            jobs, warnings = collect(job_ids, metadata, args.log_dir, args.steps_per_epoch)
            for jid, job in jobs.items():
                job["clustering_start_epoch"] = metadata[jid].get("clustering_start_epoch", 11)
            output = render(jobs, warnings, shutil.get_terminal_size((100, 35)).columns - 1, color, args.refresh)
            print(("\033[H\033[2J" if interactive else "") + output, flush=True)
            if args.once or not interactive:
                return 0
            deadline = time.monotonic() + args.refresh
            while time.monotonic() < deadline:
                readable, _, _ = select.select([sys.stdin], [], [], min(1, max(0, deadline - time.monotonic())))
                if readable and sys.stdin.read(1).lower() == "q":
                    return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if settings is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, settings)
            print("\033[?25h\033[?1049l", end="", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
