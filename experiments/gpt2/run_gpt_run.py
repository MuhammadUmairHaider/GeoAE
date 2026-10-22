#!/usr/bin/env python3
"""Run GPT-2 corpus, all-layer extraction, twelve 2x AEs and full evaluation.

All layers share one H100 and one tmpfs copy of the extraction. An exact-recipe
GPU profile sets the safe training/evaluation concurrency, up to twelve. CUDA
OOM failures retry at a lower concurrency using the normal checkpoint resume
and evaluation artifact guards. API evaluations run one layer at a time.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
ACT = ROOT / "activations_gpt2_sampled_struct5_10M"
CFG = ROOT / "experiments/gpt2/configs/extract_all.yaml"
CKDIR = "k2000_bnh_b32k_lam1_d1536_dpc_sampled_struct5_tokbias"
LAYERS = list(range(12))
GIB = 1024 ** 3
OOM_TEXT = ("cuda out of memory", "cuda error: out of memory", "torch.outofmemoryerror",
            "cublas_status_alloc_failed")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value["updated_at"] = utc()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def py(*arguments) -> list[str]:
    return [sys.executable, "-u", *map(str, arguments)]


def lifecycle(operation: str, layer: int | None = None) -> list[str]:
    command = py(ROOT / "experiments/gpt2/lifecycle.py", operation)
    if layer is not None:
        command += ["--layer", str(layer)]
    return command


def verified(command: list[str]) -> bool:
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def run(command: list[str], log: Path | None = None) -> None:
    print("[run] " + shlex.join(command), flush=True)
    if log is None:
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as output:
            output.write(f"\n[{utc()}] {shlex.join(command)}\n")
            output.flush()
            subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)


def extraction_files(source: Path) -> list[Path]:
    required = [source / "meta.json", source / "docs.npy"]
    required += [source / f"layer_{layer}.npy" for layer in LAYERS]
    required += [source / f"rows_{name}.npy" for name in ("doc", "pos", "tok", "src", "norm")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing extraction files: " + ", ".join(missing))
    return required + sorted(source.glob("norm_params_layer*.npz"))


def stage_dump(source: Path, destination: Path) -> dict:
    files = extraction_files(source)
    size = sum(path.stat().st_size for path in files)
    free = shutil.disk_usage(destination).free
    if free < size + 8 * GIB:
        raise RuntimeError(f"tmpfs has {free / GIB:.1f} GiB free; extraction needs {size / GIB:.1f} GiB plus 8 GiB")
    print(f"[stage] {len(files)} files, {size / GIB:.1f} GiB: {source} → {destination}", flush=True)
    for path in files:
        print(f"[stage] {path.name}", flush=True)
        shutil.copyfile(path, destination / path.name)
    return dict(source=str(source), path=str(destination), bytes=size, files=len(files), status="complete")


def copy_atomic(source: Path, target: Path) -> None:
    temporary = target.with_name(target.name + ".gpt_run.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def train_layer(layer: int, dump: Path) -> None:
    run(lifecycle("extraction", layer))
    run(lifecycle("table", layer))
    if verified(lifecycle("complete", layer)):
        print(f"[skip] Layer {layer} has a verified final checkpoint", flush=True)
        return
    folder = ROOT / "checkpoints/gpt2" / f"layer{layer}" / CKDIR
    command = py("-m", "geoae.training.mse", "--config", ROOT / f"experiments/gpt2/configs/layer{layer}_2x.yaml",
                 "--activations_dir", dump, "--no_wandb")
    if any(folder.glob("step_*.pt")):
        command += ["--resume", "latest"]
    run(command)
    run(lifecycle("complete", layer))


def oom_failure(log: Path) -> bool:
    # Evaluation also writes per-step logs, but relays their errors to this log.
    with log.open(errors="replace") as stream:
        return any(any(phrase in line.lower() for phrase in OOM_TEXT) for line in stream)


def stop_processes(processes: dict) -> None:
    for process, _, _ in processes.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process, stream, _ in processes.values():
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        stream.close()


def run_layers(stage: str, layers: list[int], concurrency: int, command_for, state: dict,
               state_path: Path, logs: Path, *, poll_seconds: float = 2.0) -> bool:
    """Bound children, preserve successful layers, and retry only CUDA OOMs."""
    pending = deque(layers)
    remaining_oom = []
    failed = []
    processes = {}
    attempts = {}
    rows = state.setdefault("stages", {}).setdefault(stage, {})
    rows.update(status="running", concurrency=concurrency)
    records = rows.setdefault("layers", {})
    limit = concurrency
    round_oom = False
    try:
        while pending or processes:
            while pending and len(processes) < limit:
                layer = pending.popleft()
                attempts[layer] = attempts.get(layer, 0) + 1
                previous = len(list(logs.glob(f"{stage}_layer{layer}_attempt*.log")))
                log = logs / f"{stage}_layer{layer}_attempt{previous + 1}.log"
                command = command_for(layer)
                stream = log.open("w")
                stream.write(shlex.join(command) + "\n")
                stream.flush()
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                processes[layer] = (process, stream, log)
                records[str(layer)] = dict(status="running", log=str(log), command=command, started_at=utc())
                print(f"[{stage}] layer {layer} started ({len(processes)}/{limit}); {log}", flush=True)
                write_json(state_path, state)
            completed = [layer for layer, (process, _, _) in processes.items() if process.poll() is not None]
            for layer in completed:
                process, stream, log = processes.pop(layer)
                stream.close()
                record = records[str(layer)]
                record.update(exit_code=process.returncode, finished_at=utc())
                if process.returncode == 0:
                    record["status"] = "complete"
                elif oom_failure(log):
                    record["status"] = "retry_pending"
                    remaining_oom.append(layer)
                    round_oom = True
                    # Drain active children before retrying with more free VRAM.
                    limit = max(1, limit // 2)
                    record["reason"] = "CUDA OOM; retry with lower layer concurrency"
                else:
                    record["status"] = "failed"
                    failed.append(layer)
                print(f"[{stage}] layer {layer}: {record['status']}, exit {process.returncode}", flush=True)
                write_json(state_path, state)
            if not processes and not pending and remaining_oom:
                if concurrency == 1:
                    for layer in remaining_oom:
                        records[str(layer)]["status"] = "failed"
                    failed.extend(remaining_oom)
                    remaining_oom.clear()
                else:
                    concurrency = max(1, concurrency // 2)
                    limit = concurrency
                    pending.extend(remaining_oom)
                    remaining_oom.clear()
                    rows["concurrency"] = limit
                    print(f"[{stage}] retrying CUDA OOM layers at concurrency {limit}", flush=True)
            if processes and not completed:
                time.sleep(poll_seconds)
    finally:
        stop_processes(processes)
    rows.update(status="failed" if failed else "complete", failed_layers=sorted(set(failed)),
                oom_fallback=round_oom, finished_at=utc())
    write_json(state_path, state)
    return not failed


def evaluate_command(layer: int, dump: Path, group: str, judges: str) -> list[str]:
    return py(ROOT / "experiments/gpt2/evaluate.py", "--layer", layer, "--steps", group,
              "--activations-dir", dump, "--run-name", "gpt_run", "--judges", judges)


def show_plan(args) -> None:
    dump = Path("/dev/shm/gpt_run_<jobid>/activations")
    print("Campaign: gpt_run; one Argo H100; blocks 0–11; 768 → 1536; K=2000; 50 epochs")
    print(shlex.join(py("-m", "geoae.extraction.corpus", "--config", CFG, "--seed", 42, "--workers", args.corpus_workers)))
    print(shlex.join(py("-m", "geoae.extraction.prefix", "--config", CFG)))
    print(f"Stage all 12 activation files and sidecars once under {dump}")
    print("Build each layer's token-mean table; copy normalization files back to persistent storage")
    print(shlex.join(py(ROOT / "experiments/gpt2/concurrency.py", "profile", "--config",
                        ROOT / "experiments/gpt2/configs/layer0_2x.yaml", "--activations-dir", dump,
                        "--max-concurrency", max(args.train_concurrency, args.eval_concurrency),
                        "--out", ROOT / "runs/gpt_run/gpu_profile.json")))
    print(f"Train all layers together, capped by measured VRAM (requested {args.train_concurrency})")
    print("Verify every layer's epoch-50 completion marker before evaluation")
    print(f"GPU evaluation across all layers, capped by measured VRAM (requested {args.eval_concurrency})")
    print(shlex.join(evaluate_command(0, dump, "gpu", args.judges)))
    if args.judges != "skip":
        print("API evaluation: one layer at a time, preserving pending/failed step status")
        print(shlex.join(evaluate_command(0, dump, "api", args.judges)))


def pipeline(args) -> int:
    if args.dry_run:
        show_plan(args)
        return 0
    os.chdir(ROOT)
    runs, logs = ROOT / "runs/gpt_run", ROOT / "logs/gpt_run"
    runs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    state_path = runs / "pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(run_name="gpt_run", model="gpt2", layers=LAYERS, stages={})
    state.update(status="running", job_id=os.environ.get("SLURM_JOB_ID"), started_at=utc())
    with (runs / ".pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another gpt_run pipeline process is active")
        write_json(state_path, state)
        try:
            state["stages"]["corpus"] = dict(status="running")
            write_json(state_path, state)
            run(py("-m", "geoae.extraction.corpus", "--config", CFG, "--seed", 42, "--workers", args.corpus_workers), logs / "corpus.log")
            state["stages"]["corpus"] = dict(status="complete")
            if not verified(lifecycle("extraction")):
                # Old normalization cannot be paired with a rewritten dump.
                (ACT / "meta.json").unlink(missing_ok=True)
                for path in ACT.glob("norm_params_layer*.npz"):
                    path.unlink()
                state["stages"]["extraction"] = dict(status="running")
                write_json(state_path, state)
                run(py("-m", "geoae.extraction.prefix", "--config", CFG), logs / "extraction.log")
                run(lifecycle("extraction"))
            state["stages"]["extraction"] = dict(status="complete")
            prefix = f"gpt_run_{os.environ.get('SLURM_JOB_ID', str(os.getpid()))}_"
            with tempfile.TemporaryDirectory(prefix=prefix, dir="/dev/shm") as directory:
                dump = Path(directory)
                state["stages"]["stage"] = stage_dump(ACT, dump)
                os.environ["GEOAE_GPU_INIT_LOCK"] = str(dump / ".gpu_init.lock")
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["MKL_NUM_THREADS"] = "1"
                os.environ["OPENBLAS_NUM_THREADS"] = "1"
                os.environ.setdefault("GEOAE_TRAIN_WORKERS", "1")
                os.environ.setdefault("GEOAE_VAL_WORKERS", "1")
                write_json(state_path, state)
                for layer in LAYERS:
                    if verified(lifecycle("table", layer)):
                        continue
                    table = ROOT / f"e2e/checkpoints/general/gpt2/layer{layer}/token_bias_sampled_struct5.npz"
                    table.parent.mkdir(parents=True, exist_ok=True)
                    temporary = table.with_name("token_bias_sampled_struct5.tmp.npz")
                    run(py("-m", "geoae.token_bias", "--activations_dir", dump, "--layer", layer,
                           "--val_frac", "0.05", "--out", temporary), logs / f"token_bias_layer{layer}.log")
                    copy_atomic(dump / f"norm_params_layer{layer}.npz", ACT / f"norm_params_layer{layer}.npz")
                    temporary.replace(table)
                    run(lifecycle("table", layer))
                state["stages"]["tables"] = dict(status="complete")
                profile_path = runs / "gpu_profile.json"
                run(py(ROOT / "experiments/gpt2/concurrency.py", "profile", "--config", ROOT / "experiments/gpt2/configs/layer0_2x.yaml",
                       "--activations-dir", dump, "--max-concurrency", max(args.train_concurrency, args.eval_concurrency),
                       "--out", profile_path), logs / "gpu_profile.log")
                profile = json.loads(profile_path.read_text())
                train_count = min(args.train_concurrency, int(profile["train_concurrency"]))
                eval_count = min(args.eval_concurrency, int(profile["eval_concurrency"]))
                if min(train_count, eval_count) < 1:
                    raise RuntimeError("GPU profile cannot fit one layer with the current recipe")
                state["gpu_profile"] = profile
                print(f"[profile] concurrent training layers {train_count}/12; evaluation layers {eval_count}/12", flush=True)
                incomplete = [layer for layer in LAYERS if not verified(lifecycle("complete", layer))]
                train_cmd = lambda layer: py(ROOT / "experiments/gpt2/run_gpt_run.py", "--train-layer", layer, "--activations-dir", dump)
                if not run_layers("train", incomplete, train_count, train_cmd, state, state_path, logs):
                    raise RuntimeError("Some training layers failed; see layer logs and resume this campaign")
                for layer in LAYERS:
                    run(lifecycle("complete", layer))
                # This download is shared by all layers; finish it before the wave.
                run(evaluate_command(0, dump, "wsd_data", args.judges), logs / "wsd_data.log")
                gpu_cmd = lambda layer: evaluate_command(layer, dump, "gpu", args.judges)
                if not run_layers("eval_gpu", LAYERS, eval_count, gpu_cmd, state, state_path, logs):
                    raise RuntimeError("Some GPU evaluation layers failed; complete artifacts remain available for resume")
                if args.judges != "skip":
                    api_cmd = lambda layer: evaluate_command(layer, dump, "api", args.judges)
                    if not run_layers("eval_api", LAYERS, 1, api_cmd, state, state_path, logs):
                        raise RuntimeError("Some API evaluation layers failed; see pipeline step status")
                else:
                    state["stages"]["eval_api"] = dict(status="not_requested")
                statuses = {}
                for layer in LAYERS:
                    status = ROOT / f"results/gpt_run/layer{layer}/pipeline_state.json"
                    statuses[str(layer)] = json.loads(status.read_text()).get("status", "partial")
                state["evaluation_status"] = statuses
                state["status"] = "complete" if all(v == "complete" for v in statuses.values()) else "partial"
                state["finished_at"] = utc()
                write_json(state_path, state)
                print(f"[gpt_run] {state['status']}; results/gpt_run; {state_path}", flush=True)
                return 0 if state["status"] == "complete" or args.judges == "skip" else 2
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                         error=str(error), finished_at=utc())
            write_json(state_path, state)
            raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-concurrency", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--eval-concurrency", type=int, choices=range(1, 13), default=12)
    parser.add_argument("--corpus-workers", type=int, choices=range(1, 5), default=3)
    parser.add_argument("--judges", choices=("auto", "required", "skip"), default="auto")
    parser.add_argument("--train-layer", type=int, choices=range(12), help=argparse.SUPPRESS)
    parser.add_argument("--activations-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.train_layer is not None:
        if args.activations_dir is None:
            parser.error("Internal --train-layer requires --activations-dir")
        train_layer(args.train_layer, args.activations_dir)
        return 0
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return pipeline(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"[gpt_run failed] {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
