#!/usr/bin/env python3
"""Recover failed/missing GPU evaluations after the original gpt_run job exits.

This entry point never trains, extracts, builds token tables or profiles a GPU.
It verifies final checkpoints and reuses complete evaluation artifacts. At most
two layers run together, staging only their needed activation arrays in tmpfs.
Run with --dry-run to inspect exact selections without changing any files.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.gpt2 import evaluate
from scripts.gpt2.concurrency import host_memory_snapshot, eval_host_reservation, HOST_HEADROOM_BYTES, GIB
from scripts.gpt2.run_gpt_run import run_layers

# CT re-encodes held-out corpus documents, and WSD uses only meta.skip_leading.
# Those steps need metadata, but never read the saved activation arrays.
ARRAY_STEPS = {"km", "tokstruct", "cq", "csteer", "cgen", "cgen_concept"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write_json(path, value):
    value["updated_at"] = utc()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def recipe_args(args):
    # API settings are needed to construct the shared registry, but recovery
    # selects GPU steps only and never submits API requests.
    return SimpleNamespace(run_name=args.run_name, corpus_dir=args.corpus_dir,
                           judge_model="google/gemini-2.5-flash-lite", judge_concurrency=6,
                           agreement_judges="", pair_model="anthropic/claude-sonnet-5.5",
                           pair_pairs=100, pair_controls=25)


def plans_for(args, layers):
    plans = {}
    recipe = recipe_args(args)
    for layer in layers:
        checkpoint = evaluate.checkpoint_for(layer, dry_run=args.dry_run)
        steps = evaluate.build_steps(layer, checkpoint, args.activations_dir, recipe)
        names = evaluate.select_steps("gpu", steps)
        previous = read_json(ROOT / "results" / args.run_name / f"layer{layer}/pipeline_state.json")
        if previous and previous.get("checkpoint") != str(checkpoint):
            raise ValueError(f"Layer {layer} artifacts reference a different checkpoint; refusing reuse")
        pending = []
        for step in steps:
            if step.name not in names:
                continue
            status = previous.get("steps", {}).get(step.name, {}).get("status")
            if step.last_block_only and layer != 11:
                # An interrupted evaluator may never have recorded these valid
                # skips; its old requested list still requires those records.
                if status != "skipped":
                    pending.append(step.name)
            elif not evaluate.complete(step) or status != "complete":
                pending.append(step.name)
        command = [sys.executable, "-u", str(ROOT / "scripts/gpt2/evaluate.py"), "--layer", str(layer),
                   "--steps", ",".join(pending), "--activations-dir", str(args.activations_dir),
                   "--corpus-dir", args.corpus_dir, "--run-name", args.run_name, "--judges", "skip"]
        plans[layer] = dict(checkpoint=checkpoint, registry=steps, selected=pending,
                            command=command, stage_arrays=any(
                                s.name in pending and s.name in ARRAY_STEPS and not evaluate.complete(s)
                                and not (s.last_block_only and layer != 11) for s in steps),
                            previous_state=previous)
    return plans


def staging_files(source, layers):
    """Preserve metadata verbatim; stage selected layers and all row sidecars."""
    files = [source / "meta.json", source / "docs.npy"]
    files += [source / f"rows_{name}.npy" for name in ("doc", "pos", "tok", "src", "norm")]
    files += [source / f"layer_{layer}.npy" for layer in layers]
    files += sorted(source.glob("norm_params_layer*.npz"))
    missing = [str(path) for path in files if not path.is_file()]
    missing += [str(source / f"norm_params_layer{layer}.npz") for layer in layers
                if not (source / f"norm_params_layer{layer}.npz").is_file()]
    if missing:
        raise FileNotFoundError("Missing recovery staging files: " + ", ".join(missing))
    return files


def admitted_batch(args, plans, remaining):
    """Use the existing host budget, including future tmpfs bytes; no GPU probe."""
    snapshot = host_memory_snapshot()
    reservation = eval_host_reservation()["reservation_bytes"]
    for count in range(min(args.concurrency, len(remaining)), 0, -1):
        layers = remaining[:count]
        staged_layers = [layer for layer in layers if plans[layer]["stage_arrays"]]
        files = staging_files(args.activations_dir, staged_layers) if staged_layers else []
        size = sum(path.stat().st_size for path in files)
        needed = size + count * reservation + HOST_HEADROOM_BYTES
        free = shutil.disk_usage(args.tmpfs_dir).free if files else None
        if snapshot["available_bytes"] >= needed and (free is None or free >= size + HOST_HEADROOM_BYTES):
            return layers, files, dict(host_memory=snapshot, staged_layers=staged_layers, staged_bytes=size,
                                      worker_budget_bytes=reservation, headroom_bytes=HOST_HEADROOM_BYTES,
                                      tmpfs_free_bytes=free)
    raise RuntimeError("Host memory/tmpfs cannot admit even one recovery worker and its selected activation file")


def final_outcomes(args, plans):
    outcomes = {}
    for layer, plan in plans.items():
        gpu = evaluate.select_steps("gpu", plan["registry"])
        state = read_json(ROOT / "results" / args.run_name / f"layer{layer}/pipeline_state.json")
        missing = [step.name for step in plan["registry"] if step.name in gpu
                   and not (step.last_block_only and layer != 11) and not evaluate.complete(step)]
        outcomes[str(layer)] = dict(status="complete" if not missing and state.get("status") == "complete" else "failed",
                                   missing_or_incomplete=missing, pipeline_status=state.get("status"),
                                   checkpoint=str(plan["checkpoint"]))
    return outcomes


def reconcile_campaign(args, attempt, recovery):
    path = ROOT / "runs" / args.run_name / "pipeline_state.json"
    campaign = read_json(path)
    # Preserve the original job ID, training/extraction stages and scheduler
    # history. The failed original process is evidence, not a new success exit.
    campaign.setdefault("recoveries", []).append(dict(job_id=recovery["job_id"], attempt=str(attempt.relative_to(ROOT)),
                                                     status=recovery["status"], finished_at=recovery["finished_at"]))
    campaign["recovery_job_id"] = recovery["job_id"]
    campaign["evaluation_status"] = {layer: outcome["status"] for layer, outcome in recovery["outcomes"].items()}
    prior_layers = set(campaign.get("layers", range(12)))
    covered = {int(layer) for layer in recovery["outcomes"]}
    if covered >= prior_layers:
        campaign["status"] = recovery["status"]
        if recovery["status"] == "complete":
            if "error" in campaign:
                campaign["original_error"] = campaign.pop("error")
            campaign["finished_at"] = recovery["finished_at"]
            stage = campaign.setdefault("stages", {}).setdefault("eval_gpu", {})
            stage["original_status"] = stage.get("status")
            stage["original_failed_layers"] = stage.get("failed_layers", [])
            stage.update(status="complete", failed_layers=[], recovered_by=recovery["job_id"],
                         recovery_outcomes=str((attempt / "state.json").relative_to(ROOT)))
    write_json(path, campaign)


def recover(args):
    os.chdir(ROOT)
    chosen = list(range(12)) if args.layers == "all" else list(dict.fromkeys(int(v) for v in args.layers.split(",")))
    if not chosen or any(layer not in range(12) for layer in chosen):
        raise ValueError("Layers must be in the range 0 through 11")
    plans = plans_for(args, chosen)
    if args.dry_run:
        for layer, plan in plans.items():
            if plan["selected"]:
                print(f"[dry] layer {layer}: {','.join(plan['selected'])}; at most {args.concurrency} workers; "
                      f"{'stage its activation array and shared sidecars' if plan['stage_arrays'] else 'persistent metadata only'}")
                print(shlex.join(plan["command"]))
            else:
                print(f"[dry] layer {layer}: all requested GPU artifacts complete")
        return 0
    runs = ROOT / "runs" / args.run_name
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / ".pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Original pipeline or another recovery is active; recovery must run after its job ends")
        # Re-read after acquiring the campaign lock so selection is stable.
        plans = plans_for(args, chosen)
        for layer, plan in plans.items():
            evaluate.guard_checkpoint(plan["checkpoint"], layer)
            layer_lock = ROOT / "results" / args.run_name / f"layer{layer}/.evaluation.lock"
            layer_lock.parent.mkdir(parents=True, exist_ok=True)
            with layer_lock.open("a+") as probe:
                try:
                    fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError(f"Layer {layer} evaluator is still active; retry after its process exits")
        job = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
        attempt = runs / "recovery" / f"job_{job}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        attempt.mkdir(parents=True)
        original = runs / "pipeline_state.json"
        if original.is_file():
            shutil.copyfile(original, attempt / "original_pipeline_state.json")
        manifest = runs / "campaign.json"
        if manifest.is_file():
            shutil.copyfile(manifest, attempt / "original_campaign.json")
        recovery = dict(run_name=args.run_name, job_id=job, original_job_id=read_json(original).get("job_id"),
                        started_at=utc(), status="running", concurrency=args.concurrency,
                        activations_dir=str(args.activations_dir), stages={},
                        selected_steps={str(layer): plan["selected"] for layer, plan in plans.items()},
                        commands={str(layer): plan["command"] for layer, plan in plans.items() if plan["selected"]})
        state_path = attempt / "state.json"
        for layer, plan in plans.items():
            evidence = attempt / f"layer{layer}"
            evidence.mkdir()
            source_state = ROOT / "results" / args.run_name / f"layer{layer}/pipeline_state.json"
            if source_state.is_file():
                shutil.copyfile(source_state, evidence / "original_pipeline_state.json")
            for step in plan["selected"]:
                source = ROOT / "results" / args.run_name / f"layer{layer}/logs/{step}.log"
                if source.is_file():
                    shutil.copyfile(source, evidence / f"{step}.log")
        write_json(state_path, recovery)
        logs = attempt / "logs"
        logs.mkdir()
        all_ok = True
        previous_init_lock = os.environ.get("GEOAE_GPU_INIT_LOCK")
        os.environ["GEOAE_GPU_INIT_LOCK"] = str(attempt / ".gpu_init.lock")
        try:
            # Finish shared WSD download before children can touch its common
            # temporary files. No API judging is performed.
            wsd_layers = [layer for layer, plan in plans.items() if "wsd_data" in plan["selected"]]
            if wsd_layers:
                download = plans[wsd_layers[0]]["command"].copy()
                download[download.index("--steps") + 1] = "wsd_data"
                subprocess.run(download, cwd=ROOT, check=True)
            remaining = [layer for layer, plan in plans.items() if plan["selected"]]
            recovery["batches"] = []
            while remaining:
                batch, files, admission = admitted_batch(args, plans, remaining)
                index = len(recovery["batches"])
                record = dict(layers=batch, status="running", **admission)
                recovery["batches"].append(record)
                write_json(state_path, recovery)
                # Keep only this wave's arrays in tmpfs. The directory remains
                # alive until all child evaluators (including OOM retries) exit.
                staging = (tempfile.TemporaryDirectory(prefix=f"{args.run_name}_recover_{job}_", dir=args.tmpfs_dir)
                           if files else nullcontext(str(args.activations_dir)))
                with staging as temporary:
                    dump = Path(temporary) if files else args.activations_dir
                    if files:
                        record["staging_path"] = str(dump)
                        print(f"[stage] layers {admission['staged_layers']}: {admission['staged_bytes'] / GIB:.1f} GiB → {dump}", flush=True)
                        for source in files:
                            shutil.copyfile(source, dump / source.name)
                    record["staging_complete"] = True
                    write_json(state_path, recovery)
                    def command_for(layer):
                        command = plans[layer]["command"].copy()
                        command[command.index("--activations-dir") + 1] = str(dump)
                        return command
                    batch_ok = run_layers(f"eval_batch{index}", batch, len(batch), command_for,
                                          recovery, state_path, logs)
                    all_ok = batch_ok and all_ok
                record.update(status="complete" if batch_ok else "failed", staging_cleaned=True, finished_at=utc())
                write_json(state_path, recovery)
                del remaining[:len(batch)]
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            all_ok = False
            recovery["error"] = str(exc)
            print(f"[recovery failure] {exc}", file=sys.stderr, flush=True)
        finally:
            if previous_init_lock is None:
                os.environ.pop("GEOAE_GPU_INIT_LOCK", None)
            else:
                os.environ["GEOAE_GPU_INIT_LOCK"] = previous_init_lock
        recovery["outcomes"] = final_outcomes(args, plans)
        recovery["status"] = "complete" if all_ok and all(
            value["status"] == "complete" for value in recovery["outcomes"].values()) else "failed"
        recovery["finished_at"] = utc()
        write_json(state_path, recovery)
        reconcile_campaign(args, attempt, recovery)
        print(f"[recovery] {recovery['status']}; {state_path}", flush=True)
        return 0 if recovery["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--run-name", default="gpt_run")
    parser.add_argument("--activations-dir", type=Path, default=ROOT / "activations_gpt2_sampled_struct5_10M")
    parser.add_argument("--corpus-dir", default="corpus/gpt2_sampled_struct5_v1")
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=2)
    parser.add_argument("--tmpfs-dir", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("--run-name must be one directory name")
    try:
        return recover(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[recovery abort] {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
