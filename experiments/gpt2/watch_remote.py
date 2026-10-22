#!/usr/bin/env python3
"""Evaluate completed GPT-2 layers on the GPU box, without keeping a GPU idle.

Run this lightweight watcher on Delta in tmux after submitting the campaign.
It transfers each completed layer and runs the evaluation suite serially on
geoae-box. The local status file and per-layer logs survive reconnects.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
ARM = "k2000_bnh_b32k_lam1_d1536_dpc_sampled_struct5_tokbias"
DUMP = "activations_gpt2_sampled_struct5_10M"
CORPUS = "corpus/gpt2_sampled_struct5_v1"


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def remote_command(host: str, root: str, argv: list[str], **kwargs):
    command = f"cd {shlex.quote(root)} && {shlex.join(argv)}"
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
        check=True, **kwargs,
    )


def sync_paths(host: str, remote_root: str, paths: list[Path], *, backup=False):
    """Use relative source paths; never include credentials or unrelated data."""
    present = [p for p in paths if p.exists()]
    if len(present) != len(paths):
        missing = [str(p) for p in paths if not p.exists()]
        raise FileNotFoundError(f"Missing transfer inputs: {missing}")
    argv = ["rsync", "-aL", "--relative", "--protect-args"]
    if backup:
        argv += ["--backup", "--backup-dir=runs/gpt2_2x/code_backups"]
    argv += [str(p.relative_to(ROOT)) for p in present]
    argv += [f"{host}:{remote_root}/"]
    subprocess.run(argv, cwd=ROOT, check=True)


def slurm_failed(manifest: dict, layer: int) -> str | None:
    jobs = manifest.get("jobs", {})
    train_id = jobs.get("train") or manifest.get("train_job")
    if isinstance(train_id, dict):
        train_id = train_id.get("job_id")
    if not train_id:
        return None
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", f"{train_id}_{layer}", "--format=State"],
        capture_output=True, text=True,
    )
    states = {line.strip().split("|")[0].split()[0]
              for line in result.stdout.splitlines() if line.strip()}
    failures = states & {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
    # Failed preparation leaves an afterok-dependent array pending indefinitely.
    for name in ("corpus", "extract"):
        preparation = jobs.get(name)
        if isinstance(preparation, dict):
            preparation = preparation.get("job_id")
        if preparation:
            result = subprocess.run(["sacct", "-n", "-P", "-j", str(preparation),
                                     "--format=State"], capture_output=True, text=True)
            failures |= {line.strip().split("|")[0].split()[0]
                         for line in result.stdout.splitlines() if line.strip()} & {
                             "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
    return ",".join(sorted(failures)) or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="runs/gpt2_2x/campaign.json")
    parser.add_argument("--host", default="geoae-box")
    parser.add_argument("--remote-root", default="GeoAE")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--once", action="store_true", help="Process only currently ready layers")
    args = parser.parse_args()
    manifest_path = ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    status_path = manifest_path.parent / "remote_eval_status.json"
    lock_file = (manifest_path.parent / "remote_eval.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("An evaluation watcher is already active for this campaign")
    state = json.loads(status_path.read_text()) if status_path.exists() else {"layers": {}}
    state.update(host=args.host, remote_root=args.remote_root, pid=os.getpid())
    logs = ROOT / "logs/gpt2"
    logs.mkdir(parents=True, exist_ok=True)

    # Back up any differing existing task modules before installing their changes.
    code = [ROOT / "experiments/gpt2", ROOT / "experiments/gpt2/configs"]
    for name in manifest.get("eval_code_files", ["geoae/lm_arch.py", "geoae/checkpoint.py", "geoae/hooks.py",
                                                 "geoae/evaluation/faithfulness.py", "geoae/analysis/logit_attribution.py"]):
        code.append(ROOT / name)
    remote_command(args.host, args.remote_root, ["mkdir", "-p", "runs/gpt2_2x"])
    sync_paths(args.host, args.remote_root, code, backup=True)
    print(f"Watching {args.layers}; evaluations run serially on {args.host}", flush=True)
    while True:
        for layer in args.layers:
            key = str(layer)
            previous = state["layers"].get(key, {})
            if previous.get("status") == "complete":
                continue
            if previous.get("status") == "failed" and not args.retry_failed:
                continue
            checkpoint_dir = ROOT / f"checkpoints/gpt2/layer{layer}/{ARM}"
            marker = checkpoint_dir / "training_complete.json"
            if not marker.exists():
                failure = slurm_failed(manifest, layer)
                state["layers"][key] = {"status": "failed" if failure else "waiting_for_training",
                                         "reason": failure}
                save_json(status_path, state)
                continue
            completion = json.loads(marker.read_text())
            if int(completion.get("epoch", 0)) < 50:
                raise RuntimeError(f"Refusing incomplete training marker: {marker}")
            checkpoint = Path(completion["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
            # resolve() would lose the symlink-relative transfer directory.
            if not checkpoint.is_relative_to(ROOT):
                checkpoint = checkpoint_dir / checkpoint.name
            logfile = logs / f"remote_eval_layer{layer}.log"
            try:
                gpu = remote_command(args.host, args.remote_root,
                                     ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True)
                if any(int(memory.strip()) > 2000 for memory in gpu.stdout.splitlines()):
                    state["layers"][key] = {"status": "waiting_for_evaluation_gpu"}
                    save_json(status_path, state)
                    continue
                state["layers"][key] = {"status": "transferring", "checkpoint": str(checkpoint)}
                save_json(status_path, state)
                dump = ROOT / DUMP
                files = [checkpoint, marker, ROOT / CORPUS,
                         ROOT / f"e2e/checkpoints/general/gpt2/layer{layer}/token_bias_sampled_struct5.npz",
                         dump / "meta.json", dump / f"layer_{layer}.npy",
                         dump / f"norm_params_layer{layer}.npz"]
                files += sorted(dump.glob("rows_*.npy"))
                files += sorted(dump.glob("docs.npy"))
                sync_paths(args.host, args.remote_root, files)
                state["layers"][key]["status"] = "evaluating"
                save_json(status_path, state)
                with logfile.open("a", buffering=1) as output:
                    remote_command(args.host, args.remote_root,
                                   ["env", "GEOAE_REMOTE=1", "HF_HUB_OFFLINE=0",
                                    "HF_DATASETS_OFFLINE=0", "bash", "experiments/gpt2/evaluate.sh",
                                    str(layer), "all"], stdout=output, stderr=subprocess.STDOUT)
                # Copy small reviewable results back to Delta; keep arrays/figures on the box.
                result_dir = ROOT / f"results/gpt2/layer{layer}"
                result_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-a", "--protect-args", "--include=*/",
                                "--exclude=*.npz", "--exclude=*.npy", "--exclude=*.png",
                                f"{args.host}:{args.remote_root}/results/gpt2/layer{layer}/",
                                str(result_dir) + "/"], check=True)
                pipeline = json.loads((result_dir / "pipeline_state.json").read_text())
                if pipeline.get("status") != "complete":
                    raise RuntimeError(f"Evaluation reported {pipeline.get('status')!r}; inspect pipeline_state.json")
                state["layers"][key].update(status="complete", finished=time.time())
                print(f"Layer {layer}: evaluation complete; {logfile}", flush=True)
            except Exception as error:
                state["layers"][key].update(status="failed", error=str(error), log=str(logfile))
                print(f"Layer {layer}: {error}; inspect {logfile}", flush=True)
            save_json(status_path, state)
        done = all(state["layers"].get(str(layer), {}).get("status") in {"complete", "failed"}
                   for layer in args.layers)
        if args.once or done:
            failed = [layer for layer in args.layers
                      if state["layers"].get(str(layer), {}).get("status") == "failed"]
            if failed:
                raise SystemExit(f"Evaluation failed or training interrupted for layers {failed}")
            if done:
                print("All requested layers evaluated successfully", flush=True)
            return
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
