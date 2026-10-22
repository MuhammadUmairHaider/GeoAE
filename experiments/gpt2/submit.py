#!/usr/bin/env python3
"""Submit GPT-2 corpus -> extraction/tables -> all-layer 2x GeoAE training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "k2000_bnh_b32k_lam1_d1536_dpc_sampled_struct5_tokbias"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print submissions; write no manifest or jobs")
    parser.add_argument("--concurrency", type=int, choices=range(1, 13), default=2)
    parser.add_argument("--cpu-account", default=os.environ.get("DELTA_CPU_ACCOUNT", "bimc-delta-cpu"))
    parser.add_argument("--gpu-account", default=os.environ.get("DELTA_ACCOUNT", "bbyl-delta-gpu"))
    parser.add_argument("--gpu-partition", default="gpuA100x4,gpuA40x4")
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/gpt2_2x/campaign.json")
    parser.add_argument("--resubmit", action="store_true", help="Replace an old manifest only after all its jobs leave the queue")
    args = parser.parse_args()
    manifest_path = args.manifest.absolute()
    if manifest_path.exists() and not args.dry_run:
        if not args.resubmit:
            parser.error(f"Campaign already recorded in {manifest_path}; inspect status.py before using --resubmit")
        previous = json.loads(manifest_path.read_text())
        ids = [str(job) for job in previous.get("jobs", {}).values() if job]
        if ids:
            queued = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T"],
                                    text=True, capture_output=True, check=True).stdout.strip()
            if queued:
                parser.error(f"Previous campaign still has queued/running jobs; no duplicate submitted: {queued}")

    manifest = dict(version=1, created_at=datetime.now(timezone.utc).isoformat(),
                    root=str(ROOT), model="gpt2", layers=list(range(12)), hidden_size=768,
                    latent_dim=1536, n_clusters=2000, seed=42, n_epochs=50,
                    n_tokens=10000000, context_len=1024, concurrency=args.concurrency,
                    dump="activations_gpt2_sampled_struct5_10M", corpus="corpus/gpt2_sampled_struct5_v1",
                    extraction_config="experiments/gpt2/configs/extract_all.yaml",
                    training_configs={str(l): f"experiments/gpt2/configs/layer{l}_2x.yaml" for l in range(12)},
                    checkpoint_dirs={str(l): f"checkpoints/gpt2/layer{l}/{RUN_NAME}" for l in range(12)},
                    token_tables={str(l): f"e2e/checkpoints/general/gpt2/layer{l}/token_bias_sampled_struct5.npz" for l in range(12)},
                    jobs={}, commands={}, evaluation={"host": "geoae-box", "launcher": "experiments/gpt2/watch_remote.py",
                                                    "steps": "whole", "trigger": "training_complete.json"})

    def persist() -> None:
        if args.dry_run:
            return
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, manifest_path)

    def submit(stage: str, command: list[str]) -> str:
        manifest["commands"][stage] = command
        print(shlex.join(command), flush=True)
        if args.dry_run:
            job_id = {"corpus": "CORPUS_JOB", "extract": "EXTRACT_JOB", "train": "TRAIN_ARRAY_JOB"}[stage]
        else:
            persist()
            # --parsable returns job_id or job_id;cluster. Retain IDs immediately
            # so a later submission failure cannot lose already-submitted work.
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise RuntimeError(f"Unexpected sbatch response: {result.stdout!r}")
        manifest["jobs"][stage] = job_id
        persist()
        return job_id

    if not args.dry_run:
        (ROOT / "logs/gpt2").mkdir(parents=True, exist_ok=True)
    common = ["sbatch", "--parsable", f"--chdir={ROOT}"]
    try:
        corpus = submit("corpus", common + [f"--account={args.cpu_account}", "experiments/gpt2/corpus.sbatch"])
        extract = submit("extract", common + [f"--account={args.gpu_account}",
                         f"--partition={args.gpu_partition}", f"--dependency=afterok:{corpus}",
                         "experiments/gpt2/extract.sbatch"])
        train = submit("train", common + [f"--account={args.gpu_account}",
                       f"--partition={args.gpu_partition}", f"--dependency=afterok:{extract}",
                       f"--array=0-11%{args.concurrency}", "experiments/gpt2/train.sbatch"])
    except (subprocess.CalledProcessError, OSError, RuntimeError) as error:
        manifest["submission_error"] = str(error)
        persist()
        details = getattr(error, "stderr", None) or str(error)
        print(f"Submission stopped: {details}", flush=True)
        return 1
    if args.dry_run:
        print(f"Manifest on submission: {manifest_path}")
        print("Evaluation: start experiments/gpt2/watch_remote.py after submission; it waits for verified epoch-50 markers.")
    else:
        print(f"Submitted corpus {corpus}, extract {extract}, training array {train}. Manifest: {manifest_path}")
        print("Start the serial remote evaluation watcher: python experiments/gpt2/watch_remote.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
