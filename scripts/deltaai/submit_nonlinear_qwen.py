#!/usr/bin/env python3
"""Validate both GH200 requests by default; --submit queues the two nonlinear arms."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from deltaai_qwen_common import ARMS, DEFAULT_RUN_DIR, load_and_verify, sha256


def already_submitted(run_dir: Path, arm: str) -> bool:
    """Never accidentally queue the same arm twice, including after a partial submission."""
    path = run_dir / 'submissions.jsonl'
    return path.exists() and any(json.loads(line)['arm'] == arm for line in path.read_text().splitlines() if line.strip())


def build_command(run_dir: Path, manifest: dict, arm: str, *, submit: bool) -> list[str]:
    r = manifest['slurm']
    source = Path(manifest['source_root'])
    return ['sbatch', '--parsable' if submit else '--test-only',
            f'--account={r["account"]}', f'--partition={r["partition"]}', '--nodes=1', '--ntasks=1',
            f'--gpus-per-node={r["gpus"]}', f'--cpus-per-task={r["cpus"]}',
            f'--mem={r["mem"]}', f'--time={r["time"]}', f'--chdir={source}',
            f'--job-name=qnld-{arm}', f'--output={run_dir / "logs"}/%x-%j.out', '--export=NONE',
            str(source / 'scripts/deltaai/run_nonlinear_qwen.sbatch'), str(run_dir), arm]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--submit', action='store_true')
    mode.add_argument('--print-only', action='store_true')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = load_and_verify(run_dir)
    env = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
    (run_dir / 'locks').mkdir(exist_ok=True)
    with (run_dir / 'locks' / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm in dict.fromkeys(args.arms):
            if args.submit and already_submitted(run_dir, arm):
                print(f'Skipping {arm}: already recorded in submissions.jsonl')
                continue
            command = build_command(run_dir, manifest, arm, submit=args.submit)
            print(shlex.join(command), flush=True)
            if args.print_only:
                continue
            result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            if result.stdout:
                print(result.stdout.strip(), flush=True)
            if result.stderr:
                print(result.stderr.strip(), flush=True)
            if args.submit:
                job_id = result.stdout.strip().split(';')[0]
                if not job_id.isdigit():
                    raise RuntimeError(f'Unexpected sbatch job ID: {result.stdout!r}; inspect scheduler before retrying')
                record = {'submitted_utc': datetime.now(timezone.utc).isoformat(), 'arm': arm,
                          'job_id': job_id, 'manifest_sha256': sha256(run_dir / 'manifest.json'), 'command': command}
                with (run_dir / 'submissions.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
    print('Training requests submitted or previously recorded' if args.submit else 'No jobs submitted')


if __name__ == '__main__':
    main()
