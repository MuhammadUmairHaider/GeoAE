#!/usr/bin/env python3
"""Run the complete frozen paper evaluation in one GPU allocation per arm."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from geoae.workflow.execute import directory_bytes, temporary_staging_directory
from geoae.workflow.snapshot import atomic_json as _atomic_json, sha256 as _sha256


ARMS = ('mlp', 'residual_mlp')
GROUPS = ('core', 'alignment', 'controls', 'ranges')
DEFAULT_SUITE = Path('/projects/bbyl/mhaider/GeoAE/nonlinear_qwen_20261018_deltaai/evaluation/paper_suite')
FILES = ('qwen_paper_bundle.py', 'qwen_paper_bundle.sbatch')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return _sha256(path)


def atomic_json(path, value):
    _atomic_json(path, value)


def load_runner(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    source = Path(manifest['source_root'])
    folder = source / 'experiments/nonlinear_decoders/deltaai'
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location('frozen_paper_evals', folder / 'qwen_paper_evals.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner, runner.load_manifest(root)


def prepare(root):
    runner, manifest = load_runner(root)
    if tuple(manifest['groups']) != GROUPS:
        raise ValueError('Frozen evaluation groups changed')
    if json.loads((root / 'preflight.json').read_text())['state'] != 'passed':
        raise ValueError('The frozen evaluators must have passed GPU preflight')
    directory = root / 'orchestration'
    directory.mkdir(exist_ok=True)
    existing = directory / 'manifest.json'
    if existing.exists():
        validate(root)
        return
    local = Path(__file__).resolve().parent
    for name in FILES:
        destination = directory / name
        if (local / name).resolve() != destination.resolve():
            shutil.copy2(local / name, destination)
    data = {'prepared_utc': utc(), 'layout': 'one_sequential_gpu_job_per_arm',
            'eval_dir': str(root), 'evaluation_manifest_sha256': runner.digest(root / 'manifest.json'),
            'files_sha256': {name: sha256(directory / name) for name in FILES},
            'groups': list(GROUPS), 'training_jobs': manifest['training_jobs'],
            'steps_per_arm': {arm: len(manifest['plans'][arm]) for arm in ARMS},
            'wall_time': '48:00:00', 'gpus_per_arm': 1,
            'reuse_node_local_activations': True,
            'evaluation_plans_unchanged': True}
    atomic_json(existing, data)


def validate(root):
    data = json.loads((root / 'orchestration/manifest.json').read_text())
    if data['evaluation_manifest_sha256'] != sha256(root / 'manifest.json'):
        raise ValueError('Original evaluation manifest changed')
    for name, expected in data['files_sha256'].items():
        if sha256(root / 'orchestration' / name) != expected:
            raise ValueError(f'Consolidated launcher changed: {name}')
    return data


def run_all(root, arm):
    validate(root)
    runner, manifest = load_runner(root)
    groups = tuple(manifest['groups'])
    if groups != GROUPS:
        raise ValueError('Unexpected frozen group order')
    report_path = root / 'state' / f'{arm}_bundle.json'
    report = {'arm': arm, 'job_id': os.getenv('SLURM_JOB_ID'), 'started_utc': utc(),
              'state': 'running', 'groups': {},
              'evaluation_manifest_sha256': sha256(root / 'manifest.json')}
    original_stage = runner.stage_activations
    failed = []
    with (root / 'locks' / f'{arm}_bundle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ExitStack() as scratch:
            cache_parent = None
            staged = None

            def shared_stage(frozen_manifest, _group_directory):
                nonlocal staged, cache_parent
                if staged is None:
                    if cache_parent is None:
                        original = Path(manifest['assets']['activations_dir'])
                        cache_parent = scratch.enter_context(temporary_staging_directory(
                            directory_bytes(original), prefix=f'geoae_all_{arm}_'))
                    try:
                        staged = original_stage(frozen_manifest, cache_parent)
                    except Exception:
                        shutil.rmtree(cache_parent / 'activations', ignore_errors=True)
                        raise
                return staged

            # Older frozen runners use the module function; current runners use
            # the callback and avoid reserving room for a second activation copy.
            runner.stage_activations = shared_stage
            try:
                for group in groups:
                    start = time.monotonic()
                    item = {'state': 'running', 'started_utc': utc()}
                    report['groups'][group] = item
                    atomic_json(report_path, report)
                    print(f'[bundle {arm}] start {group}', flush=True)
                    try:
                        runner.run_group(SimpleNamespace(eval_dir=root, arm=arm, group=group,
                                                         activation_stager=shared_stage))
                        item['state'] = 'complete'
                    except Exception as error:
                        failed.append(group)
                        item.update(state='failed', error=f'{type(error).__name__}: {error}')
                    item.update(finished_utc=utc(), seconds=time.monotonic() - start)
                    atomic_json(report_path, report)
                    print(f'[bundle {arm}] {group}: {item["state"]}', flush=True)
            finally:
                runner.stage_activations = original_stage
    report.update(state='failed' if failed else 'complete', finished_utc=utc(), failed_groups=failed)
    atomic_json(report_path, report)
    if failed:
        raise RuntimeError(f'All groups attempted; failures in {arm}: {failed}')


def build_command(root, manifest, arm, *, submit=False, hold=False):
    resources = manifest['slurm']
    command = ['sbatch', '--parsable' if submit else '--test-only',
               f'--account={resources["account"]}', f'--partition={resources["partition"]}',
               '--nodes=1', '--ntasks=1', '--gpus-per-node=1', '--cpus-per-task=12',
               '--mem=160G', '--time=48:00:00', '--export=NONE',
               f'--job-name=qeval-{arm}-all', f'--chdir={manifest["source_root"]}',
               f'--output={root / "logs"}/%x-%j.out',
               f'--dependency=afterok:{manifest["training_jobs"][arm]}', '--kill-on-invalid-dep=yes']
    if hold:
        command.append('--hold')
    command += [str(root / 'orchestration/qwen_paper_bundle.sbatch'), str(root), arm]
    return command


def submit(root, *, actual=False, hold=False):
    validate(root)
    runner, manifest = load_runner(root)
    path = root / 'bundle_submissions.jsonl'
    env = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
    with (root / 'locks' / 'bundle_submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
        for arm in ARMS:
            existing = next((record for record in records if record['arm'] == arm), None)
            if actual and existing:
                print(f'Already submitted {arm}: {existing["job_id"]}', flush=True)
                continue
            command = build_command(root, manifest, arm, submit=actual, hold=hold)
            result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            print(result.stdout.strip() or result.stderr.strip(), flush=True)
            if actual:
                job = result.stdout.strip().split(';')[0]
                if not job.isdigit():
                    raise ValueError(f'Inspect Slurm before retrying: unexpected response {result.stdout!r}')
                record = {'submitted_utc': utc(), 'arm': arm, 'group': 'all', 'job_id': job,
                          'command': command, 'initially_held': hold,
                          'bundle_manifest_sha256': runner.digest(root / 'orchestration/manifest.json'),
                          'evaluation_manifest_sha256': runner.digest(root / 'manifest.json')}
                with path.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                records.append(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'run', 'submit'))
    parser.add_argument('--eval-dir', type=Path, default=DEFAULT_SUITE)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--hold', action='store_true')
    args = parser.parse_args()
    root = args.eval_dir.resolve()
    if args.action == 'prepare':
        prepare(root)
    elif args.action == 'validate':
        validate(root)
        load_runner(root)
        print('Consolidated launcher and frozen evaluators verified')
    elif args.action == 'run':
        if not args.arm:
            parser.error('run requires --arm')
        run_all(root, args.arm)
    else:
        submit(root, actual=args.submit, hold=args.hold)


if __name__ == '__main__':
    main()
