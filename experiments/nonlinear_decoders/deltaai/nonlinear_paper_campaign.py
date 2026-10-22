#!/usr/bin/env python3
"""Submit one sequential training/evaluation allocation per model and decoder arm.

This orchestration module intentionally uses the Python 3.6 standard library so
the DeltaAI login node can collect status without loading PyTorch.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ARMS = ('mlp', 'residual_mlp')
MODELS = ('llama', 'gemma')
LAUNCHERS = ('nonlinear_paper_campaign.py', 'nonlinear_paper_campaign.sbatch')
DEFAULT_ROOT = Path('/projects/bbyl/mhaider/GeoAE/nonlinear_other_models_20261019_deltaai')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.{}.tmp'.format(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root):
    root = Path(root).resolve()
    target = root / 'campaign_manifest.json'
    if target.exists():
        verify(root)
        print('Existing four-arm campaign verified; no jobs submitted')
        return
    models = {}
    for model in MODELS:
        run = root / model
        training_path = run / 'training_manifest.json'
        evaluation_path = run / 'evaluation/manifest.json'
        training, evaluation = read(training_path), read(evaluation_path)
        profile = training['profile']
        if profile['model'] != model:
            raise ValueError('Wrong model profile in {}'.format(run))
        if tuple(training['arms']) != ARMS or tuple(evaluation['plans']) != ARMS:
            raise ValueError('Both matched decoder arms must be prepared: {}'.format(run))
        if training['runtime']['python'] != evaluation['runtime']['python']:
            raise ValueError('The allocation must use one verified training/evaluation environment')
        if read(run / 'evaluation/cpu_preflight.json').get('state') != 'passed':
            raise ValueError('CPU evaluation preflight has not passed: {}'.format(run))
        if profile.get('ready_manifest'):
            ready = read(profile['ready_manifest'])
            if ready.get('state') not in ('ready', 'passed', 'complete', 'verified'):
                raise ValueError('Model assets are not verified: {}'.format(model))
        resources = {}
        for arm in ARMS:
            # Use both existing projects, with two allocations on each account.
            account = ('bbyl-dtai-gh' if (model, arm) in
                       (('llama', 'mlp'), ('gemma', 'residual_mlp')) else 'bimc-dtai-gh')
            resources[arm] = {'account': account, 'qos': account, 'partition': 'ghx4',
                              'gpus': 1, 'cpus': 12, 'memory': '128G' if model == 'llama' else '160G',
                              'walltime': '12:00:00'}
        models[model] = {
            'run_dir': str(run), 'model_name': profile['model_name'],
            'training_manifest_sha256': digest(training_path),
            'evaluation_manifest_sha256': digest(evaluation_path),
            'runtime': training['runtime'], 'resources': resources,
            'evaluation_steps_per_arm': {arm: len(evaluation['plans'][arm]) for arm in ARMS},
        }
    orchestration = root / 'orchestration'
    orchestration.mkdir(parents=True, exist_ok=True)
    for name in LAUNCHERS:
        original = Path(__file__).resolve().parent / name
        destination = orchestration / name
        if original != destination:
            shutil.copy2(original, destination)
    for name in ('logs', 'state', 'locks', 'provenance'):
        (root / name).mkdir(exist_ok=True)
    atomic_json(target, {
        'schema': 1, 'prepared_utc': utc(), 'campaign_root': str(root),
        'layout': 'one_sequential_training_and_evaluation_gpu_job_per_model_arm',
        'models': models, 'arms': list(ARMS), 'job_count': 4,
        'launcher_sha256': {name: digest(orchestration / name) for name in LAUNCHERS},
        'initialization': 'From scratch, matching each paper baseline recipe; original linear checkpoints reused.',
        'walltime_basis': 'Measured Qwen full suite approximately2.3h and training2.9h; allow12h for wider models and Gemma SAE controls.',
    })
    verify(root)
    print('Prepared four sequential training/evaluation allocations; no jobs submitted')


def verify(root):
    root = Path(root)
    manifest = read(root / 'campaign_manifest.json')
    for name, expected in manifest['launcher_sha256'].items():
        if digest(root / 'orchestration' / name) != expected:
            raise ValueError('Campaign launcher changed: ' + name)
    for model, info in manifest['models'].items():
        run = Path(info['run_dir'])
        for name, key in (('training_manifest.json', 'training_manifest_sha256'),
                          ('evaluation/manifest.json', 'evaluation_manifest_sha256')):
            if digest(run / name) != info[key]:
                raise ValueError('Prepared {} manifest changed: {}'.format(model, name))
    return manifest


def build_command(root, manifest, model, arm, actual=False):
    info = manifest['models'][model]
    resources = info['resources'][arm]
    command = ['sbatch', '--parsable' if actual else '--test-only',
               '--account=' + resources['account'], '--qos=' + resources['qos'],
               '--partition=' + resources['partition'], '--nodes=1', '--ntasks=1',
               '--gpus-per-node=1', '--cpus-per-task=' + str(resources['cpus']),
               '--mem=' + resources['memory'], '--time=' + resources['walltime'], '--export=NONE',
               '--job-name=nl-{}-{}'.format(model, arm), '--chdir=' + info['run_dir'],
               '--output={}/logs/%x-%j.out'.format(root)]
    if actual:
        command.append('--hold')
    command += [str(root / 'orchestration/nonlinear_paper_campaign.sbatch'), str(root), model, arm]
    return command


def submission_records(root):
    path = Path(root) / 'submissions.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def submit(root, actual=False):
    manifest = verify(root)
    env = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
    env['TZ'] = 'UTC'
    with (root / 'locks/submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = submission_records(root)
        existing = {(record['model'], record['arm']) for record in records}
        for model in MODELS:
            for arm in ARMS:
                if actual and (model, arm) in existing:
                    print('Already submitted {} {}'.format(model, arm))
                    continue
                command = build_command(root, manifest, model, arm, actual)
                result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True, check=True)
                print(result.stdout.strip() or result.stderr.strip(), flush=True)
                if actual:
                    job = result.stdout.strip().split(';')[0]
                    if not job.isdigit():
                        raise ValueError('Inspect Slurm before retrying ambiguous submission: ' + repr(result.stdout))
                    record = {'submitted_utc': utc(), 'model': model, 'arm': arm, 'job_id': job,
                              'initially_held': True, 'command': command,
                              'campaign_manifest_sha256': digest(root / 'campaign_manifest.json')}
                    with (root / 'submissions.jsonl').open('a') as handle:
                        handle.write(json.dumps(record) + '\n')
                        handle.flush()
                        os.fsync(handle.fileno())
                    existing.add((model, arm))


def inspect_job(job):
    output = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], universal_newlines=True)
    return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)', output))


def release(root):
    manifest = verify(root)
    records = submission_records(root)
    expected = {(model, arm) for model in MODELS for arm in ARMS}
    if {(r['model'], r['arm']) for r in records} != expected or len(records) != 4:
        raise ValueError('Expected exactly four recorded allocations before release')
    receipt_path = root / 'release_receipt.json'
    previous = set(read(receipt_path).get('released_jobs', [])) if receipt_path.exists() else set()
    for info in manifest['models'].values():
        if read(Path(info['run_dir']) / 'evaluation/cpu_preflight.json').get('state') != 'passed':
            raise ValueError('Evaluation CPU preflight must pass before releasing allocations')
    before = {}
    for record in records:
        job = inspect_job(record['job_id'])
        resources = manifest['models'][record['model']]['resources'][record['arm']]
        if job['Account'] != resources['account'] or job['QOS'] != resources['qos']:
            raise ValueError('Wrong account/QOS for ' + record['job_id'])
        already_released = record['job_id'] in previous and job['JobState'] in ('RUNNING', 'COMPLETED')
        if job['TimeLimit'] != resources['walltime'] or (job['JobState'] != 'PENDING' and not already_released):
            raise ValueError('Unexpected allocation before release: ' + repr(job))
        before[record['job_id']] = job
    released = []
    path = receipt_path
    try:
        for record in records:
            job = record['job_id']
            if before[job].get('Reason') == 'JobHeldUser':
                subprocess.run(['scontrol', 'release', job], check=True)
            released.append(job)
    finally:
        atomic_json(path, {'checked_utc': utc(), 'before': before, 'released_jobs': released,
                           'state': 'complete' if len(released) == 4 else 'partial'})
    print('Released four matched training/evaluation jobs: ' + ', '.join(released))


def stage_data(dataset, temporary):
    source = Path(dataset['data_dir'])
    destination = Path(temporary) / 'activations'
    destination.mkdir()
    hashes = {}
    for name, identity in dataset['files'].items():
        original, target = source / name, destination / name
        if original.stat().st_size != identity['size']:
            raise ValueError('Activation input size changed: ' + str(original))
        shutil.copy2(original, target)
        expected = identity.get('sha256') or digest(original)
        if digest(target) != expected:
            raise ValueError('Full node-local activation checksum mismatch: ' + name)
        hashes[name] = expected
    return destination, hashes


def choose_stage(dataset):
    needed = sum(value['size'] for value in dataset['files'].values())
    memory_mb = int(os.environ.get('SLURM_MEM_PER_NODE', '0'))
    for parent in (Path('/dev/shm'), Path(os.environ.get('TMPDIR', '/tmp'))):
        if not parent.is_dir() or not os.access(str(parent), os.W_OK):
            continue
        if parent == Path('/dev/shm') and memory_mb and needed + 64 * 1024 ** 3 > memory_mb * 1024 ** 2:
            continue
        if shutil.disk_usage(str(parent)).free >= needed + 1024 ** 3:
            return parent
    # Shared NVMe mmap remains a valid, byte-identical input when other jobs
    # occupy node-local scratch. File-backed pages can be reclaimed by Linux.
    return None


def run(root, model, arm):
    manifest = verify(root)
    info = manifest['models'][model]
    run_dir = Path(info['run_dir'])
    training = read(run_dir / 'training_manifest.json')
    evaluation = read(run_dir / 'evaluation/manifest.json')
    report_path = root / 'state/{}_{}.json'.format(model, arm)
    report = {'model': model, 'arm': arm, 'job_id': os.environ.get('SLURM_JOB_ID'),
              'started_utc': utc(), 'state': 'running', 'stage': 'staging', 'stages': {},
              'resources': info['resources'][arm]}
    atomic_json(report_path, report)
    env = dict(os.environ)
    env['WANDB_MODE'] = 'disabled'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['HF_HUB_OFFLINE'] = env['HF_DATASETS_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = '1'
    assets = training['profile']['assets']
    if assets.get('hf_home'):
        env['HF_HOME'] = assets['hf_home']
        env['HF_HUB_CACHE'] = str(Path(assets['hf_home']) / 'hub')
    if assets.get('hf_datasets_cache'):
        env['HF_DATASETS_CACHE'] = assets['hf_datasets_cache']
    temporary = None
    with (root / 'locks/{}_{}.lock'.format(model, arm)).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            parent = choose_stage(training['dataset'])
            if parent is None:
                local_data = Path(training['dataset']['data_dir'])
                report['staging'] = {'mode': 'verified_source_mmap', 'path': str(local_data)}
            else:
                temporary = tempfile.TemporaryDirectory(prefix='geoae_{}_{}_'.format(model, arm), dir=str(parent))
                local_data, hashes = stage_data(training['dataset'], temporary.name)
                report['staging'] = {'mode': 'persistent_node_local_copy', 'path': str(local_data), 'sha256': hashes}
            atomic_json(report_path, report)
            for stage, source, script, action in (
                ('gpu_preflight', Path(evaluation['source_root']), 'nonlinear_paper_evals.py', 'gpu-preflight'),
                ('training', Path(training['source_root']), 'nonlinear_paper_training.py', 'run'),
                ('evaluations', Path(evaluation['source_root']), 'nonlinear_paper_evals.py', 'run'),
            ):
                start = time.monotonic()
                item = {'state': 'running', 'started_utc': utc()}
                report['stage'] = stage
                report['stages'][stage] = item
                atomic_json(report_path, report)
                child_env = dict(env)
                child_env['PYTHONPATH'] = child_env['GEOAE_ROOT'] = str(source)
                child_env['OMP_NUM_THREADS'] = child_env['MKL_NUM_THREADS'] = child_env['OPENBLAS_NUM_THREADS'] = '2' if stage == 'training' else '8'
                child_env['GEOAE_TRAIN_WORKERS'] = '2'
                child_env['GEOAE_VAL_WORKERS'] = '1'
                command = [info['runtime']['python'], '-u', str(source / 'experiments/nonlinear_decoders/deltaai' / script), action,
                           '--run-dir', str(run_dir), '--arm', arm]
                if action == 'run':
                    command += ['--activations-dir', str(local_data)]
                print('[campaign {} {}] start {}'.format(model, arm, stage), flush=True)
                result = subprocess.run(command, cwd=str(source), env=child_env)
                item.update(state='complete' if result.returncode == 0 else 'failed',
                            exit_code=result.returncode, finished_utc=utc(), seconds=time.monotonic() - start)
                atomic_json(report_path, report)
                if result.returncode:
                    raise RuntimeError('{} failed with exit {}'.format(stage, result.returncode))
            report.update(state='complete', stage='complete')
        except Exception as error:
            report.update(state='failed', error='{}: {}'.format(type(error).__name__, error))
            raise
        finally:
            if temporary is not None:
                temporary.cleanup()
            report['finished_utc'] = utc()
            atomic_json(report_path, report)


def status(root):
    manifest = read(root / 'campaign_manifest.json')
    records = submission_records(root)
    ids = [record['job_id'] for record in records]
    env = dict(os.environ, TZ='UTC')
    live = {}
    if ids:
        output = subprocess.check_output(['squeue', '-h', '-j', ','.join(ids), '-o',
                                          '%i|%T|%M|%N|%R|%a|%q|%l|%S'], env=env, universal_newlines=True)
        for line in output.splitlines():
            job, state, elapsed, node, reason, account, qos, limit, start = line.strip().split('|')
            item = dict(job_id=job, state=state, elapsed=elapsed, node=node, reason=reason,
                        account=account, qos=qos, time_limit=limit)
            if start not in ('N/A', 'Unknown', ''):
                item['estimated_start_utc' if state == 'PENDING' else 'start_utc'] = start + '+00:00'
            live[job] = item
        output = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(ids),
                                          '--format=JobIDRaw,State,Elapsed,NodeList,ExitCode,Account,QOS,Timelimit,Start,End'],
                                         env=env, universal_newlines=True)
        for line in output.splitlines():
            fields = line.split('|')
            if len(fields) >= 10 and fields[0].isdigit() and fields[0] not in live:
                job, state, elapsed, node, exit_code, account, qos, limit, start, end = fields[:10]
                live[job] = dict(job_id=job, state=state, elapsed=elapsed, node=node, exit_code=exit_code,
                                 account=account, qos=qos, time_limit=limit, start_utc=start, end_utc=end)
    report = {'checked_utc': utc(), 'campaign_root': str(root), 'cluster': 'DeltaAI',
              'layout': manifest['layout'], 'job_count': 4, 'jobs': {model: {} for model in MODELS}}
    for record in records:
        model, arm = record['model'], record['arm']
        info = manifest['models'][model]
        item = dict(live.get(record['job_id'], {'job_id': record['job_id'], 'state': 'UNKNOWN'}))
        item['resources'] = info['resources'][arm]
        item['evaluation_steps'] = info['evaluation_steps_per_arm'][arm]
        for key, path in (
            ('allocation', root / 'state/{}_{}.json'.format(model, arm)),
            ('training', Path(info['run_dir']) / 'state/{}_training.json'.format(arm)),
            ('evaluations', Path(info['run_dir']) / 'evaluation/state/{}.json'.format(arm)),
        ):
            if path.exists():
                state = read(path)
                if key == 'evaluations' and 'steps' in state:
                    state = {name: state[name] for name in state if name != 'steps'}
                    steps = read(path)['steps']
                    state['completed_steps'] = sum(value.get('state') == 'complete' for value in steps.values())
                    state['step_status'] = {name: value.get('state') for name, value in steps.items()}
                item[key] = state
        report['jobs'][model][arm] = item
    atomic_json(root / 'status.json', report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'submit', 'release', 'run', 'status'))
    parser.add_argument('--campaign', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.action == 'prepare':
        prepare(root)
    elif args.action == 'verify':
        verify(root)
        print('Campaign and both prepared model manifests verified')
    elif args.action == 'submit':
        submit(root, args.submit)
    elif args.action == 'release':
        release(root)
    elif args.action == 'run':
        if not args.model or not args.arm:
            parser.error('run requires --model and --arm')
        run(root, args.model, args.arm)
    else:
        status(root)


if __name__ == '__main__':
    main()
