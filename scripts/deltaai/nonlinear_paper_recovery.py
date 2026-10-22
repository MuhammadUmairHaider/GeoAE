#!/usr/bin/env python3
"""Recover failed model arms without changing frozen experiments or retraining Llama."""
import argparse
import contextlib
import copy
import fcntl
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import nonlinear_paper_campaign as base

DEFAULT_ROOT = Path('/projects/bbyl/mhaider/GeoAE/nonlinear_recovery_20261021_deltaai')
TASKS = ('llama_residual_mlp', 'gemma_mlp', 'gemma_residual_mlp')
LAUNCHERS = ('nonlinear_paper_recovery.py', 'nonlinear_paper_recovery.sbatch',
             'nonlinear_paper_campaign.py', 'nonlinear_model_assets.py')


def hashes(source):
    return {str(p.relative_to(source)): base.digest(p) for p in sorted(source.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.sbatch', '.tex', '.bib')}


def prepare(root, parent):
    if (root / 'manifest.json').exists():
        verify(root)
        print('Existing three-task recovery verified')
        return
    original = base.verify(parent)
    models = {}
    for model in ('llama', 'gemma'):
        run = Path(original['models'][model]['run_dir'])
        old_path = run / 'evaluation/manifest.json'
        old = base.read(old_path)
        if hashes(Path(old['source_root'])) != old['source_sha256']:
            raise ValueError('Original evaluation source changed: ' + model)
        target = root / model / 'evaluation'
        target.mkdir(parents=True, exist_ok=False)
        source = target / 'source'
        shutil.copytree(old['source_root'], str(source), ignore=shutil.ignore_patterns('__pycache__'))
        changed = {}
        for name in ('geoae/checkpoint.py', 'scripts/deltaai/nonlinear_paper_evals.py'):
            overlay = root / 'overlays' / name
            shutil.copy2(str(overlay), str(source / name))
            changed[name] = {'old': old['source_sha256'][name], 'new': base.digest(source / name)}
        manifest = copy.deepcopy(old)
        manifest.update(source_root=str(source), source_sha256=hashes(source),
                        recovery={'original_manifest': str(old_path), 'sha256': base.digest(old_path),
                                  'overlays': changed, 'created_utc': base.utc()})
        # Evaluator commands and successful-output hashes retain their exact paths.
        (target / 'results').symlink_to(run / 'evaluation/results', target_is_directory=True)
        for name in ('state', 'logs', 'locks', 'provenance'):
            (target / name).mkdir()
        if model == 'llama':
            state = base.read(run / 'evaluation/state/residual_mlp.json')
            pending = [s['name'] for s in old['plans']['residual_mlp']
                       if state['steps'].get(s['name'], {}).get('state') != 'complete']
            if pending != ['control_atlas_s5']:
                raise ValueError('Expected only the failed Llama Atlas evaluator: ' + repr(pending))
            base.atomic_json(target / 'state/residual_mlp.json', state)
            base.atomic_json(target / 'provenance/original_failed_state.json', state)
        else:
            for arm in base.ARMS:
                if any((run / 'checkpoints' / arm).glob('step_*.pt')):
                    raise ValueError('Gemma now has checkpoints; inspect before starting a recovery')
        base.atomic_json(target / 'manifest.json', manifest)
        receipt = base.read(old['profile']['ready_manifest'])
        models[model] = {'run_dir': str(run), 'evaluation_dir': str(target),
                         'evaluation_manifest_sha256': base.digest(target / 'manifest.json'),
                         'training_manifest_sha256': base.digest(run / 'training_manifest.json'),
                         'runtime': old['runtime'], 'model_name': old['profile']['model_name'],
                         'hf_model': receipt['hf_model']}
    tasks = {}
    for task in TASKS:
        model, arm = ('llama', 'residual_mlp') if task.startswith('llama') else ('gemma', task[6:])
        resources = copy.deepcopy(original['models'][model]['resources'][arm])
        if model == 'llama':
            resources.update(memory='64G', walltime='02:00:00')
        tasks[task] = {'model': model, 'arm': arm, 'resources': resources,
                       'stages': ['evaluations'] if model == 'llama' else ['gpu_preflight', 'training', 'evaluations']}
    for name in ('state', 'logs', 'locks', 'provenance'):
        (root / name).mkdir(exist_ok=True)
    base.atomic_json(root / 'manifest.json', {
        'created_utc': base.utc(), 'parent_campaign': str(parent),
        'parent_manifest_sha256': base.digest(parent / 'campaign_manifest.json'),
        'models': models, 'tasks': tasks,
        'launcher_sha256': {n: base.digest(root / 'orchestration' / n) for n in LAUNCHERS},
        'policy': 'One sequential allocation per failed arm; Llama training and fifteen valid evaluations reused.'})
    verify(root)
    print('Prepared three recovery tasks; no jobs submitted')


def verify(root):
    m = base.read(root / 'manifest.json')
    if base.digest(Path(m['parent_campaign']) / 'campaign_manifest.json') != m['parent_manifest_sha256']:
        raise ValueError('Parent campaign manifest changed')
    for name, expected in m['launcher_sha256'].items():
        if base.digest(root / 'orchestration' / name) != expected:
            raise ValueError('Recovery launcher changed: ' + name)
    for model, info in m['models'].items():
        directory = Path(info['evaluation_dir'])
        if base.digest(directory / 'manifest.json') != info['evaluation_manifest_sha256']:
            raise ValueError('Recovery evaluation manifest changed: ' + model)
        em = base.read(directory / 'manifest.json')
        if hashes(Path(em['source_root'])) != em['source_sha256']:
            raise ValueError('Recovery source changed: ' + model)
        if base.digest(Path(info['run_dir']) / 'training_manifest.json') != info['training_manifest_sha256']:
            raise ValueError('Original training manifest changed: ' + model)
    return m


def build_command(root, manifest, task, actual=False):
    t = manifest['tasks'][task]
    r = t['resources']
    c = ['sbatch', '--parsable' if actual else '--test-only', '--account=' + r['account'],
         '--qos=' + r['qos'], '--partition=' + r['partition'], '--nodes=1', '--ntasks=1',
         '--gpus-per-node=1', '--cpus-per-task=' + str(r['cpus']), '--mem=' + r['memory'],
         '--time=' + r['walltime'], '--export=NONE', '--job-name=nl-recover-' + task,
         '--chdir=' + str(root), '--output={}/logs/%x-%j.out'.format(root)]
    if actual:
        c.append('--hold')
    return c + [str(root / 'orchestration/nonlinear_paper_recovery.sbatch'), str(root), task]


def require_preflights(manifest):
    for info in manifest['models'].values():
        if base.read(Path(info['evaluation_dir']) / 'cpu_preflight.json').get('state') != 'passed':
            raise ValueError('Recovery CPU preflight has not passed')
    root = Path(manifest['models']['gemma']['evaluation_dir']).parents[1]
    result = base.read(root / 'native_validation.json')
    counts = result['counts']
    if counts['tests'] < 1 or any(counts[k] for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Native recovery regression checks have not all passed')


def submit(root, actual):
    m = verify(root)
    require_preflights(m)
    env = {k: v for k, v in os.environ.items() if not k.startswith('SBATCH_')}
    env['TZ'] = 'UTC'
    with (root / 'locks/submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = {r['task'] for r in base.submission_records(root)}
        for task in TASKS:
            if actual and task in existing:
                continue
            c = build_command(root, m, task, actual)
            r = subprocess.run(c, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, check=True)
            print(r.stdout.strip() or r.stderr.strip(), flush=True)
            if actual:
                job = r.stdout.strip().split(';')[0]
                if not job.isdigit():
                    raise ValueError('Inspect scheduler before retrying ambiguous submission: ' + r.stdout)
                record = {'task': task, 'job_id': job, 'submitted_utc': base.utc(),
                          'initially_held': True, 'command': c, 'manifest_sha256': base.digest(root / 'manifest.json')}
                with (root / 'submissions.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n'); f.flush(); os.fsync(f.fileno())


def release(root):
    m = verify(root)
    require_preflights(m)
    records = base.submission_records(root)
    if len(records) != 3 or {r['task'] for r in records} != set(TASKS):
        raise ValueError('Expected exactly the three failed-arm recovery jobs')
    path = root / 'release_receipt.json'
    prior = base.read(path).get('released_jobs', []) if path.exists() else []
    before = {}
    for record in records:
        job = base.inspect_job(record['job_id'])
        r = m['tasks'][record['task']]['resources']
        if (job['Account'], job['QOS'], job['TimeLimit']) != (r['account'], r['qos'], r['walltime']):
            raise ValueError('Unexpected recovery allocation: ' + repr(job))
        if job['JobState'] != 'PENDING' and not (record['job_id'] in prior and job['JobState'] in ('RUNNING', 'COMPLETED')):
            raise ValueError('Unexpected job state before release: ' + repr(job))
        before[record['job_id']] = job
    released = []
    try:
        for record in records:
            job = record['job_id']
            if before[job].get('Reason') == 'JobHeldUser':
                subprocess.run(['scontrol', 'release', job], check=True)
            released.append(job)
    finally:
        base.atomic_json(path, {'checked_utc': base.utc(), 'released_jobs': released, 'before': before,
                                'state': 'complete' if len(released) == 3 else 'partial'})
    print('Released recovery jobs: ' + ', '.join(released))


def model_stage_parent(needed):
    for parent in (Path(os.environ.get('TMPDIR', '/tmp')), Path('/dev/shm')):
        if parent.is_dir() and os.access(str(parent), os.W_OK) and shutil.disk_usage(str(parent)).free > needed + 1024 ** 3:
            return parent
    raise ValueError('Insufficient node-local storage for the verified LM snapshot')


def run(root, task):
    from nonlinear_model_assets import materialize_snapshot
    m = verify(root)
    t = m['tasks'][task]
    info = m['models'][t['model']]
    em = base.read(Path(info['evaluation_dir']) / 'manifest.json')
    tm = base.read(Path(info['run_dir']) / 'training_manifest.json')
    report = {'task': task, 'job_id': os.getenv('SLURM_JOB_ID'), 'state': 'running',
              'started_utc': base.utc(), 'stages': {}, 'stage': 'model_staging'}
    path = root / 'state' / (task + '.json')
    base.atomic_json(path, report)
    with (root / 'locks' / (task + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with contextlib.ExitStack() as cleanup:
            try:
                size = sum(v['size'] for v in info['hf_model']['files'].values())
                parent = model_stage_parent(size)
                temporary = cleanup.enter_context(tempfile.TemporaryDirectory(prefix='geoae_recover_lm_', dir=str(parent)))
                destination = Path(temporary) / 'model'
                # Materialization rejects any source or destination checksum mismatch.
                receipt = materialize_snapshot(info['hf_model']['snapshot'], destination, info['hf_model'])
                report['model_staging'] = receipt
                env = dict(os.environ)
                assets = em['assets']
                env.update(GEOAE_LM_SNAPSHOTS_JSON=json.dumps({info['model_name']: str(destination)}),
                           HF_HOME=assets['hf_home'], HF_HUB_CACHE=str(Path(assets['hf_home']) / 'hub'),
                           HF_DATASETS_CACHE=assets['hf_datasets_cache'], HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled',
                           TOKENIZERS_PARALLELISM='false', GEOAE_TRAIN_WORKERS='2', GEOAE_VAL_WORKERS='1')
                local_data = Path(tm['dataset']['data_dir'])
                if t['model'] == 'gemma':
                    memory = os.environ.get('SLURM_MEM_PER_NODE')
                    if parent == Path('/dev/shm') and memory:
                        os.environ['SLURM_MEM_PER_NODE'] = str(max(0, int(memory) - int(math.ceil(size / 1024 ** 2))))
                    try:
                        activation_parent = base.choose_stage(tm['dataset'])
                    finally:
                        if memory is not None:
                            os.environ['SLURM_MEM_PER_NODE'] = memory
                    if activation_parent:
                        temporary_data = cleanup.enter_context(tempfile.TemporaryDirectory(prefix='geoae_recover_data_', dir=str(activation_parent)))
                        local_data, checksums = base.stage_data(tm['dataset'], temporary_data)
                        report['activation_staging'] = {'path': str(local_data), 'sha256': checksums}
                    else:
                        report['activation_staging'] = {'mode': 'verified_source_mmap', 'path': str(local_data)}
                for stage in t['stages']:
                    started = time.monotonic()
                    item = {'state': 'running', 'started_utc': base.utc()}
                    report['stage'] = stage; report['stages'][stage] = item
                    base.atomic_json(path, report)
                    training = stage == 'training'
                    source = Path(tm['source_root'] if training else em['source_root'])
                    child = dict(env, PYTHONPATH=str(source), GEOAE_ROOT=str(source))
                    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                        child[key] = '2' if training else '8'
                    script = 'nonlinear_paper_training.py' if training else 'nonlinear_paper_evals.py'
                    action = 'gpu-preflight' if stage == 'gpu_preflight' else 'run'
                    command = [info['runtime']['python'], '-u', str(source / 'scripts/deltaai' / script), action,
                               '--run-dir', info['run_dir'], '--arm', t['arm']]
                    if not training:
                        command += ['--evaluation-dir', info['evaluation_dir']]
                    if action == 'run' and t['model'] == 'gemma':
                        command += ['--activations-dir', str(local_data)]
                    print('[recovery {}] start {}'.format(task, stage), flush=True)
                    result = subprocess.run(command, cwd=str(source), env=child)
                    item.update(state='complete' if result.returncode == 0 else 'failed',
                                exit_code=result.returncode, finished_utc=base.utc(), seconds=time.monotonic() - started)
                    base.atomic_json(path, report)
                    if result.returncode:
                        raise RuntimeError('{} failed: {}'.format(stage, result.returncode))
                report.update(state='complete', stage='complete')
            except Exception as error:
                report.update(state='failed', error='{}: {}'.format(type(error).__name__, error))
                raise
            finally:
                report['finished_utc'] = base.utc()
                base.atomic_json(path, report)


def status(root):
    m = verify(root)
    with contextlib.redirect_stdout(io.StringIO()):
        report = base.status(Path(m['parent_campaign']))
    records = base.submission_records(root)
    live = {}
    if records:
        ids = ','.join(r['job_id'] for r in records)
        env = dict(os.environ, TZ='UTC')
        out = subprocess.check_output(['squeue', '-h', '-j', ids, '-o', '%i|%T|%M|%N|%R|%S'], env=env, universal_newlines=True)
        for line in out.splitlines():
            job, state, elapsed, node, reason, start = line.split('|')
            item = dict(job_id=job, state=state, elapsed=elapsed, node=node, reason=reason)
            if start not in ('N/A', 'Unknown', ''):
                item['estimated_start_utc' if state == 'PENDING' else 'start_utc'] = start + '+00:00'
            live[job] = item
        out = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ids, '--format=JobIDRaw,State,Elapsed,NodeList,ExitCode,Start,End'], env=env, universal_newlines=True)
        for line in out.splitlines():
            fields = line.split('|')
            if len(fields) >= 7 and fields[0].isdigit() and fields[0] not in live:
                job, state, elapsed, node, code, start, end = fields[:7]
                live[job] = dict(job_id=job, state=state, elapsed=elapsed, node=node, exit_code=code, start_utc=start, end_utc=end)
    for record in records:
        task = m['tasks'][record['task']]; model, arm = task['model'], task['arm']
        previous = report['jobs'][model][arm]
        item = dict(live.get(record['job_id'], {'job_id': record['job_id'], 'state': 'UNKNOWN'}))
        item.update(previous_job_id=previous['job_id'], resources=task['resources'],
                    evaluation_steps=previous['evaluation_steps'])
        info = m['models'][model]
        for key, path in (('allocation', root / 'state' / (record['task'] + '.json')),
                          ('training', Path(info['run_dir']) / 'state' / (arm + '_training.json')),
                          ('evaluations', Path(info['evaluation_dir']) / 'state' / (arm + '.json'))):
            if path.exists():
                value = base.read(path)
                if key == 'evaluations':
                    steps = value.pop('steps', {})
                    value['completed_steps'] = sum(v.get('state') == 'complete' for v in steps.values())
                    value['step_status'] = {k: v.get('state') for k, v in steps.items()}
                item[key] = value
        report['jobs'][model][arm] = item
    report.update(checked_utc=base.utc(), recovery_root=str(root), recovery_job_count=len(records))
    base.atomic_json(root / 'status.json', report)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'verify', 'submit', 'release', 'run', 'status'))
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--parent', type=Path, default=base.DEFAULT_ROOT)
    p.add_argument('--task', choices=TASKS)
    p.add_argument('--submit', action='store_true')
    a = p.parse_args(); root = a.root.resolve()
    if a.action == 'prepare': prepare(root, a.parent.resolve())
    elif a.action == 'verify': verify(root); print('Recovery source and manifests verified')
    elif a.action == 'submit': submit(root, a.submit)
    elif a.action == 'release': release(root)
    elif a.action == 'status': status(root)
    else:
        if not a.task: p.error('run requires --task')
        run(root, a.task)


if __name__ == '__main__':
    main()
