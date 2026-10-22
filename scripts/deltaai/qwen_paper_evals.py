#!/usr/bin/env python3
"""Freeze the Delta paper evaluators and chain them after the two Qwen trainings.

The evaluation snapshot is separate from the immutable, already-running
training snapshot.  No training source, recipe or checkpoint is modified.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("mlp", "residual_mlp")
GROUPS = ("core", "alignment", "controls", "ranges")
CORE_OVERLAYS = ("geoae/model.py", "geoae/config.py", "geoae/checkpoint.py")
RUNTIME_PACKAGES = ('torch', 'numpy', 'transformers', 'datasets', 'huggingface-hub',
                    'pyarrow', 'accelerate', 'scipy', 'scikit-learn', 'pandas')
DEFAULT_TRAINING = Path('/projects/bbyl/mhaider/GeoAE/nonlinear_qwen_20261018_deltaai')


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def tree_hashes(source: Path) -> dict:
    return {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.sbatch', '.tex', '.bib')}


def asset_identities(assets: dict) -> dict:
    """Record immutable input identities; copied assets also carry full transfer hashes.

    Multi-GB arrays use size/mtime here; the full hash verification receipt is
    frozen in the manifest.  Small codebooks, task definitions and sidecars get
    complete hashes again.  The shared Hugging Face cache remains appendable.
    """
    identities = {}
    excluded = {'hf_home', 'hf_datasets_cache', 'atlas_control_classes', 'transfer_receipt'}
    for key, value in assets.items():
        if key in excluded or not isinstance(value, str):
            continue
        root = Path(value)
        if not root.is_absolute() or not root.exists():
            raise ValueError(f'Missing absolute evaluation asset {key}: {value}')
        files = sorted(root.rglob('*')) if root.is_dir() else [root]
        for path in files:
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            stat = path.stat()
            identity = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if stat.st_size <= 16 * 1024**2:
                identity['sha256'] = digest(path)
            identities[str(path)] = identity
    return identities


def verify_assets(identities: dict) -> None:
    for name, identity in identities.items():
        path = Path(name)
        stat = path.stat()
        if stat.st_size != identity['size'] or stat.st_mtime_ns != identity['mtime_ns']:
            raise ValueError(f'Evaluation asset changed: {path}')
        if 'sha256' in identity and digest(path) != identity['sha256']:
            raise ValueError(f'Evaluation asset hash changed: {path}')


def patch_wiki_reference(source: Path) -> None:
    """Use the exact first 100 Wikipedia records staged on the login node.

    This changes input transport only.  Evaluations keep the original text,
    capture, and perplexity protocol without relying on compute-node HTTP.
    """
    path = source / 'geoae/interp/range_intervention_compare.py'
    content = path.read_text()
    anchor = '    ap.add_argument("--n_wiki", type=int, default=100)'
    replacement = anchor + '\n    ap.add_argument("--wiki_texts", default="", help="Staged canonical Wikipedia reference JSON")'
    if content.count(anchor) != 1:
        raise ValueError('Canonical control parser changed; inspect before patching')
    content = content.replace(anchor, replacement)
    old = '''    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    wtxt = []
    for ex in wiki:
        wtxt.append(ex["text"][:1500])
        if len(wtxt) >= args.n_wiki:
            break'''
    new = '''    if args.wiki_texts:
        reference = json.loads(Path(args.wiki_texts).read_text())
        wtxt = (reference["texts"] if isinstance(reference, dict) else reference)[:args.n_wiki]
        if len(wtxt) < args.n_wiki:
            raise ValueError("Staged Wikipedia reference is too short")
    else:
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        wtxt = []
        for ex in wiki:
            wtxt.append(ex["text"][:1500])
            if len(wtxt) >= args.n_wiki:
                break'''
    if content.count(old) != 1:
        raise ValueError('Canonical Wikipedia input changed; inspect before patching')
    path.write_text(content.replace(old, new))


def prepare(args) -> None:
    training = args.training_run_dir.resolve()
    root = args.eval_dir.resolve()
    paper = args.paper_repo.resolve()
    launchers = Path(__file__).resolve().parent
    if (root / 'manifest.json').exists() or (root / 'source').exists():
        raise ValueError(f'Evaluation snapshot already exists: {root}')
    tm = json.loads((training / 'manifest.json').read_text())
    runtime = dict(tm['runtime'])
    if args.python:
        runtime['python'] = str(args.python.absolute())
    if not Path(runtime['python']).is_file():
        raise ValueError(f'Evaluation Python is unavailable: {runtime["python"]}')
    assets = json.loads(args.assets_json.read_text())
    from qwen_paper_plan import build_steps
    root.mkdir(parents=True, exist_ok=True)
    source = root / 'source'
    source.mkdir()
    canonical = {}
    for name in ('geoae', 'eval_out', 'docs/paper'):
        src = paper / name
        dst = source / name
        for path in sorted(src.rglob('*')):
            if not path.is_file() or path.suffix not in ('.py', '.tex', '.bib'):
                continue
            if '__pycache__' in path.parts:
                continue
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            canonical[str(target.relative_to(source))] = digest(path)
    overlays = {}
    for name in CORE_OVERLAYS:
        original = training / 'source' / name
        actual = digest(original)
        if actual != tm['source_sha256'][name]:
            raise ValueError(f'Training source changed: {name}')
        shutil.copy2(original, source / name)
        overlays[name] = actual
    target = source / 'scripts/deltaai'
    target.mkdir(parents=True)
    for path in sorted(launchers.glob('qwen_paper*')):
        if path.is_file() and path.suffix in ('.py', '.sbatch'):
            shutil.copy2(path, target / path.name)
    patch_wiki_reference(source)
    from qwen_paper_compat import patch_nonlinear_evaluators
    compatibility = patch_nonlinear_evaluators(source)
    submissions = [json.loads(line) for line in (training / 'submissions.jsonl').read_text().splitlines() if line]
    training_jobs = {item['arm']: str(item['job_id']) for item in submissions}
    if set(training_jobs) != set(ARMS):
        raise ValueError('Expected exactly the two previously submitted training arms')
    plans = {arm: build_steps(assets, training / 'checkpoints' / arm / 'step_0015250.pt',
                             root / 'results' / arm, runtime['python']) for arm in ARMS}
    paper_path = paper / 'docs/paper/main.tex'
    receipt = assets.get('transfer_receipt')
    manifest = {'prepared_utc': utc(), 'training_run_dir': str(training), 'source_root': str(source),
                'paper_origin': str(paper_path), 'paper_sha256': digest(paper_path),
                'training_manifest_sha256': digest(training / 'manifest.json'),
                'training_jobs': training_jobs, 'runtime': runtime, 'slurm': tm['slurm'],
                'runtime_package_versions': {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES},
                'canonical_source_sha256': canonical, 'nonlinear_overlays_sha256': overlays,
                'nonlinear_evaluator_patches': compatibility,
                'source_sha256': tree_hashes(source), 'assets': assets,
                'asset_identities': asset_identities(assets),
                'asset_transfer_receipt': json.loads(Path(receipt).read_text()) if receipt else None,
                'plans': plans, 'groups': list(GROUPS),
                'protocol': {'checkpoint': 'epoch50_step15250', 'model': 'Qwen/Qwen2.5-3B', 'layer': 35,
                             'core': 'Paper Table1, plus detailed geometry and token diagnostics',
                             'alignment': 'All token and sequence tasks; paired cluster>=10 row filtering; Atlas3draws',
                             'controls': 'Own two-way true positives, n_fit80,n_eval100,min_eval20; alpha0.4,0.5',
                             'ranges': 'Qwen extension of the Llama-only range-structure appendix',
                             'linear_training': 'Existing checkpoint reused; no training rerun'}}
    for directory in ('logs', 'locks', 'state', 'provenance'):
        (root / directory).mkdir(exist_ok=True)
    atomic_json(root / 'manifest.json', manifest)
    print(f'Prepared {root}: {len(plans[ARMS[0]])} steps per arm; paper {manifest["paper_sha256"]}')


def load_manifest(root: Path) -> dict:
    manifest = json.loads((root / 'manifest.json').read_text())
    if tree_hashes(Path(manifest['source_root'])) != manifest['source_sha256']:
        raise ValueError('Evaluation source changed since preparation')
    verify_assets(manifest['asset_identities'])
    for name, version in manifest.get('runtime_package_versions', {}).items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Evaluation package changed since preparation: {name}')
    return manifest


def verify_checkpoint(path: Path, arm: str, *, final: bool = True) -> dict:
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    mc = ck['config']['model']
    if mc.get('decoder_type', 'linear') != arm:
        raise ValueError(f'Wrong decoder architecture: {path}')
    if (mc['hidden_size'], mc['latent_dim'], mc['n_clusters']) != (2048, 4096, 2000):
        raise ValueError(f'Wrong model dimensions: {path}')
    if ck['config']['extraction']['model_name'] != 'Qwen/Qwen2.5-3B':
        raise ValueError(f'Wrong frozen language model: {path}')
    if ck['config']['data']['target_layer'] != 35:
        raise ValueError(f'Wrong extraction layer: {path}')
    if final and (ck['epoch'] != 50 or ck['step'] != 15250):
        raise ValueError(f'Expected final epoch50/step15250: {path}')
    if final and not bool(ck['model_state']['centroids_initialized']):
        raise ValueError(f'Centroids not initialized: {path}')
    return {'path': str(path), 'sha256': digest(path), 'epoch': int(ck['epoch']),
            'step': int(ck['step']), 'decoder_type': arm, 'val_metrics': ck.get('val_metrics', {}),
            'val_mse': float(ck['val_mse']) if ck.get('val_mse') is not None else None}


def validate_outputs(step: dict) -> None:
    for name in step['outputs']:
        path = Path(name)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing or empty evaluation output: {path}')
        if path.suffix == '.json':
            value = json.loads(path.read_text())
            required = step.get('json_required_keys', {}).get(name, [])
            if not isinstance(value, (dict, list)) or (required and not isinstance(value, dict)):
                raise ValueError(f'Unexpected output format: {path}')
            if any(key not in value for key in required):
                raise ValueError(f'Incomplete evaluation output: {path}; expected {required}')
            if isinstance(value, dict) and 'status' in value and value['status'] != 'complete':
                raise ValueError(f'Evaluation output is not complete: {path}')
            if step['name'] == 'probe_alignment':
                from qwen_paper_plan import PAPER_RUNG_NAMES
                missing = set(PAPER_RUNG_NAMES) - set(value['rungs'])
                if missing:
                    raise ValueError(f'Missing paper tasks in {path}: {sorted(missing)}')
            if step['name'].startswith('atlas_pair10'):
                missing = {'atlas14_s0', 'atlas14_s1', 'atlas14_s2'} - set(value['sets'])
                if missing:
                    raise ValueError(f'Missing Atlas draws in {path}: {sorted(missing)}')
        elif path.suffix == '.npz':
            import numpy as np
            with np.load(path, allow_pickle=False) as value:
                required = step.get('npz_required_keys', {}).get(name, [])
                if any(key not in value for key in required):
                    raise ValueError(f'Incomplete NPZ output: {path}; expected {required}')


def completed(step: dict, record: dict | None, checkpoint_sha: str) -> bool:
    if not record or record.get('state') != 'complete' or record.get('checkpoint_sha256') != checkpoint_sha:
        return False
    try:
        validate_outputs(step)
        if set(record['output_sha256']) != set(step['outputs']):
            return False
        return all(digest(Path(name)) == value for name, value in record['output_sha256'].items())
    except (OSError, ValueError, KeyError):
        return False


def stage_activations(manifest: dict, directory: Path) -> Path:
    original = Path(manifest['assets']['activations_dir'])
    training = json.loads((Path(manifest['training_run_dir']) / 'manifest.json').read_text())
    # Use the reviewed extraction's exact sidecars, including whichever norm name it uses.
    names = [p.name for p in original.iterdir() if p.is_file() and p.suffix in ('.npy', '.npz', '.json')
             and p.name != 'deltaai_transfer_validation.json']
    needed = sum((original / name).stat().st_size for name in names)
    if shutil.disk_usage(directory).free < needed + 2 * 1024**3:
        raise ValueError(f'Insufficient node-local memory for {needed / 1024**3:.1f}GiB activations')
    dest = directory / 'activations'
    dest.mkdir()
    identities = manifest['asset_identities']
    full_hashes = training.get('cross_site_transfer_validation', {}).get('files', {})
    for name in names:
        src = original / name
        shutil.copy2(src, dest / name)
        expected = full_hashes.get(name, {}).get('sha256') or identities.get(str(src), {}).get('sha256')
        if expected and digest(dest / name) != expected:
            raise ValueError(f'Node-local activation copy hash differs: {name}')
    # The original complete dump already has a cross-site full-SHA receipt.
    receipt = training.get('cross_site_transfer_validation', {})
    print(f'Staged {needed / 1024**3:.2f}GiB activations; transfer receipt present={bool(receipt)}', flush=True)
    return dest


def runtime_environment(manifest: dict) -> dict:
    env = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
    assets = manifest['assets']
    env.update({'PYTHONPATH': manifest['source_root'], 'GEOAE_ROOT': manifest['source_root'],
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled',
                'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8',
                'GEOAE_ATLAS_CLASSES_DIR': assets['atlas_classes_dir'],
                'GEOAE_CONTROL_DATASETS_DIR': assets['control_datasets_dir']})
    if assets.get('hf_home'):
        env['HF_HOME'] = assets['hf_home']
        env['HF_HUB_CACHE'] = str(Path(assets['hf_home']) / 'hub')
    if assets.get('hf_datasets_cache'):
        env['HF_DATASETS_CACHE'] = assets['hf_datasets_cache']
    return env


def run_group(args) -> None:
    root = args.eval_dir.resolve()
    manifest = load_manifest(root)
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; refusing CPU fallback')
    source = Path(manifest['source_root'])
    steps = [step for step in manifest['plans'][args.arm] if step['group'] == args.group]
    checkpoint = Path(manifest['training_run_dir']) / 'checkpoints' / args.arm / 'step_0015250.pt'
    identity = verify_checkpoint(checkpoint, args.arm)
    state_path = root / 'state' / f'{args.arm}.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'arm': args.arm, 'steps': {}}
    env = runtime_environment(manifest)
    state['updated_utc'] = utc()
    provenance = {'started_utc': utc(), 'group': args.group, 'arm': args.arm,
                  'hostname': socket.gethostname(), 'job_id': os.getenv('SLURM_JOB_ID'),
                  'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0), 'manifest_sha256': digest(root / 'manifest.json'),
                  'checkpoint': identity}
    atomic_json(root / 'provenance' / f'{args.arm}_{args.group}_{os.getenv("SLURM_JOB_ID", "local")}.json', provenance)
    failed = []
    (root / 'locks').mkdir(exist_ok=True)
    with (root / 'locks' / f'{args.arm}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix=f'geoae_eval_{args.arm}_', dir='/dev/shm') as temporary:
            local_acts = None
            for step in steps:
                name = step['name']
                if completed(step, state['steps'].get(name), identity['sha256']):
                    print(f'[skip complete] {name}', flush=True)
                    continue
                blocked = [dep for dep in step['dependencies']
                           if state['steps'].get(dep, {}).get('state') != 'complete']
                if blocked:
                    state['steps'][name] = {'state': 'blocked', 'dependencies': blocked}
                    failed.append(name)
                    atomic_json(state_path, state)
                    continue
                if step['needs_activations'] and local_acts is None:
                    local_acts = stage_activations(manifest, Path(temporary))
                command = list(step['command'])
                if step['needs_activations']:
                    command = [str(local_acts) if item == manifest['assets']['activations_dir'] else item
                               for item in command]
                for output in step['outputs']:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                record = {'state': 'running', 'started_utc': utc(), 'checkpoint_sha256': identity['sha256'],
                          'command': command, 'job_id': os.getenv('SLURM_JOB_ID')}
                state['steps'][name] = record
                atomic_json(state_path, state)
                start = time.monotonic()
                print(f'[start {args.arm}] {name}: {shlex.join(command)}', flush=True)
                try:
                    with (root / 'logs' / f'{args.arm}_{name}.log').open('w') as log:
                        result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f'Evaluator exited {result.returncode}; see {args.arm}_{name}.log')
                    validate_outputs(step)
                    record.update(state='complete', output_sha256={name: digest(Path(name)) for name in step['outputs']})
                except Exception as error:
                    record.update(state='failed', error=str(error))
                    failed.append(name)
                record.update(finished_utc=utc(), seconds=time.monotonic() - start)
                state['updated_utc'] = utc()
                atomic_json(state_path, state)
                print(f'[{record["state"]}] {name} {record["seconds"] / 60:.1f}min', flush=True)
    if args.group == 'controls':
        from qwen_paper_summary import write_control_summary
        write_control_summary(root / 'results' / args.arm)
    if failed:
        raise RuntimeError(f'Failed/blocked steps in {args.arm}/{args.group}: {failed}')


def submission_command(root: Path, manifest: dict, arm: str, group: str,
                       previous: str | None = None, *, submit: bool = False,
                       smoke: bool = False) -> list[str]:
    slurm = manifest['slurm']
    source = Path(manifest['source_root'])
    args = ['sbatch', '--parsable' if submit else '--test-only', f'--account={slurm["account"]}',
            f'--partition={slurm["partition"]}', '--nodes=1', '--ntasks=1', '--gpus-per-node=1',
            '--cpus-per-task=12', '--mem=160G', '--time=00:20:00' if smoke else '--time=48:00:00',
            '--export=NONE', f'--chdir={source}', f'--job-name=qeval-{arm}-{group}',
            f'--output={root / "logs"}/%x-%j.out']
    if not smoke:
        dependency = 'afterok:' + manifest['training_jobs'][arm]
        if previous:
            dependency += ',afterany:' + previous
        args += ['--dependency=' + dependency, '--kill-on-invalid-dep=yes']
    args += [str(source / 'scripts/deltaai/qwen_paper_evals.sbatch'), str(root), arm,
             'preflight' if smoke else group]
    return args


def submit_groups(args) -> None:
    root = args.eval_dir.resolve()
    manifest = load_manifest(root)
    env = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
    path = root / 'submissions.jsonl'
    with (root / 'locks' / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
        if args.preflight_only:
            requested = [('mlp', 'preflight')]
        else:
            requested = [(arm, group) for arm in ARMS for group in GROUPS]
        for arm, group in requested:
            existing = next((r for r in records if r['arm'] == arm and r['group'] == group), None)
            if args.submit and existing:
                print(f'Skipping previously submitted {arm}/{group}: {existing["job_id"]}')
                continue
            index = GROUPS.index(group) if group in GROUPS else 0
            predecessor = next((r['job_id'] for r in records
                                if r['arm'] == arm and index > 0 and r['group'] == GROUPS[index - 1]), None)
            if args.submit and index > 0 and not predecessor:
                raise ValueError(f'Missing preceding evaluation job for {arm}/{group}')
            command = submission_command(root, manifest, arm, group, predecessor,
                                         submit=args.submit, smoke=args.preflight_only)
            print(shlex.join(command), flush=True)
            if args.print_only:
                continue
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            print(result.stdout.strip() or result.stderr.strip(), flush=True)
            if args.submit:
                job_id = result.stdout.strip().split(';')[0]
                if not job_id.isdigit():
                    raise ValueError(f'Unrecognized sbatch response; inspect scheduler: {result.stdout!r}')
                record = {'submitted_utc': utc(), 'arm': arm, 'group': group, 'job_id': job_id,
                          'command': command, 'manifest_sha256': digest(root / 'manifest.json')}
                with path.open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                records.append(record)


def preflight(args) -> None:
    """Real CUDA LM/checkpoint/asset smoke; no optimizer or training updates."""
    root = args.eval_dir.resolve()
    manifest = load_manifest(root)
    os.environ.update(runtime_environment(manifest))
    source = Path(manifest['source_root'])
    sys.path.insert(0, str(source))
    os.chdir(source)
    import gc
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from geoae.checkpoint import load_ae_checkpoint, load_lm
    from geoae.hooks import SplicingHook, TokenIdTap
    from geoae.interp import _shared as shared, neuronlens as nl
    from geoae.interp.concept_probe import LADDER
    from geoae.interp.closest_tokens import load_baseline_kmeans
    if not torch.cuda.is_available():
        raise ValueError('Preflight requires a GPU')
    device = torch.device('cuda')
    report = {'started_utc': utc(), 'host': socket.gethostname(), 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(0), 'checks': {}}
    a = manifest['assets']
    for key in ('balanced_kmeans', 'balanced_kmeans_tokmean', 'plain_kmeans', 'plain_kmeans_tokmean'):
        codebook = load_baseline_kmeans(Path(a[key]), device, allow_erasure=True)
        report['checks'][key] = list(codebook[0].shape)
        del codebook
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', local_files_only=True)
    lm = load_lm('Qwen/Qwen2.5-3B', device=device)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    tap = TokenIdTap(lm)
    hook = SplicingHook(lm, 35)
    enc = tok(['The capital of France is', 'A short test.'], padding=True, return_tensors='pt').to(device)
    with torch.no_grad():
        base = lm(**enc, use_cache=False).logits[:, -1, :].float()
        if not torch.isfinite(base).all():
            raise ValueError('Non-finite base LM logits')
    for arm in ARMS:
        folder = Path(manifest['training_run_dir']) / 'checkpoints' / arm
        candidates = sorted(folder.glob('step_*.pt'))
        if not candidates:
            raise ValueError(f'No real checkpoint available for preflight: {arm}')
        path = candidates[0]  # stable completed epoch; the latest file may still be writing
        identity = verify_checkpoint(path, arm, final=False)
        ae, mean, std, _ = load_ae_checkpoint(path, device, allow_token_bias=True)
        hook.activate(nl.make_z_edit(None, ae, mean, std, device, tap=tap))
        with torch.no_grad():
            logits = lm(**enc, use_cache=False).logits[:, -1, :].float()
            if not torch.isfinite(logits).all():
                raise ValueError(f'Non-finite nonlinear splice: {arm}')
        hook.deactivate()
        report['checks'][arm] = identity
        del ae, mean, std, logits
        gc.collect()
        torch.cuda.empty_cache()
    tap.remove()
    for dataset in ('db14', 'biasbios', 'ag_news', 'news20', 'atlas_s0', 'atlas_s1', 'atlas_s2'):
        cfg = shared.set_dataset(dataset)
        if cfg.get('local_parquet'):
            ds = load_dataset('parquet', data_files=cfg['local_parquet'], split='train')
        else:
            ds = load_dataset(cfg['dataset_name'], split=cfg['dataset_split'])
        report['checks']['dataset_' + dataset] = len(ds)
    for name, config, split in (('cais/mmlu', 'all', 'test'), ('Salesforce/wikitext', 'wikitext-2-raw-v1', 'test')):
        report['checks'][name] = len(load_dataset(name, config, split=split))
    from qwen_paper_plan import PAPER_RUNG_NAMES
    for rung, filename, *_ in LADDER:
        if rung not in PAPER_RUNG_NAMES:
            continue
        p = Path(a['probe_cache']) / filename
        if not p.exists():
            raise ValueError(f'Missing paper task cache {p}')
        with np.load(p, allow_pickle=True) as data:
            report['checks']['cache_' + rung] = list(data.files)
    del lm
    gc.collect()
    torch.cuda.empty_cache()
    smoke_dir = root / 'preflight_outputs'
    smoke_dir.mkdir(exist_ok=True)
    for arm in ARMS:
        folder = Path(manifest['training_run_dir']) / 'checkpoints' / arm
        path = sorted(folder.glob('step_*.pt'))[0]
        # Exercise the complete stock spliced-MMLU and Qwen no-BOS PPL paths.
        # These are explicitly smoke outputs, never substituted for final scores.
        commands = [
            [sys.executable, '-u', '-m', 'geoae.interp.mmlu_splice', '--checkpoint', str(path),
             '--n', '8', '--bs', '2', '--out', str(smoke_dir / f'{arm}_mmlu8.json')],
            [sys.executable, '-u', '-m', 'geoae.interp.splice_ppl',
             '--ae', str(smoke_dir / f'{arm}_ppl2.json') + '=' + str(path), '--max_chunks', '2'],
        ]
        for index, command in enumerate(commands):
            with (smoke_dir / f'{arm}_{index}.log').open('w') as log:
                subprocess.run(command, cwd=source, env=runtime_environment(manifest), stdout=log,
                               stderr=subprocess.STDOUT, check=True)
    report.update(finished_utc=utc(), state='passed')
    atomic_json(root / 'preflight.json', report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--training-run-dir', type=Path, default=DEFAULT_TRAINING)
    prep.add_argument('--eval-dir', type=Path, default=DEFAULT_TRAINING / 'evaluation')
    prep.add_argument('--paper-repo', type=Path, default=Path('/projects/bbyl/mhaider/GeoAE'))
    prep.add_argument('--assets-json', type=Path, required=True)
    prep.add_argument('--python', type=Path, help='Isolated ARM evaluation environment; keep its symlink path')
    for action in ('run', 'preflight', 'submit', 'validate'):
        cmd = sub.add_parser(action)
        cmd.add_argument('--eval-dir', type=Path, default=DEFAULT_TRAINING / 'evaluation')
        if action == 'run':
            cmd.add_argument('--arm', choices=ARMS, required=True)
            cmd.add_argument('--group', choices=GROUPS, required=True)
        if action == 'submit':
            mode = cmd.add_mutually_exclusive_group()
            mode.add_argument('--submit', action='store_true')
            mode.add_argument('--print-only', action='store_true')
            cmd.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'run':
        run_group(args)
    elif args.action == 'submit':
        submit_groups(args)
    elif args.action == 'preflight':
        preflight(args)
    else:
        manifest = load_manifest(args.eval_dir.resolve())
        print(f'Validated paper/source/assets: {manifest["paper_sha256"]}')


if __name__ == '__main__':
    main()
