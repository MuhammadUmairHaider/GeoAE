"""DeltaAI launch requests preserve the matched recipe and cannot duplicate the arms."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'experiments/nonlinear_decoders/deltaai'))
    path = ROOT / 'experiments/nonlinear_decoders/deltaai' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'deltaai_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(run_dir):
    return {'source_root': str(run_dir / 'source'),
            'slurm': {'account': 'bbyl-dtai-gh', 'partition': 'ghx4', 'gpus': 1,
                      'cpus': 12, 'mem': '160G', 'time': '48:00:00'}}


@pytest.mark.parametrize('submit', [False, True])
def test_default_only_two_arms_explicit_deltaai_resources(tmp_path, monkeypatch, submit):
    runner = load_script('submit_nonlinear_qwen', monkeypatch)
    info = manifest(tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps(info))
    monkeypatch.setattr(runner, 'load_and_verify', lambda p: info)
    monkeypatch.setenv('SBATCH_ACCOUNT', 'wrong-account')
    monkeypatch.setenv('SBATCH_GPUS', '4')
    calls = []

    def fake_run(command, **kwargs):
        assert not any(k.startswith('SBATCH_') for k in kwargs['env'])
        calls.append(command)
        return SimpleNamespace(stdout=f'{123 + len(calls)};deltaai\n', stderr='')

    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    args = ['submit_nonlinear_qwen.py', '--run-dir', str(tmp_path)]
    if submit:
        args.append('--submit')
    monkeypatch.setattr(sys, 'argv', args)
    runner.main()
    assert [command[-1] for command in calls] == ['mlp', 'residual_mlp']
    for command in calls:
        assert ('--parsable' if submit else '--test-only') in command
        for option in ('--gpus-per-node=1', '--account=bbyl-dtai-gh', '--partition=ghx4',
                       '--cpus-per-task=12', '--mem=160G', '--time=48:00:00', '--export=NONE'):
            assert option in command
    records = tmp_path / 'submissions.jsonl'
    if submit:
        saved = [json.loads(line) for line in records.read_text().splitlines()]
        assert [r['arm'] for r in saved] == ['mlp', 'residual_mlp']
        assert [r['job_id'] for r in saved] == ['124', '125']
        # A second request must not queue duplicate training jobs.
        runner.main()
        assert len(calls) == 2
    else:
        assert not records.exists()


def test_partial_submission_only_queues_missing_arm(tmp_path, monkeypatch):
    runner = load_script('submit_nonlinear_qwen', monkeypatch)
    info = manifest(tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps(info))
    (tmp_path / 'submissions.jsonl').write_text(json.dumps({'arm': 'mlp', 'job_id': '123'}) + '\n')
    monkeypatch.setattr(runner, 'load_and_verify', lambda p: info)
    calls = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda c, **kw: (calls.append(c) or SimpleNamespace(stdout='124\n', stderr='')))
    monkeypatch.setattr(sys, 'argv', ['submit', '--run-dir', str(tmp_path), '--submit'])
    runner.main()
    assert [c[-1] for c in calls] == ['residual_mlp']


def test_prepare_preserves_virtualenv_symlink_interpreter(tmp_path, monkeypatch):
    prepare = load_script('prepare_nonlinear_qwen', monkeypatch)
    source = tmp_path / 'source'
    source.mkdir()
    venv_python = tmp_path / 'env' / 'bin' / 'python'
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    data = tmp_path / 'data'
    data.mkdir()
    table = tmp_path / 'table.npz'
    table.touch()
    baseline = tmp_path / 'baseline.pt'
    baseline.touch()
    monkeypatch.setattr(prepare, 'SOURCE_ROOT', source)
    monkeypatch.setattr(prepare, 'assert_snapshot_imports', lambda p: None)
    monkeypatch.setattr(prepare, 'inspect_dataset', lambda *a: {})
    monkeypatch.setattr(prepare, 'inspect_baseline', lambda *a: {})
    monkeypatch.setattr(prepare, 'source_hashes', lambda p: {})
    monkeypatch.setattr(prepare, 'load_and_verify', lambda p: {})
    monkeypatch.setattr(prepare, 'render_recipe', lambda *a: {})
    import geoae.config
    monkeypatch.setattr(geoae.config.Config, 'from_yaml', lambda p: None)
    monkeypatch.setattr(sys, 'argv', ['prepare', '--run-dir', str(tmp_path), '--data-dir', str(data),
                                   '--token-table', str(table), '--baseline', str(baseline),
                                   '--python', str(venv_python), '--module', 'python/miniforge3_pytorch/2.12.0'])
    prepare.main()
    saved = json.loads((tmp_path / 'manifest.json').read_text())
    assert saved['runtime']['python'] == str(venv_python)
    assert Path(saved['runtime']['python']).resolve() != Path(saved['runtime']['python'])
    assert saved['runtime']['modules'] == ['python/miniforge3_pytorch/2.12.0']


def test_recipe_changes_only_paths(monkeypatch, tmp_path):
    runner = load_script('prepare_nonlinear_qwen', monkeypatch)
    import yaml
    for arm in ('mlp', 'residual_mlp'):
        original = yaml.safe_load((ROOT / runner.TEMPLATES / f'{arm}.yaml').read_text()) if hasattr(runner, 'TEMPLATES') else yaml.safe_load((ROOT / 'experiments/nonlinear_decoders/configs/qwen2.5-3b_l35' / f'{arm}.yaml').read_text())
        rendered = runner.render_recipe(ROOT, tmp_path, Path('/data'), Path('/table.npz'), arm)
        assert rendered['data'].pop('activations_dir') == '/data'
        original['data'].pop('activations_dir')
        assert rendered['model'].pop('token_bias') == '/table.npz'
        original['model'].pop('token_bias')
        for key in ('checkpoints_dir', 'runs_dir'):
            rendered['train'].pop(key)
            original['train'].pop(key)
        assert rendered == original
        assert rendered['train']['n_epochs'] == 50
        assert rendered['data']['batch_size'] == 32768


def test_staging_refuses_insufficient_capacity_before_copy(tmp_path, monkeypatch):
    runner = load_script('run_nonlinear_qwen', monkeypatch)
    monkeypatch.setattr(runner, 'DATA_FILES', ('layer_35.npy',))
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'layer_35.npy').write_bytes(b'payload')
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda p: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match='Insufficient node-local capacity'):
        runner.stage_data({'dataset': {'data_dir': str(source)}}, tmp_path)
    assert not (tmp_path / 'activations').exists()
