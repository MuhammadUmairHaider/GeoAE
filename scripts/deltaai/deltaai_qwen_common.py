"""DeltaAI snapshot and recipe checks; reuse the reviewed Qwen data checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT / 'scripts' / 'argo'))
from nonlinear_qwen_common import (  # noqa: E402
    ARMS, DATA_FILES as DATA_FILES, TEMPLATES, assert_snapshot_imports, file_identity,
    inspect_baseline as inspect_baseline, inspect_dataset, sha256, source_hashes as argo_source_hashes,
)

DEFAULT_RUN_DIR = Path('/projects/bbyl/mhaider/GeoAE/nonlinear_qwen_20261018_deltaai')


def source_hashes(root: Path) -> dict[str, str]:
    result = argo_source_hashes(root)
    result.update({str(p.relative_to(root)): sha256(p) for p in sorted((root / 'scripts/deltaai').iterdir())
                   if p.is_file() and p.suffix in ('.py', '.sbatch', '.sh')})
    return result


def render_recipe(source: Path, run_dir: Path, data_dir: Path, table: Path, arm: str) -> dict:
    cfg = yaml.safe_load((source / TEMPLATES / f'{arm}.yaml').read_text())
    cfg['data']['activations_dir'] = str(data_dir)
    cfg['model']['token_bias'] = str(table)
    cfg['train']['checkpoints_dir'] = str(run_dir / 'checkpoints' / arm)
    cfg['train']['runs_dir'] = str(run_dir / 'runs' / arm)
    return cfg


def load_and_verify(run_dir: Path, *, data: bool = True) -> dict:
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    source = Path(manifest['source_root'])
    if source.resolve() != run_dir / 'source':
        raise ValueError('Source must be the isolated run-dir/source snapshot')
    if source_hashes(source) != manifest['source_sha256']:
        raise ValueError('Source snapshot changed since preparation; use a new run directory')
    assert_snapshot_imports(source)
    from geoae.config import Config
    dataset = manifest['dataset']
    for arm in ARMS:
        path = run_dir / 'configs' / f'{arm}.yaml'
        if sha256(path) != manifest['config_sha256'][arm]:
            raise ValueError(f'Prepared config changed: {arm}')
        actual = yaml.safe_load(path.read_text())
        expected = render_recipe(source, run_dir, Path(dataset['data_dir']), Path(dataset['token_table']), arm)
        if actual != expected:
            raise ValueError(f'Prepared config differs from the reviewed template: {arm}')
        cfg = Config.from_yaml(path)
        if cfg.data.batch_size != 32768 or cfg.train.n_epochs != 50 or cfg.train.seed != 42:
            raise ValueError(f'Matched training schedule changed: {arm}')
    if data:
        if inspect_dataset(Path(dataset['data_dir']), Path(dataset['token_table'])) != dataset:
            raise ValueError('Dataset/table changed since preparation')
        baseline = manifest['existing_linear_baseline']['identity']
        if file_identity(Path(baseline['path'])) != baseline:
            raise ValueError('Reused linear baseline checkpoint changed')
    return manifest
