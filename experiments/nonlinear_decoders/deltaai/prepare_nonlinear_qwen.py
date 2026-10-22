#!/usr/bin/env python3
"""Prepare the two Qwen nonlinear arms on DeltaAI without submitting jobs."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from deltaai_qwen_common import (
    ARMS, DEFAULT_RUN_DIR, SOURCE_ROOT, assert_snapshot_imports, inspect_baseline,
    inspect_dataset, load_and_verify, render_recipe, sha256, source_hashes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--token-table', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True, help='Native ARM64 PyTorch interpreter')
    parser.add_argument('--module', action='append', default=[], help='Module to load, repeatable')
    parser.add_argument('--setup-script', type=Path, help='Optional trusted shell module initialization script')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if SOURCE_ROOT != run_dir / 'source':
        raise SystemExit(f'Execute the prepared source snapshot at {run_dir / "source"}')
    if (run_dir / 'manifest.json').exists():
        load_and_verify(run_dir)
        print(f'Existing preparation verified: {run_dir}; no jobs submitted')
        return
    # Preserve the venv path: resolving bin/python follows its symlink into
    # the base installation and loses ARM dependency overlays.
    python = args.python.absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise SystemExit(f'Interpreter is not executable: {python}')
    data = args.data_dir.resolve(strict=True)
    table = args.token_table.resolve(strict=True)
    assert_snapshot_imports(SOURCE_ROOT)
    from geoae.config import Config
    dataset = inspect_dataset(data, table)
    baseline = inspect_baseline(args.baseline.resolve(strict=True), SOURCE_ROOT, data)
    for name in ('configs', 'checkpoints', 'runs', 'logs', 'provenance', 'locks'):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    configs = {}
    for arm in ARMS:
        for parent in ('checkpoints', 'runs'):
            path = run_dir / parent / arm
            if path.exists() and any(path.iterdir()):
                raise SystemExit(f'Refusing populated output directory: {path}')
            path.mkdir(exist_ok=True)
        path = run_dir / 'configs' / f'{arm}.yaml'
        path.write_text(yaml.safe_dump(render_recipe(SOURCE_ROOT, run_dir, data, table, arm), sort_keys=False))
        Config.from_yaml(path)
        configs[arm] = sha256(path)
    manifest = {
        'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_root': str(SOURCE_ROOT), 'source_sha256': source_hashes(SOURCE_ROOT),
        'arms': list(ARMS), 'config_sha256': configs, 'dataset': dataset,
        'existing_linear_baseline': baseline,
        'runtime': {'python': str(python), 'modules': args.module,
                    'setup_script': str(args.setup_script.resolve(strict=True)) if args.setup_script else None},
        'slurm': {'account': 'bbyl-dtai-gh', 'partition': 'ghx4', 'gpus': 1,
                  'cpus': 12, 'mem': '160G', 'time': '48:00:00'},
        'initialization': 'Both nonlinear arms start from scratch, seed 42; completed linear baseline is reused.',
        'precision': 'Unchanged FP32 training; no autocast or TF32 overrides.',
        'submission_status': 'prepared_only',
        'comparison_limitations': [
            'MLP changes parameter count and adds hidden/output biases as well as nonlinearity.',
            'MLP factors are unconstrained; residual linear skip retains reference column renormalization.',
            'Activation validation is the existing row split rather than a document holdout.',
        ],
    }
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    load_and_verify(run_dir)
    print(f'Prepared exactly two arms: {", ".join(ARMS)}; 50 epochs, 305 steps/epoch, batch 32768; no jobs submitted')


if __name__ == '__main__':
    main()
