#!/usr/bin/env python3
"""Verify ARM/CUDA, stage Qwen data, exercise full batches, then train or resume one arm."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from deltaai_qwen_common import ARMS, DATA_FILES, DEFAULT_RUN_DIR, load_and_verify, sha256


def resume_args(run_dir: Path, arm: str, cfg) -> list[str] | None:
    """Return None for complete, [] for fresh, or latest resume after exact recipe check."""
    import torch
    from geoae.training.common import resolve_resume
    folder = run_dir / 'checkpoints' / arm
    if not any(folder.iterdir()):
        return []
    path = resolve_resume('latest', folder)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint['config'] != cfg.to_dict():
        raise ValueError(f'Checkpoint recipe differs from this arm: {path}')
    if checkpoint['config']['model']['decoder_type'] != arm:
        raise ValueError(f'Cannot resume a different decoder architecture: {path}')
    if checkpoint['epoch'] >= cfg.train.n_epochs:
        return None
    print(f'Resuming {arm} at {path}, after epoch {checkpoint["epoch"]}', flush=True)
    return ['--resume', 'latest']


def stage_data(manifest: dict, parent: Path) -> Path:
    """Only stage when node-local capacity can hold all inputs plus 1 GiB headroom."""
    source = Path(manifest['dataset']['data_dir'])
    needed = sum((source / name).stat().st_size for name in DATA_FILES)
    if shutil.disk_usage(parent).free < needed + 1024**3:
        raise ValueError(f'Insufficient node-local capacity: {parent} needs {needed / 1024**3:.2f} GiB plus headroom')
    dest = parent / 'activations'
    dest.mkdir()
    for name in DATA_FILES:
        shutil.copy2(source / name, dest / name)
        # The full source dump is cross-site SHA256-verified during staging.
        # Check the complete job-local copies too rather than trusting mmap reads.
        expected = sha256(source / name)
        if sha256(dest / name) != expected:
            raise ValueError(f'Node-local copy checksum mismatch: {name}')
    print(f'Staged and checksummed {needed / 1024**3:.2f} GiB at {dest}', flush=True)
    return dest


def gpu_preflight(cfg, data_dir: Path) -> dict:
    """Two real FP32 full-batch updates, including configured density-peaks and full loss."""
    import numpy as np
    import torch
    from geoae.training.data import ActivationBuffer, split_batch
    from geoae.losses import total_loss
    from geoae.seeding import seed_everything
    from geoae.training.initialization import density_peaks_select
    from geoae.training.common import (build_model, make_optimizer, tau_schedule, lambda_schedule,
                                   geometry_schedule, zipf_alpha_schedule)
    if platform.machine() not in ('aarch64', 'arm64'):
        raise ValueError(f'DeltaAI runtime must be ARM64, got {platform.machine()}')
    if not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; refusing CPU fallback')
    import geoae.model
    source = Path(__file__).resolve().parents[3]
    if not Path(geoae.model.__file__).resolve().is_relative_to(source):
        raise ValueError(f'Model imported outside source snapshot: {geoae.model.__file__}')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    seed_everything(cfg.train.seed)
    device = torch.device('cuda')
    model = build_model(cfg, device)
    model.train()
    opt = make_optimizer(model, cfg)
    data = ActivationBuffer(data_dir, 35, val_frac=cfg.data.val_frac, split='train', return_tokens=True)
    x, tok = split_batch(data[np.arange(cfg.data.batch_size)], device)
    if len(x) != 32768:
        raise ValueError('Preflight must use the unchanged full Qwen batch')
    torch.cuda.reset_peak_memory_stats()
    report = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'python': sys.version,
              'executable': sys.executable, 'machine': platform.machine(), 'model_module': geoae.model.__file__,
              'gpu': torch.cuda.get_device_name(0),
              'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
              'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'batch_size': len(x), 'precision': str(x.dtype), 'steps': []}
    for epoch in (1, cfg.train.full_loss_start_epoch):
        if epoch >= cfg.train.clustering_start_epoch and not bool(model.centroids_initialized.item()):
            with torch.no_grad():
                # The configured density pool equals a full Qwen batch.
                if cfg.train.density_pool != len(x):
                    raise ValueError('Preflight density pool must match configured full batch')
                z_pool = model.encode(x, tok)
                centroids, _ = density_peaks_select(
                    z_pool, model.n_clusters, seed=cfg.train.seed,
                    density_power=cfg.train.density_power, min_sep_frac=cfg.train.peak_min_sep_frac,
                    knn_k=cfg.train.density_knn, refine_k=cfg.train.peak_refine_k)
                model.init_centroids_from_class_means(centroids)
                del z_pool, centroids
        model.tau = tau_schedule(epoch, cfg)
        model.zipf_alpha = zipf_alpha_schedule(epoch, cfg)
        lc, ls = lambda_schedule(epoch, cfg)
        lv, lcv, lu = geometry_schedule(epoch, cfg)
        torch.cuda.synchronize()
        start = time.monotonic()
        out = model(x, tok)
        losses = total_loss(x=x, x_hat=out.x_hat, z=out.z, centroids=model.centroids.detach(), Q=out.Q,
                            lambda_cluster=lc, lambda_sep=ls, metric=model.metric,
                            lambda_var=lv, lambda_cov=lcv, lambda_unif=lu,
                            sep_mode=cfg.loss.sep_mode, sep_margin=cfg.loss.sep_margin,
                            var_gamma=cfg.loss.var_gamma)
        if not torch.isfinite(losses['loss']):
            raise ValueError(f'Nonfinite loss in preflight epoch {epoch}')
        opt.zero_grad()
        losses['loss'].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip, error_if_nonfinite=True)
        opt.step()
        model.post_step()
        if bool(model.centroids_initialized.item()):
            model.update_centroids_ema(out.z.detach(), out.Q.detach(), move=not cfg.train.freeze_centroids)
        torch.cuda.synchronize()
        step = {'schedule_epoch': epoch, 'seconds': time.monotonic() - start,
                'loss': float(losses['loss'].detach()), 'grad_norm': float(grad_norm),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        report['steps'].append(step)
        print(f'FULL-BATCH GPU PREFLIGHT {json.dumps(step)}', flush=True)
        del out, losses
    del model, opt, data, x, tok
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--stage-root', type=Path, help='Node-local stage parent; auto chooses /dev/shm then TMPDIR or /tmp')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with (run_dir / 'locks' / f'{args.arm}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = load_and_verify(run_dir)
        from geoae.config import Config
        cfg = Config.from_yaml(run_dir / 'configs' / f'{args.arm}.yaml')
        resume = resume_args(run_dir, args.arm, cfg)
        if resume is None:
            print(f'{args.arm} already completed all 50 epochs; no training needed', flush=True)
            return
        source_data = Path(manifest['dataset']['data_dir'])
        needed = sum((source_data / name).stat().st_size for name in DATA_FILES) + 1024**3
        candidates = [args.stage_root] if args.stage_root else [Path('/dev/shm'), Path(os.environ.get('TMPDIR', '/tmp'))]
        parent = next((p for p in candidates if p.is_dir() and os.access(p, os.W_OK) and shutil.disk_usage(p).free >= needed), None)
        if parent is None:
            raise ValueError('No node-local staging filesystem has enough available capacity')
        with tempfile.TemporaryDirectory(prefix=f'geoae_qwen_{args.arm}_', dir=parent) as temporary:
            local_data = stage_data(manifest, Path(temporary))
            report = gpu_preflight(cfg, local_data)
            report.update({'validated_utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
                           'job_id': os.environ.get('SLURM_JOB_ID'), 'arm': args.arm,
                           'manifest_sha256': sha256(run_dir / 'manifest.json'),
                           'source_sha256': manifest['source_sha256'], 'runtime_modules': manifest['runtime']['modules'],
                           'loaded_modules': os.environ.get('LOADEDMODULES', '').split(':'),
                           'source_data': str(source_data), 'local_data': str(local_data), 'resume_args': resume})
            provenance = run_dir / 'provenance' / f'{args.arm}_{os.environ.get("SLURM_JOB_ID", "local")}.json'
            provenance.write_text(json.dumps(report, indent=2) + '\n')
            if args.preflight_only:
                print(f'Preflight passed: {provenance}; no training performed', flush=True)
                return
            env = dict(os.environ)
            env['GEOAE_TRAIN_ACTIVATIONS_DIR'] = str(local_data)
            env['GEOAE_DATASET_PROVENANCE'] = json.dumps({'source': manifest['dataset'], 'runtime_provenance': str(provenance)})
            command = [sys.executable, '-u', '-m', 'geoae.training.mse', '--config',
                       str(run_dir / 'configs' / f'{args.arm}.yaml'), '--no_wandb', *resume]
            print('GPU preflight passed; starting the unchanged 50-epoch training recipe', flush=True)
            subprocess.run(command, cwd=manifest['source_root'], env=env, check=True)


if __name__ == '__main__':
    main()
