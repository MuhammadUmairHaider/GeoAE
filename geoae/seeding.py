"""
Centralised determinism controls so every GeoSep experiment is reproducible and
directly comparable across runs (AE vs baseline, re-runs, ablations).

Call `seed_everything(seed)` at the top of every entry point. For DataLoaders,
pass `worker_init_fn=seed_worker` and `generator=make_generator(seed)` so the
shuffle order and per-worker RNG are reproducible too.

What it pins:
  * PYTHONHASHSEED, Python `random`, NumPy, torch CPU + all CUDA devices
  * cuDNN deterministic, benchmark off
  * TF32 off  (fp32 matmuls become bitwise-stable; negligible cost here since the
    LM runs in bf16 and k-means is on CPU)
  * torch deterministic algorithms (warn_only — never hard-crash an inference job)
  * CUBLAS_WORKSPACE_CONFIG (required for deterministic cuBLAS; set on import,
    before the first CUDA matmul)
"""
from __future__ import annotations

import os
import random

# Must be set before the first CUDA matmul, hence at import time.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

DEFAULT_SEED = 42


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True, quiet: bool = False) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    if not quiet:
        print(f"[seed] seed={seed} deterministic={deterministic} "
              f"(cudnn.deterministic, tf32 off, CUBLAS_WORKSPACE_CONFIG set)")
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init: derive a unique-but-deterministic seed per worker."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
