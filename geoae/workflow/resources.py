"""Optional process coordination for several layer jobs on one CUDA device."""
from __future__ import annotations

from functools import wraps
import os
from pathlib import Path


def serialized_density_peaks(function):
    """Serialize temporary DPC work only when the launcher supplies a lock.

    Normal training is unchanged. The output tensors stay on their original
    device; releasing cached allocator blocks does not change their values.
    A kernel must finish before another process acquires this shared lock.
    """
    @wraps(function)
    def wrapped(z, *args, **kwargs):
        lock_path = os.environ.get("GEOAE_GPU_INIT_LOCK")
        if not lock_path or z.device.type != "cuda":
            return function(z, *args, **kwargs)
        import fcntl
        import torch

        path = Path(lock_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return function(z, *args, **kwargs)
            finally:
                # The function's local distance tensors have already gone out
                # of scope here, leaving only its returned centroids resident.
                try:
                    with torch.cuda.device(z.device):
                        torch.cuda.synchronize(z.device)
                        torch.cuda.empty_cache()
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
    return wrapped
