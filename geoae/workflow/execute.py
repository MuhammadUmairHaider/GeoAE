"""Choose portable scratch space without exceeding a Slurm memory allocation."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator


GIB = 1024 ** 3


def directory_bytes(directory: Path) -> int:
    """Size a prospective copy without reading activation contents."""
    return sum(path.stat().st_size for path in directory.rglob("*")
               if path.is_file() and "__pycache__" not in path.parts)


def staging_roots() -> tuple[Path, ...]:
    """Prefer Linux shared memory, then configured and platform temporary space."""
    paths = [Path("/dev/shm")]
    if os.environ.get("TMPDIR"):
        paths.append(Path(os.environ["TMPDIR"]))
    paths.append(Path(tempfile.gettempdir()))
    return tuple(dict.fromkeys(paths))


def select_staging_root(required_bytes: int, *, disk_margin_bytes: int = 2 * GIB,
                        memory_reserve_bytes: int = 64 * GIB) -> Path:
    """Find writable scratch with disk capacity and shared-memory headroom.

    Shared-memory copies count against Slurm's RAM allocation. Keep the same
    64 GiB working-memory reserve as the campaign launcher; ordinary temporary
    files use the filesystem budget instead. No directories or data are created.
    """
    if min(required_bytes, disk_margin_bytes, memory_reserve_bytes) < 0:
        raise ValueError("Staging sizes must be nonnegative")
    memory_mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    for parent in staging_roots():
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            continue
        if (parent == Path("/dev/shm") and memory_mb
                and required_bytes + memory_reserve_bytes > memory_mb * 1024 ** 2):
            continue
        if shutil.disk_usage(parent).free >= required_bytes + disk_margin_bytes:
            return parent
    raise ValueError("No writable staging filesystem has sufficient capacity and memory headroom")


@contextmanager
def temporary_staging_directory(required_bytes: int, *, prefix: str) -> Iterator[Path]:
    """Allocate scratch only when an unfinished step needs an activation copy."""
    parent = select_staging_root(required_bytes)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as directory:
        yield Path(directory)
