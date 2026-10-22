"""Central path resolution for GeoAE configs, data, and artifacts."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# GeoAE package root (contains geoae/ subpackage)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Repository root (parent of GeoAE/) — useful when data lives outside the package
REPO_ROOT = PACKAGE_ROOT.parent

# Editable checkout resources take precedence over installed wheel resources.
CONFIGS_DIR = PACKAGE_ROOT / "configs"
if not (CONFIGS_DIR / "examples").is_dir():
    CONFIGS_DIR = Path(sys.prefix) / "share" / "geoae" / "configs"
BASE_CONFIGS_DIR = CONFIGS_DIR / "examples"
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"
E2E_CONFIGS_DIR = EXPERIMENTS_DIR / "e2e" / "configs"
ARTIFACT_ROOT = Path(os.environ.get(
    "GEOAE_ARTIFACT_ROOT", PACKAGE_ROOT.parent / "GeoAE-artifacts" / "research"
)).expanduser()


def artifact_path(path: str | Path = "") -> Path:
    """Locate archived research artifacts without changing their relative paths."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ARTIFACT_ROOT / p


def config_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a recipe and its recorded aliases inside a checkout or snapshot.

    An explicit ``base`` keeps lookup within that source snapshot, independent
    of the caller's working directory and installed package resources.
    """
    p = Path(path).expanduser()
    root = Path(base).expanduser() if base is not None else PACKAGE_ROOT
    if p.is_absolute():
        if p.exists():
            return p
        try:
            p = p.relative_to(root)
        except ValueError:
            return p
    elif base is None and p.exists():
        return p
    local = root / p
    if local.exists():
        return local
    ledger = root / "experiments/config_paths.json"
    if ledger.is_file():
        destination = json.loads(ledger.read_text())["paths"].get(p.as_posix())
        if destination:
            if base is None and destination.startswith("configs/"):
                return CONFIGS_DIR / destination.removeprefix("configs/")
            return root / destination
    return local if base is not None else p


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a working/output path without redirecting writes into archives."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or PACKAGE_ROOT) / p


def resolve_input_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Locate a read-only input locally, then in the explicit artifact store.

    Callers that also create sidecars or caches must use ``resolve_path`` or
    supply their write destinations separately. Absolute and explicitly based
    paths retain the caller's chosen location.
    """
    local = resolve_path(path, base=base)
    if base is None and not Path(path).is_absolute() and not local.exists():
        archived = artifact_path(path)
        if archived.exists():
            return archived
    return local


def default_config(name: str = "default.yaml") -> Path:
    """Return a small general training example."""
    return BASE_CONFIGS_DIR / name
