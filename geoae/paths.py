"""Central path resolution for GeoAE configs, data, and artifacts."""
from __future__ import annotations

from pathlib import Path

# GeoAE package root (contains geoae/ subpackage)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Repository root (parent of GeoAE/) — useful when data lives outside the package
REPO_ROOT = PACKAGE_ROOT.parent

# Shipped YAML configs
CONFIGS_DIR = PACKAGE_ROOT / "configs"
BASE_CONFIGS_DIR = CONFIGS_DIR / "base"
E2E_CONFIGS_DIR = CONFIGS_DIR / "e2e"


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a config path relative to *base* (default: PACKAGE_ROOT)."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or PACKAGE_ROOT) / p


def default_config(name: str = "default.yaml") -> Path:
    """Return path to a base training config."""
    return BASE_CONFIGS_DIR / name
