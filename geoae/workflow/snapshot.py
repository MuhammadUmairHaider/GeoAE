"""Small shared helpers for content identities and workflow source snapshots."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    """Hash files in bounded chunks, including large research artifacts."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Publish complete, finite JSON with the workflow's existing file format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def copy_workflow_sources(source_root: Path) -> dict[str, str]:
    """Include these helpers when preparing a new snapshot from older paper code."""
    hashes = {}
    for original in sorted(Path(__file__).parent.glob("*.py")):
        name = Path("geoae/workflow") / original.name
        target = source_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.resolve() != target.resolve():
            shutil.copy2(original, target)
        hashes[str(name)] = sha256(target)
    return hashes


def copy_evaluation_sources(origin: Path, destination: Path, *, legacy: bool = False) -> dict[str, str]:
    """Freeze runtime code and recipes without copying results, models or caches."""
    roots = ("geoae", "eval_out", "docs/paper") if legacy else (
        "geoae", "experiments", "configs", "scripts", "eval_out", "docs/paper")
    suffixes = {".py", ".tex", ".bib"} if legacy else {
        ".py", ".sbatch", ".sh", ".yaml", ".yml", ".html", ".tex", ".bib", ".sty", ".bst", ".toml", ".lock"}
    files = [path for name in roots for path in sorted((origin / name).rglob("*"))
             if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts]
    if not legacy:
        resources = ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "scripts/py", "experiments/config_paths.json",
                     "geoae/baselines/matched_topk/_reference/SOURCE.json")
        files.extend(origin / name for name in resources if (origin / name).is_file())
    hashes = {}
    for original in files:
        name = original.relative_to(origin)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        hashes[str(name)] = sha256(target)
    return hashes


def manuscript_path(origin: Path, *, legacy: bool = False) -> Path:
    """The reconciled checkout uses paper.tex; external older checkouts used main.tex."""
    names = ("main.tex", "paper.tex") if legacy else ("paper.tex",)
    for name in names:
        path = origin / "docs/paper" / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No manuscript source found under {origin / 'docs/paper'}")
