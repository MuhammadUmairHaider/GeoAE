"""Shared, read-only checks for the isolated Qwen decoder experiment."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml


ARMS = ("mlp", "residual_mlp")
TEMPLATES = Path("configs/nonlinear_decoder/qwen2.5-3b_l35")
DATA_NAME = "activations_qwen2_5_3b_l35_struct5_train10M"
TABLE_NAME = Path("e2e/checkpoints/general/qwen2.5-3B/layer35/token_bias_struct5_train10M.npz")
DEFAULT_REFERENCE_ROOT = Path("/project/def-hsajjad/dogar/GeoAE/repo")
DEFAULT_RUN_DIR = Path("/project/def-hsajjad/dogar/GeoAE/nonlinear_qwen_20261018")
DEFAULT_PYTHON = Path("/scratch/dogar/venvs/geoae/bin/python")
BASELINE_NAME = Path("checkpoints/qwen2.5-3B/layer35/k2000_bnh_b32k_lam1_d4096_dpc_struct5_train10M_tokbias/step_0015250.pt")
DATA_FILES = (
    "layer_35.npy", "rows_tok.npy", "rows_doc.npy", "rows_pos.npy", "rows_src.npy",
    "rows_norm.npy", "docs.npy", "meta.json", "norm_params_layer35.npz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, sampled: bool = False) -> dict:
    stat = path.stat()
    identity = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if sampled:
        # Avoid a 43 GB scan on the login node. Record this explicitly as a
        # sample fingerprint, never as a whole-file content checksum.
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(65536))
            handle.seek(max(0, stat.st_size - 65536))
            digest.update(handle.read(65536))
        identity["head_tail_64k_sha256"] = digest.hexdigest()
    else:
        identity["sha256"] = sha256(path)
    return identity


def source_files(root: Path) -> list[Path]:
    files = list((root / "geoae").rglob("*.py"))
    files.extend((root / TEMPLATES).glob("*.yaml"))
    files.extend((root / "scripts/argo").glob("*nonlinear_qwen*"))
    files.extend(root / name for name in ("pyproject.toml", "uv.lock") if (root / name).is_file())
    return sorted(path for path in files if path.is_file())


def source_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha256(path) for path in source_files(root)}


def inspect_dataset(data_dir: Path, table_path: Path) -> dict:
    missing = [str(data_dir / name) for name in DATA_FILES if not (data_dir / name).is_file()]
    if missing:
        raise ValueError(f"Missing activation files: {missing}")
    meta = json.loads((data_dir / "meta.json").read_text())
    layer = np.load(data_dir / "layer_35.npy", mmap_mode="r")
    if layer.shape != (10526316, 2048) or layer.dtype != np.float16:
        raise ValueError(f"Unexpected Qwen activation array: {layer.shape}, {layer.dtype}")
    if meta["model"] != "Qwen/Qwen2.5-3B" or meta["target_layer"] != 35:
        raise ValueError("Qwen model/layer metadata mismatch")
    if meta["n_tokens"] != layer.shape[0] or meta["hidden_size"] != layer.shape[1]:
        raise ValueError("Activation metadata and array dimensions differ")
    for name in ("rows_tok.npy", "rows_doc.npy", "rows_pos.npy", "rows_src.npy", "rows_norm.npy"):
        if np.load(data_dir / name, mmap_mode="r").shape != (len(layer),):
            raise ValueError(f"Sidecar row count mismatch: {name}")
    with np.load(data_dir / "norm_params_layer35.npz") as norm, np.load(table_path) as table:
        if str(table["model_name"]) != meta["model"] or int(table["layer"]) != 35:
            raise ValueError("Token table model/layer mismatch")
        if table["table"].shape != (49119, 2048):
            raise ValueError("Unexpected Qwen token table dimensions")
        for key in ("mean", "std"):
            if norm[key].shape != (2048,) or not np.isfinite(norm[key]).all():
                raise ValueError(f"Invalid normalization {key}")
            if not np.allclose(norm[key], table[f"norm_{key}"]):
                raise ValueError(f"Token table normalization {key} differs from activation dump")
        if not (norm["std"] > 0).all():
            raise ValueError("Normalization std must be positive")
        report = json.loads(str(table["report"]))
        if report["train_rows"] != 10000000:
            raise ValueError("Token table was not computed on the 10M-row training split")
        tokens = np.load(data_dir / "rows_tok.npy", mmap_mode="r")
        if int(tokens.min()) < 0 or int(tokens.max()) >= int(table["vocab_size"]):
            raise ValueError("Activation token ids exceed the table vocabulary")
    return {
        "model": meta["model"], "layer": 35, "shape": list(layer.shape), "dtype": str(layer.dtype),
        "train_rows": 10000000, "val_rows": 526316, "steps_per_epoch": 305, "total_steps": 15250,
        "data_dir": str(data_dir.resolve()), "token_table": str(table_path.resolve()),
        "row_split_limitation": meta.get("val_split_report", {}),
        "source_share": meta.get("source_share", {}),
        "files": {name: file_identity(data_dir / name, sampled=name == "layer_35.npy") for name in DATA_FILES},
        "table_identity": file_identity(table_path),
    }


def assert_snapshot_imports(source_root: Path) -> None:
    # Editable installs on Argo point at the old checkout. Make the chosen
    # snapshot first on sys.path and check the actual import location.
    sys.path.insert(0, str(source_root))
    import geoae
    from geoae import config

    for module in (geoae, config):
        if not Path(module.__file__).resolve().is_relative_to(source_root):
            raise ValueError(f"Imported {module.__name__} from outside source snapshot: {module.__file__}")


def inspect_baseline(path: Path, source_root: Path, data_dir: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint["epoch"] != 50 or checkpoint["step"] != 15250:
        raise ValueError("Existing linear baseline is not the completed 50-epoch reference")
    if tuple(checkpoint["model_state"]["decoder.weight"].shape) != (2048, 4096):
        raise ValueError("Existing baseline decoder dimensions differ from the Qwen recipe")
    reference = yaml.safe_load((source_root / TEMPLATES / "reference_bypass.yaml").read_text())
    for section, values in reference.items():
        for key, expected in values.items():
            actual = checkpoint["config"][section][key]
            if actual != expected:
                raise ValueError(f"Existing baseline differs from reference: {section}.{key}")
    with np.load(data_dir / "norm_params_layer35.npz") as norm:
        for key in ("mean", "std"):
            actual = np.asarray(checkpoint[f"norm_{key}"])
            if not np.allclose(actual, norm[key]):
                raise ValueError(f"Baseline normalization {key} differs from the shared dump")
    return {"identity": file_identity(path), "epoch": checkpoint["epoch"], "step": checkpoint["step"],
            "val_mse": float(checkpoint["val_mse"]), "decoder_type": "linear", "reused": True,
            "config": checkpoint["config"], "dataset_provenance": checkpoint.get("dataset_provenance")}


def load_and_verify(run_dir: Path, *, data: bool = True) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    source_root = Path(manifest["source_root"])
    if source_hashes(source_root) != manifest["source_sha256"]:
        raise ValueError("Source snapshot changed since preparation; use a new run directory")
    assert_snapshot_imports(source_root)
    from geoae.config import Config

    for arm in ARMS:
        recipe = run_dir / "configs" / f"{arm}.yaml"
        if sha256(recipe) != manifest["config_sha256"][arm]:
            raise ValueError(f"Prepared config changed: {arm}")
        cfg = Config.from_yaml(recipe)
        if cfg.data.batch_size != 32768 or cfg.train.n_epochs != 50 or cfg.train.seed != 42:
            raise ValueError(f"Matched training recipe changed: {arm}")
        expected = yaml.safe_load((source_root / TEMPLATES / f"{arm}.yaml").read_text())
        actual = yaml.safe_load(recipe.read_text())
        expected["data"]["activations_dir"] = manifest["dataset"]["data_dir"]
        expected["model"]["token_bias"] = manifest["dataset"]["token_table"]
        expected["train"]["checkpoints_dir"] = str(run_dir / "checkpoints" / arm)
        expected["train"]["runs_dir"] = str(run_dir / "runs" / arm)
        if actual != expected:
            raise ValueError(f"Prepared config differs from source template: {arm}")
    if data:
        current = inspect_dataset(Path(manifest["dataset"]["data_dir"]), Path(manifest["dataset"]["token_table"]))
        if current != manifest["dataset"]:
            raise ValueError("Dataset/table content or file identity changed since preparation")
        baseline = manifest["existing_linear_baseline"]["identity"]
        if file_identity(Path(baseline["path"])) != baseline:
            raise ValueError("Existing linear baseline checkpoint changed since preparation")
    return manifest
