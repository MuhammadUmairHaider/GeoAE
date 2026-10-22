#!/usr/bin/env python3
"""Prepare, validate, and train matched nonlinear decoders for a paper model.

The profile selects existing assets; the completed linear checkpoint supplies
the complete recipe. No scheduler actions or language-model loading occur here.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from geoae.paths import config_path
from geoae.workflow.snapshot import atomic_json as _atomic_json, sha256 as _sha256

ARMS = ("mlp", "residual_mlp")
SOURCE_ROOT = Path(__file__).resolve().parents[3]
MODELS = {
    "meta-llama/Llama-3.2-3B": (27, 3072, 6144),
    "google/gemma-3-4b-pt": (33, 2560, 5120),
    "Qwen/Qwen2.5-3B": (35, 2048, 4096),
}
TEMPLATES = {
    "meta-llama/Llama-3.2-3B": "llama3.2-3b_l27",
    "google/gemma-3-4b-pt": "gemma3-4b_l33",
    "Qwen/Qwen2.5-3B": "qwen2.5-3b_l35",
}
REFERENCE_CONFIGS = {
    "meta-llama/Llama-3.2-3B": Path("experiments/paper/configs/llama.yaml"),
    "google/gemma-3-4b-pt": Path("experiments/paper/configs/gemma.yaml"),
    "Qwen/Qwen2.5-3B": Path("experiments/paper/configs/qwen.yaml"),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical_config(value):
    """Compare YAML/JSON sequences with checkpoint tuples without altering recipes."""
    if isinstance(value, dict):
        return {key: canonical_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_config(item) for item in value]
    return value


def atomic_json(path: Path, value: dict):
    _atomic_json(path, value)


def sha256(path: Path) -> str:
    return _sha256(path)


def file_identity(path: Path, *, sampled=False) -> dict:
    st = path.stat()
    result = {"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if sampled:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            digest.update(stream.read(65536))
            stream.seek(max(0, st.st_size - 65536))
            digest.update(stream.read(65536))
        result["head_tail_64k_sha256"] = digest.hexdigest()
    else:
        result["sha256"] = sha256(path)
    return result


def source_hashes(source: Path, *, expected: dict | None = None) -> dict:
    files = list((source / "geoae").rglob("*.py"))
    files += [path for path in (source / "experiments/nonlinear_decoders").rglob("*")
              if path.is_file() and path.suffix in (".py", ".sbatch")]
    files += list((source / "experiments/nonlinear_decoders/configs").rglob("*.yaml"))
    files += [config_path(path, base=source) for path in REFERENCE_CONFIGS.values()
              if config_path(path, base=source).is_file()]
    files += [source / n for n in ("pyproject.toml", "uv.lock", "experiments/config_paths.json",
                                  "geoae/baselines/matched_topk/_reference/SOURCE.json")
              if (source / n).is_file()]
    return {str(p.relative_to(source)): sha256(p) for p in sorted(set(files))
            if p.suffix != ".json" or expected is None or str(p.relative_to(source)) in expected}


def assert_snapshot_imports(source: Path):
    sys.path.insert(0, str(source))
    import geoae
    from geoae import config, model
    for module in (geoae, config, model):
        if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Imported outside source snapshot: {module.__file__}")


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text())
    if "layer" in profile and "target_layer" not in profile:
        profile["target_layer"] = profile["layer"]
    actual = tuple(profile[k] for k in ("target_layer", "hidden_size", "latent_dim"))
    if MODELS.get(profile["model_name"]) != actual:
        raise ValueError(f"Unsupported model/layer/dimensions: {profile['model_name']}, {actual}")
    for key in ("activations_dir", "token_bias_table", "linear_checkpoint"):
        if key not in profile["assets"]:
            raise ValueError(f"Profile assets missing {key}")
    if profile.get("n_clusters", 2000) != 2000 or profile.get("epochs", 50) != 50:
        raise ValueError("Matched paper experiments require K2000 and 50 epochs")
    return profile


def npz_shape(path: Path, name: str) -> tuple:
    """Read an NPY member header without materializing a large token table."""
    with zipfile.ZipFile(path) as archive, archive.open(name + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(stream)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(stream)
    return shape


def inspect_dataset(profile: dict) -> dict:
    assets = profile["assets"]
    data, table_path = Path(assets["activations_dir"]), Path(assets["token_bias_table"])
    layer, dim = profile["target_layer"], profile["hidden_size"]
    required = (f"layer_{layer}.npy", "rows_tok.npy", "rows_doc.npy", "rows_pos.npy",
                "docs.npy", "meta.json", f"norm_params_layer{layer}.npz")
    missing = [str(data / n) for n in required if not (data / n).is_file()]
    if missing:
        raise ValueError(f"Missing activation files: {missing}")
    files = list(required) + [n for n in ("rows_src.npy", "rows_norm.npy") if (data / n).is_file()]
    meta = json.loads((data / "meta.json").read_text())
    x = np.load(data / f"layer_{layer}.npy", mmap_mode="r")
    if x.ndim != 2 or x.shape[1] != dim or x.dtype not in (np.float16, np.float32):
        raise ValueError(f"Activation shape/dtype mismatch: {x.shape}, {x.dtype}")
    if meta["model"] != profile["model_name"] or meta["target_layer"] != layer:
        raise ValueError("Activation model/layer mismatch")
    if meta["n_tokens"] != len(x) or meta["hidden_size"] != dim:
        raise ValueError("Activation metadata dimensions differ")
    for n in files:
        if n.startswith("rows_") and np.load(data / n, mmap_mode="r").shape != (len(x),):
            raise ValueError(f"Sidecar row count mismatch: {n}")
    train_rows = int(len(x) * (1 - 0.05))
    steps = train_rows // 32768
    if steps <= 0:
        raise ValueError("Activation cache cannot supply one complete training batch")
    if "rows" in profile and profile["rows"] != len(x):
        raise ValueError("Activation count differs from profile")
    if profile.get("final_step", steps * 50) != steps * 50:
        raise ValueError("Dataset update count differs from the paper checkpoint")
    with np.load(data / f"norm_params_layer{layer}.npz") as norm, np.load(table_path) as table:
        if str(table["model_name"]) != profile["model_name"] or int(table["layer"]) != layer:
            raise ValueError("Token table model/layer mismatch")
        if len(npz_shape(table_path, "table")) != 2 or npz_shape(table_path, "table")[1] != dim:
            raise ValueError("Token table dimensions differ")
        for key in ("mean", "std"):
            if norm[key].shape != (dim,) or not np.isfinite(norm[key]).all():
                raise ValueError(f"Invalid normalization {key}")
            if not np.allclose(norm[key], table[f"norm_{key}"]):
                raise ValueError(f"Token table normalization {key} differs")
        if not (norm["std"] > 0).all():
            raise ValueError("Normalization std must be positive")
        report = json.loads(str(table["report"]))
        if int(report["train_rows"]) != train_rows:
            raise ValueError("Token table training split differs from activation cache")
        tokens = np.load(data / "rows_tok.npy", mmap_mode="r")
        if int(tokens.min()) < 0 or int(tokens.max()) >= int(table["vocab_size"]):
            raise ValueError("Activation token IDs exceed token table vocabulary")
    return {"model": profile["model_name"], "layer": layer, "shape": list(x.shape),
            "dtype": str(x.dtype), "data_dir": str(data.resolve()), "token_table": str(table_path.resolve()),
            "train_rows": train_rows, "val_rows": len(x) - train_rows,
            "steps_per_epoch": steps, "total_steps": steps * 50,
            "files": {n: file_identity(data / n, sampled=n == f"layer_{layer}.npy") for n in files},
            "table_identity": file_identity(table_path), "row_split_limitation": meta.get("val_split_report", {})}


def inspect_baseline(profile: dict, dataset: dict) -> dict:
    import torch
    from geoae.config import Config

    path = Path(profile["assets"]["linear_checkpoint"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    cfg = checkpoint["config"]
    reference = Config.from_yaml(config_path(REFERENCE_CONFIGS[profile["model_name"]], base=SOURCE_ROOT)).to_dict()
    defaults = Config().to_dict()
    # The baseline's activation mount was temporary. Other paths are metadata,
    # while all model, objective, optimizer, and schedule values stay exact.
    for section, values in reference.items():
        for key, expected in values.items():
            if (section, key) in (("data", "activations_dir"), ("train", "checkpoints_dir"),
                                  ("train", "runs_dir"), ("model", "token_bias")):
                continue
            actual = cfg.get(section, {}).get(key, defaults[section][key])
            if canonical_config(actual) != canonical_config(expected):
                raise ValueError(f"Baseline differs from reviewed paper recipe: {section}.{key}")
    expected = {"data": {"target_layer": profile["target_layer"], "batch_size": 32768, "val_frac": 0.05},
                "model": {"hidden_size": profile["hidden_size"], "latent_dim": profile["latent_dim"],
                          "n_clusters": 2000, "nonlinearity": "gelu", "latent_norm": "batch", "metric": "euclidean"},
                "train": {"n_epochs": 50, "seed": 42}}
    for section, values in expected.items():
        for key, value in values.items():
            if cfg[section][key] != value:
                raise ValueError(f"Baseline recipe mismatch: {section}.{key}")
    if cfg["extraction"]["model_name"] != profile["model_name"]:
        raise ValueError("Baseline model differs")
    if cfg["model"].get("decoder_type", "linear") != "linear" or not cfg["model"].get("token_bias"):
        raise ValueError("Reference must be the existing linear token-bypass AE")
    if cfg["model"].get("token_bias_mode", "bypass") != "bypass":
        raise ValueError("Reference must use the encoder token bypass")
    if checkpoint["epoch"] != 50 or checkpoint["step"] != dataset["total_steps"]:
        raise ValueError("Reference is not the completed matched 50-epoch checkpoint")
    if tuple(checkpoint["model_state"]["decoder.weight"].shape) != (profile["hidden_size"], profile["latent_dim"]):
        raise ValueError("Reference decoder dimensions differ")
    with np.load(Path(dataset["data_dir"]) / f"norm_params_layer{profile['target_layer']}.npz") as norm:
        for name in ("mean", "std"):
            if not np.allclose(np.asarray(checkpoint[f"norm_{name}"]), norm[name]):
                raise ValueError(f"Baseline normalization {name} differs from shared cache")
    return {"identity": file_identity(path), "epoch": int(checkpoint["epoch"]),
            "step": int(checkpoint["step"]), "val_mse": float(checkpoint["val_mse"]),
            "config": cfg, "reused": True, "dataset_provenance": checkpoint.get("dataset_provenance")}


def render_recipe(baseline_config: dict, run_dir: Path, dataset: dict, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    cfg = copy.deepcopy(baseline_config)
    cfg["data"]["activations_dir"] = dataset["data_dir"]
    cfg["model"]["token_bias"] = dataset["token_table"]
    cfg["model"].update(decoder_type=arm,
                         decoder_hidden_dim=cfg["model"]["latent_dim"] // (8 if arm == "residual_mlp" else 1),
                         decoder_nonlinearity="gelu", decoder_output_bias=arm == "mlp")
    cfg["train"]["checkpoints_dir"] = str(run_dir / "checkpoints" / arm)
    cfg["train"]["runs_dir"] = str(run_dir / "runs" / arm)
    return cfg


def cpu_preflight() -> dict:
    """Exercise both decoder code paths on small CPU tensors; never read activation values."""
    import torch
    from geoae.model import GeoAE
    report = {"device": "cpu", "activation_rows_loaded": 0, "arms": {}}
    for arm in ARMS:
        torch.manual_seed(42)
        model = GeoAE(hidden_size=8, latent_dim=16, n_clusters=4, nonlinearity="gelu", latent_norm="batch",
                      decoder_type=arm, decoder_hidden_dim=16 if arm == "mlp" else 2,
                      decoder_output_bias=arm == "mlp")
        x = torch.randn(8, 8)
        out = model(x)
        loss = (out.x_hat - x).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError(f"CPU decoder preflight failed: {arm}")
        report["arms"][arm] = {"loss": float(loss.detach()), "input_shape": list(x.shape),
                               "latent_shape": list(out.z.shape), "output_shape": list(out.x_hat.shape)}
    return report


def prepare(profile: dict, run_dir: Path, source: Path, runtime: dict) -> dict:
    if source.resolve() != (run_dir / "source").resolve():
        raise ValueError("Training source must already be frozen at run-dir/source")
    manifest_path = run_dir / "training_manifest.json"
    if manifest_path.exists():
        manifest = validate(run_dir)
        if manifest["profile"] != profile or manifest["runtime"] != runtime:
            raise ValueError("Existing preparation differs from requested profile/runtime")
        return manifest
    assert_snapshot_imports(source)
    from geoae.config import Config
    dataset = inspect_dataset(profile)
    baseline = inspect_baseline(profile, dataset)
    hashes = {}
    for arm in ARMS:
        for parent in ("checkpoints", "runs"):
            path = run_dir / parent / arm
            if path.exists() and any(path.iterdir()):
                raise ValueError(f"Refusing populated output directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        path = run_dir / "configs" / f"{arm}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(render_recipe(baseline["config"], run_dir, dataset, arm), sort_keys=False))
        Config.from_yaml(path)
        hashes[arm] = sha256(path)
    for name in ("logs", "provenance", "locks", "state"):
        (run_dir / name).mkdir(exist_ok=True)
    manifest = {"schema": 1, "created_utc": utc(), "source_root": str(source.resolve()),
                "source_sha256": source_hashes(source), "profile": profile, "dataset": dataset,
                "existing_linear_baseline": baseline, "config_sha256": hashes, "runtime": runtime,
                "arms": list(ARMS), "initialization": "Independent from-scratch training, seed42; no linear retraining",
                "precision": "Reference FP32; no autocast, AMP, or TF32 overrides",
                "cpu_preflight": cpu_preflight()}
    atomic_json(manifest_path, manifest)
    return validate(run_dir)


def validate(run_dir: Path, *, data=True) -> dict:
    manifest = json.loads((run_dir / "training_manifest.json").read_text())
    source = Path(manifest["source_root"])
    if (source.resolve() != (run_dir / "source").resolve() or
            source_hashes(source, expected=manifest["source_sha256"]) != manifest["source_sha256"]):
        raise ValueError("Source snapshot changed since preparation")
    assert_snapshot_imports(source)
    from geoae.config import Config
    for arm in ARMS:
        path = run_dir / "configs" / f"{arm}.yaml"
        expected = render_recipe(manifest["existing_linear_baseline"]["config"], run_dir, manifest["dataset"], arm)
        if (sha256(path) != manifest["config_sha256"][arm] or
                canonical_config(yaml.safe_load(path.read_text())) != canonical_config(expected)):
            raise ValueError(f"Frozen recipe changed: {arm}")
        Config.from_yaml(path)
    if data:
        if inspect_dataset(manifest["profile"]) != manifest["dataset"]:
            raise ValueError("Dataset/table changed since preparation")
        identity = manifest["existing_linear_baseline"]["identity"]
        if file_identity(Path(identity["path"])) != identity:
            raise ValueError("Reused baseline changed since preparation")
    return manifest


def verify_staged_data(manifest: dict, directory: Path):
    for name, identity in manifest["dataset"]["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != identity["size"]:
            raise ValueError(f"Staged activation file missing/size mismatch: {path}")
        current = file_identity(path, sampled="head_tail_64k_sha256" in identity)
        field = "sha256" if "sha256" in identity else "head_tail_64k_sha256"
        if current[field] != identity[field]:
            raise ValueError(f"Staged activation fingerprint mismatch: {path}")


def stage_data(manifest: dict, parent: Path) -> Path:
    """Reusable by the campaign wrapper; retain the source dtype and full-copy hashes."""
    source = Path(manifest["dataset"]["data_dir"])
    needed = sum(v["size"] for v in manifest["dataset"]["files"].values())
    if shutil.disk_usage(parent).free < needed + 1024**3:
        raise ValueError("Insufficient node-local capacity for activation cache plus1GiB")
    dest = parent / "activations"
    dest.mkdir()
    for name in manifest["dataset"]["files"]:
        shutil.copy2(source / name, dest / name)
        if sha256(dest / name) != sha256(source / name):
            raise ValueError(f"Staged full-file checksum mismatch: {name}")
    return dest


def resume_args(run_dir: Path, arm: str, cfg, total_steps: int) -> list | None:
    import torch
    from geoae.training.common import resolve_resume
    folder = run_dir / "checkpoints" / arm
    if not any(folder.iterdir()):
        return []
    path = resolve_resume("latest", folder)
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if canonical_config(ck["config"]) != canonical_config(cfg.to_dict()):
        raise ValueError(f"Resume checkpoint recipe differs: {path}")
    if ck["epoch"] == cfg.train.n_epochs and ck["step"] == total_steps:
        return None
    if ck["epoch"] >= cfg.train.n_epochs or ck["step"] >= total_steps:
        raise ValueError("Resume checkpoint exceeds or contradicts the expected training schedule")
    return ["--resume", "latest"]


def gpu_preflight(cfg, data_dir: Path) -> dict:
    """Two real full FP32 batch updates, including density peaks and the full loss."""
    import torch
    from geoae.training.data import ActivationBuffer, split_batch
    from geoae.losses import total_loss
    from geoae.seeding import seed_everything
    from geoae.training.initialization import density_peaks_select
    from geoae.training.common import (build_model, make_optimizer, tau_schedule, lambda_schedule,
                                   geometry_schedule, zipf_alpha_schedule)
    if platform.machine() not in ("aarch64", "arm64") or not torch.cuda.is_available():
        raise ValueError("DeltaAI training requires ARM64 and CUDA; refusing CPU fallback")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    seed_everything(cfg.train.seed)
    device = torch.device("cuda")
    model = build_model(cfg, device)
    model.train()
    opt = make_optimizer(model, cfg)
    data = ActivationBuffer(data_dir, cfg.data.target_layer, val_frac=cfg.data.val_frac,
                            split="train", return_tokens=True)
    x, tok = split_batch(data[np.arange(cfg.data.batch_size)], device)
    if len(x) != cfg.data.batch_size or x.dtype != torch.float32:
        raise ValueError("GPU preflight requires the complete unchanged FP32 training batch")
    torch.cuda.reset_peak_memory_stats()
    report = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "python": sys.version,
              "executable": sys.executable, "machine": platform.machine(),
              "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
              "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "batch_size": len(x), "precision": str(x.dtype), "steps": []}
    for epoch in (1, cfg.train.full_loss_start_epoch):
        if epoch >= cfg.train.clustering_start_epoch and not bool(model.centroids_initialized.item()):
            with torch.no_grad():
                if cfg.train.density_pool != len(x):
                    raise ValueError("Full-batch preflight density pool differs from recipe")
                z = model.encode(x, tok)
                centers, _ = density_peaks_select(z, model.n_clusters, seed=cfg.train.seed,
                    density_power=cfg.train.density_power, min_sep_frac=cfg.train.peak_min_sep_frac,
                    knn_k=cfg.train.density_knn, refine_k=cfg.train.peak_refine_k)
                model.init_centroids_from_class_means(centers)
                del z, centers
        model.tau = tau_schedule(epoch, cfg)
        model.zipf_alpha = zipf_alpha_schedule(epoch, cfg)
        lc, ls = lambda_schedule(epoch, cfg)
        lv, lcv, lu = geometry_schedule(epoch, cfg)
        torch.cuda.synchronize()
        started = time.monotonic()
        out = model(x, tok)
        losses = total_loss(x=x, x_hat=out.x_hat, z=out.z, centroids=model.centroids.detach(), Q=out.Q,
            lambda_cluster=lc, lambda_sep=ls, metric=model.metric, lambda_var=lv, lambda_cov=lcv, lambda_unif=lu,
            sep_mode=cfg.loss.sep_mode, sep_margin=cfg.loss.sep_margin, var_gamma=cfg.loss.var_gamma)
        if not torch.isfinite(losses["loss"]):
            raise ValueError(f"Nonfinite full-batch loss: epoch{epoch}")
        opt.zero_grad()
        losses["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip, error_if_nonfinite=True)
        opt.step()
        model.post_step()
        if bool(model.centroids_initialized.item()):
            model.update_centroids_ema(out.z.detach(), out.Q.detach(), move=not cfg.train.freeze_centroids)
        torch.cuda.synchronize()
        step = {"schedule_epoch": epoch, "seconds": time.monotonic()-started,
                "loss": float(losses["loss"].detach()), "grad_norm": float(norm),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        report["steps"].append(step)
        print("FULL-BATCH GPU PREFLIGHT " + json.dumps(step), flush=True)
        del out, losses
    del model, opt, data, x, tok
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run(run_dir: Path, arm: str, activations_dir: Path, *, preflight_only=False):
    with (run_dir / "locks" / f"{arm}_training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = validate(run_dir)
        from geoae.config import Config
        cfg = Config.from_yaml(run_dir / "configs" / f"{arm}.yaml")
        resume = resume_args(run_dir, arm, cfg, manifest["dataset"]["total_steps"])
        state = run_dir / "state" / f"{arm}_training.json"
        if resume is None:
            atomic_json(state, {"state": "complete", "arm": arm, "verified_utc": utc(),
                                "final_checkpoint": str(run_dir / "checkpoints" / arm / f"step_{manifest['dataset']['total_steps']:07d}.pt")})
            print(f"{arm} already completed all50epochs", flush=True)
            return
        verify_staged_data(manifest, activations_dir)
        atomic_json(state, {"state": "preflight", "arm": arm, "started_utc": utc(), "job_id": os.getenv("SLURM_JOB_ID")})
        try:
            report = gpu_preflight(cfg, activations_dir)
        except Exception as error:
            atomic_json(state, {"state": "failed", "stage": "preflight", "arm": arm,
                                "finished_utc": utc(), "error": str(error)})
            raise
        report.update({"validated_utc": utc(), "hostname": socket.gethostname(), "job_id": os.getenv("SLURM_JOB_ID"),
                       "arm": arm, "manifest_sha256": sha256(run_dir / "training_manifest.json"),
                       "source_sha256": manifest["source_sha256"], "local_data": str(activations_dir),
                       "resume_args": resume, "loaded_modules": os.getenv("LOADEDMODULES", "").split(":")})
        provenance = run_dir / "provenance" / f"{arm}_{os.getenv('SLURM_JOB_ID', 'local')}.json"
        atomic_json(provenance, report)
        if preflight_only:
            return
        env = dict(os.environ)
        env["GEOAE_TRAIN_ACTIVATIONS_DIR"] = str(activations_dir)
        env["GEOAE_DATASET_PROVENANCE"] = json.dumps({"source": manifest["dataset"], "runtime_provenance": str(provenance)})
        atomic_json(state, {"state": "running", "arm": arm, "started_utc": utc(), "job_id": os.getenv("SLURM_JOB_ID"),
                            "resume_args": resume, "preflight": str(provenance)})
        command = [sys.executable, "-u", "-m", "geoae.training.mse", "--config", str(run_dir / "configs" / f"{arm}.yaml"),
                   "--no_wandb", *resume]
        try:
            subprocess.run(command, cwd=manifest["source_root"], env=env, check=True)
            if resume_args(run_dir, arm, cfg, manifest["dataset"]["total_steps"]) is not None:
                raise ValueError("Training exited before producing the matched final checkpoint")
        except Exception as error:
            atomic_json(state, {"state": "failed", "arm": arm, "finished_utc": utc(), "error": str(error)})
            raise
        atomic_json(state, {"state": "complete", "arm": arm, "finished_utc": utc(), "job_id": os.getenv("SLURM_JOB_ID"),
                            "final_step": manifest["dataset"]["total_steps"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--profile", type=Path, required=True)
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--source-root", type=Path)
    prep.add_argument("--python", type=Path, default=Path(sys.executable))
    prep.add_argument("--module", action="append", default=[])
    check = sub.add_parser("validate")
    check.add_argument("--run-dir", type=Path, required=True)
    check.add_argument("--skip-data", action="store_true")
    train = sub.add_parser("run")
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--arm", choices=ARMS, required=True)
    train.add_argument("--activations-dir", type=Path, required=True, help="Persistent staged cache; never removed/restaged")
    train.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.command == "prepare":
        runtime = {"python": str(args.python.absolute()), "modules": args.module}
        if not os.access(runtime["python"], os.X_OK):
            raise ValueError("Runtime interpreter is not executable")
        result = prepare(load_profile(args.profile), root, (args.source_root or root / "source").resolve(), runtime)
        print(json.dumps({"prepared": str(root), "profile": result["profile"]["model_name"],
                          "steps_per_epoch": result["dataset"]["steps_per_epoch"], "total_steps": result["dataset"]["total_steps"]}))
    elif args.command == "validate":
        validate(root, data=not args.skip_data)
        print(f"Verified preparation: {root}")
    else:
        run(root, args.arm, args.activations_dir.resolve(), preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
