#!/usr/bin/env python3
"""Train matched Llama/Gemma middle/final-layer AEs and run detailed evaluations.

Stages run serially and resume from verified artifacts. --dry-run prints commands
without importing torch, downloading data, training, or writing files.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
import hashlib
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.depth.evaluate import PROFILES, CANONICAL_ARMS, NEW_ARMS
from experiments.gpt2.evaluate import checkpoint_for, guard_checkpoint, write_state, profile_paths


def arm_paths(profile: dict, run_name: str) -> dict:
    return profile_paths(profile["layer"], argparse.Namespace(profile=profile, run_name=run_name))


def config_paths(arm: str) -> tuple[Path, Path]:
    profile = PROFILES[arm]
    return Path(profile["extract_config"]), Path(profile["train_config"])


def check_table(profile: dict) -> Path:
    """Reject another layer's table or a table from different train statistics."""
    import numpy as np
    layer = profile["layer"]
    dump = Path(profile["activations_dir"])
    path = arm_paths(profile, "unused")["table"]
    with np.load(path, allow_pickle=False) as table, np.load(dump / f"norm_params_layer{layer}.npz") as norm:
        if int(table["layer"]) != layer or str(table["model_name"]) != profile["model_name"]:
            raise ValueError(f"Wrong model/layer in {path}")
        if Path(str(table["activations"])).resolve() != dump.resolve():
            raise ValueError(f"Token table was built from another activation dump: {path}")
        values, ids = table["table"], table["token_ids"]
        if values.shape != (len(ids), profile["hidden_size"]) or not len(ids) or not np.isfinite(values).all():
            raise ValueError(f"Empty, nonfinite, or wrong-shaped token table: {path}")
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Invalid token IDs in {path}")
        if ids.min() < 0 or ids.max() >= int(table["vocab_size"]):
            raise ValueError(f"Token IDs exceed the table vocabulary: {path}")
        if float(table["shrink_k"]) != 30.0 or int(table["min_count"]) != 10:
            raise ValueError(f"Wrong token-bias recipe in {path}")
        report = json.loads(str(table["report"]))
        rows = json.loads((dump / "meta.json").read_text())["n_tokens"]
        if report.get("train_rows") != int(rows * 0.95):
            raise ValueError(f"Token table was not built on the full 95% train split: {path}")
        for key in ("mean", "std"):
            if norm[key].shape != (profile["hidden_size"],):
                raise ValueError(f"Wrong normalization {key} shape in {dump}")
            if not np.array_equal(table[f"norm_{key}"], norm[key]):
                raise ValueError(f"Token table and dump {key} disagree: {path}")
        if not np.isfinite(norm["mean"]).all() or not np.isfinite(norm["std"]).all() or (norm["std"] <= 0).any():
            raise ValueError(f"Invalid normalization in {dump}")
    return path


def checked_checkpoint(path: Path, training_config: Path, expected_provenance: dict | None = None) -> dict:
    """Validate the full training recipe before reusing or resuming a checkpoint."""
    import torch
    from experiments.depth.config import CampaignConfig
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = CampaignConfig.from_yaml(training_config).to_dict()
    comparable = {section: dict(values) for section, values in saved["config"].items()}
    # These controls belong to a separate experiment and are excluded from the
    # campaign schema. Accept only inert defaults in any intervening checkpoint.
    if comparable.get("train", {}).get("centroid_init") == "dpc":
        legacy_defaults = {"lloyd_rows": 1_583_334, "lloyd_iters": 100, "freeze_centroids": False}
        for key, value in legacy_defaults.items():
            if key in comparable["train"]:
                if comparable["train"][key] != value:
                    raise ValueError(f"Checkpoint training configuration differs from {training_config}")
                comparable["train"].pop(key)
    if comparable != expected:
        raise ValueError(f"Checkpoint training configuration differs from {training_config}")
    recorded = saved.get("dataset_provenance")
    if recorded is not None:
        from experiments.depth.dataset import SOURCE_FINGERPRINT
        if (recorded.get("source_fingerprint") != SOURCE_FINGERPRINT
                or recorded.get("model") != expected["extraction"]["model_name"]
                or recorded.get("layers") != [expected["data"]["target_layer"]]):
            raise ValueError(f"Checkpoint was trained with a different source recipe: {path}")
        if expected_provenance is not None:
            for key in ("source_fingerprint", "model", "layers", "corpus_manifest_sha256"):
                if recorded.get(key) != expected_provenance.get(key):
                    raise ValueError(f"Checkpoint dataset {key} differs from the current frozen dump: {path}")
    epoch, step = int(saved["epoch"]), int(saved["step"])
    if not 0 <= epoch <= expected["train"]["n_epochs"] or step < 0:
        raise ValueError(f"Invalid checkpoint progress epoch={epoch}, step={step}: {path}")
    return saved


def check_training(path: Path, profile: dict, training_config: Path, expected_provenance: dict | None = None) -> dict:
    guard_checkpoint(path, profile["layer"], profile)
    saved = checked_checkpoint(path, training_config, expected_provenance)
    return {"checkpoint": str(path), "epoch": saved["epoch"], "step": saved["step"],
            "model": profile["model_name"], "layer": profile["layer"],
            "latent_dim": profile["latent_dim"], "ae_mode": profile.get("mode", "bypass"),
            "token_bias_mode": "bypass" if profile.get("mode", "bypass") == "bypass" else "none",
            "dataset_provenance": saved.get("dataset_provenance"),
            "config": str(training_config), "written_at": datetime.now(timezone.utc).isoformat()}


def ensure_dump_location(profile: dict, data_root: Path | None) -> None:
    if data_root is None:
        return
    dump = Path(profile["activations_dir"])
    target = data_root.resolve() / "activations" / dump.name
    if dump.exists() or dump.is_symlink():
        if dump.resolve() != target:
            raise ValueError(f"{dump} already exists outside the requested data root {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    dump.symlink_to(target, target_is_directory=True)


def stage_dump(profile: dict, staging_root: Path) -> Path:
    """Copy immutable arrays to owned RAM storage while preserving source identity."""
    source = Path(profile["activations_dir"])
    layer = profile["layer"]
    files = [source / f"layer_{layer}.npy", source / "meta.json", source / "docs.npy",
             source / f"norm_params_layer{layer}.npz"]
    files += [source / f"rows_{suffix}.npy" for suffix in ("doc", "pos", "tok", "src", "norm")]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"Cannot stage an incomplete dump: {missing}")
    identity = {"source": str(source.resolve()), "layer": layer,
                "files": {path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                          for path in files},
                "meta_sha256": hashlib.sha256((source / "meta.json").read_bytes()).hexdigest()}
    staging_root.mkdir(parents=True, exist_ok=True)
    target = staging_root / source.name
    marker = target / ".stage_complete.json"
    if marker.is_file():
        old = json.loads(marker.read_text())
        if old != identity:
            raise ValueError(f"Staged dump has another source identity: {target}")
        if all((target / path.name).is_file() and (target / path.name).stat().st_size == path.stat().st_size
               for path in files):
            return target
    elif target.exists():
        # Interrupted copies have an ownership marker; never remove unrelated data.
        owner = target / ".stage_owner.json"
        if not owner.is_file() or json.loads(owner.read_text()) != identity:
            raise ValueError(f"Refusing to overwrite an unowned staging directory: {target}")
    need = sum(path.stat().st_size for path in files)
    existing = sum((target / path.name).stat().st_size for path in files if (target / path.name).is_file())
    if shutil.disk_usage(staging_root).free < max(0, need - existing) + 2 * 1024**3:
        raise ValueError(f"Insufficient node-local space to stage {need / 1024**3:.1f} GiB from {source}")
    target.mkdir(parents=True, exist_ok=True)
    (target / ".stage_owner.json").write_text(json.dumps(identity) + "\n")
    print(f"[stage] {source} → {target} ({need / 1024**3:.1f} GiB)", flush=True)
    for path in files:
        shutil.copy2(path, target / path.name)
    marker.write_text(json.dumps(identity) + "\n")
    return target


def select_stages(raw: str) -> list[str]:
    order = ["corpus", "extract", "table", "train", "evaluate", "report"]
    selected = set()
    for name in raw.replace(",", " ").split():
        if name == "all":
            selected.update(order)
        elif name == "training":
            selected.update(order[:4])
        elif name in order:
            selected.add(name)
        else:
            raise ValueError(f"Unknown stage {name!r}; choose all, training, or {', '.join(order)}")
    if not selected:
        raise ValueError("Choose at least one stage")
    return [name for name in order if name in selected]


def run_arm(arm: str, args) -> bool:
    profile = PROFILES[arm]
    layer = profile["layer"]
    extraction, training = config_paths(arm)
    dump = Path(profile["activations_dir"])
    directory = Path("checkpoints") / profile["family"] / f"layer{layer}" / profile["ckdir"]
    paths = arm_paths(profile, args.run_name)
    table = paths["table"]
    stages = select_stages(args.steps)
    out = paths["out"]
    state = {"arm": arm, "model": profile["model_name"], "layer": layer,
             "extraction_config": str(extraction), "training_config": str(training),
             "requested": stages, "steps": {}, "status": "running", "ae_mode": profile.get("mode", "bypass")}
    state_path = out / "trainer_state.json"
    py = [sys.executable, "-u"]
    commands = {
        "corpus": [*py, "-m", "geoae.extraction.corpus", "--config", str(extraction), "--workers", "3"],
        "extract": [*py, "-m", "geoae.extraction.prefix", "--config", str(extraction)],
        "table": [*py, "-m", "geoae.token_bias", "--activations_dir", str(dump), "--layer", str(layer),
                  "--val_frac", "0.05", "--out", str(table)],
        "train": [*py, "-m", "geoae.training.mse", "--config", str(training), "--no_wandb", "--campaign_dpc"],
        "evaluate": [*py, "experiments/depth/evaluate.py", arm, "--run-name", args.run_name,
                     "--steps", args.eval_steps, "--judges", args.judges],
        "report": [*py, "experiments/depth/report.py", "--arm", arm, "--run-name", args.run_name],
    }
    if args.force_evals:
        commands["evaluate"].append("--force")
    if args.dry_run:
        print(f"[trainer] {arm}: {profile['model_name']} layer {layer}, {profile['hidden_size']} → {profile['latent_dim']}")
        if args.data_root:
            print(f"[dry] dump {dump} → {args.data_root.resolve() / 'activations' / dump.name}")
        if getattr(args, "stage_dump_root", None):
            print(f"[dry] stage {dump} → {args.stage_dump_root / dump.name}")
        for stage in stages:
            print(f"[dry] {stage}: {shlex.join(commands[stage])}")
        return True

    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        lock_paths = [out / ".training.lock"]
        if any(s in stages for s in ("corpus", "extract", "table", "train")):
            directory.mkdir(parents=True, exist_ok=True)
            lock_paths.append(directory / ".midlayer.lock")
        for lock_path in lock_paths:
            lock = locks.enter_context(lock_path.open("a+"))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another trainer is already using {lock_path.parent}")
        write_state(state_path, state)
        try:
            if any(s in stages for s in ("extract", "table", "train")):
                ensure_dump_location(profile, args.data_root)
            from experiments.depth.dataset import check_dump, check_corpus, validate_extraction_config
            provenance = validate_extraction_config(extraction)
            state["dataset_provenance"] = provenance
            for stage in stages:
                command = commands[stage]
                skip = False
                environment = os.environ.copy()
                if stage == "corpus":
                    try:
                        check_corpus(extraction)
                        skip = True
                    except FileNotFoundError:
                        pass
                    # An incomplete compatible corpus can be completed incrementally.
                    except ValueError:
                        pass
                if stage in ("extract", "table", "train"):
                    if stage != "extract" or (dump / "meta.json").exists():
                        provenance = check_dump(extraction)["dataset_provenance"]
                        state["dataset_provenance"] = provenance
                    skip = stage == "extract" and (dump / "meta.json").exists()
                if stage == "table" and table.exists():
                    check_table(profile)
                    skip = True
                if stage == "train":
                    if profile.get("mode", "bypass") == "bypass":
                        check_table(profile)
                    try:
                        checkpoint = checkpoint_for(layer, profile=profile)
                    except FileNotFoundError:
                        checkpoint = None
                    if checkpoint:
                        epoch = checked_checkpoint(checkpoint, training, provenance)["epoch"]
                        if epoch == profile["n_epochs"]:
                            marker = check_training(checkpoint, profile, training, provenance)
                            write_state(directory / "training_complete.json", marker)
                            skip = True
                        else:
                            command = [*command, "--resume", "latest"]
                if stage in ("train", "evaluate") and not skip:
                    provenance = check_dump(extraction)["dataset_provenance"]
                    state["dataset_provenance"] = provenance
                    staged = stage_dump(profile, args.stage_dump_root) if getattr(args, "stage_dump_root", None) else dump
                    if stage == "train":
                        environment["GEOAE_TRAIN_ACTIVATIONS_DIR"] = str(staged.resolve())
                        environment["GEOAE_DATASET_PROVENANCE"] = json.dumps(provenance, sort_keys=True)
                    else:
                        environment["GEOAE_EVAL_ACTIVATIONS_DIR"] = str(staged.resolve())
                    state["runtime_activations_dir"] = str(staged.resolve())
                if skip:
                    print(f"[skip] {stage}: verified complete", flush=True)
                else:
                    print(f"[run] {arm} {stage}: {shlex.join(command)}", flush=True)
                    state["steps"][stage] = {"status": "running", "command": command}
                    write_state(state_path, state)
                    subprocess.run(command, check=True, env=environment)
                    if stage == "corpus":
                        check_corpus(extraction)
                    elif stage == "extract":
                        check_dump(extraction)
                    elif stage == "table":
                        check_table(profile)
                    elif stage == "train":
                        marker = check_training(checkpoint_for(layer, profile=profile), profile, training, provenance)
                        write_state(directory / "training_complete.json", marker)
                state["steps"][stage] = {"status": "complete", "reused": skip, "command": command}
                if stage == "evaluate":
                    ev_state = json.loads((out / "pipeline_state.json").read_text())
                    state["steps"][stage]["status"] = ev_state["status"]
                write_state(state_path, state)
            state["status"] = ("complete" if all(s["status"] == "complete" for s in state["steps"].values()) else "partial")
            write_state(state_path, state)
            return True
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
            state["status"] = "failed"
            state["error"] = str(exc)
            write_state(state_path, state)
            print(f"[fail] {arm}: {exc}", file=sys.stderr)
            return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=(*PROFILES, "all", "new"), default="new")
    parser.add_argument("--steps", default="all", help="all, training, or corpus/extract/table/train/evaluate/report")
    parser.add_argument("--eval-steps", default="whole", help="Detailed evaluation groups; see evaluate.py --help")
    parser.add_argument("--run-name", default="midlayer_run")
    parser.add_argument("--judges", choices=("auto", "skip", "required"), default="auto")
    parser.add_argument("--force-evals", action="store_true")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("DELTA_DATA"), help="Bulk-data root for new dump symlinks")
    parser.add_argument("--stage-dump-root", type=Path, default=None, help="Owned node-local directory for verified dump copies")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        select_stages(args.steps)
        if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
            parser.error("--run-name must be one directory name")
        os.chdir(ROOT)
        arms = list(CANONICAL_ARMS if args.arm == "all" else NEW_ARMS if args.arm == "new" else [args.arm])
        results = [run_arm(arm, args) for arm in arms]
        return 0 if all(results) else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[abort] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
