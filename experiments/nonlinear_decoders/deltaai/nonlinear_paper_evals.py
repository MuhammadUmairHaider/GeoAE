#!/usr/bin/env python3
"""Prepare and run one model's full paper suite inside an existing GPU allocation.

This module never submits jobs. The campaign launcher trains an arm and invokes
``run`` with the same verified node-local activation directory. Evaluation
source, plans, inputs, final checkpoints, and successful outputs are identified
independently, so a restart can retain completed work without accepting stale
results from another model or checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
import re

from geoae.workflow.execute import directory_bytes, temporary_staging_directory
from geoae.workflow.snapshot import copy_evaluation_sources, manuscript_path

from nonlinear_paper_plan import build_steps
from qwen_paper_evals import (ARMS, GROUPS, CORE_OVERLAYS as CORE_OVERLAYS, RUNTIME_PACKAGES,
                              atomic_json, digest, tree_hashes, verify_assets,
                              runtime_environment, utc, paper_checkout, verify_training_core,
                              validate_outputs as validate_standard_outputs)


def evaluation_root(args) -> Path:
    """Allow a recovery snapshot to retain the original training/results paths."""
    override = getattr(args, "evaluation_dir", None)
    return override.resolve() if override is not None else args.run_dir.resolve() / "evaluation"


def profile_identity(profile: dict) -> dict:
    keys = ("model_name", "target_layer", "hidden_size", "latent_dim",
            "n_clusters", "epochs", "final_step")
    result = {key: profile[key] for key in keys}
    for key in keys[1:]:
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"Invalid model profile field {key}: {result[key]!r}")
    return result


def checkpoint_path(manifest: dict, arm: str) -> Path:
    if arm not in ARMS:
        raise ValueError(f"Unknown decoder arm: {arm}")
    step = manifest["profile"]["final_step"]
    return Path(manifest["training_run_dir"]) / "checkpoints" / arm / f"step_{step:07d}.pt"


def verify_checkpoint(path: Path, arm: str, profile: dict) -> dict:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    cfg, expected = ck["config"], profile_identity(profile)
    model = cfg["model"]
    if model.get("decoder_type", "linear") != arm:
        raise ValueError(f"Wrong decoder architecture: {path}")
    if tuple(model[key] for key in ("hidden_size", "latent_dim", "n_clusters")) != tuple(
            expected[key] for key in ("hidden_size", "latent_dim", "n_clusters")):
        raise ValueError(f"Wrong model dimensions: {path}")
    if cfg["extraction"]["model_name"] != expected["model_name"]:
        raise ValueError(f"Wrong frozen language model: {path}")
    if cfg["data"]["target_layer"] != expected["target_layer"]:
        raise ValueError(f"Wrong extraction layer: {path}")
    if (ck["epoch"], ck["step"]) != (expected["epochs"], expected["final_step"]):
        raise ValueError(f"Expected final epoch {expected['epochs']}, step {expected['final_step']}: {path}")
    if not bool(ck["model_state"]["centroids_initialized"]):
        raise ValueError(f"Centroids not initialized: {path}")
    return {"path": str(path), "sha256": digest(path), "epoch": int(ck["epoch"]),
            "step": int(ck["step"]), "decoder_type": arm,
            "val_metrics": ck.get("val_metrics", {}), "val_mse": ck.get("val_mse")}


def asset_identities(assets: dict) -> dict:
    """Hash small immutable inputs; record size/mtime for large staged arrays."""
    from qwen_paper_evals import asset_identities as identities
    nonpaths = {"model_name", "target_layer", "hidden_size", "latent_dim", "n_clusters", "sae"}
    return identities({key: value for key, value in assets.items() if key not in nonpaths})


def prepare(args) -> None:
    paper = paper_checkout(getattr(args, "paper_repo", None))
    training, root = args.run_dir.resolve(), args.run_dir.resolve() / "evaluation"
    profile = json.loads(args.profile.read_text())
    profile_identity(profile)
    tm_path = training / "training_manifest.json"
    tm = json.loads(tm_path.read_text())
    if profile_identity(tm["profile"]) != profile_identity(profile):
        raise ValueError("Evaluation profile differs from the frozen training profile")
    for key in ("linear_checkpoint", "activations_dir", "token_bias_table"):
        if profile["assets"][key] != tm["profile"]["assets"][key]:
            raise ValueError(f"Evaluation {key} differs from the frozen training input")
    runtime = dict(tm["runtime"])
    if args.python:
        runtime["python"] = str(args.python.absolute())
    if not Path(runtime["python"]).is_file():
        raise ValueError("The evaluation Python executable does not exist")
    if root.exists():
        raise ValueError(f"Evaluation directory already exists: {root}")
    # Validate the complete model-specific plan before creating a snapshot.
    plans = {arm: build_steps(profile, training / "checkpoints" / arm /
                              f"step_{profile['final_step']:07d}.pt", root / "results" / arm,
                              runtime["python"]) for arm in ARMS}
    identities = asset_identities(profile["assets"])
    source = root / "source"
    paper_path = manuscript_path(paper)
    training_core = verify_training_core(tm, paper)
    source.mkdir(parents=True)
    canonical = copy_evaluation_sources(paper, source)
    verify_training_core(tm, source)
    overlays = {}
    compatibility = {}
    manifest = {"prepared_utc": utc(), "training_run_dir": str(training),
                "training_manifest_sha256": digest(tm_path), "source_root": str(source),
                "profile": profile, "runtime": runtime, "assets": profile["assets"],
                "plans": plans, "groups": list(GROUPS),
                "paper_origin": str(paper_path),
                "paper_sha256": digest(paper_path),
                "canonical_source_sha256": canonical, "nonlinear_overlays_sha256": overlays,
                "training_core_sha256": training_core,
                "nonlinear_evaluator_patches": compatibility,
                "runtime_package_versions": {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES},
                "source_sha256": tree_hashes(source),
                "asset_identities": identities,
                "activation_files": list(tm["dataset"]["files"]),
                "protocol": {"groups_sequential_in_training_allocation": True,
                             "checkpoint": "exact final epoch and step from model profile",
                             "controls": "fresh jointly correct examples; Gemma includes SAE all features and top one",
                             "linear_training": "reuse original checkpoint without retraining"}}
    for directory in ("logs", "locks", "state", "provenance"):
        (root / directory).mkdir()
    atomic_json(root / "manifest.json", manifest)
    print(f"Prepared {profile['model_name']}: {len(plans[ARMS[0]])} evaluators per arm", flush=True)


def load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if tree_hashes(Path(manifest["source_root"]), expected=manifest["source_sha256"]) != manifest["source_sha256"]:
        raise ValueError("Evaluation source changed since preparation")
    training = Path(manifest["training_run_dir"]) / "training_manifest.json"
    if digest(training) != manifest["training_manifest_sha256"]:
        raise ValueError("Training manifest changed since evaluation preparation")
    verify_assets(manifest["asset_identities"])
    for name, version in manifest["runtime_package_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Evaluation package changed since preparation: {name}")
    return manifest


def command_sha256(step: dict) -> str:
    return hashlib.sha256(json.dumps(step["command"], separators=(",", ":")).encode()).hexdigest()


def validate_outputs(step: dict) -> None:
    validate_standard_outputs(step)
    if step["name"] == "probe_jigsaw":
        data = json.loads(Path(step["outputs"][0]).read_text())
        missing = {"toxicity", "toxicity_pattern"} - set(data.get("rungs", {}))
        if missing:
            raise ValueError(f"Missing Jigsaw diagnostics: {sorted(missing)}")
    if step["name"].startswith("control_"):
        data = json.loads(Path(step["outputs"][0]).read_text())
        if not data["concepts"]:
            raise ValueError("Control evaluator produced no scored concepts")
        command = step["command"]
        spaces = command[command.index("--substrates") + 1].split(",")
        for concept, record in data["concepts"].items():
            for space in spaces:
                if space not in record.get("ref", {}) or any(
                        f"{space}_st_global_a{alpha}" not in record for alpha in ("0.4", "0.5")):
                    raise ValueError(f"Incomplete control concept {concept}, space {space}")


def completed(step: dict, record: dict | None, checkpoint_sha: str) -> bool:
    if not record or record.get("state") != "complete" or record.get("checkpoint_sha256") != checkpoint_sha:
        return False
    if record.get("command_sha256") != command_sha256(step):
        return False
    try:
        validate_outputs(step)
        return set(record["output_sha256"]) == set(step["outputs"]) and all(
            digest(Path(name)) == value for name, value in record["output_sha256"].items())
    except (OSError, ValueError, KeyError):
        return False


def verify_staged_activations(manifest: dict, staged: Path) -> None:
    original = Path(manifest["assets"]["activations_dir"])
    declared = manifest.get("activation_files")
    if declared is None and manifest.get("training_run_dir"):
        training = json.loads((Path(manifest["training_run_dir"]) / "training_manifest.json").read_text())
        declared = list(training["dataset"]["files"])
    sources = [original / name for name in declared] if declared is not None else original.iterdir()
    for source in sources:
        if not source.is_file() or source.suffix not in (".npy", ".npz", ".json"):
            continue
        if source.name == "deltaai_transfer_validation.json":
            continue
        target = staged / source.name
        identity = manifest["asset_identities"].get(str(source), {})
        if not target.is_file() or target.stat().st_size != source.stat().st_size:
            raise ValueError(f"Wrong node-local activation copy: {target}")
        if "sha256" in identity and digest(target) != identity["sha256"]:
            raise ValueError(f"Wrong node-local activation sidecar hash: {target}")


def help_command(step: dict, python: str) -> list[str]:
    command = step["command"]
    index = command.index(python)
    prefix = command[index:index + 2]
    if prefix[1] != "-u":
        raise ValueError("Expected an explicit unbuffered evaluation interpreter")
    if command[index + 2] == "-m":
        return prefix + command[index + 2:index + 4] + ["--help"]
    return prefix + [command[index + 2], "--help"]


def cpu_preflight(args) -> dict:
    """Validate offline inputs and command parsers without loading LM tensors."""
    root = evaluation_root(args)
    manifest = load_manifest(root)
    source, profile, assets = Path(manifest["source_root"]), manifest["profile"], manifest["assets"]
    env = runtime_environment(manifest)
    report = {"started_utc": utc(), "device": "cpu", "lm_weight_tensors_loaded": False,
              "activation_values_loaded": False, "checks": {}}
    report_path = root / "cpu_preflight.json"
    os.environ.update(env)
    sys.path.insert(0, str(source))
    import numpy as np
    from transformers import AutoConfig
    from geoae.checkpoint import load_tokenizer
    from transformers.utils import cached_file
    from datasets import load_dataset
    from nonlinear_paper_training import npz_shape
    try:
        commands = {}
        for step in manifest["plans"][ARMS[0]]:
            command = help_command(step, manifest["runtime"]["python"])
            key = tuple(command)
            if key not in commands:
                result = subprocess.run(command, cwd=source, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, timeout=180)
                if result.returncode:
                    raise ValueError(f"Evaluator import/help failed: {shlex.join(command)}\n{result.stdout[-6000:]}")
                commands[key] = result.stdout
            options = {value for value in step["command"] if value.startswith("--")}
            missing = [option for option in options if not re.search(re.escape(option) + r"(?=[\s,=\]]|$)", commands[key])]
            if missing:
                raise ValueError(f"Unsupported options in {step['name']}: {missing}")
            report["checks"][step["name"] + "_parser"] = "passed"
            print(f"[CPU preflight] {step['name']} imports and options passed", flush=True)
        config = AutoConfig.from_pretrained(profile["model_name"], local_files_only=True)
        text = getattr(config, "text_config", config)
        if text.hidden_size != profile["hidden_size"] or text.num_hidden_layers - 1 != profile["target_layer"]:
            raise ValueError("Cached LM config has the wrong final-layer dimensions")
        tokenizer = load_tokenizer(profile["model_name"], local_files_only=True)
        encoded = tokenizer("A short offline tokenizer test.")["input_ids"]
        if not encoded:
            raise ValueError("Cached tokenizer returned no tokens")
        index = cached_file(profile["model_name"], "model.safetensors.index.json", local_files_only=True,
                            _raise_exceptions_for_missing_entries=False)
        if index:
            names = set(json.loads(Path(index).read_text())["weight_map"].values())
        else:
            names = {"model.safetensors"}
        for name in sorted(names):
            path = cached_file(profile["model_name"], name, local_files_only=True)
            if not path or not Path(path).is_file() or Path(path).stat().st_size == 0:
                raise ValueError(f"Missing cached LM shard {name}")
        report["checks"]["offline_model"] = {"model_name": profile["model_name"], "shards": len(names),
                                               "hidden_size": text.hidden_size, "tokenizer_test_tokens": len(encoded)}
        for key in ("balanced_kmeans", "balanced_kmeans_tokmean", "plain_kmeans", "plain_kmeans_tokmean"):
            shape = npz_shape(Path(assets[key]), "centroids")
            if shape != (profile["n_clusters"], profile["hidden_size"]):
                raise ValueError(f"Wrong codebook dimensions for {key}: {shape}")
            report["checks"][key] = list(shape)
        from geoae.analysis.concept_probe import LADDER
        from nonlinear_paper_plan import PAPER_RUNG_NAMES
        for rung, filename, *_ in LADDER:
            if rung not in set(PAPER_RUNG_NAMES) | {"toxicity_pattern"}:
                continue
            path = Path(assets["probe_cache"]) / filename
            with np.load(path, allow_pickle=False) as data:
                report["checks"]["probe_" + rung] = list(data.files)
        from geoae.evaluation import cohorts as shared
        control_steps = [step for step in manifest["plans"][ARMS[0]] if step["group"] == "controls"]
        datasets = [step["command"][step["command"].index("--dataset") + 1] for step in control_steps]
        for dataset in datasets:
            cfg = shared.set_dataset(dataset)
            data = (load_dataset("parquet", data_files=cfg["local_parquet"], split="train")
                    if cfg.get("local_parquet") else load_dataset(cfg["dataset_name"], split=cfg["dataset_split"]))
            report["checks"]["dataset_" + dataset] = len(data)
        for name, config, split in (("cais/mmlu", "all", "test"), ("Salesforce/wikitext", "wikitext-2-raw-v1", "test")):
            report["checks"][name] = len(load_dataset(name, config, split=split))
        if assets.get("sae"):
            from safetensors import safe_open
            from geoae.baselines.gemma_scope import _validate_config
            directory = Path(assets["sae_dir"])
            config = json.loads((directory / "config.json").read_text())
            _validate_config(config, profile["model_name"], profile["target_layer"])
            with safe_open(str(directory / "params.safetensors"), framework="pt", device="cpu") as tensors:
                shapes = {name: tensors.get_slice(name).get_shape() for name in tensors.keys()}
            expected = {"w_enc": [profile["hidden_size"], config["width"]],
                        "w_dec": [config["width"], profile["hidden_size"]],
                        "b_enc": [config["width"]], "b_dec": [profile["hidden_size"]], "threshold": [config["width"]]}
            if shapes != expected:
                raise ValueError(f"Gemma SAE header dimensions differ: {shapes}")
            if npz_shape(Path(assets["sae_kmeans"]), "centroids") != (profile["n_clusters"], config["width"]):
                raise ValueError("The SAE code-space codebook has the wrong dimensions")
            report["checks"]["sae_headers"] = shapes
        report.update(state="passed", finished_utc=utc())
    except Exception as error:
        report.update(state="failed", error=f"{type(error).__name__}: {error}", finished_utc=utc())
        atomic_json(report_path, report)
        raise
    atomic_json(report_path, report)
    print("CPU preflight passed; no LM tensors or activation values loaded", flush=True)
    return report


def gpu_preflight(manifest: dict, checkpoint: Path, arm: str, root: Path, *, fresh: bool = False) -> None:
    """Exercise the real model and nonlinear splice inside this allocation."""
    import gc
    import torch
    from geoae.checkpoint import load_tokenizer
    sys.path.insert(0, manifest["source_root"])
    from geoae.checkpoint import load_ae_checkpoint, load_lm
    from geoae.hooks import SplicingHook, TokenIdTap
    from geoae.evaluation import interventions as nl
    device = torch.device("cuda")
    name = manifest["profile"]["model_name"]
    tokenizer = load_tokenizer(name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    lm = load_lm(name, device=device)
    tap = TokenIdTap(lm)
    hook = SplicingHook(lm, manifest["profile"]["target_layer"])
    if fresh:
        from geoae.config import Config
        from geoae.model import GeoAE
        from geoae.token_bias import load_table
        import numpy as np
        config_path = Path(manifest["training_run_dir"]) / "configs" / f"{arm}.yaml"
        cfg = Config.from_yaml(config_path)
        mc = cfg.model
        table = load_table(Path(manifest["assets"]["token_bias_table"]))
        # The canonical paper train_common predates nonlinear decoders. Build
        # explicitly from the reviewed config instead of calling that builder.
        ae = GeoAE(hidden_size=mc.hidden_size, latent_dim=mc.latent_dim,
                   n_clusters=mc.n_clusters, nonlinearity=mc.nonlinearity,
                   latent_norm=mc.latent_norm, metric=mc.metric,
                   decoder_type=mc.decoder_type, decoder_hidden_dim=mc.decoder_hidden_dim,
                   decoder_nonlinearity=mc.decoder_nonlinearity, decoder_output_bias=mc.decoder_output_bias,
                   token_bias_rows=len(table["token_ids"]), vocab_size=int(table["vocab_size"]),
                   token_bias_mode=mc.token_bias_mode).to(device)
        ae.load_token_bias(torch.as_tensor(table["token_ids"]), torch.as_tensor(table["table"]))
        ae.eval()
        del table
        if ae.decoder_type != arm:
            raise ValueError("Fresh preflight constructed the wrong decoder architecture")
        layer = manifest["profile"]["target_layer"]
        with np.load(Path(manifest["assets"]["activations_dir"]) / f"norm_params_layer{layer}.npz") as norm:
            mean = torch.as_tensor(norm["mean"], device=device)
            std = torch.as_tensor(norm["std"], device=device)
        identity = {"configuration_sha256": digest(config_path), "initialization": "fresh untrained smoke only"}
    else:
        ae, mean, std, _ = load_ae_checkpoint(checkpoint, device, allow_token_bias=True)
        identity = {"checkpoint_sha256": digest(checkpoint), "initialization": "final trained checkpoint"}
    inputs = tokenizer(["The capital of France is", "A short test."], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        clean = lm(**inputs, use_cache=False).logits[:, -1, :].float()
        hook.activate(nl.make_z_edit(None, ae, mean, std, device, tap=tap))
        spliced = lm(**inputs, use_cache=False).logits[:, -1, :].float()
        hook.deactivate()
        hook.activate(nl.make_z_edit(lambda z: z + 0.01, ae, mean, std, device, tap=tap))
        edited = lm(**inputs, use_cache=False).logits[:, -1, :].float()
        hook.deactivate()
        if not all(torch.isfinite(value).all() for value in (clean, spliced, edited)):
            raise ValueError("Non-finite logits in the final nonlinear splice")
        if manifest["assets"].get("sae"):
            from geoae.baselines.sae_space import load_sae_space
            sae = load_sae_space(manifest["assets"]["sae"], device)
            sample = torch.randn(2, manifest["profile"]["hidden_size"], device=device)
            if not torch.isfinite(sae.decode_raw(sae.encode_raw(sample))).all():
                raise ValueError("Non-finite Gemma SAE GPU round trip")
            del sae, sample
    tap.remove()
    suffix = "fresh" if fresh else "final"
    atomic_json(root / "provenance" / f"{arm}_gpu_preflight_{suffix}.json", {
        "state": "passed", "finished_utc": utc(), "model_name": name,
        "decoder_type": ae.decoder_type, "hidden_size": manifest["profile"]["hidden_size"],
        "latent_dim": manifest["profile"]["latent_dim"],
        **identity, "gpu": torch.cuda.get_device_name(0),
        "within_training_allocation": True})
    del lm, ae, mean, std, clean, spliced, edited, inputs, hook, tap
    gc.collect()
    torch.cuda.empty_cache()


def run(args) -> None:
    root = evaluation_root(args)
    manifest = load_manifest(root)
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; refusing CPU fallback")
    checkpoint = checkpoint_path(manifest, args.arm)
    identity = verify_checkpoint(checkpoint, args.arm, manifest["profile"])
    path = root / "state" / f"{args.arm}.json"
    state = json.loads(path.read_text()) if path.exists() else {"arm": args.arm, "steps": {}}
    env, source = runtime_environment(manifest), Path(manifest["source_root"])
    os.environ.update(env)
    gpu_preflight(manifest, checkpoint, args.arm, root)
    state.update(state="running", started_utc=utc(), checkpoint=identity)
    atomic_json(root / "provenance" / f"{args.arm}_{os.getenv('SLURM_JOB_ID', 'local')}.json",
                {"started_utc": utc(), "host": socket.gethostname(), "job_id": os.getenv("SLURM_JOB_ID"),
                 "gpu": torch.cuda.get_device_name(0), "checkpoint": identity,
                 "manifest_sha256": digest(root / "manifest.json")})
    failed = []
    with (root / "locks" / f"{args.arm}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ExitStack() as scratch:
            staged = args.activations_dir.resolve() if args.activations_dir else None
            if staged is not None:
                verify_staged_activations(manifest, staged)
            for step in manifest["plans"][args.arm]:
                name = step["name"]
                if completed(step, state["steps"].get(name), identity["sha256"]):
                    print(f"[skip complete] {name}", flush=True)
                    continue
                by_name = {item["name"]: item for item in manifest["plans"][args.arm]}
                blocked = [name for name in step["dependencies"] if not completed(
                    by_name[name], state["steps"].get(name), identity["sha256"])]
                if blocked:
                    state["steps"][name] = {"state": "blocked", "dependencies": blocked}
                    failed.append(name)
                    atomic_json(path, state)
                    continue
                command = list(step["command"])
                if step["needs_activations"]:
                    if staged is None:
                        original = Path(manifest["assets"]["activations_dir"])
                        needed = directory_bytes(original)
                        temporary = scratch.enter_context(temporary_staging_directory(
                            needed, prefix=f"geoae_{args.arm}_"))
                        if shutil.disk_usage(temporary).free < needed + 2 * 1024**3:
                            raise ValueError("Insufficient node-local memory for evaluation activations")
                        staged = Path(temporary) / "activations"
                        shutil.copytree(original, staged, ignore=shutil.ignore_patterns("__pycache__"))
                        verify_staged_activations(manifest, staged)
                    command = [str(staged) if value == manifest["assets"]["activations_dir"] else value
                               for value in command]
                for output in step["outputs"]:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                record = {"state": "running", "started_utc": utc(), "checkpoint_sha256": identity["sha256"],
                          "command_sha256": command_sha256(step), "command": command,
                          "group": step["group"], "job_id": os.getenv("SLURM_JOB_ID")}
                state["steps"][name] = record
                atomic_json(path, state)
                start = time.monotonic()
                print(f"[start {args.arm}] {name}: {shlex.join(command)}", flush=True)
                try:
                    with (root / "logs" / f"{args.arm}_{name}.log").open("w") as log:
                        result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f"Evaluator exited {result.returncode}; see {args.arm}_{name}.log")
                    validate_outputs(step)
                    record.update(state="complete", output_sha256={name: digest(Path(name)) for name in step["outputs"]})
                except Exception as error:
                    failed.append(name)
                    record.update(state="failed", error=f"{type(error).__name__}: {error}")
                record.update(finished_utc=utc(), seconds=time.monotonic() - start)
                state["updated_utc"] = utc()
                atomic_json(path, state)
                print(f"[{record['state']}] {name}: {record['seconds'] / 60:.1f} min", flush=True)
            try:
                from nonlinear_paper_summary import write_control_summary
                assets = manifest["assets"]
                datasets = assets.get("summary_control_datasets")
                if datasets is None:
                    from nonlinear_paper_plan import CONTROL_DATASETS
                    datasets = [name for name in CONTROL_DATASETS if not name.startswith("atlas_")]
                    datasets += assets.get("atlas_control_datasets", ["atlas_s0", "atlas_s1", "atlas_s2"])
                write_control_summary(root / "results" / args.arm, sae=bool(assets.get("sae")), datasets=datasets)
                state.pop("control_summary_error", None)
            except Exception as error:
                failed.append("control_summary")
                state["control_summary_error"] = f"{type(error).__name__}: {error}"
            state.update(state="failed" if failed else "complete", failed_steps=failed, finished_utc=utc())
            atomic_json(path, state)
    if failed:
        raise RuntimeError(f"All independent evaluations attempted; failed or blocked: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--profile", type=Path, required=True)
    prepare_parser.add_argument("--run-dir", type=Path, required=True)
    prepare_parser.add_argument("--paper-repo", type=Path,
        help="Reconciled external checkout with the exact frozen training core; historical layouts are rejected")
    prepare_parser.add_argument("--python", type=Path)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--arm", choices=ARMS, required=True)
    run_parser.add_argument("--activations-dir", type=Path)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--run-dir", type=Path, required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--run-dir", type=Path, required=True)
    gpu_parser = commands.add_parser("gpu-preflight")
    gpu_parser.add_argument("--run-dir", type=Path, required=True)
    gpu_parser.add_argument("--arm", choices=ARMS, required=True)
    for command_parser in (run_parser, validate_parser, preflight_parser, gpu_parser):
        command_parser.add_argument("--evaluation-dir", type=Path,
                                    help="Separate immutable recovery evaluation snapshot")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        run(args)
    elif args.action == "preflight":
        cpu_preflight(args)
    elif args.action == "gpu-preflight":
        root = evaluation_root(args)
        manifest = load_manifest(root)
        os.environ.update(runtime_environment(manifest))
        import torch
        if not torch.cuda.is_available():
            raise ValueError("GPU preflight requires CUDA")
        gpu_preflight(manifest, checkpoint_path(manifest, args.arm), args.arm, root, fresh=True)
    else:
        load_manifest(evaluation_root(args))
        print("Verified evaluation source, assets, training identity, and runtime")


if __name__ == "__main__":
    main()
