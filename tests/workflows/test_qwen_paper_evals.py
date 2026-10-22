"""Paper evaluation provenance, resumability and Slurm dependency protection."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def runner(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "experiments/nonlinear_decoders/deltaai"))
    spec = importlib.util.spec_from_file_location("qwen_paper_evals", root / "experiments/nonlinear_decoders/deltaai/qwen_paper_evals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_manifest(runner, root):
    source = root / "source"
    source.mkdir(parents=True)
    (source / "evaluator.py").write_text("RESULT = 1\n")
    for directory in ("locks", "logs", "state", "provenance"):
        (root / directory).mkdir()
    manifest = {"source_root": str(source), "source_sha256": runner.tree_hashes(source),
                "asset_identities": {}, "slurm": {"account": "bbyl-dtai-gh", "partition": "ghx4"},
                "training_jobs": {"mlp": "3410182", "residual_mlp": "3410183"}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("training_matches", [True, False])
@pytest.mark.parametrize("external_checkout", [True, False])
def test_default_preparation_freezes_current_paper_without_legacy_patches(
        runner, tmp_path, monkeypatch, training_matches, external_checkout):
    import qwen_paper_plan

    checkout = Path(__file__).resolve().parents[2]
    training = tmp_path / "training"
    training.mkdir()
    training_source = training / "source"
    for name in runner.CORE_OVERLAYS:
        target = training_source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkout / name, target)
    if not training_matches:
        (training_source / "geoae/model.py").write_text("# Different, legitimately frozen training implementation\n")
    core_hashes = {name: runner.digest(training_source / name) for name in runner.CORE_OVERLAYS}
    runner.atomic_json(training / "manifest.json", {"runtime": {"python": sys.executable, "modules": []},
        "slurm": {"account": "fixture", "partition": "fixture"},
        "source_root": str(training_source), "source_sha256": core_hashes})
    (training / "submissions.jsonl").write_text(''.join(
        json.dumps({"arm": arm, "job_id": index + 1}) + "\n" for index, arm in enumerate(runner.ARMS)))
    asset = tmp_path / "fixture.asset"
    asset.write_text("fixture")
    assets = {key: str(asset) for key in qwen_paper_plan.REQUIRED_ASSETS}
    assets["atlas_control_classes"] = {"atlas_s0": [0, 2], "atlas_s1": "1,3", "atlas_s2": [4, 5]}
    assets_json = tmp_path / "assets.json"
    runner.atomic_json(assets_json, assets)
    monkeypatch.setattr(runner, "patch_wiki_reference", lambda *args: pytest.fail("Canonical code needs no text patch"))
    output = tmp_path / "evaluation"
    args = SimpleNamespace(training_run_dir=training, eval_dir=output, assets_json=assets_json,
                           paper_repo=checkout if external_checkout else None, python=None)
    if not training_matches:
        with pytest.raises(ValueError, match="Evaluation core differs from frozen training source: geoae/model.py"):
            runner.prepare(args)
        assert not output.exists(), "A core mismatch must not leave a prepared snapshot"
        return
    runner.prepare(args)
    manifest = runner.load_manifest(output)
    assert manifest["paper_origin"] == str(checkout / "docs/paper/paper.tex")
    assert manifest["nonlinear_overlays_sha256"] == manifest["nonlinear_evaluator_patches"] == {}
    assert manifest["training_core_sha256"] == core_hashes
    assert len(manifest["plans"]["mlp"]) == 18
    source = Path(manifest["source_root"])
    assert runner.digest(source / "geoae/model.py") == runner.digest(checkout / "geoae/model.py")
    for step in manifest["plans"]["mlp"]:
        command = step["command"]
        target = (command[command.index("-m") + 1].replace(".", "/") + ".py" if "-m" in command
                  else next(argument for argument in command if argument.endswith(".py")))
        assert (source / target).is_file(), f"Missing evaluator in source snapshot: {target}"
    assert not (source / "runs").exists()


def test_historical_paper_checkout_is_rejected_before_writing_a_snapshot(runner, tmp_path):
    old = tmp_path / "historical"
    (old / "geoae/interp").mkdir(parents=True)
    (old / "geoae/interp/range_intervention_compare.py").write_text("# old layout\n")
    (old / "docs/paper").mkdir(parents=True)
    (old / "docs/paper/main.tex").write_text("% preserved historical paper\n")
    before = runner.tree_hashes(old)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="reconciled source layout.*historical"):
        runner.prepare(SimpleNamespace(paper_repo=old, training_run_dir=tmp_path / "training",
                                       eval_dir=output, assets_json=tmp_path / "assets.json", python=None))
    assert not output.exists()
    assert runner.tree_hashes(old) == before


def test_core_verification_rejects_tampered_training_even_when_evaluation_matches(runner, tmp_path):
    source = tmp_path / "source"
    (source / "geoae").mkdir(parents=True)
    for name in runner.CORE_OVERLAYS:
        (source / name).write_text("# frozen\n")
    manifest = {"source_root": str(source), "source_sha256": runner.tree_hashes(source)}
    (source / "geoae/checkpoint.py").write_text("# tampered\n")
    with pytest.raises(ValueError, match="Training core source changed: geoae/checkpoint.py"):
        runner.verify_training_core(manifest, source)


def test_bundle_supplied_activation_copy_needs_no_second_scratch_reservation(runner, tmp_path, monkeypatch):
    root = tmp_path / "evaluation"
    manifest = frozen_manifest(runner, root)
    output = root / "results/fixture.json"
    manifest.update(training_run_dir=str(tmp_path / "training"), assets={"activations_dir": "/original/activations"},
                    plans={"mlp": [{"name": "fixture", "group": "core", "needs_activations": True,
                                    "dependencies": [], "command": ["evaluate", "/original/activations"],
                                    "outputs": [str(output)]}]})
    monkeypatch.setattr(runner, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(runner, "verify_checkpoint", lambda *args: {"sha256": "checkpoint"})
    monkeypatch.setattr(runner, "runtime_environment", lambda *args: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "fixture GPU")
    monkeypatch.setattr(runner, "temporary_staging_directory", lambda *args, **kwargs:
                        pytest.fail("Bundle already owns its activation copy"))

    def evaluate(command, **kwargs):
        assert command == ["evaluate", "/existing/staged/activations"]
        output.write_text('{"valid": true}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", evaluate)
    runner.run_group(SimpleNamespace(eval_dir=root, arm="mlp", group="core",
                                    activation_stager=lambda *args: Path("/existing/staged/activations")))
    assert json.loads((root / "state/mlp.json").read_text())["steps"]["fixture"]["state"] == "complete"


def test_source_and_small_asset_contents_cannot_change_with_preserved_size_and_mtime(runner, tmp_path):
    asset = tmp_path / "codebook.npz"
    asset.write_bytes(b"original")
    identities = runner.asset_identities({"balanced_kmeans": str(asset)})
    runner.verify_assets(identities)
    original_stat = asset.stat()
    asset.write_bytes(b"tampered")
    os.utime(asset, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(ValueError, match="hash changed"):
        runner.verify_assets(identities)

    root = tmp_path / "evaluation"
    manifest = frozen_manifest(runner, root)
    assert runner.load_manifest(root) == manifest
    source = Path(manifest["source_root"]) / "evaluator.py"
    source.write_text("RESULT = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        runner.load_manifest(root)


@pytest.mark.parametrize("resource", ["experiments/config_paths.json", "geoae/baselines/matched_topk/_reference/SOURCE.json"])
def test_runtime_source_resources_are_pinned_but_old_manifest_schemas_still_load(runner, tmp_path, resource):
    root = tmp_path / "evaluation"
    manifest = frozen_manifest(runner, root)
    source = Path(manifest["source_root"])
    additions = (resource, "README.md", "LICENSE", "scripts/py", "experiments/template.html",
                 "docs/paper/acl.sty", "docs/paper/acl_natbib.bst", "configs/sites/site.yml")
    for name in additions:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"version": 1}\n')
    # Historical manifests never hashed these resources and keep their schema.
    assert runner.load_manifest(root) == manifest
    manifest["source_sha256"] = runner.tree_hashes(source)
    assert set(additions).issubset(manifest["source_sha256"])
    runner.atomic_json(root / "manifest.json", manifest)
    runner.load_manifest(root)
    for name in additions:
        path = source / name
        previous = path.stat()
        path.write_text('{"version": 2}\n')
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        with pytest.raises(ValueError, match="source changed"):
            runner.load_manifest(root)
        path.write_text('{"version": 1}\n')


def test_completed_requires_matching_checkpoint_valid_outputs_and_original_output_hash(runner, tmp_path):
    output = tmp_path / "mmlu.json"
    output.write_text(json.dumps({"meta": {"n": 2000}, "preds": [1]}))
    step = {"name": "mmlu", "outputs": [str(output)], "json_required_keys": {str(output): ["meta", "preds"]}}
    record = {"state": "complete", "checkpoint_sha256": "checkpoint-A",
              "output_sha256": {str(output): runner.digest(output)}}
    assert runner.completed(step, record, "checkpoint-A")
    assert not runner.completed(step, record, "checkpoint-B")
    assert not runner.completed(step, dict(record, state="failed"), "checkpoint-A")
    output.write_text(json.dumps({"meta": {"n": 2000}, "preds": [2]}))
    assert not runner.completed(step, record, "checkpoint-A")
    output.write_text("{partial")
    assert not runner.completed(step, record, "checkpoint-A")


def test_output_validation_checks_named_json_and_npz_members(runner, tmp_path):
    import numpy as np
    output = tmp_path / "atlas.npz"
    np.savez(output, linear=np.array([1, 2]))
    step = {"name": "atlas_assign", "outputs": [str(output)], "npz_required_keys": {str(output): ["linear", "geoae"]}}
    with pytest.raises(ValueError, match="Incomplete NPZ"):
        runner.validate_outputs(step)
    np.savez(output, linear=np.array([1, 2]), geoae=np.array([2, 1]))
    runner.validate_outputs(step)
    score = tmp_path / "scores.json"
    score.write_text(json.dumps({"meta": {}}))
    with pytest.raises(ValueError, match="Incomplete evaluation output"):
        runner.validate_outputs({"name": "atlas_pair10", "outputs": [str(score)],
                                 "json_required_keys": {str(score): ["meta", "sets"]}})


def test_paper_task_validation_rejects_silently_skipped_tasks_and_missing_atlas_draws(runner, tmp_path):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"rungs": {"domain": {}}}))
    with pytest.raises(ValueError, match="Missing paper tasks"):
        runner.validate_outputs({"name": "probe_alignment", "outputs": [str(probe)]})
    atlas = tmp_path / "atlas.json"
    atlas.write_text(json.dumps({"sets": {"atlas14_s0": {}}}))
    with pytest.raises(ValueError, match="Missing Atlas draws"):
        runner.validate_outputs({"name": "atlas_pair10", "outputs": [str(atlas)]})


@pytest.mark.parametrize("wrapped", [False, True])
def test_staged_wikipedia_transport_reads_original_texts_without_network(runner, tmp_path, wrapped):
    import argparse
    source = tmp_path / "source"
    target = source / "geoae/interp/range_intervention_compare.py"
    target.parent.mkdir(parents=True)
    target.write_text('''def reference_texts(args):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_wiki", type=int, default=100)
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    wtxt = []
    for ex in wiki:
        wtxt.append(ex["text"][:1500])
        if len(wtxt) >= args.n_wiki:
            break
    return wtxt
''')
    runner.patch_wiki_reference(source)
    texts = [f"Original unshuffled text {index}" for index in range(100)]
    path = tmp_path / "wikipedia.json"
    path.write_text(json.dumps({"texts": texts} if wrapped else texts))

    def no_network(*args, **kwargs):
        pytest.fail("Staged Wikipedia transport unexpectedly accessed the network")

    namespace = {"argparse": argparse, "json": json, "Path": Path, "load_dataset": no_network}
    exec(compile(target.read_text(), str(target), "exec"), namespace)
    assert namespace["reference_texts"](SimpleNamespace(wiki_texts=str(path), n_wiki=100)) == texts


def checkpoint(path, **updates):
    data = {"config": {"model": {"decoder_type": "mlp", "hidden_size": 2048,
                                 "latent_dim": 4096, "n_clusters": 2000},
                       "extraction": {"model_name": "Qwen/Qwen2.5-3B"}, "data": {"target_layer": 35}},
            "epoch": 50, "step": 15250, "model_state": {"centroids_initialized": torch.tensor(True)}}
    for key, value in updates.items():
        data[key] = value
    torch.save(data, path)
    return data


def test_final_checkpoint_guard_rejects_warmup_wrong_architecture_and_uninitialized_centroids(runner, tmp_path):
    path = tmp_path / "ae.pt"
    data = checkpoint(path)
    assert runner.verify_checkpoint(path, "mlp")["step"] == 15250
    with pytest.raises(ValueError, match="Wrong decoder"):
        runner.verify_checkpoint(path, "residual_mlp")
    checkpoint(path, epoch=5, step=1525)
    with pytest.raises(ValueError, match="final epoch50"):
        runner.verify_checkpoint(path, "mlp")
    assert runner.verify_checkpoint(path, "mlp", final=False)["epoch"] == 5
    checkpoint(path, model_state={"centroids_initialized": torch.tensor(False)})
    with pytest.raises(ValueError, match="Centroids not initialized"):
        runner.verify_checkpoint(path, "mlp")
    data["config"]["data"]["target_layer"] = 30
    torch.save(data, path)
    with pytest.raises(ValueError, match="Wrong extraction layer"):
        runner.verify_checkpoint(path, "mlp")


def test_scheduler_dependencies_wait_for_own_training_and_serialize_group_attempts(runner, tmp_path):
    manifest = frozen_manifest(runner, tmp_path / "evaluation")
    root = tmp_path / "evaluation"
    first = runner.submission_command(root, manifest, "mlp", "core", submit=True)
    later = runner.submission_command(root, manifest, "mlp", "alignment", "999", submit=True)
    assert "--dependency=afterok:3410182" in first
    assert "--dependency=afterok:3410182,afterany:999" in later
    assert "--kill-on-invalid-dep=yes" in later
    for option in ("--gpus-per-node=1", "--account=bbyl-dtai-gh", "--partition=ghx4", "--export=NONE",
                   "--cpus-per-task=12", "--mem=160G"):
        assert option in first
    preflight = runner.submission_command(root, manifest, "mlp", "preflight", submit=True, smoke=True)
    assert not any(value.startswith("--dependency=") for value in preflight)
    assert "--time=00:20:00" in preflight
    assert preflight[-1] == "preflight"


def test_submission_is_durable_and_retry_cannot_duplicate_evaluations(runner, tmp_path, monkeypatch):
    root = tmp_path / "evaluation"
    frozen_manifest(runner, root)
    monkeypatch.setenv("SBATCH_ACCOUNT", "inherited-wrong-account")
    calls = []

    def submit(command, **kwargs):
        assert not any(name.startswith("SBATCH_") for name in kwargs["env"])
        calls.append(command)
        return SimpleNamespace(stdout=f"{9000 + len(calls)};deltaai\n", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", submit)
    args = SimpleNamespace(eval_dir=root, submit=True, print_only=False, preflight_only=False)
    runner.submit_groups(args)
    assert len(calls) == 8
    records = [json.loads(line) for line in (root / "submissions.jsonl").read_text().splitlines()]
    assert [(record["arm"], record["group"]) for record in records] == [
        (arm, group) for arm in runner.ARMS for group in runner.GROUPS]
    for index in (1, 2, 3, 5, 6, 7):
        assert ",afterany:" + records[index - 1]["job_id"] in " ".join(calls[index])
    assert all(record["manifest_sha256"] == runner.digest(root / "manifest.json") for record in records)
    runner.submit_groups(args)
    assert len(calls) == 8


def test_dry_run_does_not_write_submission_receipts(runner, tmp_path, monkeypatch):
    root = tmp_path / "evaluation"
    frozen_manifest(runner, root)
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs:
                        (calls.append(command) or SimpleNamespace(stdout="", stderr="test successful")))
    runner.submit_groups(SimpleNamespace(eval_dir=root, submit=False, print_only=False, preflight_only=False))
    assert len(calls) == 8 and all("--test-only" in command for command in calls)
    assert not (root / "submissions.jsonl").exists()


def test_runtime_environment_uses_isolated_source_and_offline_staged_datasets(runner, monkeypatch):
    monkeypatch.setenv("SBATCH_GPUS", "4")
    manifest = {"source_root": "/eval/source", "assets": {
        "atlas_classes_dir": "/assets/atlas", "control_datasets_dir": "/assets/control",
        "hf_home": "/hf/home", "hf_datasets_cache": "/hf/datasets"}}
    env = runner.runtime_environment(manifest)
    assert env["PYTHONPATH"] == env["GEOAE_ROOT"] == "/eval/source"
    assert env["HF_HUB_OFFLINE"] == env["HF_DATASETS_OFFLINE"] == "1"
    assert env["HF_HOME"] == "/hf/home" and env["HF_DATASETS_CACHE"] == "/hf/datasets"
    assert env["GEOAE_ATLAS_CLASSES_DIR"] == "/assets/atlas"
    assert "SBATCH_GPUS" not in env


def test_failed_step_blocks_dependents_but_independent_results_survive_and_resume(runner, tmp_path, monkeypatch):
    root = tmp_path / "evaluation"
    manifest = frozen_manifest(runner, root)
    manifest.update(training_run_dir=str(tmp_path / "training"), assets={
        "atlas_classes_dir": str(tmp_path / "atlas"), "control_datasets_dir": str(tmp_path / "control")})
    steps = [{"name": name, "group": "core", "command": ["fake-evaluator", name],
              "outputs": [str(root / "results" / (name + ".json"))],
              "needs_activations": False, "dependencies": ["first"] if name == "dependent" else [],
              "json_required_keys": {str(root / "results" / (name + ".json")): ["value"]}}
             for name in ("first", "dependent", "independent")]
    manifest["plans"] = {"mlp": steps}
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "test GPU")
    monkeypatch.setattr(runner, "verify_checkpoint", lambda *args, **kwargs: {"sha256": "final-checkpoint"})
    calls = []
    fail_first = True

    def evaluator(command, **kwargs):
        name = command[-1]
        calls.append(name)
        if fail_first and name == "first":
            return SimpleNamespace(returncode=1)
        (root / "results" / (name + ".json")).write_text(json.dumps({"value": name}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", evaluator)
    args = SimpleNamespace(eval_dir=root, arm="mlp", group="core")
    with pytest.raises(RuntimeError, match="Failed/blocked steps"):
        runner.run_group(args)
    state = json.loads((root / "state/mlp.json").read_text())
    assert {name: record["state"] for name, record in state["steps"].items()} == {
        "first": "failed", "dependent": "blocked", "independent": "complete"}
    assert calls == ["first", "independent"]
    fail_first = False
    runner.run_group(args)
    assert calls == ["first", "independent", "first", "dependent"]
    state = json.loads((root / "state/mlp.json").read_text())
    assert all(record["state"] == "complete" for record in state["steps"].values())
