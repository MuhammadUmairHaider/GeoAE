"""Middle-layer trainers reject cross-model artifacts and unsafe resume recipes."""
from __future__ import annotations

import fcntl
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geoae.config import Config
from scripts.midlayer import run


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arm,parent_layer", [("llama", 27), ("gemma4b", 33)])
def test_middle_layer_configs_preserve_parent_recipe(arm, parent_layer):
    profile = run.PROFILES[arm]
    extraction, training = run.config_paths(arm)
    current = Config.from_yaml(ROOT / training)
    parent = Config.from_yaml(ROOT / str(training).replace(
        f"_l{profile['layer']}_", f"_l{parent_layer}_"))
    current_dict, parent_dict = current.to_dict(), parent.to_dict()
    for section, key in (("data", "target_layer"), ("data", "activations_dir"),
                         ("model", "token_bias"), ("train", "checkpoints_dir")):
        current_dict[section][key] = parent_dict[section][key]
    assert current_dict == parent_dict

    middle_extract = Config.from_yaml(ROOT / extraction).extraction
    parent_extract = Config.from_yaml(ROOT / str(extraction).replace(
        f"_l{profile['layer']}_", "_")).extraction
    for field in ("layers", "target_layer", "activations_dir"):
        setattr(parent_extract, field, getattr(middle_extract, field))
    assert middle_extract == parent_extract
    assert middle_extract.layers == [profile["layer"]]
    assert middle_extract.hidden_size == current.model.hidden_size == profile["hidden_size"]
    assert middle_extract.activations_dir == current.data.activations_dir == profile["activations_dir"]
    assert current.model.token_bias_mode == "bypass"
    assert current.model.latent_dim == 2 * current.model.hidden_size
    assert middle_extract.dtype == ("float32" if arm == "gemma4b" else "float16")


@pytest.fixture
def token_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = dict(run.PROFILES["llama"], hidden_size=3)
    dump = Path(profile["activations_dir"])
    dump.mkdir()
    (dump / "meta.json").write_text(json.dumps({"n_tokens": 20}))
    mean, std = np.zeros(3, np.float32), np.ones(3, np.float32)
    np.savez(dump / "norm_params_layer14.npz", mean=mean, std=std)
    path = Path("e2e/checkpoints/general") / profile["family"] / "layer14" / profile["token_bias_filename"]
    path.parent.mkdir(parents=True)
    payload = dict(layer=14, model_name=profile["model_name"], activations=str(dump),
                   table=np.zeros((2, 3), np.float16), token_ids=np.array([1, 2]), vocab_size=10,
                   shrink_k=30.0, min_count=10, norm_mean=mean, norm_std=std,
                   report=json.dumps({"train_rows": 19}))
    np.savez(path, **payload)
    return profile, dump, path, payload


def test_token_table_accepts_matching_source_and_normalization(token_artifacts):
    profile, _, path, _ = token_artifacts
    assert run.check_table(profile) == path


@pytest.mark.parametrize("changes,message", [
    ({"layer": 27}, "Wrong model/layer"),
    ({"model_name": "google/gemma-3-4b-pt"}, "Wrong model/layer"),
    ({"activations": "another_dump"}, "another activation dump"),
    ({"table": np.ones((2, 2))}, "wrong-shaped"),
    ({"table": np.full((2, 3), np.nan)}, "nonfinite"),
    ({"token_ids": np.array([2, 2])}, "Invalid token IDs"),
    ({"token_ids": np.array([1.0, 2.0])}, "Invalid token IDs"),
    ({"token_ids": np.array([-1, 2])}, "exceed the table vocabulary"),
    ({"token_ids": np.array([1, 10])}, "exceed the table vocabulary"),
    ({"shrink_k": 10.0}, "Wrong token-bias recipe"),
    ({"min_count": 1}, "Wrong token-bias recipe"),
    ({"report": json.dumps({"train_rows": 18})}, "full 95% train split"),
    ({"norm_mean": np.ones(3, np.float32)}, "mean disagree"),
    ({"norm_std": np.full(3, 2.0, np.float32)}, "std disagree"),
])
def test_token_table_rejects_mismatched_or_corrupt_inputs(token_artifacts, changes, message):
    profile, _, path, payload = token_artifacts
    np.savez(path, **(payload | changes))
    with pytest.raises(ValueError, match=message):
        run.check_table(profile)


@pytest.mark.parametrize("mean,std,message", [
    (np.zeros(2), np.ones(2), "normalization mean shape"),
    (np.zeros(3), np.zeros(3), "Invalid normalization"),
    (np.full(3, np.inf), np.ones(3), "Invalid normalization"),
])
def test_token_table_requires_valid_normalization_vectors(token_artifacts, mean, std, message):
    profile, dump, path, payload = token_artifacts
    np.savez(dump / "norm_params_layer14.npz", mean=mean, std=std)
    np.savez(path, **(payload | {"norm_mean": mean, "norm_std": std}))
    with pytest.raises(ValueError, match=message):
        run.check_table(profile)


def checkpoint_fixture(arm="llama", epoch=50):
    profile = run.PROFILES[arm]
    config = ROOT / run.config_paths(arm)[1]
    checkpoint = dict(config=Config.from_yaml(config).to_dict(), epoch=epoch, step=14450,
                      model_state=dict(tb_table=torch.zeros((2, 3)), centroids_initialized=True))
    return profile, config, checkpoint


def test_final_training_guard_accepts_matching_final_checkpoint(monkeypatch):
    profile, config, checkpoint = checkpoint_fixture()
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    marker = run.check_training(Path("final.pt"), profile, config)
    assert marker["epoch"] == 50
    assert marker["layer"] == 14
    assert marker["token_bias_mode"] == "bypass"


@pytest.mark.parametrize("epoch", [0, 10, 49])
def test_final_training_guard_rejects_warmup_or_partial_checkpoint(monkeypatch, epoch):
    profile, config, checkpoint = checkpoint_fixture(epoch=epoch)
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    with pytest.raises(ValueError, match="Checkpoint guard failed"):
        run.check_training(Path("partial.pt"), profile, config)


@pytest.mark.parametrize("section,key,value", [
    ("loss", "lambda_cluster", 2.0),
    ("data", "target_layer", 27),
    ("model", "token_bias", "another_table.npz"),
    ("model", "token_bias_mode", "decoder_only"),
    ("train", "seed", 43),
    ("train", "lr", 1e-3),
])
def test_resume_checks_full_training_recipe_before_loading_model(monkeypatch, section, key, value):
    _, config, checkpoint = checkpoint_fixture(epoch=20)
    checkpoint["config"][section][key] = value
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    with pytest.raises(ValueError, match="training configuration differs"):
        run.checked_checkpoint(Path("partial.pt"), config)


def test_matching_partial_checkpoint_is_resumable(monkeypatch):
    _, config, checkpoint = checkpoint_fixture(epoch=20)
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    assert run.checked_checkpoint(Path("partial.pt"), config)["epoch"] == 20


def test_legacy_dpc_checkpoint_resumes_after_harmless_config_schema_additions(monkeypatch):
    _, config, checkpoint = checkpoint_fixture(epoch=28)
    for key in ("lloyd_rows", "lloyd_iters", "freeze_centroids"):
        checkpoint["config"]["train"].pop(key)
    original = deepcopy(checkpoint["config"])
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    assert run.checked_checkpoint(Path("legacy.pt"), config)["epoch"] == 28
    assert checkpoint["config"] == original


def test_legacy_resume_still_rejects_changed_centroid_behavior(tmp_path, monkeypatch):
    import yaml
    _, config, checkpoint = checkpoint_fixture(epoch=28)
    for key in ("lloyd_rows", "lloyd_iters", "freeze_centroids"):
        checkpoint["config"]["train"].pop(key)
    settings=yaml.safe_load(config.read_text())
    settings["train"]["freeze_centroids"] = True
    modified = tmp_path / "frozen.yaml"
    modified.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    with pytest.raises(ValueError, match="Unknown config key"):
        run.checked_checkpoint(Path("legacy.pt"), modified)


@pytest.mark.parametrize("key,value", [("source_fingerprint", "older-nine-source-mix"),
                                      ("model", "google/gemma-3-4b-pt"), ("layers", [27])])
def test_resume_rejects_checkpoint_provenance_from_another_dataset_scope(monkeypatch, key, value):
    from scripts.midlayer import dataset

    profile, config, checkpoint = checkpoint_fixture(epoch=20)
    checkpoint["dataset_provenance"] = dataset.validate_extraction_config(ROOT / profile["extract_config"])
    checkpoint["dataset_provenance"][key] = value
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    with pytest.raises(ValueError, match="different source recipe"):
        run.checked_checkpoint(Path("partial.pt"), config)


def test_completion_marker_retains_verified_dataset_provenance(monkeypatch):
    from scripts.midlayer import dataset

    profile, config, checkpoint = checkpoint_fixture()
    provenance = dataset.validate_extraction_config(ROOT / profile["extract_config"])
    provenance["corpus_manifest_sha256"] = "a" * 64
    checkpoint["dataset_provenance"] = deepcopy(provenance)
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    marker = run.check_training(Path("final.pt"), profile, config)
    assert marker["dataset_provenance"] == provenance
    assert marker["dataset_provenance"]["activations_dir"] == checkpoint["config"]["data"]["activations_dir"]


def test_plain_training_uses_staged_arrays_and_preserves_canonical_dataset_identity(tmp_path, monkeypatch):
    from scripts.midlayer import dataset

    monkeypatch.chdir(tmp_path)
    arm = "llama_mid_plain"
    profile = run.PROFILES[arm]
    extraction, training = (ROOT / path for path in run.config_paths(arm))
    canonical = Config.from_yaml(training).to_dict()
    provenance = dataset.validate_extraction_config(extraction)
    provenance["corpus_manifest_sha256"] = "b" * 64
    staged = tmp_path / "ram" / profile["activations_dir"]
    monkeypatch.setattr(run, "config_paths", lambda selected: (extraction, training))
    monkeypatch.setattr(dataset, "check_dump", lambda *a, **k: {"dataset_provenance": deepcopy(provenance)})
    monkeypatch.setattr(run, "stage_dump", lambda *a, **k: staged)
    monkeypatch.setattr(run, "check_table", lambda *a, **k: pytest.fail("Plain training required a token table"))
    calls = []

    def checkpoint_for(*a, **k):
        if not calls:
            raise FileNotFoundError("fresh arm")
        return Path("final.pt")

    def launch(command, *, check, env):
        assert check
        calls.append((command, env))
        assert Config.from_yaml(training).to_dict() == canonical
        assert env["GEOAE_TRAIN_ACTIVATIONS_DIR"] == str(staged)
        assert json.loads(env["GEOAE_DATASET_PROVENANCE"]) == provenance
        assert json.loads(env["GEOAE_DATASET_PROVENANCE"])["activations_dir"] == canonical["data"]["activations_dir"]

    monkeypatch.setattr(run, "checkpoint_for", checkpoint_for)
    monkeypatch.setattr(run.subprocess, "run", launch)
    monkeypatch.setattr(run, "check_training", lambda *a, **k: {"dataset_provenance": deepcopy(provenance)})
    args = training_args()
    args.stage_dump_root = tmp_path / "ram"
    assert run.run_arm(arm, args)
    assert len(calls) == 1
    state = json.loads((tmp_path / "results/test/llama3.2-3B/layer14/plain/trainer_state.json").read_text())
    assert state["status"] == "complete"
    assert state["dataset_provenance"] == provenance
    assert state["runtime_activations_dir"] == str(staged)


def training_args(run_name="test"):
    return SimpleNamespace(steps="train", run_name=run_name, eval_steps="whole", judges="skip",
                           force_evals=False, dry_run=False, data_root=None)


def test_changed_partial_recipe_aborts_before_training_subprocess(tmp_path, monkeypatch):
    from scripts.midlayer import dataset

    monkeypatch.chdir(tmp_path)
    profile, config, checkpoint = checkpoint_fixture(epoch=20)
    checkpoint["config"]["loss"]["lambda_cluster"] = 2.0
    extraction = ROOT / profile["extract_config"]
    monkeypatch.setattr(run, "config_paths", lambda arm: (extraction, config))
    monkeypatch.setattr(dataset, "check_dump", lambda *a, **k: {
        "dataset_provenance": dataset.validate_extraction_config(extraction)})
    monkeypatch.setattr(run, "check_table", lambda *a, **k: Path("table.npz"))
    monkeypatch.setattr(run, "checkpoint_for", lambda *a, **k: Path("partial.pt"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: checkpoint)
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: pytest.fail("Mismatched recipe started training"))
    assert run.run_arm("llama", training_args()) is False
    state = json.loads((tmp_path / "results/test/llama3.2-3B/layer14/trainer_state.json").read_text())
    assert state["status"] == "failed"
    assert "training configuration differs" in state["error"]


def test_different_result_names_cannot_train_same_artifacts_concurrently(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = run.PROFILES["llama"]
    directory = Path("checkpoints") / profile["family"] / "layer14" / profile["ckdir"]
    directory.mkdir(parents=True)
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: pytest.fail("Locked artifact started training"))
    with (directory / ".midlayer.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another trainer"):
            run.run_arm("llama", training_args("another_result_name"))


def test_trainer_dry_run_has_no_subprocess_or_filesystem_writes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: pytest.fail("Dry run launched a subprocess"))
    assert run.main(["--arm", "all", "--dry-run", "--data-root", str(tmp_path / "bulk")]) == 0
    output = capsys.readouterr().out
    assert "evaluate.py llama_mid_bypass --run-name" in output
    assert "evaluate.py gemma4b_mid_bypass --run-name" in output
    assert output.count("[dry] train:") == 8
    assert "--val_frac 0.05" in output
    assert "layer 14" in output and "layer 17" in output
    assert not list(tmp_path.iterdir())


def test_dump_location_refuses_replacing_existing_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = run.PROFILES["llama"]
    original = Path(profile["activations_dir"])
    original.mkdir()
    (original / "sentinel").write_text("existing data")
    with pytest.raises(ValueError, match="outside the requested data root"):
        run.ensure_dump_location(profile, tmp_path / "bulk")
    assert (original / "sentinel").read_text() == "existing data"
    assert not (tmp_path / "bulk").exists()
