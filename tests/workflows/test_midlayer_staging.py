"""Node-local staging must preserve dataset identity and the serialized recipe."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geoae.config import Config
from experiments.depth import dataset, run

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def staging_case(tmp_path, monkeypatch):
    source = tmp_path / "bulk" / "dump"
    source.mkdir(parents=True)
    np.save(source / "layer_1.npy", np.random.default_rng(42).normal(size=(16, 4)).astype(np.float32))
    np.save(source / "layer_9.npy", np.zeros((16, 4), np.float32))
    for name in ("docs", "rows_doc", "rows_pos", "rows_tok", "rows_src", "rows_norm"):
        np.save(source / f"{name}.npy", np.arange(16))
    np.savez(source / "norm_params_layer1.npz", mean=np.zeros(4), std=np.ones(4))
    (source / "meta.json").write_text(json.dumps({"n_tokens": 16,
        "source_fingerprint": dataset.SOURCE_FINGERPRINT, "source_mix": dict(dataset.USER_MIX)}))
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    return dict(activations_dir=str(source), layer=1), source, tmp_path / "ram"


def test_staging_copies_selected_layer_and_all_sidecars(staging_case):
    profile, source, staging = staging_case
    target = run.stage_dump(profile, staging)
    expected = {"layer_1.npy", "meta.json", "docs.npy", "norm_params_layer1.npz",
                "rows_doc.npy", "rows_pos.npy", "rows_tok.npy", "rows_src.npy", "rows_norm.npy"}
    assert {p.name for p in target.iterdir() if not p.name.startswith(".stage_")} == expected
    for name in expected:
        assert (target / name).read_bytes() == (source / name).read_bytes()
    identity = json.loads((target / ".stage_complete.json").read_text())
    assert identity["source"] == str(source.resolve())
    assert identity["layer"] == 1
    assert set(identity["files"]) == expected


def test_verified_staging_reuses_complete_copy(staging_case, monkeypatch):
    profile, _, staging = staging_case
    target = run.stage_dump(profile, staging)
    monkeypatch.setattr(run.shutil, "copy2", lambda *a, **k: pytest.fail("A complete dump was copied again"))
    assert run.stage_dump(profile, staging) == target


@pytest.mark.parametrize("changed", ["metadata", "activation", "source"])
def test_staging_rejects_changed_source_identity(staging_case, tmp_path, changed):
    profile, source, staging = staging_case
    target = run.stage_dump(profile, staging)
    before = (target / "meta.json").read_bytes()
    if changed == "metadata":
        (source / "meta.json").write_text('{"different_recipe": true}')
    elif changed == "activation":
        np.save(source / "layer_1.npy", np.ones((17, 4), np.float32))
    else:
        other = tmp_path / "other" / "dump"
        run.shutil.copytree(source, other)
        profile = dict(profile, activations_dir=str(other))
    with pytest.raises(ValueError, match="another source identity"):
        run.stage_dump(profile, staging)
    assert (target / "meta.json").read_bytes() == before


def test_staging_refuses_unowned_directory(staging_case):
    profile, source, staging = staging_case
    target = staging / source.name
    target.mkdir(parents=True)
    (target / "sentinel").write_text("user data")
    with pytest.raises(ValueError, match="unowned staging directory"):
        run.stage_dump(profile, staging)
    assert (target / "sentinel").read_text() == "user data"


def test_staging_repairs_owned_interrupted_copy(staging_case):
    profile, source, staging = staging_case
    target = run.stage_dump(profile, staging)
    (target / ".stage_complete.json").unlink()
    (target / "rows_tok.npy").unlink()
    assert run.stage_dump(profile, staging) == target
    assert (target / "rows_tok.npy").read_bytes() == (source / "rows_tok.npy").read_bytes()


def test_staging_rejects_missing_sidecar_and_insufficient_capacity(staging_case, monkeypatch):
    profile, source, staging = staging_case
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="Insufficient node-local space"):
        run.stage_dump(profile, staging)
    assert not (staging / source.name).exists()
    (source / "rows_src.npy").unlink()
    with pytest.raises(ValueError, match="incomplete dump"):
        run.stage_dump(profile, staging)


def test_training_runtime_copy_keeps_canonical_config_and_provenance(staging_case, tmp_path, monkeypatch):
    import geoae.training.mse as training

    profile, source, staging = staging_case
    staged = run.stage_dump(profile, staging)
    cfg = Config()
    cfg.data.activations_dir = str(tmp_path / "canonical_dump")
    cfg.data.target_layer, cfg.data.batch_size, cfg.data.val_frac = 1, 4, .25
    cfg.model.hidden_size, cfg.model.latent_dim, cfg.model.n_clusters = 4, 6, 2
    cfg.train.n_epochs = 1
    cfg.train.checkpoints_dir = str(tmp_path / "checkpoints")
    cfg.train.diag_every, cfg.train.reinit_every = 1000, 1000
    original = cfg.to_dict()
    provenance = {"source_fingerprint": dataset.SOURCE_FINGERPRINT,
                  "source_mix": dict(dataset.USER_MIX), "model": cfg.extraction.model_name,
                  "layers": [1], "activations_dir": cfg.data.activations_dir}
    monkeypatch.setenv("GEOAE_TRAIN_ACTIVATIONS_DIR", str(staged))
    monkeypatch.setenv("GEOAE_DATASET_PROVENANCE", json.dumps(provenance))
    monkeypatch.setenv("GEOAE_TRAIN_WORKERS", "0")
    monkeypatch.setenv("GEOAE_VAL_WORKERS", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    training.train(cfg, use_wandb=False)
    checkpoint = torch.load(next(Path(cfg.train.checkpoints_dir).glob("step_*.pt")),
                            map_location="cpu", weights_only=False)
    assert checkpoint["config"] == original == cfg.to_dict()
    assert checkpoint["config"]["data"]["activations_dir"] != str(staged)
    assert checkpoint["dataset_provenance"] == provenance
    assert checkpoint["epoch"] == 1 and checkpoint["step"] == 3
    assert "tb_table" not in checkpoint["model_state"]
    assert not Path(cfg.data.activations_dir).exists()
    assert (source / "layer_1.npy").read_bytes() == (staged / "layer_1.npy").read_bytes()


@pytest.mark.parametrize("stage", ["train", "evaluate"])
def test_plain_runner_uses_staged_runtime_without_requiring_token_table(tmp_path, monkeypatch, stage):
    monkeypatch.chdir(tmp_path)
    arm = "llama_mid_plain"
    profile = run.PROFILES[arm]
    extraction, training = (ROOT / p for p in run.config_paths(arm))
    provenance = dataset.validate_extraction_config(extraction)
    staged = tmp_path / "ram" / "verified_dump"
    staged.mkdir(parents=True)
    monkeypatch.setattr(run, "config_paths", lambda _: (extraction, training))
    monkeypatch.setattr(dataset, "check_dump", lambda *a, **k: {"dataset_provenance": provenance})
    monkeypatch.setattr(run, "check_table", lambda *a, **k: pytest.fail("Plain training checked a token table"))
    monkeypatch.setattr(run, "stage_dump", lambda *a, **k: staged)
    checkpoints = iter([None, Path("final.pt")])

    def checkpoint_for(*args, **kwargs):
        value = next(checkpoints)
        if value is None:
            raise FileNotFoundError("not trained yet")
        return value

    monkeypatch.setattr(run, "checkpoint_for", checkpoint_for)
    monkeypatch.setattr(run, "check_training", lambda *a, **k: {"dataset_provenance": provenance})
    commands = []
    args = SimpleNamespace(steps=stage, run_name="test", eval_steps="evals", judges="skip",
                           force_evals=False, dry_run=False, data_root=None, stage_dump_root=tmp_path / "ram")
    out = run.arm_paths(profile, args.run_name)["out"]

    def subprocess_run(command, **kwargs):
        commands.append((command, kwargs["env"]))
        if stage == "evaluate":
            (out / "pipeline_state.json").write_text('{"status": "complete"}')

    monkeypatch.setattr(run.subprocess, "run", subprocess_run)
    assert run.run_arm(arm, args)
    assert len(commands) == 1
    command, environment = commands[0]
    variable = "GEOAE_TRAIN_ACTIVATIONS_DIR" if stage == "train" else "GEOAE_EVAL_ACTIVATIONS_DIR"
    assert environment[variable] == str(staged.resolve())
    assert "--activations_dir" not in command
    if stage == "train":
        assert json.loads(environment["GEOAE_DATASET_PROVENANCE"]) == provenance
        assert command[command.index("--config") + 1] == str(training)
    state = json.loads((out / "trainer_state.json").read_text())
    assert state["runtime_activations_dir"] == str(staged.resolve())
    assert state["dataset_provenance"] == provenance
    assert state["ae_mode"] == "plain" and state["status"] == "complete"
