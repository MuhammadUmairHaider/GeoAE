"""Matched recipes, bounded cache inspection, and safe checkpoint reuse."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_training", ROOT / "scripts/deltaai/nonlinear_paper_training.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("name,dim,latent,width", [
    ("llama3.2-3b_l27", 3072, 6144, 768),
    ("gemma3-4b_l33", 2560, 5120, 640),
])
@pytest.mark.parametrize("arm", runner.ARMS)
def test_exact_baseline_recipe_changes_only_decoder_and_output_mounts(name, dim, latent, width, arm, tmp_path):
    from geoae.config import Config
    folder = ROOT / "configs/nonlinear_decoder" / name
    original = yaml.safe_load((folder / "reference_bypass.yaml").read_text())
    source = copy.deepcopy(original)
    rendered = runner.render_recipe(original, tmp_path, {"data_dir": "/cache", "token_table": "/table.npz"}, arm)
    assert original == source
    assert rendered["model"]["hidden_size"] == dim
    assert rendered["model"]["latent_dim"] == latent
    assert rendered["model"]["decoder_hidden_dim"] == (latent if arm == "mlp" else width)
    assert rendered["model"]["decoder_output_bias"] == (arm == "mlp")
    assert rendered["data"]["activations_dir"] == "/cache"
    assert rendered["model"]["token_bias"] == "/table.npz"
    for section, key in (("data", "activations_dir"), ("model", "token_bias"),
                         ("train", "checkpoints_dir"), ("train", "runs_dir")):
        rendered[section].pop(key)
        source[section].pop(key)
    for key in ("decoder_type", "decoder_hidden_dim", "decoder_nonlinearity", "decoder_output_bias"):
        rendered["model"].pop(key)
    assert rendered == source
    cfg = Config.from_yaml(folder / f"{arm}.yaml")
    assert cfg.train.n_epochs == 50 and cfg.train.seed == 42 and cfg.data.batch_size == 32768
    assert cfg.model.decoder_hidden_dim == (latent if arm == "mlp" else width)


@pytest.fixture
def cache(tmp_path):
    data = tmp_path / "cache"
    data.mkdir()
    n, d = 40000, 8
    np.save(data / "layer_27.npy", np.zeros((n, d), dtype=np.float32))
    for name in ("rows_tok", "rows_doc", "rows_pos", "rows_src", "rows_norm"):
        np.save(data / (name + ".npy"), np.zeros(n, dtype=np.int32))
    np.save(data / "docs.npy", np.asarray(["one", "two"]))
    (data / "meta.json").write_text(json.dumps({"model": "fixture", "target_layer": 27,
                                                "n_tokens": n, "hidden_size": d}))
    np.savez(data / "norm_params_layer27.npz", mean=np.zeros(d), std=np.ones(d))
    table = tmp_path / "table.npz"
    np.savez(table, model_name="fixture", layer=27, table=np.zeros((10, d), dtype=np.float32),
             norm_mean=np.zeros(d), norm_std=np.ones(d), vocab_size=10,
             report=json.dumps({"train_rows": int(n * .95)}))
    profile = {"model_name": "fixture", "target_layer": 27, "hidden_size": d, "latent_dim": 2*d,
               "final_step": 50, "assets": {"activations_dir": str(data), "token_bias_table": str(table)}}
    return profile


def test_dataset_inspection_preserves_float32_and_computes_actual_steps(cache):
    result = runner.inspect_dataset(cache)
    assert result["dtype"] == "float32"
    assert result["shape"] == [40000, 8]
    assert result["train_rows"] == 38000 and result["val_rows"] == 2000
    assert result["steps_per_epoch"] == 1 and result["total_steps"] == 50
    assert "head_tail_64k_sha256" in result["files"]["layer_27.npy"]
    assert runner.npz_shape(Path(cache["assets"]["token_bias_table"]), "table") == (10, 8)


def test_dataset_rejects_different_update_count(cache):
    cache["final_step"] = 100
    with pytest.raises(ValueError, match="update count"):
        runner.inspect_dataset(cache)


def test_dataset_rejects_foreign_token_normalization(cache):
    p = Path(cache["assets"]["activations_dir"]) / "norm_params_layer27.npz"
    np.savez(p, mean=np.zeros(8), std=np.ones(8)*2)
    with pytest.raises(ValueError, match="Token table normalization"):
        runner.inspect_dataset(cache)


def test_staged_cache_is_verified_and_never_deleted(cache, tmp_path):
    manifest = {"dataset": runner.inspect_dataset(cache)}
    parent = tmp_path / "stage"
    parent.mkdir()
    stage = runner.stage_data(manifest, parent)
    runner.verify_staged_data(manifest, stage)
    assert np.load(stage / "layer_27.npy", mmap_mode="r").dtype == np.float32
    with (stage / "rows_tok.npy").open("r+b") as stream:
        stream.seek(-4, 2)
        stream.write(b"bad!")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        runner.verify_staged_data(manifest, stage)
    assert stage.is_dir()


def test_cpu_preflight_uses_no_activation_cache():
    report = runner.cpu_preflight()
    assert report["activation_rows_loaded"] == 0
    assert set(report["arms"]) == set(runner.ARMS)
    for value in report["arms"].values():
        assert value["input_shape"] == value["output_shape"] == [8, 8]
        assert np.isfinite(value["loss"])


def test_resume_accepts_only_exact_recipe_and_complete_final_step(tmp_path):
    from geoae.config import Config
    cfg = Config.from_yaml(ROOT / "configs/nonlinear_decoder/llama3.2-3b_l27/mlp.yaml")
    folder = tmp_path / "checkpoints/mlp"
    folder.mkdir(parents=True)
    assert runner.resume_args(tmp_path, "mlp", cfg, 14450) == []
    p = folder / "step_0014450.pt"
    ck = {"config": cfg.to_dict(), "epoch": 50, "step": 14450}
    torch.save(ck, p)
    assert runner.resume_args(tmp_path, "mlp", cfg, 14450) is None
    ck["config"]["model"]["decoder_type"] = "residual_mlp"
    torch.save(ck, p)
    with pytest.raises(ValueError, match="recipe differs"):
        runner.resume_args(tmp_path, "mlp", cfg, 14450)
    ck["config"] = cfg.to_dict()
    ck["step"] = 14449
    torch.save(ck, p)
    with pytest.raises(ValueError, match="contradicts"):
        runner.resume_args(tmp_path, "mlp", cfg, 14450)


def test_profile_rejects_wrong_model_dimensions(tmp_path):
    p = tmp_path / "profile.json"
    obj = {"model_name": "meta-llama/Llama-3.2-3B", "layer": 27, "hidden_size": 2560,
           "latent_dim": 6144, "assets": {"activations_dir": "/data", "token_bias_table": "/table", "linear_checkpoint": "/linear"}}
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="dimensions"):
        runner.load_profile(p)
    obj["hidden_size"] = 3072
    p.write_text(json.dumps(obj))
    assert runner.load_profile(p)["target_layer"] == 27


def test_source_snapshot_changes_are_detectable(tmp_path):
    (tmp_path / "geoae").mkdir()
    p = tmp_path / "geoae/model.py"
    p.write_text("original")
    first = runner.source_hashes(tmp_path)
    p.write_text("mutated")
    assert first != runner.source_hashes(tmp_path)


def test_existing_preparation_is_idempotent_but_refuses_profile_changes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    saved = {"profile": {"model": "llama"}, "runtime": {"python": sys.executable, "modules": []}}
    (tmp_path / "training_manifest.json").write_text("{}")
    monkeypatch.setattr(runner, "validate", lambda *args: saved)
    assert runner.prepare(saved["profile"], tmp_path, source, saved["runtime"]) is saved
    with pytest.raises(ValueError, match="profile/runtime"):
        runner.prepare({"model": "gemma"}, tmp_path, source, saved["runtime"])


def test_checkpoint_tuple_betas_match_yaml_reference_without_mutation(tmp_path, monkeypatch):
    folder = ROOT / "configs/nonlinear_decoder/llama3.2-3b_l27"
    config = yaml.safe_load((folder / "reference_bypass.yaml").read_text())
    assert config["train"]["betas"] == [.9, .999]
    config["train"]["betas"] = (.9, .999)
    checkpoint = {"config": config, "epoch": 50, "step": 14450, "val_mse": .03,
                  "model_state": {"decoder.weight": SimpleNamespace(shape=(3072, 6144))},
                  "norm_mean": np.zeros(3072), "norm_std": np.ones(3072)}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    path = tmp_path / "reference.pt"
    path.write_bytes(b"trusted checkpoint fixture")
    np.savez(tmp_path / "norm_params_layer27.npz", mean=np.zeros(3072), std=np.ones(3072))
    profile = {"model_name": "meta-llama/Llama-3.2-3B", "target_layer": 27,
               "hidden_size": 3072, "latent_dim": 6144,
               "assets": {"linear_checkpoint": str(path)}}
    inspected = runner.inspect_baseline(profile, {"total_steps": 14450, "data_dir": str(tmp_path)})
    assert inspected["config"]["train"]["betas"] == (.9, .999)
    assert checkpoint["config"]["train"]["betas"] == (.9, .999)
    reloaded = json.loads(json.dumps(inspected))
    assert reloaded["config"]["train"]["betas"] == [.9, .999]
    assert runner.canonical_config(reloaded["config"]) == runner.canonical_config(config)


def test_resume_accepts_tuple_list_serialization_but_rejects_changed_betas(tmp_path):
    from geoae.config import Config
    cfg = Config.from_yaml(ROOT / "configs/nonlinear_decoder/llama3.2-3b_l27/mlp.yaml")
    folder = tmp_path / "checkpoints/mlp"
    folder.mkdir(parents=True)
    saved = cfg.to_dict()
    saved["train"]["betas"] = tuple(saved["train"]["betas"])
    path = folder / "step_0014450.pt"
    checkpoint = {"config": saved, "epoch": 50, "step": 14450}
    torch.save(checkpoint, path)
    assert runner.resume_args(tmp_path, "mlp", cfg, 14450) is None
    saved["train"]["betas"] = (.9, .998)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="recipe differs"):
        runner.resume_args(tmp_path, "mlp", cfg, 14450)
