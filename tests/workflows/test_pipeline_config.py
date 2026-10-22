"""The pipeline uses the supplied recipe consistently without rewriting it."""
import sys
from pathlib import Path

import pytest
import yaml

import geoae.workflow.pipeline as pipeline
from geoae.config import Config


@pytest.mark.parametrize("overrides", [[], ["--layer", "4", "--n_clusters", "5", "--latent_dim", "9", "--n_tokens", "23"]])
def test_pipeline_recipe_controls_all_phases(tmp_path, monkeypatch, overrides):
    cfg = Config()
    cfg.extraction.layers = [2]
    cfg.extraction.target_layer = cfg.data.target_layer = 2
    cfg.extraction.model_name = "fixture/frozen-head"
    cfg.model.n_clusters, cfg.model.latent_dim = 3, 6
    cfg.train.seed = 17
    cfg.train.checkpoints_dir = str(tmp_path / "custom-checkpoints")
    checkpoint = Path(cfg.train.checkpoints_dir) / "best_val.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    recipe = tmp_path / "custom.yaml"
    original = yaml.safe_dump(cfg.to_dict())
    recipe.write_text(original)
    commands, recipes = [], []

    def capture(command):
        commands.append(command)
        if "--config" in command:
            path = Path(command[command.index("--config") + 1])
            recipes.append(Config.from_yaml(path).to_dict())

    monkeypatch.setattr(pipeline, "run", capture)
    monkeypatch.setattr(sys, "argv", ["geoae-pipeline", "all", "--config", str(recipe), *overrides])
    pipeline.main()
    assert [cmd[2] for cmd in commands] == ["geoae.extraction.prefix", "geoae.training.mse", "geoae.evaluation.faithfulness"]
    assert recipes[0] == recipes[1]
    effective = recipes[0]
    assert effective["train"]["checkpoints_dir"] == str(checkpoint.parent)
    assert effective["extraction"]["model_name"] == "fixture/frozen-head"
    layer = 4 if overrides else 2
    assert effective["data"]["target_layer"] == effective["extraction"]["target_layer"] == layer
    assert effective["extraction"]["layers"] == [layer]
    assert effective["model"]["n_clusters"] == (5 if overrides else 3)
    assert effective["model"]["latent_dim"] == (9 if overrides else 6)
    assert effective["extraction"]["n_tokens"] == (23 if overrides else cfg.extraction.n_tokens)
    evaluation = commands[-1]
    assert evaluation[evaluation.index("--checkpoint") + 1] == str(checkpoint)
    assert evaluation[evaluation.index("--layer") + 1] == str(layer)
    assert all(cmd[cmd.index("--seed") + 1] == "17" for cmd in (commands[0], evaluation))
    assert recipe.read_text() == original
    assert not Path(commands[0][commands[0].index("--config") + 1]).exists()


def test_default_pipeline_retains_run_id_checkpoint_directory(tmp_path, monkeypatch):
    cfg = Config()
    (tmp_path / "default.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    monkeypatch.setattr(pipeline, "BASE_CONFIGS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["geoae-pipeline", "train", "--n_clusters", "7"])
    captured = []

    def capture(command):
        captured.append(Config.from_yaml(command[command.index("--config") + 1]))

    monkeypatch.setattr(pipeline, "run", capture)
    pipeline.main()
    assert captured[0].train.checkpoints_dir == "checkpoints/layer27_k7_d2048"
