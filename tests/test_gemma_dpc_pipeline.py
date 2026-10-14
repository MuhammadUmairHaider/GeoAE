"""CPU-only orchestration/guard tests. No Hub access, LM loading or GPU jobs."""
from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml

from geoae.config import Config
from geoae.gemma_dpc import (
    DEFAULT_CONFIG, Pipeline, Stage, completed, inspect_checkpoint, main,
    output_signature, reduce_atlas, run, validate_activations, validate_outputs, write_json,
)
from geoae.gemma_activation_probe import range_metrics


@pytest.fixture
def pipeline(tmp_path):
    return Pipeline(DEFAULT_CONFIG, tmp_path / "experiment")


def checkpoint(p, path, *, epoch=50, initialized=True, lr=None):
    torch.save({
        "config": p.cfg.to_dict(), "epoch": epoch, "step": 150,
        "val_mse": .01,
        "model_state": {"centroids_initialized": torch.tensor(initialized), "weight": torch.ones(2)},
        "norm_mean": np.zeros(1152, np.float32), "norm_std": np.ones(1152, np.float32),
        "opt_state": {"param_groups": [{"lr": p.cfg.train.lr if lr is None else lr}]},
    }, path)


def test_plan_is_non_mutating(pipeline, capsys):
    main(["plan", "--run-dir", str(pipeline.root), "--stages", "number"])
    assert not pipeline.root.exists()
    text = capsys.readouterr().out
    assert "Nothing launched" in text
    assert "--saliency dprime" in text
    assert "Llama" not in text


def test_recipe_and_yaml_roundtrip(pipeline):
    p = pipeline
    assert p.cfg.model.latent_dim == 2304
    assert p.cfg.model.n_clusters == 2000
    assert p.cfg.train.centroid_init == "dpc"
    assert p.cfg.train.reinit_mode == "peaks"
    p.root.mkdir()
    p.effective_config.write_text(yaml.safe_dump(p.cfg.to_dict()))
    assert Config.from_yaml(p.effective_config).to_dict() == p.cfg.to_dict()


def test_reject_unsafe_paths_and_configs(tmp_path):
    from geoae.paths import PACKAGE_ROOT
    with pytest.raises(ValueError, match="dedicated"):
        Pipeline(run_dir=PACKAGE_ROOT)
    with pytest.raises(ValueError, match="comma"):
        Pipeline(run_dir=tmp_path / "bad,run")
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg["extraction"]["dtype"] = "float16"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="float32"):
        Pipeline(path, tmp_path / "run")


def test_dependency_order_and_isolated_paths(pipeline):
    stages = pipeline.selected(["number"])
    assert [s.name for s in stages] == ["preflight", "extract", "validate_activations", "train", "number"]
    visited = set()
    for s in pipeline.selected(None):
        assert set(s.deps) <= visited
        assert all(p.is_relative_to(pipeline.root) for p in s.outputs)
        visited.add(s.name)
    with pytest.raises(ValueError, match="Unknown stage"):
        pipeline.selected(["typo"])
    probe = pipeline.stages["probe"].command
    assert "auto" not in probe and "--exclude_anchors" not in probe
    assert all(x in probe[probe.index("--baselines") + 1] for x in
               ["balanced_kmeans=", "balanced_dpc=", "plain_kmeans="])
    assert pipeline.stages["mmlu"].cwd == pipeline.evals
    assert pipeline.stages["mmlu"].outputs == (pipeline.evals / "results_ccc_mmlu.json",)
    assert "--concepts" not in pipeline.stages["range_biasbios"].command


@pytest.mark.parametrize("epoch,initialized,lr,match", [
    (10, False, None, "final epoch"),
    (50, False, None, "Centroids"),
    (50, True, .00003, "learning rate"),
])
def test_checkpoint_guards(pipeline, tmp_path, epoch, initialized, lr, match):
    path = tmp_path / "checkpoint.pt"
    checkpoint(pipeline, path, epoch=epoch, initialized=initialized, lr=lr)
    with pytest.raises(ValueError, match=match):
        inspect_checkpoint(path, pipeline.cfg, require_final=True)


def test_valid_final_checkpoint(pipeline, tmp_path):
    path = tmp_path / "checkpoint.pt"
    checkpoint(pipeline, path)
    info = inspect_checkpoint(path, pipeline.cfg, require_final=True)
    assert info["epoch"] == 50 and info["centroids_initialized"]
    assert len(info["sha256"]) == 64
    pipeline.cfg.model.latent_dim += 1
    with pytest.raises(ValueError, match="model config"):
        inspect_checkpoint(path, pipeline.cfg, require_final=True)


def test_activation_scan(pipeline):
    p = pipeline
    p.acts.mkdir(parents=True)
    p.cfg.extraction.n_tokens = 100
    x = np.ones((100, 1152), np.float32)
    np.save(p.acts / "layer_25.npy", x)
    meta = {"model": "google/gemma-3-1b-pt", "layers": [25], "n_tokens": 100,
            "domain_tokens": {"web": 40, "math": 30, "code": 30}}
    write_json(p.acts / "meta.json", meta)
    assert validate_activations(p)["all_rows_finite"]
    x[70, 100] = np.inf
    np.save(p.acts / "layer_25.npy", x)
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_activations(p)


def test_atlas_label_reductions():
    labels = [[1, 2], [1], [1, 2], [1, 3], [], [2]]
    idx, y = reduce_atlas(labels, "single", min_support=1)
    assert idx.tolist() == [1, 5] and y.tolist() == [1, 2]
    idx, y = reduce_atlas(labels, "rarest", min_support=1)
    assert y.tolist() == [2, 1, 2, 3, 2]
    _, y = reduce_atlas(labels, "commonest", min_support=1)
    assert y.tolist() == [1, 1, 1, 1, 2]
    idx, y = reduce_atlas(labels, "rarest", min_support=3)
    assert idx.tolist() == [0, 2, 5]


def test_partial_or_changed_output_not_complete(pipeline):
    p = pipeline
    stage = p.stages["number"]
    write_json(stage.outputs[0], {"status": "running", "arms": {}})
    assert not completed(p, stage)
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_outputs(stage)
    write_json(stage.outputs[0], {"status": "complete", "arms": {str(i): {} for i in range(144)}})
    write_json(p.root / "state/number.json", {"outputs": output_signature(stage.outputs)})
    assert completed(p, stage)
    stage.outputs[0].write_text(stage.outputs[0].read_text() + "\n")
    with pytest.raises(RuntimeError, match="changed"):
        completed(p, stage)


def test_runner_does_not_adopt_unowned_directory(pipeline):
    pipeline.root.mkdir()
    sentinel = pipeline.root / "user-data.txt"
    sentinel.write_text("preserve me")
    with pytest.raises(RuntimeError, match="Nonempty"):
        run(pipeline, [], False)
    assert sentinel.read_text() == "preserve me"


def test_runner_real_local_child_and_skip(pipeline):
    # Exercise manifest/config serialization and checkpoint-free stage execution.
    import sys
    p = pipeline
    out = p.root / "dummy.json"
    code = "from pathlib import Path; Path('dummy.json').write_text('{}')"
    stage = Stage("dummy", (), (sys.executable, "-c", code), (out,), p.root)
    p.stages = {"dummy": stage}
    run(p, [stage], False)
    assert completed(p, stage)
    stamp = out.stat().st_mtime_ns
    run(p, [stage], False)
    assert out.stat().st_mtime_ns == stamp
    p.stages["dummy"] = replace(stage, command=stage.command + ("changed",))
    with pytest.raises(RuntimeError, match="changed"):
        run(p, [p.stages["dummy"]], False)


def test_probe_distinguishes_fp16_and_bf16_and_does_not_clip():
    x = np.array([[1, 100000], [3, 80000]], np.float32)
    before = x.copy()
    result = range_metrics(x)
    np.testing.assert_array_equal(x, before)
    assert result["channels_over_float16_max"] == [1]
    assert result["fraction_tokens_over_float16_max"] == 1
    assert result["elements_over_bfloat16_max"] == 0
    assert result["counterfactual_fp16_clip_relative_squared_change"] > 0
    assert range_metrics(np.array([[np.inf]], np.float32))["safe_for_training"] is False


def test_every_model_evaluation_uses_sealed_checkpoint(pipeline):
    for stage in pipeline.stages.values():
        command = list(stage.command)
        if "--checkpoint" in command:
            assert command[command.index("--checkpoint") + 1] == str(pipeline.final)
        assert not any("best_val.pt" in x for x in command)


@pytest.mark.parametrize("stage_name", [
    "extract", "balanced_kmeans", "balanced_dpc", "plain_kmeans", "concept_cache",
    "ravel_cache", "mmlu", "cq_balanced_kmeans", "probe", "geometry",
    "range_db14", "range_ag_news", "range_biasbios", "steer_db14", "number",
])
def test_stage_cli_contract_without_executing(pipeline, monkeypatch, stage_name):
    """Parse actual stage arguments, then stop before any model/data I/O."""
    import argparse
    import importlib
    import sys

    class Parsed(Exception):
        pass

    stage = pipeline.stages[stage_name]
    argv = list(stage.command)
    module_index = argv.index("-m") + 1
    module = importlib.import_module(argv[module_index])
    parse = argparse.ArgumentParser.parse_args

    def parse_and_stop(parser, *args, **kwargs):
        parse(parser, *args, **kwargs)
        raise Parsed()

    monkeypatch.setattr(sys, "argv", [argv[module_index], *argv[module_index + 1:]])
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_and_stop)
    with pytest.raises(Parsed):
        module.main()
