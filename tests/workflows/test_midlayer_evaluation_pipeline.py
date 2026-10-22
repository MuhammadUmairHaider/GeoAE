"""Middle-block evaluations must use the correct model, token bypass and context."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from experiments.gpt2 import evaluate as research
from experiments.depth import evaluate


def options(arm="llama", **overrides):
    args = evaluate.parser().parse_args([arm, "--dry-run"])
    for name, value in overrides.items():
        setattr(args, name, value)
    return evaluate.options_for(arm, args)


def plan(arm):
    args = options(arm)
    layer = args.profile["layer"]
    checkpoint = research.checkpoint_for(layer, dry_run=True, profile=args.profile)
    return research.build_steps(layer, checkpoint, Path(args.activations_dir), args), args


@pytest.mark.parametrize("arm", ["llama", "gemma4b"])
def test_whole_recipe_includes_every_supported_research_group(arm):
    stages, args = plan(arm)
    names = research.select_steps("whole", stages)
    assert {"km", "cache", "bench", "probe_acts", "wsd_data", "core_eval", "tokstruct", "probe", "mmlu",
            "cq", "dispersion", "number", "number_hb", "steer", "range_db14", "range_bios", "geometry",
            "biasprobe", "tpp", "ct_base", "ct_tok", "ct", "wsd", "cgen", "cgen_concept", "cjudge",
            "cjudge_concept", "judge", "judge_tok", "judge3", "atlas", "atlas_tok", "pair"} <= names
    assert "dispersion" in research.select_steps("evals", stages)
    assert "dispersion" in research.select_steps("gpu", stages)
    assert "pair_import" not in names
    assert not any(step.api for step in stages if step.name in research.select_steps("gpu", stages))
    table = {step.name: step for step in stages}
    assert "--skip_next" in table["tokstruct"].commands[0]
    assert table["csteer"].last_block_only
    assert not any(table[name].last_block_only for name in ("cgen", "cgen_concept", "cjudge", "cjudge_concept"))
    assert table["cgen"].commands[0][table["cgen"].commands[0].index("--corpus_dir") + 1] == args.corpus_dir
    assert len(table["geometry"].outputs) == 3 * len(research.GEOMETRY_RUNGS.split(","))
    assert len(table["geometry"].optional_outputs) == len(research.GEOMETRY_RUNGS.split(","))
    assert all("/umap/" in str(path) for path in table["geometry"].optional_outputs)


@pytest.mark.parametrize("arm", ["llama", "gemma4b"])
def test_artifacts_and_explicit_layers_are_model_isolated(arm):
    stages, args = plan(arm)
    profile, layer = args.profile, args.profile["layer"]
    paths = research.profile_paths(layer, args)
    assert str(paths["out"]) == f"results/midlayer_run/{profile['family']}/layer{layer}"
    assert f"{profile['family']}/layer{layer}" in str(paths["table"])
    assert str(paths["cache"]) == f"cache/{profile['family']}/layer{layer}"
    model_directory = profile["model_name"].split("/")[-1].replace("Llama-", "llama")
    assert paths["bb"].parts[-2:] == (model_directory, "last")
    assert paths["db"].parts[-3:] == (model_directory, "last_ids", "unprompted")
    if arm == "llama":
        assert str(paths["bb"]) == "biasbios/activations/llama3.2-3B/layer14/llama3.2-3B/last"
        assert str(paths["db"]) == "dbpedia/activations/llama3.2-3B/layer14/llama3.2-3B/last_ids/unprompted"
    for step in stages:
        for output in step.outputs:
            if str(output).startswith("results/"):
                assert output.is_relative_to(paths["out"]), (step.name, output)
        for command in step.commands:
            if "--layer" in command:
                assert command[command.index("--layer") + 1] == str(layer)
            if "--model_name" in command:
                assert command[command.index("--model_name") + 1] == profile["model_name"]
    rendered = " ".join(str(word) for step in stages for command in step.commands for word in command)
    assert "checkpoints/gpt2" not in rendered and "cache/gpt2" not in rendered


def test_profile_checkpoints_use_numeric_latest_and_exclude_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = evaluate.PROFILES["gemma4b"]
    directory = Path("checkpoints") / profile["family"] / "layer17" / profile["ckdir"]
    directory.mkdir(parents=True)
    for name in ("best_val.pt", "step_99.pt", "step_100.pt", "step_incomplete.pt"):
        (directory / name).touch()
    assert research.checkpoint_for(17, profile=profile).name == "step_100.pt"


@pytest.mark.parametrize("mutation,expected", [
    ({"model_name": "other/model"}, "model="),
    ({"target_layer": 33}, "layer="),
    ({"hidden_size": 1152}, "hidden_size="),
    ({"token_bias_mode": "decoder_only"}, "token_bias_mode="),
    ({"epoch": 49}, "epoch="),
    ({"centroids_initialized": False}, "initialized="),
    ({"token_bias": False}, "token_bias="),
])
def test_guard_refuses_wrong_or_unfinished_bypass_checkpoint(tmp_path, mutation, expected):
    import torch
    profile = evaluate.PROFILES["gemma4b"]
    value = {"epoch": 50, "config": {
        "extraction": {"model_name": profile["model_name"]}, "data": {"target_layer": 17},
        "model": {"hidden_size": 2560, "latent_dim": 5120, "n_clusters": 2000, "token_bias_mode": "bypass"},
    }, "model_state": {"tb_table": torch.ones(1, 2), "centroids_initialized": torch.tensor(True)}}
    path = tmp_path / "step_100.pt"
    torch.save(value, path)
    research.guard_checkpoint(path, 17, profile)
    for key, new in mutation.items():
        if key == "model_name":
            value["config"]["extraction"][key] = new
        elif key == "target_layer":
            value["config"]["data"][key] = new
        elif key in ("hidden_size", "token_bias_mode"):
            value["config"]["model"][key] = new
        elif key == "centroids_initialized":
            value["model_state"][key] = torch.tensor(new)
        elif key == "token_bias":
            value["model_state"].pop("tb_table")
        else:
            value[key] = new
    torch.save(value, path)
    with pytest.raises(ValueError, match=expected):
        research.guard_checkpoint(path, 17, profile)


def test_middle_layer_direct_head_skip_is_recorded_but_context_generation_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = options(dry_run=False, steps="csteer,cgen", judges="skip")
    out = research.profile_paths(14, args)["out"]
    final_only = research.Step("csteer", [[sys.executable, "-c", "raise RuntimeError('must skip')"]],
                               [out / "direct.json"], last_block_only=True)
    target = out / "generated.json"
    generation = research.Step("cgen", [[sys.executable, "-c", f"from pathlib import Path; Path({str(target)!r}).write_text('{{}}')"]],
                                [target])
    with patch.object(research, "build_steps", return_value=[final_only, generation]), \
            patch.object(research, "code_files", return_value=[]):
        assert research.run_layer(14, args)
    state = json.loads((out / "pipeline_state.json").read_text())
    assert state["model"] == "meta-llama/Llama-3.2-3B"
    assert state["status"] == "complete"
    assert state["steps"]["csteer"]["status"] == "skipped"
    assert "requires layer 27" in state["steps"]["csteer"]["reason"]
    assert state["steps"]["cgen"]["status"] == "complete"


def test_all_recipe_dry_run_and_code_transfer_are_tensor_free():
    command = [sys.executable, str(evaluate.ROOT / "experiments/depth/evaluate.py")]
    dry = subprocess.run([*command, "all", "--dry-run", "--judges", "skip"], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert "results/midlayer_run/llama3.2-3B/layer14" in dry.stdout
    assert "results/midlayer_run/gemma3-4B/layer17" in dry.stdout
    assert dry.stdout.count("[dry] cgen:") == 8
    assert dry.stdout.count("[dry] dispersion:") == 8
    assert "requires layer 27" in dry.stdout and "requires layer 33" in dry.stdout
    transfer = subprocess.run([*command, "all", "--print-code-files"], capture_output=True, text=True)
    assert transfer.returncode == 0, transfer.stderr
    files = json.loads(transfer.stdout)
    assert "experiments/depth/evaluate.py" in files and "experiments/gpt2/evaluate.py" in files
    assert "experiments/paper/analysis/paper_dispersion.py" in files
    assert "geoae/evaluation/dispersion.py" in files


@pytest.mark.parametrize("changed", ["checkpoint", "activations_dir", "corpus_dir"])
def test_completed_metrics_cannot_be_adopted_for_different_inputs(tmp_path, monkeypatch, changed):
    monkeypatch.chdir(tmp_path)
    args = options(dry_run=False, steps="dummy")
    out = research.profile_paths(14, args)["out"]
    out.mkdir(parents=True)
    state_path = out / "pipeline_state.json"
    checkpoint = research.checkpoint_for(14, dry_run=True, profile=args.profile)
    original = {"checkpoint": str(checkpoint), "activations_dir": args.activations_dir,
                "corpus_dir": args.corpus_dir, "steps": {"dummy": {"status": "complete"}}}
    original[changed] = "previous-input"
    state_path.write_text(json.dumps(original))
    stage = research.Step("dummy", [[sys.executable, "-c", "raise RuntimeError('must not launch')"]],
                          [out / "metric.json"])
    with patch.object(research, "build_steps", return_value=[stage]):
        with pytest.raises(ValueError, match=f"different {changed}"):
            research.run_layer(14, args)
    assert json.loads(state_path.read_text()) == original


def test_missing_optional_umap_is_explicitly_partial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = options(dry_run=False, steps="geometry")
    out = research.profile_paths(14, args)["out"]
    target = out / "pca.png"
    step = research.Step("geometry", [[sys.executable, "-c", f"from pathlib import Path; Path({str(target)!r}).write_bytes(b'figure')"]],
                         [target], optional_outputs=[out / "umap.png"])
    with patch.object(research, "build_steps", return_value=[step]), \
            patch.object(research, "code_files", return_value=[]):
        assert research.run_layer(14, args)
    state = json.loads((out / "pipeline_state.json").read_text())
    assert state["status"] == "partial"
    assert state["steps"]["geometry"]["status"] == "partial"
    assert state["steps"]["geometry"]["optional_missing"] == [str(out / "umap.png")]


def test_checkpoint_free_prep_can_precede_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = options(dry_run=False, steps="dummy", checkpoint="new-final.pt")
    checkpoint = Path(args.checkpoint)
    checkpoint.touch()
    out = research.profile_paths(14, args)["out"]
    out.mkdir(parents=True)
    old = {"checkpoint": "<latest-step>.pt", "activations_dir": args.activations_dir,
           "corpus_dir": args.corpus_dir, "requested": ["km"], "steps": {"km": {"status": "complete"}}}
    (out / "pipeline_state.json").write_text(json.dumps(old))
    target = out / "metric.json"
    metric = research.Step("dummy", [[sys.executable, "-c", f"from pathlib import Path; Path({str(target)!r}).write_text('{{}}')"]],
                           [target], requires=[checkpoint])
    with patch.object(research, "build_steps", return_value=[metric]), \
            patch.object(research, "guard_checkpoint"), patch.object(research, "code_files", return_value=[]):
        assert research.run_layer(14, args)
    state = json.loads((out / "pipeline_state.json").read_text())
    assert state["checkpoint"] == "new-final.pt"
    assert state["steps"]["km"]["status"] == "complete"
    assert state["steps"]["dummy"]["status"] == "complete"


@pytest.mark.parametrize("arm", evaluate.CANONICAL_ARMS)
def test_every_recipe_flag_is_declared_by_its_tool(arm):
    stages, _ = plan(arm)
    for step in stages:
        for command in step.commands:
            source = (Path(command[command.index("-m") + 1].replace(".", "/") + ".py")
                      if "-m" in command else Path(command[2]))
            tree = ast.parse((research.ROOT / source).read_text())
            declared = {argument.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                        for argument in node.args if isinstance(argument, ast.Constant) and isinstance(argument.value, str)}
            used = {word for word in command[3:] if word.startswith("--")}
            assert used <= declared, (step.name, source, used - declared)


def test_variant_sets_include_six_new_runs_and_eight_total_comparisons():
    assert len(evaluate.CANONICAL_ARMS) == 8
    assert len(evaluate.NEW_ARMS) == 6
    assert set(evaluate.CANONICAL_ARMS) - set(evaluate.NEW_ARMS) == {
        "llama_last_bypass", "gemma4b_last_bypass"}
    assert {evaluate.PROFILES[arm]["mode"] for arm in evaluate.CANONICAL_ARMS} == {"bypass", "plain"}


@pytest.mark.parametrize("family", ["llama", "gemma4b"])
@pytest.mark.parametrize("position", ["mid", "last"])
def test_plain_and_bypass_share_controls_and_keep_metrics_separate(family, position):
    bypass_steps, bypass = plan(f"{family}_{position}_bypass")
    plain_steps, plain = plan(f"{family}_{position}_plain")
    layer = plain.profile["layer"]
    expected_layer = {("llama", "mid"): 14, ("llama", "last"): 27,
                      ("gemma4b", "mid"): 17, ("gemma4b", "last"): 33}[family, position]
    assert layer == expected_layer
    bp = research.profile_paths(layer, bypass)
    pp = research.profile_paths(layer, plain)
    assert bp["out"].name == "bypass" and pp["out"].name == "plain"
    assert bp["out"].parent == pp["out"].parent
    for name in bp.keys() - {"out"}:
        assert bp[name] == pp[name], name
    assert plain.activations_dir == bypass.activations_dir
    assert plain.corpus_dir == bypass.corpus_dir
    plain_checkpoint = research.checkpoint_for(layer, dry_run=True, profile=plain.profile)
    bypass_checkpoint = research.checkpoint_for(layer, dry_run=True, profile=bypass.profile)
    assert plain_checkpoint != bypass_checkpoint
    assert "tokbias" not in str(plain_checkpoint)
    assert "tokbias" not in plain.profile["train_config"]
    assert research.select_steps("whole", plain_steps) == research.select_steps("whole", bypass_steps)
    if position == "last":
        assert "--skip_next" not in next(step for step in plain_steps if step.name == "tokstruct").commands[0]
        assert plain.profile["extract_config"] == bypass.profile["extract_config"]
    for step in plain_steps:
        for output in step.outputs:
            if str(output).startswith("results/"):
                assert output.is_relative_to(pp["out"])


@pytest.mark.parametrize("arm", ["llama_mid_plain", "llama_last_plain", "gemma4b_mid_plain", "gemma4b_last_plain"])
@pytest.mark.parametrize("mutation,expected", [
    ({"tb_table": True}, "token_bias="),
    ({"token_bias": "unexpected/table.npz"}, "token_bias_config="),
    ({"target_layer": -1}, "layer="),
    ({"hidden_size": 1}, "hidden_size="),
    ({"epoch": 49}, "epoch="),
])
def test_plain_guard_refuses_hidden_bypass_wrong_layer_and_unfinished_checkpoint(tmp_path, arm, mutation, expected):
    import torch
    profile = evaluate.PROFILES[arm]
    layer = profile["layer"]
    value = {"epoch": 50, "config": {
        "extraction": {"model_name": profile["model_name"]}, "data": {"target_layer": layer},
        "model": {"hidden_size": profile["hidden_size"], "latent_dim": profile["latent_dim"],
                  "n_clusters": 2000, "token_bias": "", "token_bias_mode": "bypass"},
    }, "model_state": {"centroids_initialized": torch.tensor(True)}}
    path = tmp_path / "step_100.pt"
    torch.save(value, path)
    research.guard_checkpoint(path, layer, profile)
    for key, new in mutation.items():
        if key == "tb_table":
            value["model_state"][key] = torch.zeros(1, profile["hidden_size"])
        elif key == "target_layer":
            value["config"]["data"][key] = new
        elif key == "epoch":
            value[key] = new
        else:
            value["config"]["model"][key] = new
    torch.save(value, path)
    with pytest.raises(ValueError, match=expected):
        research.guard_checkpoint(path, layer, profile)
