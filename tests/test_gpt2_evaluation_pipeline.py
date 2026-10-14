"""Guard the research pipeline's model/layer isolation and execution reporting."""
from argparse import Namespace
import ast
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from scripts.gpt2 import evaluate


def options(**overrides):
    values = dict(judge_model="judge/model", judge_concurrency=6,
                  agreement_judges="judge/one,judge/two", pair_model="pair/model",
                  pair_pairs=100, pair_controls=25, steps="whole", judges="skip",
                  dry_run=True, force=False, activations_dir="activations_gpt2_sampled_struct5_10M",
                  corpus_dir="corpus/gpt2_sampled_struct5_v1", run_name="gpt_run")
    values.update(overrides)
    return Namespace(**values)


def stages(layer):
    ck = Path(f"checkpoints/gpt2/layer{layer}") / evaluate.CKDIR / "step_00014500.pt"
    return evaluate.build_steps(layer, ck, Path("activations_gpt2_sampled_struct5_10M"), options())


def test_whole_pipeline_covers_existing_research_groups():
    plan = stages(0)
    names = evaluate.select_steps("whole", plan)
    assert {"km", "cache", "bench", "probe_acts", "wsd_data", "core_eval", "tokstruct", "probe",
            "mmlu", "cq", "number", "range_bios", "geometry", "biasprobe", "tpp", "ct", "wsd",
            "judge", "judge_tok", "judge3", "atlas", "atlas_tok", "pair"} <= names
    assert "pair_import" not in names  # importing human answers requires a separate action
    assert not any(s.api for s in plan if s.name in evaluate.select_steps("gpu", plan))


def test_layer_paths_and_explicit_layer_arguments_are_isolated():
    for layer in (0, 7, 11):
        plan = stages(layer)
        command_text = " ".join(word for step in plan for cmd in step.commands for word in cmd)
        assert "llama" not in command_text.lower()
        assert "gemma" not in command_text.lower()
        assert f"checkpoints/gpt2/layer{layer}" in command_text
        assert f"cache_ids/gpt2/layer{layer}" in command_text
        assert f"cache/gpt2/layer{layer}" in command_text
        for step in plan:
            for output in step.outputs:
                if str(output).startswith("results/"):
                    assert str(output).startswith(f"results/gpt_run/layer{layer}/")
            for command in step.commands:
                if "--layer" in command:
                    assert command[command.index("--layer") + 1] == str(layer)


def test_intermediate_layers_skip_only_direct_head_metrics():
    plan = stages(5)
    table = {s.name: s for s in plan}
    assert "--skip_next" in table["tokstruct"].commands[0]
    assert not table["core_eval"].last_block_only
    assert not table["range_db14"].last_block_only
    assert not table["number"].last_block_only
    assert {s.name for s in plan if s.last_block_only} == {
        "csteer", "cgen", "cgen_concept", "cjudge", "cjudge_concept"}
    assert "--skip_next" not in {s.name: s for s in stages(11)}["tokstruct"].commands[0]


def test_all_command_flags_exist_in_tool_argument_parsers():
    for step in stages(11):
        for command in step.commands:
            source = Path(command[command.index("-m") + 1].replace(".", "/") + ".py") if "-m" in command else Path(command[2])
            tree = ast.parse((evaluate.ROOT / source).read_text())
            declared = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                        for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
            used = {word for word in command[3:] if word.startswith("--")}
            assert used <= declared, (step.name, source, used - declared)


def test_guard_accepts_model_hidden_size_and_rejects_wrong_layer(tmp_path):
    import torch
    from geoae.config import Config
    from geoae.model import GeoAE
    cfg = Config.from_yaml(evaluate.ROOT / "configs/base/gpt2/layer4_2x.yaml")
    ae = GeoAE(hidden_size=cfg.model.hidden_size, latent_dim=cfg.model.latent_dim,
               n_clusters=cfg.model.n_clusters, token_bias_rows=1, vocab_size=2)
    ae.centroids_initialized.fill_(True)
    checkpoint = tmp_path / "step_999.pt"
    torch.save({"epoch": 50, "config": cfg.to_dict(), "model_state": ae.state_dict()}, checkpoint)
    evaluate.guard_checkpoint(checkpoint, 4)
    try:
        evaluate.guard_checkpoint(checkpoint, 5)
    except ValueError as exc:
        assert "layer=4" in str(exc)
    else:
        raise AssertionError("Checkpoint from a different decoder block was accepted")


def test_final_checkpoint_is_numerical_latest_not_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = Path("checkpoints/gpt2/layer2") / evaluate.CKDIR
    folder.mkdir(parents=True)
    for name in ("best_val.pt", "step_999.pt", "step_1000.pt", "step_bad.pt"):
        (folder / name).touch()
    assert evaluate.checkpoint_for(2).name == "step_1000.pt"


def test_manifest_failure_requires_complete_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = options(steps="test", dry_run=False)
    # A successful process with no requested metric artifact cannot be reported complete.
    missing = evaluate.Step("test", [[sys.executable, "-c", "pass"]], [Path("results/gpt_run/layer0/missing.json")])
    with patch.object(evaluate, "build_steps", return_value=[missing]):
        assert not evaluate.run_layer(0, args)
    state = json.loads(Path("results/gpt_run/layer0/pipeline_state.json").read_text())
    assert state["status"] == "failed"
    assert state["steps"]["test"]["exit_code"] == 0


def test_api_skips_are_reported_as_partial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    judge = evaluate.Step("test", [[sys.executable, "-c", "pass"]], [Path("judge.json")], api=True)
    with patch.object(evaluate, "build_steps", return_value=[judge]):
        assert evaluate.run_layer(0, options(steps="test", dry_run=False, judges="skip"))
    state = json.loads(Path("results/gpt_run/layer0/pipeline_state.json").read_text())
    assert state["status"] == "partial"
    assert state["steps"]["test"]["status"] == "skipped"


def test_cli_supports_watcher_spelling_and_layer_validation():
    tool = str(evaluate.ROOT / "scripts/gpt2/evaluate.py")
    good = subprocess.run([sys.executable, tool, "11", "all", "--dry-run"], capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert "[dry] cgen:" in good.stdout
    assert "--skip_next" not in good.stdout
    bad = subprocess.run([sys.executable, tool, "12", "all", "--dry-run"], capture_output=True, text=True)
    assert bad.returncode != 0
