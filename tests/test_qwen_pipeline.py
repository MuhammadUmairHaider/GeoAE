"""The approved Qwen pair uses the last block and the ten-source 10M recipe."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.gpt2 import evaluate as research
from scripts.midlayer import dataset, evaluate


ROOT = Path(__file__).resolve().parents[1]


def options(arm):
    args = evaluate.parser().parse_args([arm, "--dry-run"])
    return evaluate.options_for(arm, args)


def plan(arm):
    args = options(arm)
    checkpoint = research.checkpoint_for(35, dry_run=True, profile=args.profile)
    return research.build_steps(35, checkpoint, Path(args.activations_dir), args), args


def test_qwen_is_explicit_without_changing_existing_campaign_selections():
    assert evaluate.QWEN_ARMS == ("qwen_last_bypass", "qwen_last_plain")
    assert len(evaluate.CANONICAL_ARMS) == 8
    assert len(evaluate.NEW_ARMS) == 6
    assert not set(evaluate.QWEN_ARMS) & set(evaluate.CANONICAL_ARMS)
    assert not set(evaluate.QWEN_ARMS) & set(evaluate.NEW_ARMS)


@pytest.mark.parametrize("arm", evaluate.QWEN_ARMS)
def test_qwen_uses_approved_dimensions_and_exact_training_data(arm):
    profile = evaluate.PROFILES[arm]
    extraction = dataset.validate_extraction_config(ROOT / profile["extract_config"])
    training = yaml.safe_load((ROOT / profile["train_config"]).read_text())
    raw_extraction = yaml.safe_load((ROOT / profile["extract_config"]).read_text())["extraction"]
    assert profile["model_name"] == training["extraction"]["model_name"] == extraction["model"] == "Qwen/Qwen2.5-3B"
    assert profile["layer"] == training["data"]["target_layer"] == 35
    assert extraction["layers"] == [35]
    assert profile["layer"] == profile["n_layers"] - 1
    assert profile["hidden_size"] == training["model"]["hidden_size"] == raw_extraction["hidden_size"] == 2048
    assert profile["latent_dim"] == training["model"]["latent_dim"] == 4096
    assert profile["n_clusters"] == training["model"]["n_clusters"] == 2000
    assert profile["n_epochs"] == training["train"]["n_epochs"] == 50
    assert profile["activations_dir"] == training["data"]["activations_dir"] == extraction["activations_dir"]
    assert profile["corpus_dir"] == extraction["corpus_dir"]
    assert extraction["source_mix"] == dict(dataset.USER_MIX)
    assert extraction["source_fingerprint"] == dataset.SOURCE_FINGERPRINT
    assert profile["n_tokens"] == extraction["n_tokens"] == 10_526_316
    assert profile["n_train_rows"] == extraction["training_rows"] == 10_000_000
    assert extraction["n_tokens"] - extraction["training_rows"] == 526_316
    assert profile["val_frac"] == training["data"]["val_frac"] == 0.05
    assert training["train"]["centroid_init"] == "dpc"
    assert training["train"]["ema_decay"] == 0.9
    assert training["train"]["ema_hard"] is True
    # Keep the other experiment's centroid controls out of the approved schema.
    assert not {"lloyd_rows", "lloyd_iters", "freeze_centroids"} & training["train"].keys()
    paths = research.profile_paths(35, options(arm))
    assert training["train"]["checkpoints_dir"] == str(research.checkpoint_for(35, dry_run=True, profile=profile).parent)
    assert bool(training["model"]["token_bias"]) == (profile["mode"] == "bypass")
    if profile["mode"] == "bypass":
        assert training["model"]["token_bias"] == str(paths["table"])


def test_qwen_pair_shares_data_and_controls_with_separate_checkpoints_and_results():
    bypass = options("qwen_last_bypass")
    plain = options("qwen_last_plain")
    bp, pp = research.profile_paths(35, bypass), research.profile_paths(35, plain)
    assert bp["out"] == Path("results/midlayer_run/qwen2.5-3B/layer35/bypass")
    assert pp["out"] == Path("results/midlayer_run/qwen2.5-3B/layer35/plain")
    for key in bp.keys() - {"out"}:
        assert bp[key] == pp[key], key
    assert bypass.activations_dir == plain.activations_dir
    assert bypass.corpus_dir == plain.corpus_dir
    bcfg = yaml.safe_load((ROOT / bypass.profile["train_config"]).read_text())
    pcfg = yaml.safe_load((ROOT / plain.profile["train_config"]).read_text())
    bcfg["model"]["token_bias"] = ""
    bcfg["train"]["checkpoints_dir"] = pcfg["train"]["checkpoints_dir"]
    assert bcfg == pcfg


@pytest.mark.parametrize("arm", evaluate.QWEN_ARMS)
def test_qwen_evaluations_use_last_block_and_model_specific_artifacts(arm):
    stages, args = plan(arm)
    paths = research.profile_paths(35, args)
    names = research.select_steps("whole", stages)
    assert {"core_eval", "mmlu", "cq", "dispersion", "number", "number_hb", "steer",
            "range_db14", "range_bios", "geometry", "biasprobe", "tpp", "ct", "wsd",
            "cgen", "cgen_concept", "cjudge", "cjudge_concept", "csteer", "judge", "pair"} <= names
    assert not any(step.api for step in stages if step.name in research.select_steps("gpu", stages))
    assert "--skip_next" not in next(step for step in stages if step.name == "tokstruct").commands[0]
    assert paths["bb"].parts[-2:] == ("Qwen2.5-3B", "last")
    assert paths["db"].parts[-3:] == ("Qwen2.5-3B", "last_ids", "unprompted")
    for step in stages:
        for output in step.outputs:
            if str(output).startswith("results/"):
                assert output.is_relative_to(paths["out"]), (step.name, output)
        for command in step.commands:
            if "--layer" in command:
                assert command[command.index("--layer") + 1] == "35"
            if "--model_name" in command:
                assert command[command.index("--model_name") + 1] == "Qwen/Qwen2.5-3B"
            source = (Path(command[command.index("-m") + 1].replace(".", "/") + ".py")
                      if "-m" in command else Path(command[2]))
            tree = ast.parse((ROOT / source).read_text())
            declared = {argument.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                        for argument in node.args if isinstance(argument, ast.Constant)
                        and isinstance(argument.value, str)}
            used = {word for word in command[3:] if word.startswith("--")}
            assert used <= declared, (step.name, source, used - declared)


@pytest.mark.parametrize("arm", evaluate.QWEN_ARMS)
def test_qwen_checkpoint_guard_rejects_other_models_and_wrong_bypass_mode(tmp_path, arm):
    import torch

    profile = evaluate.PROFILES[arm]
    config = yaml.safe_load((ROOT / profile["train_config"]).read_text())
    state = {"centroids_initialized": torch.tensor(True)}
    if profile["mode"] == "bypass":
        state["tb_table"] = torch.ones(1, 2048)
    checkpoint = {"epoch": 50, "config": config, "model_state": state}
    path = tmp_path / "step_100.pt"
    torch.save(checkpoint, path)
    research.guard_checkpoint(path, 35, profile)
    wrong = deepcopy(checkpoint)
    wrong["config"]["extraction"]["model_name"] = "meta-llama/Llama-3.2-3B"
    torch.save(wrong, path)
    with pytest.raises(ValueError, match="model="):
        research.guard_checkpoint(path, 35, profile)
    wrong = deepcopy(checkpoint)
    if profile["mode"] == "bypass":
        wrong["model_state"].pop("tb_table")
    else:
        wrong["model_state"]["tb_table"] = torch.ones(1, 2048)
    torch.save(wrong, path)
    with pytest.raises(ValueError, match="token_bias="):
        research.guard_checkpoint(path, 35, profile)


def test_offline_qwen_last_block_hooks_and_native_readout():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from geoae.hooks import SplicingHook, TokenIdTap
    from geoae.interp._context_logits import (
        capture_context_residuals, final_residual_logits, prepare_final_readout, validate_layer,
    )
    from geoae.lm_arch import locate_lm_parts

    # Preserve Qwen's native decoder implementation and 36-block indexing,
    # while avoiding network access, pretrained weights, or GPU allocation.
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=36,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        attention_dropout=0.0, use_cache=False,
    )).eval()
    _, layers, norm, head = locate_lm_parts(model)
    assert len(layers) == 36
    assert norm is model.model.norm and head is model.lm_head
    assert validate_layer(model, 35) == 0
    ids = torch.tensor([[2, 3, 4, 5]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        native = model(input_ids=ids, attention_mask=mask).logits
        assert prepare_final_readout(model, ids, mask)["transform"] == "identity"
        selected, final = capture_context_residuals(model, 35, ids, mask)
        assert torch.equal(selected, final)
        assert torch.allclose(final_residual_logits(model, final), native, atol=1e-6)
        hook, tap = SplicingHook(model, 35), TokenIdTap(model)
        def zero_with_token_ids(hidden):
            assert torch.equal(tap.ids_for(hidden), ids.flatten())
            return torch.zeros_like(hidden)
        hook.activate(zero_with_token_ids)
        try:
            zeroed = model(input_ids=ids, attention_mask=mask).logits
        finally:
            hook.deactivate()
            tap.remove()
        assert torch.count_nonzero(zeroed) == 0
        assert not torch.allclose(zeroed, native)
        assert torch.equal(model(input_ids=ids, attention_mask=mask).logits, native)
