"""Architecture-agnostic LM part location (CPU, no downloads).

Models are built from tiny hand-written configs, so these run offline and pin
the module layouts that geoae.extract / SplicingHook / e2e logits depend on:
Llama and Qwen3 expose `lm.model.layers`, while Gemma 3 4b/12b/27b load as
Gemma3ForConditionalGeneration, whose `lm.model` has NO `.layers` and NO
`.norm` — the decoder lives at `lm.model.language_model`.
GPT-2 uses `lm.transformer.h` and `lm.transformer.ln_f`.
"""
import pytest
import torch

from geoae.hooks import SplicingHook, install_padding_position_hook
from geoae.lm_arch import decoder_layers, describe, hidden_size, locate_lm_parts

TEXT = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8)


def make_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    return LlamaForCausalLM(LlamaConfig(**TEXT))


def make_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    return Qwen3ForCausalLM(Qwen3Config(**TEXT))


def make_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=TEXT["vocab_size"], n_embd=TEXT["hidden_size"],
        n_layer=TEXT["num_hidden_layers"], n_head=TEXT["num_attention_heads"],
        n_positions=32, bos_token_id=0, eos_token_id=1,
    ))


def make_gemma3_multimodal():
    """The layout AutoModelForCausalLM returns for google/gemma-3-12b-pt."""
    from transformers import Gemma3Config
    from transformers.models.gemma3 import Gemma3ForConditionalGeneration
    cfg = Gemma3Config(
        text_config={**TEXT, "sliding_window": 8},
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, image_size=16, patch_size=8),
    )
    return Gemma3ForConditionalGeneration(cfg)


def make_gemma3_text():
    """The text-only architecture used by google/gemma-3-1b-pt."""
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM
    return Gemma3ForCausalLM(Gemma3TextConfig(**TEXT, sliding_window=8))


BUILDERS = [make_llama, make_qwen3, make_gemma3_multimodal, make_gemma3_text, make_gpt2]
IDS = ["llama", "qwen3", "gemma3-multimodal", "gemma3-text", "gpt2"]


@pytest.fixture(params=BUILDERS, ids=IDS)
def lm(request):
    m = request.param()
    m.eval()
    return m


def test_locate_parts(lm):
    decoder, layers, final_norm, lm_head = locate_lm_parts(lm)
    assert len(layers) == TEXT["num_hidden_layers"]
    assert layers is decoder_layers(lm)
    assert lm_head.weight.shape[0] == TEXT["vocab_size"]
    assert hidden_size(lm) == TEXT["hidden_size"]
    assert callable(final_norm)
    assert f"{TEXT['num_hidden_layers']} layers" in describe(lm)


def test_gemma3_decoder_is_not_lm_model():
    """Guards the exact trap: `lm.model.layers` raises on the Gemma 3 wrapper."""
    lm = make_gemma3_multimodal()
    assert not hasattr(lm.model, "layers")
    assert not hasattr(lm.model, "norm")
    assert decoder_layers(lm) is lm.model.language_model.layers


def test_gpt2_decoder_parts():
    lm = make_gpt2()
    decoder, layers, final_norm, head = locate_lm_parts(lm)
    assert decoder is lm.transformer
    assert layers is lm.transformer.h
    assert final_norm is lm.transformer.ln_f
    assert head is lm.lm_head
    assert not hasattr(decoder, "layers")


def test_capture_hook_sees_residual(lm):
    """geoae.extract's hook shape/dtype contract."""
    from geoae.extract import register_hooks

    captured, handles, nonfinite = register_hooks(lm, [1], store_dtype=torch.float16)
    with torch.no_grad():
        lm(input_ids=torch.randint(0, TEXT["vocab_size"], (1, 7)))
    for h in handles:
        h.remove()
    assert captured[1].shape == (1, 7, TEXT["hidden_size"])
    assert captured[1].dtype == torch.float16
    assert int(nonfinite[1]) == 0


def test_register_hooks_rejects_bad_layer(lm):
    from geoae.extract import register_hooks

    with pytest.raises(ValueError, match="out of range"):
        register_hooks(lm, [TEXT["num_hidden_layers"]])


def test_splicing_hook_replaces_residual(lm):
    """Splicing the TRUE residual back in must reproduce the original logits."""
    ids = torch.randint(0, TEXT["vocab_size"], (1, 7))
    with torch.no_grad():
        ref = lm(input_ids=ids).logits

    hook = SplicingHook(lm, 1)
    cap = []
    hook.activate(lambda hs: (cap.append(hs.clone()) or hs))
    with torch.no_grad():
        lm(input_ids=ids)
    hook.deactivate()

    hook.activate(lambda hs: cap[0])
    with torch.no_grad():
        spliced = lm(input_ids=ids).logits
    hook.deactivate()
    assert torch.allclose(ref, spliced, atol=1e-4)


def test_head_only_path_matches_full_forward(lm):
    """Last-layer e2e shortcut: lm_head(final_norm(residual_L-1)) == logits."""
    from geoae.e2e.logits import LogitsComputer

    n_layers = TEXT["num_hidden_layers"]
    ids = torch.randint(0, TEXT["vocab_size"], (1, 7))
    lc = LogitsComputer(lm, n_layers - 1)
    assert lc.is_last and not lc.needs_input_ids

    cap = []
    hook = SplicingHook(lm, n_layers - 1)
    hook.activate(lambda hs: (cap.append(hs.clone()) or hs))
    with torch.no_grad():
        ref = lm(input_ids=ids).logits[:, -1, :]
    hook.deactivate()

    with torch.no_grad():
        head = lc.head_logits(cap[0][:, -1, :])
    assert torch.allclose(ref.float(), head.float(), atol=1e-4)


def test_intermediate_logits_preserve_values_and_reconstruction_gradient(lm):
    """Full-pipeline KL/steering paths must support intermediate GPT-2 blocks."""
    from geoae.e2e.logits import LogitsComputer

    lm.requires_grad_(False)
    ids = torch.randint(0, TEXT["vocab_size"], (1, 7))
    mask = torch.ones_like(ids)
    captured = []
    hook = SplicingHook(lm, 1)
    hook.activate(lambda hs: (captured.append(hs[:, -1].detach().clone()) or hs))
    with torch.no_grad():
        reference = lm(input_ids=ids, attention_mask=mask).logits[:, -1]
    hook.deactivate()

    reconstruction = captured[0].requires_grad_(True)
    lc = LogitsComputer(lm, 1)
    assert lc.needs_input_ids and not lc.is_last
    spliced = lc.student_logits(reconstruction, ids, mask)
    assert torch.allclose(reference, spliced, atol=1e-4)
    spliced.square().sum().backward()
    assert reconstruction.grad is not None
    assert torch.isfinite(reconstruction.grad).all()
    assert reconstruction.grad.abs().sum() > 0


@pytest.mark.parametrize("side", ["left", "right"])
@torch.no_grad()
def test_gpt2_padded_capture_and_splice_match_individual_documents(side):
    lm = make_gpt2().eval()
    positions = install_padding_position_hook(lm)
    docs = [torch.tensor([[3, 4, 5]]), torch.tensor([[6, 7, 8, 9, 10]])]
    reference = [lm(input_ids=ids).logits[0, -1] for ids in docs]
    if side == "left":
        ids = torch.tensor([[0, 0, 3, 4, 5], [6, 7, 8, 9, 10]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        last = [4, 4]
    else:
        ids = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 10]])
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        last = [2, 4]
    # The same hook used by causal AE splices must preserve padded forwards.
    hook = SplicingHook(lm, 1)
    hook.activate(lambda hs: hs)
    batched = lm(input_ids=ids, attention_mask=mask).logits
    hook.deactivate()
    for row, index in enumerate(last):
        torch.testing.assert_close(batched[row, index], reference[row], atol=1e-5, rtol=1e-5)
    # NeuronLens's last-real-token capture must also agree across batch shapes.
    from geoae.interp.neuronlens import capture_post_layer
    together = capture_post_layer(lm, {"input_ids": ids, "attention_mask": mask}, 1)
    for row, doc in enumerate(docs):
        alone = capture_post_layer(lm, {"input_ids": doc, "attention_mask": torch.ones_like(doc)}, 1)
        torch.testing.assert_close(torch.from_numpy(together[row]), torch.from_numpy(alone[0]),
                                   atol=1e-5, rtol=1e-5)
    positions.remove()


@torch.no_grad()
def test_gpt2_cached_padding_positions_match_individual_document():
    lm = make_gpt2().eval()
    positions = install_padding_position_hook(lm)
    batch = lm(input_ids=torch.tensor([[0, 0, 3, 4]]),
               attention_mask=torch.tensor([[0, 0, 1, 1]]), use_cache=True)
    single = lm(input_ids=torch.tensor([[3, 4]]), use_cache=True)
    padded_next = lm(input_ids=torch.tensor([[5]]), attention_mask=torch.tensor([[0, 0, 1, 1, 1]]),
                     past_key_values=batch.past_key_values, use_cache=True).logits
    single_next = lm(input_ids=torch.tensor([[5]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                     past_key_values=single.past_key_values, use_cache=True).logits
    torch.testing.assert_close(padded_next, single_next, atol=1e-5, rtol=1e-5)
    positions.remove()


@torch.no_grad()
def test_gpt2_padding_hook_preserves_explicit_positions():
    lm = make_gpt2().eval()
    ids = torch.tensor([[0, 0, 3, 4]])
    mask = torch.tensor([[0, 0, 1, 1]])
    explicit = torch.tensor([[2, 3, 4, 5]])
    reference = lm(ids, None, mask, None, explicit).logits
    positions = install_padding_position_hook(lm)
    torch.testing.assert_close(lm(ids, None, mask, None, explicit).logits, reference)
    torch.testing.assert_close(lm(input_ids=ids, attention_mask=mask, position_ids=explicit).logits,
                               reference)
    positions.remove()
