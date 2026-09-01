"""Architecture-agnostic LM part location (CPU, no downloads).

Models are built from tiny hand-written configs, so these run offline and pin
the module layouts that geoae.extract / SplicingHook / e2e logits depend on:
Llama and Qwen3 expose `lm.model.layers`, while Gemma 3 4b/12b/27b load as
Gemma3ForConditionalGeneration, whose `lm.model` has NO `.layers` and NO
`.norm` — the decoder lives at `lm.model.language_model`.
"""
import pytest
import torch

from geoae.hooks import SplicingHook
from geoae.lm_arch import decoder_layers, describe, hidden_size, locate_lm_parts

TEXT = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8)


def make_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    return LlamaForCausalLM(LlamaConfig(**TEXT))


def make_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    return Qwen3ForCausalLM(Qwen3Config(**TEXT))


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


BUILDERS = [make_llama, make_qwen3, make_gemma3_multimodal]
IDS = ["llama", "qwen3", "gemma3-multimodal"]


@pytest.fixture(params=BUILDERS, ids=IDS)
def lm(request):
    m = request.param()
    m.eval()
    return m


def test_locate_parts(lm):
    decoder, layers, final_norm, lm_head = locate_lm_parts(lm)
    assert len(layers) == TEXT["num_hidden_layers"]
    assert layers is decoder.layers
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
