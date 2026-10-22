"""Few-shot capture and generation must share GPT-2's finite context budget."""
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from geoae.hooks import install_padding_position_hook
import geoae.evaluation.cohorts as shared


@pytest.fixture
def case(monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    monkeypatch.setattr(shared, "PROMPT_HEAD", "{}")
    monkeypatch.setattr(shared, "CLASSES", ["class"])
    monkeypatch.setattr(shared, "MAX_NEW", 6)
    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "word": 2, "class": 3},
                                 unk_token="<pad>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>",
        model_max_length=32, padding_side="left",
    )
    torch.manual_seed(0)
    lm = GPT2LMHeadModel(GPT2Config(
        vocab_size=4, n_embd=8, n_layer=2, n_head=2, n_positions=32,
        bos_token_id=0, eos_token_id=None, pad_token_id=0,
    )).eval()
    return lm, tokenizer


@pytest.mark.parametrize("padding_hook", [False, True])
@torch.no_grad()
def test_full_prompt_overflows_on_cached_step_with_or_without_padding_hook(case, padding_hook):
    """CPU reproducer for CUDA srcIndex assertions: position table index 32."""
    lm, tokenizer = case
    positions = install_padding_position_hook(lm) if padding_hook else None
    enc = tokenizer(["word " * 32], padding=True, return_tensors="pt")
    assert enc["input_ids"].shape[-1] == 32
    with pytest.raises(IndexError, match="index out of range"):
        lm.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=0)
    if positions:
        positions.remove()


@pytest.mark.parametrize("max_new", [None, 3])
def test_predict_and_capture_share_reserved_prompt_and_stay_within_positions(case, max_new):
    lm, tokenizer = case
    positions = install_padding_position_hook(lm)
    lengths, last_states, used_positions = [], [], []
    block_hook = lm.transformer.h[0].register_forward_hook(
        lambda module, args, output: (lengths.append(output.shape[1]),
                                      last_states.append(output[:, -1].detach().clone()), None)[-1]
    )
    position_hook = lm.transformer.wpe.register_forward_pre_hook(
        lambda module, args: (used_positions.append(args[0].max().item()), None)[-1]
    )
    texts = ["word " * 80, "word " * 5]
    captured = shared.capture_h(lm, tokenizer, texts, 0, "cpu", max_new=max_new)
    reserve = 6 if max_new is None else max_new
    expected_prompt = 32 - reserve
    assert lengths == [expected_prompt]
    result = shared.predict(lm, tokenizer, texts, [0, 0], "cpu", max_new=max_new)
    assert lengths[1] == expected_prompt
    assert len(result["pred"]) == len(result["correct"]) == 2
    torch.testing.assert_close(torch.from_numpy(captured), last_states[1].float())
    assert max(used_positions) < 32
    assert lengths[2:] == [1] * (reserve - 1)
    block_hook.remove()
    position_hook.remove()
    positions.remove()


def test_other_architectures_and_direct_tokenization_keep_existing_limit(case):
    _, tokenizer = case
    lm = SimpleNamespace(config=SimpleNamespace(model_type="llama", max_position_embeddings=32))
    assert shared._prompt_max_length(lm, 6) == 1024
    assert shared._prompt_max_length(None) == 1024
    enc = shared._tok(tokenizer, ["word " * 80], "cpu", lm=lm)
    assert enc["input_ids"].shape[-1] == 80


@pytest.mark.parametrize("max_new", [0, -1, 32, 33])
def test_gpt2_rejects_continuation_without_a_prompt_budget(case, max_new):
    lm, _ = case
    with pytest.raises(ValueError, match="GPT-2.*32-position context window"):
        shared._prompt_max_length(lm, max_new)


@pytest.mark.parametrize("valid_budget", [False, True])
def test_joint_correct_cache_records_and_checks_gpt2_context_budget(case, tmp_path, monkeypatch, valid_budget):
    lm, tokenizer = case
    path = tmp_path / "correct.json"
    cached = {
        "prompt_sha": hashlib.sha1(b"{}").hexdigest()[:12], "dataset": "fixture",
        "model_name": "gpt2", "ae_sha": "weights", "docs": [{"text": "cached", "label": 0}],
    }
    if valid_budget:
        cached.update(prompt_max_length=26, max_new_tokens=6)
    path.write_text(json.dumps(cached))
    calls = []

    def predict(lm, tokenizer, texts, labels, **kwargs):
        calls.append(list(texts))
        return {"correct": [True] * len(texts)}

    monkeypatch.setattr(shared, "predict", predict)
    hook = SimpleNamespace(activate=lambda fn: None, deactivate=lambda: None)
    docs = shared.build_joint_correct_set(
        [{"text": "fresh", "label": 0}], {"text_column": "text"}, tokenizer,
        lm, "cpu", hook, lambda hs: hs, path, 1, "fixture", model_name="gpt2", ae_sha="weights",
    )
    assert docs[0]["text"] == ("cached" if valid_budget else "fresh")
    assert len(calls) == (0 if valid_budget else 2)
    if not valid_budget:
        saved = json.loads(path.read_text())
        assert saved["prompt_max_length"] == 26 and saved["max_new_tokens"] == 6
