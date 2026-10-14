#!/usr/bin/env python3
"""Check real pretrained GPT-2 extraction and token-aware splice plumbing.

The randomly initialized AE here checks execution only; its numbers are not
training or evaluation results for the campaign.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer

from geoae.checkpoint import load_lm
from geoae.e2e.logits import LogitsComputer
from geoae.evaluate import experiment_loss_recovered, experiment_per_cluster_ablation
from geoae.extract_sampled import register_capture, forward_batch
from geoae.hooks import SplicingHook
from geoae.lm_arch import describe
from geoae.model import GeoAE


def main():
    if not torch.cuda.is_available():
        raise SystemExit("Run this smoke check on a GPU worker")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    lm = load_lm("gpt2", device=device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    texts = ["The cat sat beside the window.", "Paris is the capital of France."]
    ids = [tokenizer(text)["input_ids"] for text in texts]
    captured, handles = register_capture(lm, list(range(12)))
    with torch.no_grad():
        forward_batch(lm, ids, tokenizer.eos_token_id, device)
    for handle in handles:
        handle.remove()
    assert len(captured) == 12
    assert all(torch.isfinite(value).all() and value.shape[-1] == 768
               for value in captured.values())
    input_ids = torch.tensor([ids[0]], device=device)
    with torch.no_grad():
        original = lm(input_ids=input_ids).logits
        for layer in (0, 5, 11):
            hook = SplicingHook(lm, layer)
            hook.activate(lambda hidden: hidden)
            try:
                identity = lm(input_ids=input_ids).logits
            finally:
                hook.deactivate()
            torch.testing.assert_close(identity, original, rtol=0, atol=0)
        reconstructed = LogitsComputer(lm, 11).head_logits(captured[11][0, len(ids[0]) - 1])
        torch.testing.assert_close(reconstructed, original[0, -1], rtol=0.03, atol=0.25)

    ae = GeoAE(hidden_size=768, latent_dim=1536, n_clusters=4,
               nonlinearity="gelu", latent_norm="batch",
               token_bias_rows=1, vocab_size=50257).to(device).eval()
    norm = {"mean": torch.zeros(768, device=device), "std": torch.ones(768, device=device)}
    result = {"model": describe(lm), "all_layer_extraction": True,
              "identity_splice": True, "last_layer_head": True, "token_aware_splice": {}}
    for layer in (0, 11):
        loss = experiment_loss_recovered(lm, ae, norm, tokenizer, texts[:1], layer, device)
        ablation = experiment_per_cluster_ablation(lm, ae, norm, tokenizer,
                                                  texts[:1], layer, device, top_k_tokens=2)
        result["token_aware_splice"][str(layer)] = {"loss_recovered": loss,
                                                   "ablation_clusters": len(ablation)}
    output = Path("runs/gpt2_2x/smoke_gpu.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: real GPT-2 extraction and splice plumbing; {output}")


if __name__ == "__main__":
    main()
