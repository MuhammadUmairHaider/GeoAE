"""Next-token signatures at arbitrary blocks with the original document context.

Intermediate residuals cannot be unembedded directly: the remaining attention
blocks need every position of the document. Run those blocks normally, stop at
the final norm, and unembed only the sampled positions to bound vocabulary-sized
memory. This also retains the model's native masks and rotary/absolute positions.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from geoae.lm_arch import decoder_layers, locate_lm_parts


class _ReachedFinalNorm(Exception):
    pass


def validate_layer(lm, layer: int) -> int:
    """Validate a zero-based decoder block and return the number of later blocks."""
    n_layers = len(decoder_layers(lm))
    if not 0 <= layer < n_layers:
        raise ValueError(f"layer {layer} out of range [0, {n_layers})")
    return n_layers - layer - 1


@torch.no_grad()
def prepare_final_readout(lm, input_ids, attention_mask=None):
    """Verify the native readout against a short full forward before using it.

    Gemma text-only and multimodal wrappers differ across Transformers versions
    in whether the configured final softcap is executed. Compare both readouts
    against actual native logits, retaining the one the loaded model executes.
    Matching shapes during this check also avoids GEMM rounding differences.
    """
    existing = getattr(lm, "_geoae_signature_readout", None)
    if existing is not None:
        return existing
    _, _, norm, head = locate_lm_parts(lm)
    # Eight positions suffice to test the position-wise output transform and
    # keep the complete vocabulary allocation small, including Gemma's head.
    ids = input_ids[:1, :8]
    mask = torch.ones_like(ids) if attention_mask is None else attention_mask[:1, :8]
    captured = []
    handle = norm.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    try:
        reference = lm(input_ids=ids, attention_mask=mask, use_cache=False).logits
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("expected one final-norm call when validating the signature readout")
    direct = head(norm(captured[0].to(head.weight.dtype)))
    cfg = getattr(lm.config, "text_config", lm.config)
    cap = getattr(cfg, "final_logit_softcapping", None)
    candidates = [(None, direct)]
    if cap is not None:
        candidates.append((cap, torch.tanh(direct / cap) * cap))
    errors = [(float((candidate.float() - reference.float()).abs().max()), softcap, candidate)
              for softcap, candidate in candidates]
    error, softcap, candidate = min(errors, key=lambda row: row[0])
    if not torch.allclose(candidate.float(), reference.float(), atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"unsupported native final-logit transform (max error {error:g})")
    report = {"transform": "softcap" if softcap is not None else "identity", "softcap": softcap,
              "max_validation_error": error, "validated_positions": ids.numel()}
    lm._geoae_signature_readout = report
    return report


def final_residual_logits(lm, residual: torch.Tensor) -> torch.Tensor:
    """Apply the final readout validated by prepare_final_readout."""
    report = getattr(lm, "_geoae_signature_readout", None)
    if report is None:
        raise RuntimeError("call prepare_final_readout before computing residual signatures")
    _, _, norm, head = locate_lm_parts(lm)
    logits = head(norm(residual.to(head.weight.dtype)))
    if report["softcap"] is not None:
        logits = torch.tanh(logits / report["softcap"]) * report["softcap"]
    return logits


@torch.no_grad()
def capture_context_residuals(lm, layer: int, input_ids, attention_mask):
    """Return (selected-block residual, final residual) for a full input batch.

    All later decoder blocks execute on the full sequence. Stopping before the
    final norm avoids materializing B x T x vocabulary logits. Both hooks are
    removed even if the forward fails, so later generation remains unmodified.
    """
    validate_layer(lm, layer)
    _, layers, norm, _ = locate_lm_parts(lm)
    captured = {}

    def capture(_module, _args, output):
        captured["selected"] = (output[0] if isinstance(output, tuple) else output).detach()

    def stop(_module, args):
        captured["final"] = args[0].detach()
        raise _ReachedFinalNorm

    hook = layers[layer].register_forward_hook(capture)
    stop_hook = norm.register_forward_pre_hook(stop)
    try:
        try:
            lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        except _ReachedFinalNorm:
            pass
    finally:
        hook.remove()
        stop_hook.remove()
    if "selected" not in captured or "final" not in captured:
        raise RuntimeError("LM forward did not reach the selected block and final norm")
    return captured["selected"], captured["final"]


@torch.no_grad()
def context_signature_batches(lm, layer, stream, device, pad_id, *, batch_docs=2, chunk=1024):
    """Yield aligned raw residuals, token IDs, document IDs, and next-token probabilities.

    `stream` yields (input_ids[1,T], domain, sampled_positions), as provided by
    closest_tokens.stream_sampled_docs. Right padding preserves each document's
    original token positions. Document IDs follow this held-out stream's order.
    """
    if batch_docs <= 0 or chunk <= 0:
        raise ValueError("batch_docs and chunk must be positive")
    iterator = enumerate(stream)
    while True:
        documents = list(itertools.islice(iterator, batch_docs))
        if not documents:
            return
        length = max(ids.shape[-1] for _, (ids, _, _) in documents)
        ids_batch = torch.full((len(documents), length), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids_batch)
        for bi, (_, (ids, _, _)) in enumerate(documents):
            ids_batch[bi, :ids.shape[-1]] = ids[0].to(device)
            mask[bi, :ids.shape[-1]] = 1
        prepare_final_readout(lm, ids_batch, mask)
        selected, final = capture_context_residuals(lm, layer, ids_batch, mask)
        for bi, (di, (ids, domain, positions)) in enumerate(documents):
            positions = np.asarray(positions, dtype=np.int64)
            if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= ids.shape[-1]):
                raise ValueError(f"invalid sampled positions for document {di}")
            for start in range(0, len(positions), chunk):
                p = torch.as_tensor(positions[start:start + chunk], device=device)
                raw = selected[bi].index_select(0, p)
                # A no_grad decorator alone does not cover a generator's iteration.
                with torch.no_grad():
                    logits = final_residual_logits(lm, final[bi].index_select(0, p))
                    probs = torch.softmax(logits.float(), -1)
                yield raw, ids_batch[bi, p], np.full(len(p), di, dtype=np.int64), probs, domain
