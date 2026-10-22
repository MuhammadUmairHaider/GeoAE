"""
Map an autoencoder reconstruction to next-token logits via the frozen LLM.

The reconstruction is provided in RAW activation space (already denormalised,
i.e. x_hat * std + mean), shaped (B, D) — one last-token residual per document.

Two regimes, selected automatically from the target layer index:

  last-layer   (layer_idx == n_layers - 1)
      The target layer IS the final decoder layer, so the only computation
      between the residual stream and the logits is `final_norm` then `lm_head`,
      both applied position-wise. The last-token logits depend ONLY on the
      last-token residual:
          logits = lm_head(norm(recon))
      No sequence, no attention, no input_ids needed. This is the exact special
      case of an on-the-fly full forward: under a position-wise head every other
      position is irrelevant to the last-token logits. We therefore skip the
      provably-irrelevant lower-stack forward — the result is identical.

  intermediate (layer_idx < n_layers - 1)
      Subsequent decoder layers mix across positions via attention, so we must
      run the full forward from input_ids with a splicing hook that replaces the
      last-token residual at the target layer with `recon`, then read the logits
      at the last position. Inputs are assumed LEFT-padded, so the last token is
      at position -1 for every row in the batch.

Gradients: the LLM parameters are frozen (requires_grad_(False)) but NONE of the
forward passes here are wrapped in no_grad — gradient must flow through the
frozen tail back into `recon`, and hence the autoencoder.
"""
from __future__ import annotations



from torch import Tensor

from geoae.hooks import SplicingHook
from geoae.lm_arch import locate_lm_parts  # noqa: F401  (re-exported for callers)


class LogitsComputer:
    """
    Turns a (B, D) last-token reconstruction into (B, V) last-token logits.

    Parameters
    ----------
    lm        : a frozen *ForCausalLM (Llama/Qwen-style).
    layer_idx : the residual-stream layer the AE operates on (the hook layer).
    """

    def __init__(self, lm, layer_idx: int):
        self.lm = lm
        self.layer_idx = int(layer_idx)
        self.base, self.layers, self.final_norm, self.lm_head = locate_lm_parts(lm)
        self.n_layers = len(self.layers)
        if not (0 <= self.layer_idx < self.n_layers):
            raise ValueError(
                f"layer_idx {self.layer_idx} out of range [0, {self.n_layers})"
            )
        self.is_last = self.layer_idx == self.n_layers - 1
        self.head_dtype = self.lm_head.weight.dtype
        self._hook = None if self.is_last else SplicingHook(lm, self.layer_idx)

    @property
    def needs_input_ids(self) -> bool:
        """Intermediate layers require the full sequence; last layer does not."""
        return not self.is_last

    # ------------------------------------------------------------------ #
    # Last-layer path: position-wise head only.
    # ------------------------------------------------------------------ #
    def head_logits(self, residual_last: Tensor) -> Tensor:
        """
        residual_last : (B, D) RAW last-token residual at the final layer.
        Returns         (B, V) logits, in the head's dtype (cast to fp32 for KL).

        Used both for the student (recon) and, at precompute time, for the
        teacher (original activation).
        """
        h = residual_last.to(self.head_dtype)
        return self.lm_head(self.final_norm(h))

    # ------------------------------------------------------------------ #
    # Intermediate path: full forward + splice of the last position.
    # ------------------------------------------------------------------ #
    def spliced_logits(
        self,
        input_ids: Tensor,        # (B, T) LEFT-padded
        attention_mask: Tensor,   # (B, T)
        recon_last: Tensor,       # (B, D) RAW
    ) -> Tensor:
        """
        Run the full forward, replacing the last-token residual at `layer_idx`
        with `recon_last`, and return the (B, V) logits at the last position.
        """
        rl = recon_last.to(self.head_dtype)

        def fn(hs: Tensor) -> Tensor:
            # hs: (B, T, D). Replace only the last position; keep the graph for rl.
            out = hs.clone()
            out[:, -1, :] = rl.to(hs.dtype)
            return out

        self._hook.activate(fn)
        try:
            out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            self._hook.deactivate()
        return out.logits[:, -1, :]

    # ------------------------------------------------------------------ #
    # Unified entry point.
    # ------------------------------------------------------------------ #
    def student_logits(
        self,
        recon_last: Tensor,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if self.is_last:
            return self.head_logits(recon_last)
        if input_ids is None or attention_mask is None:
            raise ValueError(
                "Intermediate-layer target requires input_ids and attention_mask."
            )
        return self.spliced_logits(input_ids, attention_mask, recon_last)
