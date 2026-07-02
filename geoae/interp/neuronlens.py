"""
NeuronLens (arXiv 2502.06809) applied to an arbitrary neuron space, for the
base-model-neurons vs AE-neurons comparison.

Faithful to github.com/MuhammadUmairHaider/NeuronLens (models/lama.py MaskLayer,
utils.compute_masks / mask_range_llma):
  * saliency per concept c : s_j^c = mean_{x in c} |a_j|   (a = h or z)
  * salient dims           : top-p% (or top-q) of s_j^c
  * concept range          : AR(c,j) = [mu - tao*sigma, mu + tao*sigma], tao=2.0,
                             mu/sigma from concept-c samples (salient dims only)
  * gate (concept erase)   : mask = (lo <= a_j <= hi); a_j = where(mask, rep_j, a_j)
                             applied post-layer on the decoder output (same point as
                             reference model.mask_layer); rep_j = 0 in the reference.
                             Non-salient dims have lo=+inf, hi=-inf so never gated.
  * full / naive mask      : tao = inf  => salient dims always replaced with rep_j

Two substrates, ONE method:
  h-space : gate the residual hidden units directly (the literal NeuronLens).
  z-space : encode (h-mean)/std -> z, gate z_j, decode -> x_hat, *std+mean, so the
            rest of the model sees the gated AE reconstruction (full h replaced, not
            per-dim h gate). Replacement in z is 0 by default (GELU latents are >= 0).
            Report the no-gate AE-recon as the z-substrate baseline; deltas for z must
            be read against that baseline, not clean h.

The gate functions return a `replacement_fn(hs)` for evaluate.SplicingHook.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Activation capture at the same point as reference model.mask_layer (post-layer,
# pre-final-norm).  Do NOT use hidden_states[layer+1] for the last decoder layer:
# in HF that index is after RMSNorm, not the decoder output.
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_post_layer(lm, enc, layer: int) -> np.ndarray:
    """(B, D) activations at orig_lens-1 from decoder layer `layer` output."""
    captured: list[torch.Tensor] = []

    def _hook(_module, _inp, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured.append(hs)

    handle = lm.model.layers[layer].register_forward_hook(_hook)
    lm(**enc, use_cache=False)
    handle.remove()

    hs = captured[0].float()
    # Index the last ATTENDED (real) token, robust to left- OR right-padding.
    # (orig_lens-1 is only correct for right-padding; this harness left-pads.)
    am = enc["attention_mask"]
    last_idx = (am.shape[1] - 1 - torch.flip(am, dims=[1]).argmax(dim=1)).to(hs.device)
    rows = torch.arange(hs.shape[0], device=hs.device)
    return hs[rows, last_idx, :].cpu().numpy()


# ---------------------------------------------------------------------------
# Saliency + range fitting (pure numpy; a = h or z activations, (N, D))
# ---------------------------------------------------------------------------

def saliency(acts: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """(C, D) mean absolute activation per concept."""
    return np.stack([
        np.abs(acts[labels == c]).mean(axis=0) if (labels == c).any()
        else np.zeros(acts.shape[1], dtype=acts.dtype)
        for c in range(n_classes)
    ])


def select_top(saliency_c: np.ndarray, p: float | None = None, q: int | None = None) -> np.ndarray:
    """Boolean mask (D,) of salient dims: top-q by count, else top-p fraction."""
    D = len(saliency_c)
    k = int(q) if q is not None else int(round(p * D))
    k = max(1, min(k, D))
    idx = np.argsort(-saliency_c)[:k]
    m = np.zeros(D, dtype=bool)
    m[idx] = True
    return m


def fit_ranges(acts_c: np.ndarray, salient: np.ndarray, tao: float):
    """Per-dim Gaussian range over concept-c samples.
    Returns (lo, hi, mu, sigma) each (D,); non-salient dims get lo=+inf, hi=-inf
    (an empty interval) so the gate never touches them."""
    mu = acts_c.mean(axis=0)
    sigma = acts_c.std(axis=0)
    lo = np.full_like(mu, np.inf)
    hi = np.full_like(mu, -np.inf)
    if np.isinf(tao):
        lo[salient] = -np.inf
        hi[salient] = np.inf
    else:
        lo[salient] = mu[salient] - tao * sigma[salient]
        hi[salient] = mu[salient] + tao * sigma[salient]
    return lo, hi, mu, sigma


# ---------------------------------------------------------------------------
# Gate hook factories (return replacement_fn(hs) for SplicingHook)
# ---------------------------------------------------------------------------

def make_h_gate(lo: np.ndarray, hi: np.ndarray, avg: np.ndarray, device) -> "callable":
    """Gate residual units directly: where lo<=h_j<=hi -> rep_j (the `avg` arg;
    0 by default, matching the reference MaskLayer), else pass through."""
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=device).view(1, 1, -1)
    hi_t = torch.as_tensor(hi, dtype=torch.float32, device=device).view(1, 1, -1)
    avg_t = torch.as_tensor(avg, dtype=torch.float32, device=device).view(1, 1, -1)

    def fn(hs: Tensor) -> Tensor:
        h = hs.float()
        mask = (h >= lo_t) & (h <= hi_t)
        out = torch.where(mask, avg_t.to(h.device), h)
        return out.to(hs.dtype)
    return fn


def make_z_gate(lo: np.ndarray, hi: np.ndarray, avg_z: np.ndarray,
                ae, mean: Tensor, std: Tensor, device, gate: bool = True) -> "callable":
    """Encode -> (optionally) gate z_j -> decode -> denormalise. With gate=False
    this is the pure AE-reconstruction splice (the z-substrate baseline)."""
    lo_t = torch.as_tensor(lo, dtype=torch.float32, device=device).view(1, -1)
    hi_t = torch.as_tensor(hi, dtype=torch.float32, device=device).view(1, -1)
    avg_t = torch.as_tensor(avg_z, dtype=torch.float32, device=device).view(1, -1)
    m = mean.to(device).float()
    s = std.to(device).float()

    @torch.no_grad()
    def fn(hs: Tensor) -> Tensor:
        B, T, D = hs.shape
        x = (hs.reshape(B * T, D).float() - m) / s
        z = ae.encoder(x)
        if gate:
            mask = (z >= lo_t) & (z <= hi_t)
            z = torch.where(mask, avg_t, z)
        x_hat = ae.decoder(z)
        recon = (x_hat * s + m).reshape(B, T, D)
        return recon.to(hs.device).to(hs.dtype)
    return fn


def make_h_steer(r: np.ndarray, alpha: float, device) -> "callable":
    """Linear steering in residual space: h' = h - alpha * r."""
    r_t = torch.as_tensor(r, dtype=torch.float32, device=device).view(1, 1, -1)

    def fn(hs: Tensor) -> Tensor:
        h = hs.float()
        out = h - float(alpha) * r_t.to(h.device)
        return out.to(hs.dtype)

    return fn


def make_z_steer(r: np.ndarray, alpha: float, ae, mean: Tensor, std: Tensor, device) -> "callable":
    """Linear steering in latent space: encode -> z' = z - alpha * r -> decode."""
    r_t = torch.as_tensor(r, dtype=torch.float32, device=device).view(1, -1)
    m = mean.to(device).float()
    s = std.to(device).float()

    @torch.no_grad()
    def fn(hs: Tensor) -> Tensor:
        B, T, D = hs.shape
        x = (hs.reshape(B * T, D).float() - m) / s
        z = ae.encoder(x)
        z = z - float(alpha) * r_t.to(z.device)
        x_hat = ae.decoder(z)
        recon = (x_hat * s + m).reshape(B, T, D)
        return recon.to(hs.device).to(hs.dtype)

    return fn


# ---------------------------------------------------------------------------
# Separability metrics (the user's hypothesis: z ranges more separable than h)
# ---------------------------------------------------------------------------

def _bhattacharyya(mu1, s1, mu2, s2, eps=1e-8):
    """Bhattacharyya coefficient of two 1-D Gaussians (1=identical, 0=disjoint)."""
    v1, v2 = s1 ** 2 + eps, s2 ** 2 + eps
    return np.sqrt(2 * s1 * s2 / (v1 + v2)) * np.exp(-((mu1 - mu2) ** 2) / (4 * (v1 + v2)))


def range_overlap(mus: np.ndarray, sigmas: np.ndarray, salient_union: np.ndarray) -> float:
    """Mean pairwise concept-range overlap over salient dims. mus/sigmas: (C, D).
    Lower = concepts occupy more separable ranges."""
    C = mus.shape[0]
    dims = np.where(salient_union)[0]
    if len(dims) == 0:
        return float("nan")
    vals = []
    for a in range(C):
        for b in range(a + 1, C):
            bc = _bhattacharyya(mus[a, dims], sigmas[a, dims], mus[b, dims], sigmas[b, dims])
            vals.append(bc.mean())
    return float(np.mean(vals))


def range_purity(acts: np.ndarray, labels: np.ndarray, ranges: dict) -> float:
    """For each concept c and its salient dims j with range AR(c,j): when
    a_j in AR(c,j), the fraction of samples truly labelled c. Higher = the range
    is a cleaner concept indicator. Averaged over (c, j). `ranges[c] = (lo,hi,salient)`."""
    purities = []
    for c in ranges:
        lo, hi, salient = ranges[c]
        is_c = (labels == c)
        for j in np.where(salient)[0]:
            inside = (acts[:, j] >= lo[j]) & (acts[:, j] <= hi[j])
            n_in = inside.sum()
            if n_in > 0:
                purities.append((inside & is_c).sum() / n_in)
    return float(np.mean(purities)) if purities else float("nan")
