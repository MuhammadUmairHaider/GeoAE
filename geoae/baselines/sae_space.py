"""One interface over the SAEs scored beside GeoAE on representation-level evals.

Spec strings (``load_sae_space``):
  matched:<path/to/last.pt>   data-matched (Batch)TopK SAE from geoae.baselines.matched_topk. Token-bypass: its input is
                              (h - mean)/std - token_bias(tok) with the frozen GeoAE token table, so every
                              encode/decode REQUIRES token ids (a missing id tensor raises, never zero bias).
                              Inference uses the calibrated EMA threshold (MatchedSAE.encode), not per-row TopK.
  scope:<id or local dir>     Gemma Scope 2 JumpReLU SAE. <id> is "16k"/"262k" (shorthand for
                              resid_post_all/layer_33_width_<id>_l0_small), a full relative Hub path, or an
                              existing local directory holding config.json + params.safetensors (or a root
                              holding the Hub path). Raw residual in, raw residual out, fp32 throughout.

Interface: ``n_features``, ``needs_tokens``, ``encode_raw(h_raw, tok) -> codes``,
``decode_raw(codes, tok) -> h_raw``, and chunked helpers ``top1_labels`` / ``reconstruct_raw`` that never
materialise more than ``chunk`` rows of codes (262k-wide SAEs: 4096 dense rows would be 4 GB per temporary).
"""
from __future__ import annotations

from pathlib import Path

import torch

_CODE_BUDGET = 2 ** 28  # elements of one dense code chunk (~1 GB fp32)


def chunk_rows(n_features: int, cap: int = 4096) -> int:
    return int(max(64, min(cap, _CODE_BUDGET // max(n_features, 1))))


class SAESpace:
    needs_tokens = False
    has_decoder = True      # False: no meaningful reconstruction (random / restricted controls)
    n_features = 0
    name = ""

    @property
    def chunk_features(self):   # width of the dense codes a chunk really materialises
        return self.n_features

    @property
    def n_labels(self):
        return self.n_features

    def assign_labels(self, h_raw, tok=None, chunk=None):
        return self.top1_labels(h_raw, tok, chunk)

    def encode_raw(self, h_raw, tok=None):  # pragma: no cover - interface
        raise NotImplementedError

    def decode_raw(self, codes, tok=None):  # pragma: no cover - interface
        raise NotImplementedError

    def _check_tok(self, h, tok):
        if self.needs_tokens and tok is None:
            raise ValueError(f"{self.name}: token-bypass SAE needs current-token ids (tok=None); "
                             "refusing to fall back to a zero token bias")
        if tok is not None and tok.shape[0] != h.shape[0]:
            raise ValueError("tok must have one id per row")

    @torch.no_grad()
    def top1_labels(self, h_raw, tok=None, chunk=None):
        """Top-1 active feature per row (argmax of codes); rows with no active feature get -1."""
        self._check_tok(h_raw, tok)
        chunk = chunk or chunk_rows(self.chunk_features)
        out = []
        for i in range(0, len(h_raw), chunk):
            codes = self.encode_raw(h_raw[i:i + chunk], None if tok is None else tok[i:i + chunk])
            val, idx = codes.max(1)
            out.append(torch.where(val > 0, idx, torch.full_like(idx, -1)))
        return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=h_raw.device)

    @torch.no_grad()
    def reconstruct_raw(self, h_raw, tok=None, chunk=None):
        """decode_raw(encode_raw(h)), chunked. Returns fp32 [N, D]."""
        self._check_tok(h_raw, tok)
        chunk = chunk or chunk_rows(self.chunk_features)
        out = []
        for i in range(0, len(h_raw), chunk):
            t = None if tok is None else tok[i:i + chunk]
            out.append(self.decode_raw(self.encode_raw(h_raw[i:i + chunk], t), t))
        return torch.cat(out)


class MatchedSpace(SAESpace):
    needs_tokens = True

    def __init__(self, model, name="matched"):
        self.model, self.name = model, name
        self.n_features = int(model.core.W_dec.shape[0] if hasattr(model.core, "W_dec")
                              else model.core.decoder.weight.shape[1])

    @property
    def decoder_directions(self):
        return self.model.decoder_directions

    @torch.no_grad()
    def encode_raw(self, h_raw, tok=None):
        self._check_tok(h_raw, tok)
        return self.model.encode(h_raw.float(), tok.long())

    @torch.no_grad()
    def decode_raw(self, codes, tok=None):
        if tok is None:
            raise ValueError(f"{self.name}: token-bypass SAE needs current-token ids to decode")
        return self.model.decode(codes, tok.long())


class ScopeSpace(SAESpace):
    needs_tokens = False

    def __init__(self, model, name="scope"):
        self.model, self.name, self.n_features = model, name, int(model.n_features)

    @property
    def decoder_directions(self):
        return self.model.decoder_directions

    @torch.no_grad()
    def encode_raw(self, h_raw, tok=None):
        return self.model.encode(h_raw.float())

    @torch.no_grad()
    def decode_raw(self, codes, tok=None):
        return self.model.decode(codes)


class RandomSpace(SAESpace):
    """Random-SAE null: seeded unit-norm decoder rows W; code = (matched-preprocessed input) @ W^T.

    The input goes through the same normalisation and frozen token-table subtraction as a matched SAE
    (so the null gets the same token treatment). No reconstruction is defined."""
    needs_tokens = True
    has_decoder = False

    def __init__(self, width, seed, transform, device, name="random"):
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(width, transform.norm_mean.numel(), generator=g)
        self.W = (w / w.norm(dim=1, keepdim=True)).to(device)
        self.transform, self.name, self.n_features = transform, name, int(width)

    @property
    def decoder_directions(self):
        return self.W

    @torch.no_grad()
    def encode_raw(self, h_raw, tok=None):
        self._check_tok(h_raw, tok)
        return self.transform.preprocess(h_raw.float(), tok.long()) @ self.W.T

    def decode_raw(self, codes, tok=None):
        raise NotImplementedError("random-SAE null has no decoder")


class RestrictedSpace(SAESpace):
    """C3: keep only a feature subset (e.g. the K most frequent), so top-1 has as many labels as a K-cluster arm."""
    has_decoder = False

    def __init__(self, base, feature_ids, name=""):
        self.base, self.name = base, name or base.name
        self.sel = torch.as_tensor(feature_ids, dtype=torch.long).cpu()
        self.n_features = len(self.sel)
        self.needs_tokens = base.needs_tokens

    @property
    def chunk_features(self):
        return self.base.chunk_features

    @torch.no_grad()
    def encode_raw(self, h_raw, tok=None):
        return self.base.encode_raw(h_raw, tok)[:, self.sel.to(h_raw.device)]

    def decode_raw(self, codes, tok=None):
        raise NotImplementedError("restricted SAE has no decoder")


class CodeKMeans:
    """C1: nearest centroid (centroids [K, F]) of an SAE's code vector. Same labelling interface as SAESpace."""
    has_decoder = False

    def __init__(self, base, centroids, name=""):
        self.base, self.name = base, name or base.name
        self.C = torch.as_tensor(centroids, dtype=torch.float32)
        self.n_features = self.n_labels = int(self.C.shape[0])
        self.needs_tokens = base.needs_tokens

    @property
    def chunk_features(self):
        return self.base.chunk_features

    @torch.no_grad()
    def assign_labels(self, h_raw, tok=None, chunk=None):
        base = self.base
        base._check_tok(h_raw, tok)
        C = self.C.to(h_raw.device)
        chunk = chunk or chunk_rows(base.chunk_features)
        out = []
        for i in range(0, len(h_raw), chunk):
            codes = base.encode_raw(h_raw[i:i + chunk], None if tok is None else tok[i:i + chunk])
            out.append(torch.cdist(codes, C).argmin(1))
        return torch.cat(out)


def load_sae_space(spec: str, device="cuda", name: str = "") -> SAESpace:
    if "#top" in spec:   # C3: <spec>#top<K>=<counts.npz>, counts[F] = per-feature activation frequency
        base_spec, _, tail = spec.rpartition("#top")
        k, _, counts_path = tail.partition("=")
        import numpy as np
        counts = np.load(counts_path)["counts"]
        sel = np.argsort(-counts, kind="stable")[:int(k)]
        base = load_sae_space(base_spec, device, name)
        return RestrictedSpace(base, torch.as_tensor(sel).to(device), name or base.name)
    kind, _, rest = spec.partition(":")
    if not rest:
        raise ValueError(f"SAE spec must be 'matched:<last.pt>' or 'scope:<id|dir>', got {spec!r}")
    if kind == "matched":
        from geoae.baselines.matched_topk.common import load_matched
        model, _ckpt = load_matched(rest, device)
        return MatchedSpace(model, name or Path(rest).parent.name)
    if kind == "scope":
        from geoae.baselines.gemma_scope import load_gemma_scope
        p = Path(rest)
        if p.is_dir():
            model = load_gemma_scope(local_dir=p, device=device)  # dir with the two files, or root of the default path
        else:
            sae_path = f"resid_post_all/layer_33_width_{rest}_l0_small" if "/" not in rest else rest
            model = load_gemma_scope(sae_path=sae_path, device=device)
        return ScopeSpace(model, name or rest)
    if kind == "random":   # random:<width>:<seed>:<matched last.pt supplying the input transform>
        width, seed, path = rest.split(":", 2)
        from geoae.baselines.matched_topk.common import BypassTransform
        st = torch.load(path, map_location="cpu", weights_only=False)["transform_state"]
        tf = BypassTransform(st["norm_mean"], st["norm_std"], st["tb_table"], st["tb_index"]).to(device)
        return RandomSpace(int(width), int(seed), tf, device, name or f"rand{width}")
    raise ValueError(f"Unknown SAE kind {kind!r} in {spec!r}")


def load_sae_kmeans(spec_at_path: str, device="cuda", name: str = "") -> CodeKMeans:
    """'<sae spec>@<centroids.npz>' (centroids fitted in that SAE's code space by geoae/baselines/sae_controls_fit.py)."""
    import numpy as np
    spec, _, path = spec_at_path.rpartition("@")
    if not spec:
        raise ValueError(f"expected '<sae spec>@<centroids.npz>', got {spec_at_path!r}")
    return CodeKMeans(load_sae_space(spec, device, name), np.load(path)["centroids"], name)


def parse_saes(arg: str) -> dict:
    """'name=spec,name=spec' -> {name: spec}. Specs contain ':' but never ','."""
    out = {}
    for item in filter(None, (arg or "").split(",")):
        n, s = item.split("=", 1)
        out[n] = s
    return out
