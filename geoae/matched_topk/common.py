"""Frozen GeoAE input/output transform and pinned reference trainer factory."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import uuid

import torch
from torch import nn

from ._reference.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from ._reference.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE, MatryoshkaBatchTopKTrainer,
)

ROOT = Path("results/delta/g4b_l33_d5120/evals/gemma_scope_compare/matched_topk")
GEOAE_CHECKPOINT = "checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/best_val.pt"
TOKEN_VISITS = 500_000_000
BATCH_SIZE = 2048
GROUP_FRACTIONS = [1/32, 1/16, 1/8, 1/4, 1/2+1/32]
METHODS = ("batch_topk", "matryoshka")


def model_name(method, width, k=20):
    """Keep the original k20 paths and isolate additional sparsity arms."""
    if method not in METHODS or width <= 0 or k <= 0:
        raise ValueError("Invalid SAE method, width, or k")
    name = f"{method}_{width}"
    return name if k == 20 else f"{name}_k{k}"


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def reference_provenance():
    root = Path(__file__).parent / "_reference"
    provenance = json.loads((root / "SOURCE.json").read_text())
    for filename, expected in provenance["files"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Modified reference source: {filename}")
    return provenance


def make_trainer(method, width, *, activation_dim=2560, k=20, visits=TOKEN_VISITS,
                 batch_size=BATCH_SIZE, seed=42, device="cuda"):
    steps = math.ceil(visits / batch_size)
    if steps <= 1251:
        raise ValueError("Schedule requires >1251 steps; smoke runs retain the full schedule")
    config = dict(steps=steps, activation_dim=activation_dim, dict_size=width, k=k,
                  layer=33, lm_name="google/gemma-3-4b-pt", submodule_name="model.layers.33.output",
                  lr=5e-5, auxk_alpha=1/32, warmup_steps=1000,
                  decay_start=int(.8*steps), k_anneal_steps=max(1, int(.01*steps)),
                  threshold_beta=.999, threshold_start_step=1000, seed=seed, device=device)
    if method == "batch_topk":
        return BatchTopKTrainer(**config)
    if method == "matryoshka":
        return MatryoshkaBatchTopKTrainer(**config, group_fractions=GROUP_FRACTIONS,
                                        group_weights=None)
    raise ValueError(f"Unknown method: {method}")


class BypassTransform(nn.Module):
    """Same per-dimension normalization and frozen token table as GeoAE."""
    def __init__(self, mean, std, table, index):
        super().__init__()
        if not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("Invalid normalization")
        self.register_buffer("norm_mean", mean.detach().float().clone())
        self.register_buffer("norm_std", std.detach().float().clone())
        self.register_buffer("tb_table", table.detach().clone())
        self.register_buffer("tb_index", index.detach().long().clone())

    @classmethod
    def from_geoae(cls, ae, mean, std):
        if not ae.has_token_bias or ae.token_bias_mode != "bypass":
            raise ValueError("Requires a GeoAE with encoder token bypass")
        return cls(mean, std, ae.tb_table, ae.tb_index)

    def token_bias(self, tok):
        ids = self.tb_index[tok.long()]
        return self.tb_table[ids.clamp_min(0)].float() * (ids >= 0).unsqueeze(-1)

    def preprocess(self, raw, tok):
        return (raw.float() - self.norm_mean) / self.norm_std - self.token_bias(tok)

    def restore(self, bypass, tok):
        return (bypass + self.token_bias(tok)) * self.norm_std + self.norm_mean

    @torch.no_grad()
    def assert_geoae_equivalence(self, ae, raw, tok):
        captured = []
        hook = ae.encoder.register_forward_pre_hook(lambda _m, args: captured.append(args[0].clone()))
        try:
            z = ae.encode((raw.float()-self.norm_mean)/self.norm_std, tok)
        finally:
            hook.remove()
        ours = self.preprocess(raw, tok)
        torch.testing.assert_close(ours, captured[0], rtol=0, atol=0)
        restored = self.restore(ae.decoder(z), tok)
        original = ae.decode(z, tok)*self.norm_std+self.norm_mean
        torch.testing.assert_close(restored, original, rtol=0, atol=0)
        return {"rows": len(raw), "encoder_input_max_abs_error": 0.0,
                "decode_restore_max_abs_error": 0.0, "assertion": "bitwise_equal"}


def decoder_rows(core):
    return core.decoder.weight.T if isinstance(core, BatchTopKSAE) else core.W_dec


class MatchedSAE(nn.Module):
    def __init__(self, core, transform, metadata=None):
        super().__init__(); self.core = core; self.transform = transform
        self.metadata = metadata or {}

    def encode(self, raw, tok):
        if self.core.threshold < 0:
            raise ValueError("Inference threshold is not calibrated")
        return self.core.encode(self.transform.preprocess(raw, tok), use_threshold=True)

    def decode(self, features, tok):
        return self.transform.restore(self.core.decode(features), tok)

    @property
    def decoder_directions(self):
        return decoder_rows(self.core)*self.transform.norm_std

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.core.parameters())


def load_matched(path, device="cuda"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["recipe"]
    if cfg["method"] == "batch_topk":
        core = BatchTopKSAE(cfg["activation_dim"], cfg["width"], cfg["k"])
    else:
        groups = checkpoint["model_state"]["group_sizes"].tolist()
        core = MatryoshkaBatchTopKSAE(cfg["activation_dim"], cfg["width"], cfg["k"], groups)
    core.load_state_dict(checkpoint["model_state"])
    state = checkpoint["transform_state"]
    transform = BypassTransform(state["norm_mean"], state["norm_std"], state["tb_table"], state["tb_index"])
    model = MatchedSAE(core, transform, checkpoint["provenance"]).to(device).eval()
    model.requires_grad_(False)
    return model, checkpoint
