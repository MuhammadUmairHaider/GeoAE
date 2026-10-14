"""
CPU-only unit tests for the e2e KL pieces (no LLM download required).

Run:
    python -m pytest geoae/e2e/tests/test_e2e_losses.py -q
    # or, without pytest:
    python geoae/e2e/tests/test_e2e_losses.py
"""
from __future__ import annotations




import torch
import torch.nn as nn
import torch.nn.functional as F

from geoae.e2e.losses import kl_loss
from geoae.e2e.logits import LogitsComputer


# --------------------------------------------------------------------------- #
# Fake Llama/Qwen-shaped model: .model.layers, .model.norm, .lm_head
# --------------------------------------------------------------------------- #

class _FakeBase(nn.Module):
    def __init__(self, n_layers, D):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(n_layers)])
        self.norm = nn.LayerNorm(D)


class FakeLM(nn.Module):
    def __init__(self, n_layers=4, D=16, V=32):
        super().__init__()
        self.model = _FakeBase(n_layers, D)
        self.lm_head = nn.Linear(D, V, bias=False)


# --------------------------------------------------------------------------- #
# kl_loss
# --------------------------------------------------------------------------- #

def test_kl_zero_when_equal():
    logits = torch.randn(8, 50)
    assert kl_loss(logits, logits).item() < 1e-6


def test_kl_nonnegative():
    t = torch.randn(8, 50)
    s = torch.randn(8, 50)
    assert kl_loss(t, s).item() >= 0.0


def test_kl_matches_torch_reference():
    t = torch.randn(6, 40)
    s = torch.randn(6, 40)
    ours = kl_loss(t, s)
    # F.kl_div(log_q, p) == sum p (log p - log q); batchmean divides by batch.
    p = F.softmax(t, dim=-1)
    log_q = F.log_softmax(s, dim=-1)
    ref = F.kl_div(log_q, p, reduction="batchmean")
    assert torch.allclose(ours, ref, atol=1e-6)


def test_kl_teacher_detached():
    t = torch.randn(4, 30, requires_grad=True)
    s = torch.randn(4, 30, requires_grad=True)
    kl_loss(t, s).backward()
    assert t.grad is None or torch.allclose(t.grad, torch.zeros_like(t))
    assert s.grad is not None and s.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# LogitsComputer regime detection + head path
# --------------------------------------------------------------------------- #

def test_last_layer_detection_and_head_path():
    D, V, n = 16, 32, 4
    lm = FakeLM(n_layers=n, D=D, V=V)
    lc = LogitsComputer(lm, layer_idx=n - 1)
    assert lc.is_last and not lc.needs_input_ids

    x = torch.randn(5, D, requires_grad=True)
    logits = lc.head_logits(x)
    assert logits.shape == (5, V)
    # grad flows through the frozen head back into the input
    logits.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_intermediate_requires_input_ids():
    lm = FakeLM(n_layers=4, D=16, V=32)
    lc = LogitsComputer(lm, layer_idx=1)
    assert (not lc.is_last) and lc.needs_input_ids
    try:
        lc.student_logits(torch.randn(3, 16))
    except ValueError:
        return
    raise AssertionError("expected ValueError when input_ids missing")


def test_layer_idx_bounds():
    lm = FakeLM(n_layers=4, D=16, V=32)
    for bad in (-1, 4, 99):
        try:
            LogitsComputer(lm, layer_idx=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for layer_idx={bad}")


# --------------------------------------------------------------------------- #
# Manual runner
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
