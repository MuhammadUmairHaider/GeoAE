import math

import numpy as np
import torch

from geoae.evaluation.perplexity import bootstrap_ratio_ci
from geoae.evaluation.perplexity import keep_bos
from geoae.evaluation.perplexity import make_chunks
from geoae.evaluation.perplexity import summarise
from geoae.evaluation.perplexity import zero_non_bos


def test_chunks_cover_every_token_once():
    ids = list(range(100, 100 + 23))
    ch = make_chunks(ids, 1, 8)
    assert all(c[0] == 1 and len(c) <= 8 for c in ch)
    assert [t for c in ch for t in c[1:]] == ids
    assert len(ch[-1]) == 1 + 23 % 7  # partial tail kept
    assert len(make_chunks(ids, 1, 8, max_chunks=2)) == 2


def test_one_token_tail_kept():
    ch = make_chunks(list(range(8)), 1, 8)
    assert len(ch) == 2 and ch[-1] == [1, 7]


def test_summarise():
    s = summarise(math.log(2) * 10, 10, 20, 5)
    assert abs(s["ppl"] - 2) < 1e-9
    assert abs(s["bits_per_byte"] - 0.5) < 1e-9
    assert abs(s["word_ppl"] - 4) < 1e-9


def test_bootstrap_contains_point():
    rng = np.random.default_rng(1)
    tok = np.full(40, 100.0)
    d = rng.normal(5, 2, 40)
    lo, hi = bootstrap_ratio_ci(d, tok, 500, 0)
    assert lo <= d.sum() / tok.sum() <= hi


def test_keep_bos():
    hs = torch.randn(1, 6, 4)
    out = keep_bos(lambda h: h * 0 + 7)(hs)
    assert torch.equal(out[:, 0], hs[:, 0]) and (out[:, 1:] == 7).all()
    z = zero_non_bos(hs)
    assert torch.equal(z[:, 0], hs[:, 0]) and (z[:, 1:] == 0).all()
