"""Labelled extractors share safe pooling and retain their importable helpers."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import geoae.extraction.biasbios as bias
import geoae.extraction.dbpedia as dbpedia
from geoae.extraction import common


@pytest.mark.parametrize("module", [bias, dbpedia])
def test_existing_extractor_helpers_remain_available(module):
    assert module.register_hook is common.register_hook
    assert module.output_dtype is common.output_dtype
    assert module.pool is common.pool


@pytest.mark.parametrize("tuple_output", [False, True])
def test_capture_detaches_and_handle_stops_recording(tuple_output):
    class Block(nn.Module):
        def forward(self, x):
            return (2 * x, None) if tuple_output else 2 * x

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([Block()])
    captured, handle = common.register_hook(model, 0)
    x = torch.arange(12.).reshape(1, 3, 4).requires_grad_()
    model.model.layers[0](x)
    assert len(captured) == 1 and not captured[0].requires_grad
    assert captured[0].device.type == "cpu"
    torch.testing.assert_close(captured[0], 2 * x)
    handle.remove()
    model.model.layers[0](x)
    assert len(captured) == 1


def test_gemma_storage_preserves_large_residuals_and_reports_unsafe_casts():
    gemma = SimpleNamespace(config=SimpleNamespace(model_type="gemma3"))
    llama = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    assert common.output_dtype(gemma) == np.dtype("float32")
    assert common.output_dtype(llama) == np.dtype("float16")
    assert common.output_dtype(gemma, "float16") == np.dtype("float16")
    h = torch.tensor([[70000., 2.], [90000., 6.]])
    np.testing.assert_array_equal(common.pool(h, "last", common.output_dtype(gemma)), [90000., 6.])
    np.testing.assert_array_equal(common.pool(h, "mean", common.output_dtype(gemma)), [80000., 4.])
    with pytest.raises(ValueError, match="overflow"):
        common.pool(h, "last")
    with pytest.raises(ValueError, match="Non-finite"):
        common.pool(torch.tensor([[float("nan")]]), "mean", np.float32)
    with pytest.raises(ValueError, match="activation dtype"):
        common.output_dtype(llama, "float64")


def test_small_pooling_retains_original_fp16_values():
    h = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    np.testing.assert_array_equal(common.pool(h, "last"), np.array([3., 4.], np.float16))
    np.testing.assert_array_equal(common.pool(h, "mean"), np.array([2., 3.], np.float16))
