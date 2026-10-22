"""Compatibility patches preserve canonical features and current nonlinear semantics."""
import ast
import importlib.util
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geoae.model import GeoAE


@pytest.fixture
def compat():
    path = Path(__file__).resolve().parents[1] / "scripts/deltaai/qwen_paper_compat.py"
    spec = importlib.util.spec_from_file_location("qwen_paper_compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def canonical(compat, tmp_path):
    original = Path(os.getenv("GEOAE_PAPER_REPO", "/home/exouser/code/GeoAE"))
    if not all((original / name).is_file() for name in compat.PATCHES):
        pytest.skip("Canonical paper checkout unavailable; deployed evaluator behavior has separate tests")
    source = tmp_path / "source"
    for name in compat.PATCHES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / name, target)
    return source


def definitions(path, names):
    """Compile actual patched function/class definitions using the current core."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    namespace = {"torch": torch, "Tensor": torch.Tensor, "np": np, "GeoAE": GeoAE}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_patch_is_only_reviewed_hunks_and_preserves_newer_layer_support(compat, canonical):
    original = {name: (canonical / name).read_text() for name in compat.PATCHES}
    receipt = compat.patch_nonlinear_evaluators(canonical)
    assert set(receipt) == set(compat.PATCHES)
    for name, replacements in compat.PATCHES.items():
        updated = (canonical / name).read_text()
        assert receipt[name]["before_sha256"] != receipt[name]["after_sha256"]
        assert receipt[name]["hunks"] == len(replacements)
        for before, after in reversed(replacements):
            assert updated.count(after) == 1
            updated = updated.replace(after, before, 1)
        assert updated == original[name]
    steering = (canonical / "geoae/interp/concept_steer_generate.py").read_text()
    assert "downstream_blocks = validate_layer(lm, args.layer)" in steering
    assert '"intervention_forward": "complete_lm_at_selected_block"' in steering


def test_changed_anchor_fails_before_any_source_file_is_mutated(compat, canonical):
    path = canonical / "geoae/interp/faithfulness_ablate.py"
    path.write_text(path.read_text().replace("return self.ae.decoder(v) * self.std", "return v"))
    original = {name: (canonical / name).read_bytes() for name in compat.PATCHES}
    with pytest.raises(ValueError, match="anchor changed"):
        compat.patch_nonlinear_evaluators(canonical)
    assert {name: (canonical / name).read_bytes() for name in compat.PATCHES} == original


@pytest.mark.parametrize("decoder", ["mlp", "residual_mlp"])
@torch.no_grad()
def test_patched_canonical_removal_adds_exact_state_conditioned_decoded_change(compat, canonical, decoder):
    compat.patch_nonlinear_evaluators(canonical)
    functions = definitions(canonical / "geoae/interp/concept_remove_generate.py", ["make_edit_splice"])
    ae = GeoAE(hidden_size=2, latent_dim=4, n_clusters=2, nonlinearity="linear",
               decoder_type=decoder, decoder_hidden_dim=4, decoder_output_bias=True).eval()
    if decoder == "residual_mlp":
        ae.decoder.output.weight[:, :2].copy_(torch.eye(2))
        ae.decoder.skip.bias.fill_(10.0)
    h = torch.tensor([[[1.0, -0.2], [-2.0, 1.0], [0.3, 0.4]]])
    mean, std = torch.zeros(2), torch.ones(2)
    change = torch.tensor([0.2, -0.1, 0.3, 0.1])
    diagnostics = {name: [] for name in ("fire", "edit_norm", "pos_fire_tgt", "pos_fire_comp")}
    tap = SimpleNamespace(ids_for=lambda hs: torch.zeros(hs.shape[0] * hs.shape[1], dtype=torch.long))
    splice = functions["make_edit_splice"]("z", "delta", lambda z: z + change,
        np.full(4, -10.0), np.full(4, 10.0), mean, std, ae, tap, "cpu", diagnostics,
        n_tgt=1, pos_gate="none")
    z = ae.encode(h.reshape(-1, 2))
    expected = h + (ae.decode(z + change) - ae.decode(z)).reshape_as(h)
    torch.testing.assert_close(splice(h), expected)


@pytest.mark.parametrize("decoder,bias", [("mlp", False), ("residual_mlp", False), ("linear", True)])
def test_patched_fixed_direction_evaluators_refuse_unsupported_decoder(compat, canonical, decoder, bias):
    compat.patch_nonlinear_evaluators(canonical)
    ae = GeoAE(hidden_size=2, latent_dim=4, n_clusters=2, nonlinearity="linear",
               decoder_type=decoder, decoder_output_bias=bias).eval()
    steer = definitions(canonical / "geoae/interp/concept_steer_generate.py", ["decode_direction"])
    dla = definitions(canonical / "geoae/interp/logit_attribution.py", ["compute_dla"])
    faith = definitions(canonical / "geoae/interp/faithfulness_ablate.py", ["AEEnc"])
    wrapper = faith["AEEnc"].__new__(faith["AEEnc"])
    wrapper.ae, wrapper.std = ae, torch.ones(2)
    for call in (lambda: steer["decode_direction"](ae, torch.ones(4)),
                 lambda: dla["compute_dla"](ae, torch.ones(2), torch.ones(3, 2), None, False),
                 lambda: wrapper.to_raw_dir(torch.ones(4))):
        with pytest.raises(ValueError, match="requires a bias-free linear decoder"):
            call()
