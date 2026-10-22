"""Public imports stay lightweight; credentials resolve only at loading boundaries."""
import os
import subprocess
import sys

from geoae.checkpoint import load_tokenizer


def test_plain_import_preserves_environment_and_defers_torch():
    script = """
import os, sys
before = dict(os.environ)
import geoae
assert dict(os.environ) == before
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'geoae.hf_auth' not in sys.modules
from geoae import Config
assert Config().model.n_clusters == 128
assert 'torch' not in sys.modules
assert 'GeoSepAE' in dir(geoae)
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True,
                            env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert result.returncode == 0, result.stderr


def test_lazy_public_exports_keep_class_identity():
    import geoae
    from geoae.config import Config
    from geoae.model import AEOutput, GeoAE

    assert geoae.Config is Config
    assert geoae.GeoAE is GeoAE and geoae.GeoSepAE is GeoAE
    assert geoae.AEOutput is AEOutput


def test_tokenizer_resolves_auth_before_forwarding_original_options(monkeypatch):
    from geoae import hf_auth
    from transformers import AutoTokenizer

    events = []
    tokenizer = object()

    def auth(**kwargs):
        events.append(("auth", kwargs))
        return False

    def construct(name, *args, **kwargs):
        events.append((name, args, kwargs))
        return tokenizer

    monkeypatch.setattr(hf_auth, "ensure_hf_login", auth)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", construct)
    loaded = load_tokenizer("fixture/tokenizer", "optional-argument", local_files_only=True, revision="frozen")
    assert loaded is tokenizer
    assert events == [("auth", {"verbose": False}),
                      ("fixture/tokenizer", ("optional-argument",), {"local_files_only": True, "revision": "frozen"})]
