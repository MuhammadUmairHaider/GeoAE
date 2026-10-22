"""
GeoAE — concept-cluster autoencoder over frozen LLM residual streams.

Train a geometrically separated concept dictionary on layer activations, with
faithfulness measured by MSE reconstruction or end-to-end KL on next-token logits.
"""

from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "AEOutput",
    "Config",
    "DataConfig",
    "ExtractionConfig",
    "GeoAE",
    "LossConfig",
    "ModelConfig",
    "TrainConfig",
    "__version__",
    "ensure_hf_login",
]

_EXPORTS = {
    **{name: ("geoae.config", name) for name in (
        "Config", "DataConfig", "ExtractionConfig", "LossConfig", "ModelConfig", "TrainConfig")},
    "AEOutput": ("geoae.model", "AEOutput"),
    "GeoAE": ("geoae.model", "GeoAE"),
    "GeoSepAE": ("geoae.model", "GeoAE"),
    "ensure_hf_login": ("geoae.hf_auth", "ensure_hf_login"),
}


def __getattr__(name):
    """Load public classes on demand; importing GeoAE never resolves credentials."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
