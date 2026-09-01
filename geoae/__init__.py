"""
GeoAE — concept-cluster autoencoder over frozen LLM residual streams.

Train a geometrically separated concept dictionary on layer activations, with
faithfulness measured by MSE reconstruction or end-to-end KL on next-token logits.
"""

from geoae.config import (
    Config,
    DataConfig,
    ExtractionConfig,
    LossConfig,
    ModelConfig,
    TrainConfig,
)
from geoae.hf_auth import ensure_hf_login
from geoae.model import AEOutput, GeoAE

# Resolve an HF token once, at import, so gated checkpoints (Gemma 3, Llama)
# work from every entry point — including the many that build an AutoTokenizer
# before touching load_lm. Quiet and offline: it only exports HF_TOKEN from an
# already-present env var, a git-ignored token file, or an `hf auth login`
# credential. No token found is a no-op; load_lm still raises its GATED_HINT.
ensure_hf_login(verbose=False)

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

# Backward-compatible alias (GeoSepAE was the research codebase name)
GeoSepAE = GeoAE
