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
from geoae.model import AEOutput, GeoAE

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
]

# Backward-compatible alias (GeoSepAE was the research codebase name)
GeoSepAE = GeoAE
