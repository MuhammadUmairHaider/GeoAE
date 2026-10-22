# GeoAE

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

**GeoAE** (Geometric concept Autoencoder) trains a concept-cluster autoencoder over frozen LLM residual-stream activations. Latent concepts are encouraged to be geometrically separated via Sinkhorn-balanced soft clustering, while faithfulness is measured either by **MSE reconstruction** (base pipeline) or **end-to-end KL divergence** on next-token logits (e2e pipeline). Causal splice hooks validate that clusters carry interpretable, behaviorally meaningful information.

This package consolidates the research codebases `geosep/` and the planned `geosepae/` refactor into a single publishing-ready layout. The original trees are left intact for reference.

## Features

- **Concept-cluster autoencoder** (`GeoAE`) with linear / ReLU / GeLU encoders and unit-norm decoder
- **Sinkhorn-balanced clustering** with separation and usage regularization
- **MSE training** on memory-mapped activations from diverse corpora
- **End-to-end KL training** with frozen LM in the loop (`geoae.e2e`)
- **DBpedia-14 benchmark** for supervised cluster evaluation
- **Interpretability suite**: closest tokens, DLA, NeuronLens concept control, steering comparisons
- **Reproducible training** via centralized seeding and YAML configs

## Installation

From the `GeoAE/` directory:

```bash
cd GeoAE
pip install -e ".[dev]"
```

Requirements: Python 3.10+, CUDA-capable GPU for training and extraction (tests run on CPU).

## Quick start

### 1. Extract activations

```bash
geoae-extract --config configs/base/small_test.yaml --n_tokens 100000
```

### 2. Train (MSE faithfulness)

```bash
geoae-train --config configs/base/layer_27.yaml --no_wandb
```

### 3. Causal evaluation

```bash
geoae-evaluate --checkpoint checkpoints/best_val.pt --experiment all --n_eval 200
```

### 4. Full pipeline

```bash
geoae-pipeline all --layer 27 --n_clusters 128 --no_wandb
```

### 5. End-to-end KL training

```bash
# Precompute teacher logits (cached mode)
geoae-e2e-extract --config configs/e2e/llama3.2-3B/layer27/unprompted_last_gelu.yaml --split train

# Train with KL faithfulness
geoae-e2e-train --config configs/e2e/llama3.2-3B/layer27/unprompted_last_gelu.yaml --no_wandb
```

## Python API

```python
from geoae import GeoAE, Config
from geoae.data import ActivationBuffer, ShuffledActivationLoader
from geoae.losses import total_loss

cfg = Config.from_yaml("configs/base/default.yaml")
model = GeoAE(
    hidden_size=cfg.model.hidden_size,
    latent_dim=cfg.model.latent_dim,
    n_clusters=cfg.model.n_clusters,
    nonlinearity=cfg.model.nonlinearity,
)
```

`GeoSepAE` is retained as an alias for backward compatibility.

## Project layout

```
GeoAE/
├── geoae/              # Installable Python package
│   ├── model.py        # GeoAE architecture
│   ├── train.py        # MSE training loop
│   ├── extract.py      # Activation extraction
│   ├── evaluate.py     # Causal splice evaluation
│   ├── e2e/            # KL faithfulness variant
│   ├── dbpedia/        # DBpedia-14 benchmark
│   └── interp/         # Interpretability tools
├── configs/
│   ├── base/           # MSE pipeline YAML configs
│   └── e2e/            # E2E KL configs
├── tests/              # Unit tests (CPU)
├── docs/               # Detailed documentation
├── pyproject.toml
└── README.md
```

See [docs/architecture.md](docs/architecture.md) for design details.

## Documentation

| Guide | Description |
|-------|-------------|
| [Architecture](docs/architecture.md) | Model, losses, data flow |
| [Training](docs/training.md) | Base AE and e2e workflows |
| [Configuration](docs/configuration.md) | YAML config reference |
| [Interpretability](docs/interpretability.md) | Analysis and causal tools |
| [Experiments](docs/experiments.md) | Running benchmark pipelines |

## Tests

```bash
cd GeoAE
pytest
```

## Citation

If you use GeoAE in your research, please cite:

```bibtex
@software{geoae2026,
  title  = {GeoAE: Concept-Cluster Autoencoder for LLM Residual Streams},
  author = {GeoAE Contributors},
  year   = {2026},
  url    = {https://github.com/your-org/RepresentationAE}
}
```

Replace the URL and author list with your publication details when available.

## License

MIT License — see [LICENSE](LICENSE).

## Relation to `geosep/`

`geosep/` remains the original research workspace with experiment artifacts. **GeoAE** is the cleaned, installable successor. New development should target `GeoAE/`; consider deprecating `geosep/` once checkpoints and configs are migrated.
