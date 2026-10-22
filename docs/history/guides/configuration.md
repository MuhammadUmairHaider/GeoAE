# Configuration Guide

YAML configs have five sections: `extraction`, `data`, `model`, `loss`, `train`.

```python
from geoae import Config
cfg = Config.from_yaml("configs/base/layer_27.yaml")
```

## Key knobs

| Section | Important keys |
|---------|----------------|
| extraction | `model_name`, `layers`, `n_tokens`, `activations_dir` |
| data | `batch_size`, `val_frac`, `target_layer` |
| model | `latent_dim`, `n_clusters`, `nonlinearity`, `metric` |
| loss | `lambda_cluster`, `lambda_sep`, `tau_start`, `tau_end` |
| train | `n_epochs`, schedule epochs, `centroid_init`, `teacher_mode`, `seed` |

Paths resolve relative to the **GeoAE/** package root (`geoae.paths.PACKAGE_ROOT`).

Config trees: `configs/base/` (MSE), `configs/e2e/` (KL).
