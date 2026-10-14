# GeoAE E2E KL variant

KL faithfulness replaces MSE. See [docs/training.md](../docs/training.md).

```bash
geoae-e2e-extract --config configs/e2e/llama3.2-3B/layer27/unprompted_last_gelu.yaml --split train
geoae-e2e-train --config configs/e2e/llama3.2-3B/layer27/unprompted_last_gelu.yaml --no_wandb
```

Streaming: `python -m geoae.e2e.train_stream --config configs/e2e/general/llama3.2-3B/layer16/kl_gelu.yaml`
