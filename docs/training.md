# Training Workflows

## MSE pipeline

```bash
geoae-extract --config configs/base/small_test.yaml --n_tokens 100000
geoae-train --config configs/base/layer_27.yaml --no_wandb
geoae-evaluate --checkpoint checkpoints/best_val.pt --experiment all
```

Or orchestrated: `geoae-pipeline all --layer 27 --no_wandb`

## E2E KL pipeline

```bash
geoae-e2e-extract --config configs/e2e/llama3.2-3B/layer27/unprompted_last_gelu.yaml --split train
geoae-e2e-train --config configs/e2e/llama3.2-3B/layer27/unprompted_last_gelu.yaml --no_wandb
```

Streaming (any layer, no cache): `python -m geoae.e2e.train_stream --config configs/e2e/general/llama3.2-3B/layer16/kl_gelu.yaml`

## Schedule

Three phases: recon-only epochs → +cluster loss → +sep loss with τ annealing.

## Reproducibility

`geoae.seeding.seed_everything(seed)` at all entry points.
