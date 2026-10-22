# Small configuration examples

These generic examples demonstrate extraction and basic AE training settings. Scientific campaign recipes live under [experiments](../../experiments/README.md). Config names are descriptive examples, not current experiment status.

From the project root on a configured cluster:

```bash
geoae train --config configs/examples/small_test.yaml --no_wandb
```

The YAML must point to an existing activation dataset. Local CPU verification instead uses `bash scripts/remote/setup_laptop.sh --smoke-only`; it uses synthetic fixtures and does not load models or start training runs.
