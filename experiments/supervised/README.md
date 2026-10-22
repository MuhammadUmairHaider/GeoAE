# Supervised continuation experiments

Can labeled continuation improve alignment without degrading reconstruction or model behavior?

The encoder-only and full-model continuation recipes are separate controls. Their resume checkpoint and optimizer state matter; recorded schedules alone do not prove a clean restart or exact continuation. Historical methodological notes and results should accompany comparisons.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_diverse_10M`. Checkpoint output in the example: `checkpoints/llama3.2-3B/layer27/d12288_dpc_ft_sup_enc`. The historical report names `logs/ft_sup_enc_d12288.log`, but that log is absent from both recovered local copies. Use the original cluster checkpoint/optimizer evidence before interpreting a continuation. [The artifact catalog](../../docs/artifacts.json) locates the recovered material.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/supervised/configs/llama3.2-3b_l27_d12288_dpc_ft_sup_enc.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.

Additional experiment tools are grouped under [analysis programs](analysis/). Their inputs and compute requirements depend on the selected program; use the shared `geoae` commands for reusable training and evaluation.
