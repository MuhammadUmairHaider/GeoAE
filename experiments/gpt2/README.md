# GPT-2 compatibility experiments

Can post-hoc GeoAE extraction, token bypass and splicing operate across all GPT-2 blocks?

These are post-hoc AE recipes on a frozen GPT-2, not the GeoPretrain language-model training project. Each layer owns its token-bias table and checkpoint directory. The preserved GPU compatibility smoke record and code backups are provenance; they are not a replacement implementation.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_gpt2_sampled_struct5_10M`. Checkpoint output in the example: `checkpoints/gpt2/layer11/k2000_bnh_b32k_lam1_d1536_dpc_sampled_struct5_tokbias`. Original evidence is preserved at [`runs/gpt2_2x`](../../../GeoAE-artifacts/research/runs/gpt2_2x) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/gpt2/configs/layer11_2x.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.
