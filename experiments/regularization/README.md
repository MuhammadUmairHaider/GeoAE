# Objective and encoder ablations

Which effects come from pull/separation, variance/covariance, staging and encoder nonlinearity?

This family retains the component-loss and tanh controls plus earlier MSE/hinge variants. Each YAML preserves its original data, loss coefficients and schedule. These are specific hypotheses; using one as a new default would require a separate research decision.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_gemma3_4b_sampled_struct5_10M`. Checkpoint output in the example: `checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias_nopull`. Original evidence is preserved at [`results/delta/g4b_l33_d5120`](../../../GeoAE-artifacts/research/results/delta/g4b_l33_d5120) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/regularization/configs/gemma3-4b_l33_k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias_nopull.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.

Additional experiment tools are grouped under [analysis programs](analysis/). Their inputs and compute requirements depend on the selected program; use the shared `geoae` commands for reusable training and evaluation.
