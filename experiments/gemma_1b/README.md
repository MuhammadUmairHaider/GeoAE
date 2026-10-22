# Gemma 1B transfer experiments

Does the clustered-AE recipe transfer across depth and width in Gemma 3 1B?

This campaign includes the original standalone DPC pipeline and the later sampled structured-data bypass arms. The standalone recipe writes under its run directory; the later recipes reuse shared activation dumps. Preserve their differing data contracts when interpreting old reports.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_gemma3_1b_sampled_struct5_10M`. Checkpoint output in the example: `checkpoints/gemma3-1B/layer25/k2000_bnh_b32k_lam1_d2304_dpc_sampled_struct5_tokbias`. Original evidence is preserved at [`results/delta/g1b_l25_d2304`](../../../GeoAE-artifacts/research/results/delta/g1b_l25_d2304) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/gemma_1b/configs/gemma3-1b_l25_k2000_bnh_b32k_lam1_d2304_dpc_sampled_struct5_tokbias.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.
