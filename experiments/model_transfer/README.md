# Gemma 12B transfer experiments

Does the reconstruction/clustering recipe transfer to the larger Gemma model?

The extraction and MSE-only recipes are retained together. These are historical model-transfer controls and are separate from the Gemma 4B final-paper and Gemma 1B campaigns. Model loading, activation extraction and training belong on the cluster.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_gemma3_12b`. Checkpoint output in the example: `checkpoints/gemma3-12B/layer47/mse_only_k2000`. Original evidence is preserved at [`runs`](../../../GeoAE-artifacts/research/runs) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/model_transfer/configs/gemma3-12b_l47_mse_only.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.
