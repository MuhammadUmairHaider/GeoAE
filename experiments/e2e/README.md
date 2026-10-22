# KL faithfulness experiments

What changes when training preserves downstream model predictions rather than only activation reconstruction?

These configs use the end-to-end KL trainers, including general-corpus and DBpedia-conditioned variants. The frozen model/head and teacher mode are part of the input contract. General configs remain under their historical model/layer subdirectories so streaming and cached trainers can share the same recipe.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_diverse`. Checkpoint output in the example: `e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu`. Original evidence is preserved at [`runs`](../../../GeoAE-artifacts/research/runs) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train-kl --config experiments/e2e/configs/general/llama3.2-3B/layer27/kl_gelu.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.
