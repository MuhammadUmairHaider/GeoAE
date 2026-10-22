# Activation sampling experiments

Which results depend on prefix sampling, document positions and corpus mixtures?

These are older extraction controls and the diverse-corpus training baseline. Final sampled-paper extraction recipes are owned by [paper](../paper/README.md). Original activation, corpus and checkpoint paths remain unchanged inside the YAMLs. Extraction loads language models and writes large activation arrays; use cluster storage.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_diverse_10M`. Original evidence is preserved at [`corpus/sampled_v1/manifest.json`](../../../GeoAE-artifacts/research/corpus/sampled_v1/manifest.json) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae extract --config experiments/data_sampling/configs/llama3.2-3b_extract.yaml
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.
