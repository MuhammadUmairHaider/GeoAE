# DBpedia training controls

How do document pooling, prompting and supervised initialization change a dataset-specific AE?

These are dataset-trained controls, separate from evaluation-only DBpedia labels for the general-corpus GeoAE. Prompted/unprompted and last/mean pooling are distinct inputs. Use the recorded model, layer, train split and pooling when extracting activations or comparing checkpoints.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `dbpedia/activations/llama3.2-3B/last/unprompted`. Checkpoint output in the example: `dbpedia/checkpoints/llama3.2-3B/layer27/unprompted_last_gelu`. Original evidence is preserved at [`dbpedia/activations`](../../../GeoAE-artifacts/research/dbpedia/activations) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/dbpedia/configs/llama3.2-3B/layer27/unprompted_last_gelu.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.

Additional experiment tools are grouped under [launch and workflow scripts](scripts/). Their inputs and compute requirements depend on the selected program; use the shared `geoae` commands for reusable training and evaluation.
