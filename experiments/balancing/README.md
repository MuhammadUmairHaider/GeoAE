# Balancing and schedule experiments

Which effects come from balanced assignments, centroid updates and staged objectives?

This family owns no-Sinkhorn, Lloyd/frozen-codebook, Zipf and no-staging schedules, including the 50/100-epoch campaigns. The nobal directory names distinguish complete historical recipes; shared K or width does not make them duplicates. Preserve frozen codebooks and original checkpoint selection when comparing saved outputs.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_sampled_10M`. Checkpoint output in the example: `checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_nosinkhorn`. Original evidence is preserved at [`results/delta/nosinkhorn`](../../../GeoAE-artifacts/research/results/delta/nosinkhorn) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/balancing/configs/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_nosinkhorn.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.

Additional experiment tools are grouped under [analysis programs](analysis/). Their inputs and compute requirements depend on the selected program; use the shared `geoae` commands for reusable training and evaluation.
