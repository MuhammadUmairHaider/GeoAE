# Initialization experiments

How do random/k-means++, density peaks and labeled anchors affect learned partitions?

These recipes retain unsupervised baselines and seeded/anchor controls, including Atlas anchors and peak reseeding. Some arms use labels for initialization; they are not interchangeable with the paper’s unsupervised training. Component ablations that change initialization have one owner here and can be compared with the paper baseline.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_diverse_10M`. Checkpoint output in the example: `checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_seeded`. Original evidence is preserved at [`results/delta/d6144`](../../../GeoAE-artifacts/research/results/delta/d6144) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/initialization/configs/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_seeded.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.
