# Latent width experiments

Which geometry and reconstruction changes follow from changing the latent width?

These recipes preserve the original Llama width arms, including sampled and older diverse-activation inputs. Compare arms with matched data, bypass, checkpoint selection and evaluation protocol. Cluster-count changes belong to [cluster_count](../cluster_count/README.md); compound loss changes are retained under [regularization](../regularization/README.md).

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_sampled_10M`. Checkpoint output in the example: `checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias`. Original evidence is preserved at [`results/delta/d12288`](../../../GeoAE-artifacts/research/results/delta/d12288) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/width/configs/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d12288_dpc_sampled_tokbias.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.


The saved-result report tools live in [analysis](analysis/). They read historical inputs through `geoae.paths.resolve_input_path`; new outputs default to `results/width/` in the working checkout:

```bash
scripts/py experiments/width/analysis/extract_report.py
scripts/py experiments/width/analysis/build_report.py --input results/width/report_data.json --output results/width/geoae_round2_report.html
```

The supplied local archive contains `results/delta/summary.json`, but neither `report_data.json` nor `logs/panel_usage.json` was found in the preserved artifact tree. The extractor can rebuild its structured report input from available result files; it cannot reconstruct the missing cluster-usage measurements. Restore that recorded input from the cluster before treating the width report as complete. The HTML builder requires its explicit input and fails if it is absent; it does not substitute invented measurements.
