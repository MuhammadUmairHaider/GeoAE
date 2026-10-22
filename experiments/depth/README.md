# Representation depth experiments

How does the same GeoAE recipe behave at other transformer blocks?

Llama block 14 and Gemma block 17 have separate bypass/no-bypass and extraction recipes. The filenames state the actual block; do not infer paper appendix depth from the family name. Dataset/model identity and block indexing must match each checkpoint and evaluation cache.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_llama3_3b_l14_sampled_struct5_train10M`. Checkpoint output in the example: `checkpoints/llama3.2-3B/layer14/k2000_bnh_b32k_lam1_d6144_dpc_sampled_struct5_train10M_tokbias`. Original evidence is preserved at [`runs`](../../../GeoAE-artifacts/research/runs) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/depth/configs/llama3.2-3b_l14_k2000_bnh_b32k_lam1_d6144_dpc_sampled_struct5_tokbias.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.

The later final-block recipes with `train10M` in their filenames retain the
matched-depth extraction, token-table and checkpoint paths. Six later recipes were recovered exactly from the preserved code checkout; their original hashes are recorded in [the workflow ledger](../../docs/history/audits/workflow_remaining_paths.json). Four distinct extraction/bypass recipes remain here. The Llama plain recipe also owns the identical older paper `struct5` variant, while the Gemma plain recipe shares [the paper no-bypass config](../paper/configs/gemma_no_bypass.yaml). [The consolidation receipt](../../docs/history/audits/config_consolidation.json) records their complete effective equality.
