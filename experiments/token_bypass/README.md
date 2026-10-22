# Token bypass experiments

Which effects come from removing the per-token mean before encoding, routing it around reconstruction, or changing its strength?

This family retains the decoder-only and scaled-bypass controls plus the independent-seed recipes. The matched standard bypass/no-bypass pair is owned by [paper](../paper/README.md); width changes are owned by [width](../width/README.md). These links avoid duplicate editable baselines. Token tables must match the model, tokenizer, activation normalization and training split of the checkpoint.

The [configs](configs/) retain their cluster data/checkpoint paths; repeated comments were compacted with exact parsed-value checks in [the cleanup receipt](../../docs/history/audits/config_cleanup.json). [config_paths.json](../config_paths.json) maps previous paths; [config_migration.json](../config_migration.json) records their source hashes. Historical results and receipts are preserved in the sibling artifact store, including [the decoder-only source record](../../../GeoAE-artifacts/research/runs/decoderonly_d6144); [docs/artifacts.json](../../docs/artifacts.json) is the artifact catalog.

From the project root on a configured cluster, this example executes the decoder-only control:

```bash
geoae train --config experiments/token_bypass/configs/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly.yaml --no_wandb
```

The example reads `activations_sampled_10M`; inspect its YAML for the token table and checkpoint output. Keep seed, data, token-table and document-pool differences explicit when comparing saved runs. Opening the experiment index does not start a new run.

Additional experiment tools are grouped under [analysis programs](analysis/). Their inputs and compute requirements depend on the selected program; use the shared `geoae` commands for reusable training and evaluation.
