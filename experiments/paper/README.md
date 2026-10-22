# Paper experiments

How do dense clustered representations compare with residual-stream k-means and Gemma Scope?

These are the final-layer model recipes and matched bypass/no-bypass controls associated with the paper. Historical sampled-data variants remain distinct because their activation directories differ. Geometry and sequence/control evaluations use the bypass variant; token-level alignment uses its no-bypass partner. The paper and appendix also depend on the depth, width and cluster-count families.

The main recipes are:

| Model | Token bypass | No bypass |
| --- | --- | --- |
| Llama 3.2 3B | [llama.yaml](configs/llama.yaml) | [llama_no_bypass.yaml](configs/llama_no_bypass.yaml) |
| Gemma 3 4B | [gemma.yaml](configs/gemma.yaml) | [gemma_no_bypass.yaml](configs/gemma_no_bypass.yaml) |
| Qwen 2.5 3B | [qwen.yaml](configs/qwen.yaml) | [qwen_no_bypass.yaml](configs/qwen_no_bypass.yaml) |

These names follow the recorded paper arms. Historical sampling variants remain separate; in particular, Gemma's no-bypass arm uses its recorded `struct5_train10M` input and checkpoint path.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_sampled_10M`. Checkpoint output in the example: `checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias`. Original evidence is preserved at [`results/delta/d6144`](../../../GeoAE-artifacts/research/results/delta/d6144) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/paper/configs/llama.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.

Additional experiment tools are grouped under [analysis programs](analysis/) and [launch and workflow scripts](scripts/). Their inputs and compute requirements depend on the selected program; use the shared `geoae` commands for reusable training and evaluation.


Analysis programs and table generators resolve saved relative inputs in the checkout first, then in the sibling artifact store (`GEOAE_ARTIFACT_ROOT` can select another read-only mirror). Generated reports, summaries and figures use new local output paths. Pass input and output paths separately when processing an explicitly located archived directory.

The local artifact mirror is incomplete for some historical report programs. In particular, `runs/gemma3_1b_l25_d2304_dpc_seed42/manifest.json` (the default input to `analysis/update_gemma_report.py`) is absent. The Atlas K-scaling program under `../cluster_count/analysis/plot_atlas_k_scaling.py` also needs the missing `results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/kscale/` results. These tools remain available, but require their recorded cluster inputs or an explicit complete input mirror; no replacement experiment is started automatically.
