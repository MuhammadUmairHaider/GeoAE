# Nonlinear decoder experiments

Does a nonlinear inverse map improve reconstruction, geometry, alignment or control relative to the linear decoder?

Full MLP and residual-MLP recipes share the token-bypass encoder and staged clustering setup. This is follow-up research beyond the paper’s linear-decoder architecture. The linear references share the paper family’s [Llama](../paper/configs/llama.yaml), [Gemma](../paper/configs/gemma.yaml), and [Qwen](../paper/configs/qwen.yaml) recipes. The preserved [Qwen campaign note](notes/qwen_campaign_20261018.md) describes the original launch and evaluation snapshots; dates and job states there are historical.

The [configs](configs/) are editable recipes. [config_paths.json](../config_paths.json) maps every previous configuration path to its single current owner; [config_migration.json](../config_migration.json) records original source byte hashes. Comment cleanup and explicit `no_sinkhorn` flags are documented in [the cleanup receipt](../../docs/history/audits/config_cleanup.json); canonical names and equivalent recipes are documented in [the consolidation receipt](../../docs/history/audits/config_consolidation.json). Dataset and checkpoint paths retain their original cluster meaning.

Activation input in the example: `activations_qwen2_5_3b_l35_struct5_train10M`. Checkpoint output in the example: `checkpoints/nonlinear_decoder/qwen2.5-3B/layer35/mlp_seed42`. Original evidence is preserved at [`runs/nonlinear_decoder_validation`](../../../GeoAE-artifacts/research/runs/nonlinear_decoder_validation) in the sibling artifact store; [the artifact catalog](../../docs/artifacts.json) records its location.

From the project root on a configured cluster, this example executes the recipe:

```bash
geoae train --config experiments/nonlinear_decoders/configs/qwen2.5-3b_l35/mlp.yaml --no_wandb
```

Read the YAML and confirm the input/output paths before starting a new run. Historical source snapshots, hashes and manifests remain in the artifact store. Opening this README or running local setup does not start an experiment.

The evaluation `prepare` commands under `deltaai/` use this reconciled checkout by default. `--paper-repo` can select another complete checkout with the same layout and exactly the model, config, and checkpoint-loader source hashes recorded by training. A different core implementation is rejected before a snapshot is created; loading the same checkpoint shapes does not prove equivalent evaluation behavior.

Historical `geoae/interp` / `eval_out` checkouts are not accepted by the maintained preparers, because their commands and imports form a different runtime layout. Replay those experiments with their existing frozen launchers and manifests. The recovery runner recognizes both recorded launcher layouts and preserves the old command paths, completion state, and source hashes; any recovery overlay must be supplied explicitly at the original path and cannot depend on absent refactor modules.

Use [plan.py](plan.py) for source preparation and campaign commands, and [analyze.py](analyze.py) to rebuild saved comparison tables and figures. [Saved result references](results/README.md) locate the evidence without duplicating it in the source tree.
