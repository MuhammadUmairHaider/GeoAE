# Experiments

Choose a scientific question here, then use its configs and analysis/launch recipes. The shared model, losses, training, data and evaluation implementations live in `geoae/`; experiments reuse them. GPU work and large datasets stay on Delta or Argo. Local setup and reading saved results do not start experiments.

| Family | Question | Configs |
| --- | --- | ---: |
| [paper](paper/README.md) | How do dense clustered representations compare with residual-stream k-means and Gemma Scope? | 13 |
| [token_bypass](token_bypass/README.md) | Which effects come from token mean removal, bypass strength and seed controls? | 12 |
| [nonlinear_decoders](nonlinear_decoders/README.md) | Does a nonlinear inverse map improve reconstruction, geometry, alignment or control relative to the linear decoder? | 6 |
| [width](width/README.md) | Which geometry and reconstruction changes follow from changing the latent width? | 8 |
| [cluster_count](cluster_count/README.md) | How does the learned partition change with the number of cluster centers? | 6 |
| [balancing](balancing/README.md) | Which effects come from balanced assignments, centroid updates and staged objectives? | 32 |
| [initialization](initialization/README.md) | How do random/k-means++, density peaks and labeled anchors affect learned partitions? | 6 |
| [regularization](regularization/README.md) | Which effects come from pull/separation, variance/covariance, staging and encoder nonlinearity? | 18 |
| [data_sampling](data_sampling/README.md) | Which results depend on prefix sampling, document positions and corpus mixtures? | 5 |
| [depth](depth/README.md) | How does the same GeoAE recipe behave at other transformer blocks? | 13 |
| [supervised](supervised/README.md) | Can labeled continuation improve alignment without degrading reconstruction or model behavior? | 2 |
| [e2e](e2e/README.md) | What changes when training preserves downstream model predictions rather than only activation reconstruction? | 23 |
| [dbpedia](dbpedia/README.md) | How do document pooling, prompting and supervised initialization change a dataset-specific AE? | 23 |
| [gemma_1b](gemma_1b/README.md) | Does the clustered-AE recipe transfer across depth and width in Gemma 3 1B? | 8 |
| [model_transfer](model_transfer/README.md) | Does the reconstruction/clustering recipe transfer to the larger Gemma model? | 2 |
| [gpt2](gpt2/README.md) | Can post-hoc GeoAE extraction, token bypass and splicing operate across all GPT-2 blocks? | 13 |
| [sae_comparison](sae_comparison/README.md) | How do pretrained and matched sparse autoencoders compare with GeoAE? | — |
| [matched_topk](matched_topk/README.md) | How do data-matched BatchTopK and Matryoshka SAE baselines behave? | — |
| [concept_steering](concept_steering/README.md) | Which semantic edits transfer across latent representations and control settings? | — |
| [number_control](number_control/README.md) | Do continuous latent edits shift numeric predictions as intended? | — |
| [compositionality](compositionality/README.md) | How does representation geometry capture compositional visual-language contrasts? | — |

Small generic training examples live in [configs/examples](../configs/examples/). The [configuration path map](config_paths.json), [migration ledger](config_migration.json) and [tab-separated map](config_moves.tsv) account for all 196 original YAML paths and their 193 current owners. Five additional distinct recipes recovered from the older checkout bring the total to 198 editable YAMLs: 190 here and eight generic examples. The three nonlinear linear-decoder references share the paper recipes, and two restored depth recipes share identical existing owners. Six main-paper recipes have readable names. [The consolidation receipt](../docs/history/audits/config_consolidation.json) records exact equality of the complete effective configs. Original source byte hashes remain in the ledger; [the cleanup receipt](../docs/history/audits/config_cleanup.json) records comment compaction and six explicit `no_sinkhorn` flags. Dataset, checkpoint and run paths retain their original meaning.

Saved results, launch receipts, frozen source snapshots and reports are stored outside the code tree in `../GeoAE-artifacts/research/`. [docs/artifacts.json](../docs/artifacts.json) is their catalog. New analysis programs should accept an artifact root or use `geoae.paths.artifact_path` rather than reconstructing the old repository layout. Frozen snapshots are evidence and must not be globally rewritten when source paths change.

For the paper authoring source and its known input gaps, see [docs/paper](../docs/paper/README.md). The nonlinear-decoder family is separate follow-up work; existing runs and checkpoints remain usable.
