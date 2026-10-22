# Usage

Use this Mac for source editing, small CPU checks and reading saved results. Keep language-model loading, dataset extraction and GPU training/evaluation on Delta or Argo. The source tree and external evidence store are separate; neither opening the project nor running setup starts an experiment.

From the project root, install the supported locked environment and run its offline CPU smoke checks:

```bash
bash scripts/setup_laptop.sh
bash scripts/setup_laptop.sh --smoke-only
```

Use `.venv/bin/python` as the IDE interpreter. Setup preserves `uv.lock` and does not download research models or datasets. [The setup status](history/migration/LAPTOP_SETUP_STATUS.md) and [handoff](history/migration/LAPTOP_HANDOFF.md) describe cluster access and the recovered workspace.

The `geoae` command exposes training, extraction, evaluation, analysis and baselines. Use help to discover supported names and the selected implementation's arguments:

```bash
geoae --help
geoae commands
geoae train --help
geoae train-kl --help
geoae extract --help
geoae evaluate --help
geoae analyze --help
geoae baseline --help
geoae prepare --help
```

The common evaluation names are `geometry`, `alignment`, `atlas-alignment`, `mmlu`, `perplexity`, `control`, `number-control` and `faithfulness`. Baselines include `kmeans`, `balanced-kmeans`, `token-erasure`, and the `matched-train` / `matched-evaluate` / `matched-report` SAE controls. For example, `geoae evaluate control --help` shows the intervention evaluator's options. Other commands use their descriptive module names with dashes. Historical installed entry points such as `geoae-train` remain available and call the same implementations.

Choose configs by [scientific question](../experiments/README.md). Generic examples live in [configs/examples](../configs/examples/); the [path map](../experiments/config_paths.json) resolves previous config names. On a configured cluster, an explicit MSE-training invocation is:

```bash
geoae train --config experiments/paper/configs/llama.yaml --no_wandb
```

KL training uses `geoae train-kl` for cached activations or `geoae train-stream` for streaming, with an [e2e recipe](../experiments/e2e/README.md). Extraction uses `geoae extract` for the prefix workflow or `geoae extract-sampled` for positions sampled across documents. `geoae corpus` builds a corpus mixture, `geoae token-bias` prepares fixed token statistics, and `geoae prepare` lists benchmark/cache preparation tools. These commands execute work; inspect their arguments and the YAML's activation, token-table and checkpoint paths first. A training config is a recorded recipe, not a statement that its input files exist on the laptop. Cluster launchers preserve the original remote storage layout and frozen source snapshots.

An activation directory must contain the requested layer arrays and their metadata. Bypass training additionally needs aligned token IDs and the matching fixed token table. Normalization and token statistics must come from the training split. Cached semantic evaluations must agree with the model, layer, pooling, tokenization and document-selection protocol of the checkpoint. Reusing a similar filename is insufficient.

The default evidence store is the sibling `../GeoAE-artifacts/research/`, preserving original `runs/`, `results/`, `reports/`, `logs/` and metadata subtrees. [docs/artifacts.json](artifacts.json) records its contents. For another machine or checkout, set the artifact root to the equivalent location:

```bash
export GEOAE_ARTIFACT_ROOT=/path/to/GeoAE-artifacts/research
```

Analysis code should accept an artifact-root option or resolve saved inputs through `geoae.paths.artifact_path`. New outputs should have a new run/output directory. Existing manifests, hashes, source archives, checkpoint paths and paid judge caches are evidence and should retain their original content. Report/export tools consume saved results; generation or judge commands can instead run models or paid APIs, so select the intended command explicitly.

For paper rendering, use the [paper build instructions](paper/README.md). For research scope, current local entry points and comparison limits, read [research.md](research.md). For the path from an activation to an intervention, read [architecture.md](architecture.md).
