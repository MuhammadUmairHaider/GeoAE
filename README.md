# GeoAE

GeoAE learns structured autoencoders over frozen language-model activations. It supports MSE and KL training, token bypasses, linear and nonlinear decoders, clustering and SAE baselines, and geometry, alignment, faithfulness and intervention evaluations.

Shared implementations live in `geoae/`. Each scientific question has a directory under `experiments/` containing its recipes, workflow, analysis and evidence references. The [paper](docs/paper/README.md) describes the linear-decoder method; [nonlinear decoders](experiments/nonlinear_decoders/README.md) are subsequent work.

Start with the supported CPU environment:

```bash
bash scripts/setup_laptop.sh
.venv/bin/geoae --help
.venv/bin/geoae commands
```

Use `.venv/bin/python` in your editor. Setup installs the locked environment and runs offline checks. Keep model loading, activation extraction and GPU work on Delta or Argo. See [usage](docs/usage.md) for execution instructions and optional dependencies.

```text
geoae/                  shared implementation
  training/             MSE, cached KL and streaming KL
  extraction/           corpora, activations and benchmark caches
  baselines/            k-means, token erasure and SAE controls
  evaluation/           geometry, alignment, faithfulness and control
  analysis/             probes, inspection and reporting
  workflow/             execution, resources and source snapshots
experiments/            one directory per research question
  paper/                main linear-decoder recipes and tables
  nonlinear_decoders/   nonlinear follow-up and cluster workflows
  ...                   depth, width, initialization and other controls
configs/                generic examples and cluster profiles
scripts/                environment setup and small launch helpers
tests/                  offline CPU verification
docs/                   architecture, usage, paper and history
```

The [architecture guide](docs/architecture.md) explains how the pieces fit. The [experiment index](experiments/README.md) is the starting point for research recipes. Use command help before executing a workflow:

```bash
geoae train --help
geoae extract-sampled --help
geoae evaluate geometry --help
geoae baseline --help
geoae analyze --help
```

The seven historical installed commands, including `geoae-train` and `geoae-e2e-train`, remain available. The [source relocation ledger](docs/history/audits/source_paths.json) records previous module names, and the [config map](experiments/config_paths.json) resolves previous YAML paths. Optional extras are `sae` for the matched SAE reference, `judges` for explicitly requested API evaluation, and `plots` for UMAP plotting.

Saved runs, datasets, logs and reports are preserved in the sibling `../GeoAE-artifacts/research/` store. The [artifact catalog](docs/artifacts.json) records their locations and preservation manifest. Set `GEOAE_ARTIFACT_ROOT` when reading them from another location. Scientific inputs, remote checkpoints and frozen source snapshots retain their original paths and hashes.

Run the offline suite with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest
```

Read [research state and evidence](docs/research.md) before interpreting saved results. Historical discussions and migration evidence remain indexed in [docs/history](docs/history/CHAT_HISTORY.md). The [cleanup report](docs/history/audits/cleanup_20261022.md) records the changes, validation, limitations and rollback path; the [original audit](docs/history/audits/CODEBASE_CLEANUP_PLAN.md) explains the rationale.

Licensed under the [MIT license](LICENSE).
