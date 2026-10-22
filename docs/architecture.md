# Architecture

GeoAE learns a dense continuous representation of a frozen language model's activations while organizing that representation around cluster centers. Reconstruction uses the full latent vector. A cluster identifier is an interpretation of the latent, not the decoder's input.

The ordinary workflow is corpus preparation, activation extraction, normalization, AE training, then evaluation against the corresponding residual-stream baseline. [Experiment families](../experiments/README.md) own the scientific recipes. The reusable implementation is organized by its job:

| Area | Responsibility | Start reading |
| --- | --- | --- |
| Core | AE architecture, decoder variants and clustering objectives | [model.py](../geoae/model.py), [losses.py](../geoae/losses.py) |
| Training | Activation buffers, schedules, checkpoint tracking, MSE and KL training | [training/mse.py](../geoae/training/mse.py), [training/common.py](../geoae/training/common.py) |
| Extraction | Corpora, sampled/prefix activations and benchmark caches | [extraction/sampled.py](../geoae/extraction/sampled.py), [extraction/corpus.py](../geoae/extraction/corpus.py) |
| Baselines | Ordinary/balanced k-means, token erasure and SAE controls | [baselines](../geoae/baselines/) |
| Evaluation | Geometry, faithfulness, semantic alignment and interventions | [evaluation](../geoae/evaluation/) |
| Analysis | Inspection, probes, reports, plots and judge diagnostics | [analysis](../geoae/analysis/) |
| Workflow | Pipeline/resource planning and command dispatch | [workflow](../geoae/workflow/) |
| Loading and paths | Checkpoint compatibility, model layers and artifact locations | [checkpoint.py](../geoae/checkpoint.py), [lm_arch.py](../geoae/lm_arch.py), [paths.py](../geoae/paths.py) |

For the paper's token-bypass model, raw activation `h` is standardized with training statistics: `x = (h - mean) / std`. A fixed token table supplies the standardized token mean `b[t]`. The encoder computes `z = GELU(BatchNorm(W_e(x - b[t]) + bias))`; reconstruction is `x_hat = D(z) + b[t]`. The paper uses a bias-free linear decoder whose columns are normalized after updates. The nonlinear follow-up also supports a full MLP and a linear skip plus residual MLP. [model.py](../geoae/model.py) owns those choices, including token-aware `encode`, `decode` and `decoded_delta`.

Training combines reconstruction with cluster attraction/separation and variance/covariance regularization. Recipes determine when each objective starts, how centers are initialized, and how inactive centers are refreshed. The usual centers are EMA buffers. Sinkhorn assignments balance training batches; evaluation normally assigns points to their nearest center. A batch-balanced assignment must not be substituted for an independent semantic prediction. Seeded initialization and supervised continuation are explicit experimental controls, separate from unsupervised paper training.

Two model interventions answer different questions. A full reconstruction splice replaces `h` with `mean + std * (D(z) + b[t])` and measures faithfulness. A residual-preserving latent edit uses `h + std * (D(z_edited) - D(z))` and measures the edit without also imposing reconstruction error. This exact decoder difference is required for nonlinear decoders; decoding the latent difference alone is only equivalent for a linear decoder. The token bypass cancels from the difference but token IDs remain necessary when obtaining the latent.

Use the shared checkpoint loader rather than rebuilding architectures in individual evaluators. It restores the saved decoder configuration, normalization, token tables, centers, BatchNorm state and assignment settings. A tool must opt into token-aware loading before consuming a bypass checkpoint. Keep this contract and model-specific layer indexing consistent across extraction, caches and splicing.

Configs resolve data/checkpoint paths from the project root, retaining their existing cluster meaning. Scientific evidence lives outside the source tree; [the artifact catalog](artifacts.json) and `geoae.paths.artifact_path` locate it. `resolve_input_path` can find saved read-only inputs locally or in that store. Working/output paths use `resolve_path`, which never redirects a writer into the archive. Inputs that also receive normalization sidecars or caches must use an explicit working copy. Configuration and source relocation ledgers under [experiments](../experiments/config_paths.json) and [history/audits](history/audits/source_paths.json) preserve the previous entry points without keeping duplicate implementations.
