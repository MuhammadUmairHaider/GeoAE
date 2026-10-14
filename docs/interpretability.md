# Interpretability Tools

All tools live under `geoae.interp`.

| Command | Purpose |
|---------|---------|
| `python -m geoae.interp.closest_tokens` | Max-activating corpus examples per cluster |
| `python -m geoae.interp.logit_attribution` | DLA per centroid |
| `python -m geoae.interp.dla_inventory` | Distinct feature inventory |
| `python -m geoae.interp.clustering_quality` | Geometric + functional metrics |
| `python -m geoae.interp.causal_concept_compare` | NeuronLens h vs z concept control |
| `python -m geoae.interp.steering_concept_compare` | Mean-diff steering comparison |
| `python -m geoae.interp.inference_benchmark` | Perplexity / KL under splice |
| `python -m geoae.interp.cluster_baseline` | Raw k-means baseline |

Core causal validation: `geoae.evaluate` with `SplicingHook` and `load_ae_from_checkpoint`.

NeuronLens primitives: `geoae.interp.neuronlens`.
