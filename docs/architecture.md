# Architecture Overview

GeoAE learns **K concept clusters** over frozen LLM residual activations. See the main [README](../README.md) for quick start.

## Pipeline

Frozen LLM → extract `layer_L.npy` → normalize → GeoAE (encode → Sinkhorn Q → decode) → MSE or KL faithfulness → causal splice validation.

## Core modules

- `geoae.model.GeoAE` — encoder, Sinkhorn assignment, unit-norm decoder, EMA centroids
- `geoae.losses` — recon, cluster, sep, Sinkhorn
- `geoae.e2e` — KL faithfulness via `LogitsComputer`
- `geoae.evaluate.SplicingHook` — layer-L residual replacement for causal tests
