# Running Experiments

## Unsupervised (diverse corpus)

1. `geoae-extract --config configs/base/full_run_v2.yaml`
2. `geoae-train --config configs/base/layer_27.yaml`
3. `geoae-evaluate --checkpoint checkpoints/best_val.pt`

KL variant: `python -m geoae.e2e.train_stream --config configs/e2e/general/llama3.2-3B/layer27/kl_gelu.yaml`

## DBpedia-14

```bash
python -m geoae.dbpedia.extract --mode unprompted --pooling last --layer 27
geoae-e2e-train --config configs/e2e/llama3.2-3B/layer27/unprompted_last_gelu.yaml
python -m geoae.dbpedia.evaluate --checkpoint checkpoints/best_val.pt
```

## Artifacts (gitignored)

All paths resolve relative to `GeoAE/` — run commands from this directory. Bundled artifacts:

- `e2e/checkpoints/` — trained e2e KL checkpoints (Llama layers 16/20/24/27, Qwen layer 31), including `closest_tokens.json` / DLA outputs per run
- `results_*.json`, `results.txt`, `clustering_quality_comparison.json` — experiment results
- `dbpedia/` — correct-prediction JSONs, per-model configs, results, and `run_experiment.sh`
- `activations_diverse/`, `activations_qwen_diverse/` — extracted activations (113G + 38G), moved here from `geosep/`; the old `geosep/activations_*` paths are now symlinks pointing back into `GeoAE/`.

## Layer configs

`configs/base/layer_{16,20,24,27}.yaml` and matching `configs/e2e/general/.../kl_gelu.yaml`.
