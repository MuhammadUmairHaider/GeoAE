# Gemma 3 4B / layer 33 — MSE-only run: clustering quality + DB14 steering

Run finished 2026-08-21. `checkpoints/gemma3-4B/layer33/mse_only_k2000/best_val.pt`
= **epoch 42, step 76,566, val_mse 0.00243** (z-space ⇒ FVE ≈ 0.9976), tau 0.264.
W&B final-diag: silhouette −0.0364, effective_k 2000, min centroid dist 9.995,
dying 128, balance entropy 0.037.

FVE 0.998 with `latent_dim == hidden_size` and no sparsity term is **not** a win —
same near-identity caveat as 12B. The commands below are the deciding evidence.

Everything is resolved from the checkpoint (`google/gemma-3-4b-pt`, layer 33), so
no extra config is needed. Run from the repo root.

---

## 1. Raw k-means baseline (must run FIRST — none exists for 4B/L33)

`clustering_quality` compares against a raw-residual k-means npz, and there is no
`baseline_kmeans_k2000.npz` under `checkpoints/gemma3-4B/`. Fit it with the
**corrected** settings — the 12B baseline was under-fit (early-stopped at step
15/1171 on 500k tokens), which contaminated the first 12B comparison:

```bash
uv run python -u -m geoae.interp.fit_baseline_kmeans \
  --checkpoint checkpoints/gemma3-4B/layer33/mse_only_k2000/best_val.pt \
  --activations activations_gemma3_4b/layer_33.npy \
  --n_clusters 2000 \
  --n_sample 1500000 \
  --max_no_improvement 0 \
  --seed 42 \
  --out checkpoints/gemma3-4B/layer33/baseline_kmeans_k2000.npz \
  2>&1 | tee logs/fit_baseline_kmeans_gemma3_4b_l33.log
```

`--max_no_improvement 0` = None = run the full budget. `--n_sample 1500000` is the
script default (the 12B fit used 500k). Peak RAM ≈ 15 GB for the sample (107 GB
free); the slow part is 1.5M scattered row reads out of the 80 GB npy.

## 2. Clustering quality — MSE-only AE vs raw

```bash
uv run python -u -m geoae.interp.clustering_quality \
  --baseline checkpoints/gemma3-4B/layer33/baseline_kmeans_k2000.npz \
  --checkpoints checkpoints/gemma3-4B/layer33/mse_only_k2000/best_val.pt \
  --names raw-kmeans mse-only \
  --activations_dir activations_gemma3_4b \
  --layer 33 \
  --n_sample 500000 \
  --seed 0 \
  --out clustering_quality_gemma3_4b_l33_mse_only.json \
  2>&1 | tee logs/clustering_quality_gemma3_4b_l33_mse_only.log
```

`--n_sample 500000` matches the 12B probe so the two are comparable as a
model-scale pair. Dead-centroid masking is now built into the metrics
(`clustering_quality.py:351`), so no separate live-centroids rerun is needed.

## 3. DB14 concept steering — h (raw residual) vs z (AE latent)

Both spaces are steered in the same pass on the same docs, so the base comparison
is built in; there is no separate baseline run.

```bash
uv run python -u -m geoae.interp.steering_concept_compare \
  --checkpoint checkpoints/gemma3-4B/layer33/mse_only_k2000/best_val.pt \
  --dataset db14 --layer 33 \
  --correct_json dbpedia/joint_correct_db14_gemma3_4b_l33_mse_only_ep42.json \
  --out results_steering_db14_gemma3_4b_l33_mse_only_ep42.json \
  2>&1 | tee logs/steering_db14_gemma3_4b_l33_mse_only_ep42.log
```

The joint-correct set does not exist yet for this model and is built on the first
run (~15 min at 12B, faster at 4B), then cached and keyed by `model_name` + AE sha —
a later checkpoint will correctly refuse the stale cache rather than reuse it.
If the LM OOMs during that build, lower `--jc_batch_size` (default 16).

Defaults, matching the 12B runs: `--n_fit 80 --n_eval 50 --n_comp 80
--alphas 0.5 1.0 2.0 4.0 8.0 --seed 42` → 1120 fit docs, 700 eval docs, 14 concepts.

Smoke test first if you want (~2 concepts, small n): add `--smoke` and write to a
`_SMOKE.json` out path.

## 4. Optional — closest_tokens (the other half of the semantics evidence)

```bash
uv run python -u -m geoae.interp.closest_tokens \
  --checkpoint checkpoints/gemma3-4B/layer33/mse_only_k2000/best_val.pt \
  --n_tokens 500000 \
  --out closest_tokens_gemma3_4b_l33_mse_only.json \
  2>&1 | tee logs/closest_tokens_gemma3_4b_l33_mse_only.log
```

---

## Notes

- Disk is at 93% (38 GB free). All four outputs are small (the baseline npz is
  ~20 MB), so nothing here is at risk, but don't start another extraction alongside.
- `h sel` is not comparable across runs — the joint-correct doc set is filtered by
  each AE's own reconstruction. Within a run, h-vs-z is a clean paired comparison.

## 5. Emotions concept steering (`emotions_train`)

Parallel of the 12B L47 kl-lm045 emotions run, with `--n_eval 100 --n_comp 100`:

```bash
uv run python -u -m geoae.interp.steering_concept_compare \
  --checkpoint checkpoints/gemma3-4B/layer33/mse_only_k2000/best_val.pt \
  --dataset emotions_train --layer 33 \
  --correct_json dbpedia/joint_correct_emotions_gemma3_4b_l33_mse_only_ep42.json \
  --out results_steering_emotions_gemma3_4b_l33_mse_only_ep42.json \
  --n_eval 100 --n_comp 100 \
  2>&1 | tee logs/steering_emotions_gemma3_4b_l33.log
```

Its own joint-correct cache — separate dataset, separate model, so it builds fresh
even after the DB14 run in §3.
