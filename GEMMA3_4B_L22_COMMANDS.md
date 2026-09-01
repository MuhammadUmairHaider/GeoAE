# Gemma 3 4B / layer 22 — MSE-only run: concept steering (then clustering)

Run finished 2026-08-25. `checkpoints/gemma3-4B/layer22/mse_only_k2000/best_val.pt`
= **epoch 39, step 88,842, val_mse 0.001421**, tau 0.325, ae_sha `8e43e089cefb`.
Final W&B diag (epoch 50): silhouette −0.0221, effective_k 2000/2000, dying 45,
min/mean/max centroid dist 8.43 / 41.66 / 101.20, balance entropy 0.032.

Paired with `checkpoints/gemma3-4B/layer33/mse_only_k2000/best_val.pt`
(epoch 42, val_mse 0.002425, ae_sha `6846b0226cad`) — same model, same recipe,
65% vs 97% depth. Rows 0..7,860,950 of the two activation dumps are the same
tokens, so the two runs are paired over that prefix.

**FVE 0.999 at `latent_dim == hidden_size` with no sparsity term is not a win** —
same near-identity caveat as 12B/L47 and 4B/L33. Steering is the deciding evidence.

GPU is free (A100-40GB, 0 MiB used). Disk 84 GB free.

---

## 0. Status of the L33 comparison — it needs a rerun

`logs/steering_db14_gemma3_4b_l33_mse_only_ep42.log` **died at concept 4 of 14**
(Aug 23 23:27, right before the L22 training started) and wrote no results json.
There is no `results_steering_db14_gemma3_4b_l33_mse_only_ep42.json`.

Its joint-correct cache IS still valid (`ae_sha 6846b0226cad` matches the live
checkpoint), so the rerun skips the ~15 min build and goes straight to steering.

What the partial log already showed, on the 5 concepts it reached: `h sel` and
`z sel` track each other almost exactly (e.g. −0.838/−0.838, −0.675/−0.662,
−0.325/−0.338, −0.188/−0.188, −0.037/−0.037). That is the near-identity signature.
The L22 run is the test of whether a mid-depth layer breaks that tie.

## 1. DB14 concept steering — L22 (the new run)

```bash
uv run python -u -m geoae.interp.steering_concept_compare \
  --checkpoint checkpoints/gemma3-4B/layer22/mse_only_k2000/best_val.pt \
  --dataset db14 --layer 22 \
  --correct_json dbpedia/joint_correct_db14_gemma3_4b_l22_mse_only_ep39.json \
  --out results_steering_db14_gemma3_4b_l22_mse_only_ep39.json \
  2>&1 | tee logs/steering_db14_gemma3_4b_l22_mse_only_ep39.log
```

Builds its own joint-correct set on the first run (new ae_sha), then caches it.
If the LM OOMs during that build, lower `--jc_batch_size` (default 16).

Smoke first if you want (~2 concepts, small n, ~5 min):

```bash
uv run python -u -m geoae.interp.steering_concept_compare \
  --checkpoint checkpoints/gemma3-4B/layer22/mse_only_k2000/best_val.pt \
  --dataset db14 --layer 22 --smoke \
  --correct_json dbpedia/joint_correct_db14_gemma3_4b_l22_SMOKE.json \
  --out results_steering_db14_gemma3_4b_l22_SMOKE.json
```

## 2. DB14 concept steering — L33 rerun (completes the pair)

```bash
uv run python -u -m geoae.interp.steering_concept_compare \
  --checkpoint checkpoints/gemma3-4B/layer33/mse_only_k2000/best_val.pt \
  --dataset db14 --layer 33 \
  --correct_json dbpedia/joint_correct_db14_gemma3_4b_l33_mse_only_ep42.json \
  --out results_steering_db14_gemma3_4b_l33_mse_only_ep42.json \
  2>&1 | tee logs/steering_db14_gemma3_4b_l33_mse_only_ep42_rerun.log
```

## 3. Emotions steering — L22 (L33 already done)

`results_steering_emotions_gemma3_4b_l33_mse_only_ep42.json` exists, so only L22
is missing. Same `--n_eval 100 --n_comp 100` as the L33 run:

```bash
uv run python -u -m geoae.interp.steering_concept_compare \
  --checkpoint checkpoints/gemma3-4B/layer22/mse_only_k2000/best_val.pt \
  --dataset emotions_train --layer 22 \
  --correct_json dbpedia/joint_correct_emotions_gemma3_4b_l22_mse_only_ep39.json \
  --out results_steering_emotions_gemma3_4b_l22_mse_only_ep39.json \
  --n_eval 100 --n_comp 100 \
  2>&1 | tee logs/steering_emotions_gemma3_4b_l22.log
```

Defaults for §1/§2, matching every prior run: `--n_fit 80 --n_eval 50 --n_comp 80
--alphas 0.5 1.0 2.0 4.0 8.0 --seed 42` → 1120 fit docs, 700 eval docs, 14 concepts.

---

## Later — clustering quality (after steering)

No `baseline_kmeans_k2000.npz` exists under `checkpoints/gemma3-4B/layer22/`.
Fit it first, with the corrected full-budget settings:

```bash
uv run python -u -m geoae.interp.fit_baseline_kmeans \
  --checkpoint checkpoints/gemma3-4B/layer22/mse_only_k2000/best_val.pt \
  --activations activations_gemma3_4b/layer_22.npy \
  --n_clusters 2000 --n_sample 1500000 --max_no_improvement 0 --seed 42 \
  --out checkpoints/gemma3-4B/layer22/baseline_kmeans_k2000.npz \
  2>&1 | tee logs/fit_baseline_kmeans_gemma3_4b_l22.log

uv run python -u -m geoae.interp.clustering_quality \
  --baseline checkpoints/gemma3-4B/layer22/baseline_kmeans_k2000.npz \
  --checkpoints checkpoints/gemma3-4B/layer22/mse_only_k2000/best_val.pt \
  --names raw-kmeans mse-only \
  --activations_dir activations_gemma3_4b --layer 22 \
  --n_sample 500000 --seed 0 \
  --out clustering_quality_gemma3_4b_l22_mse_only.json \
  2>&1 | tee logs/clustering_quality_gemma3_4b_l22_mse_only.log
```

## Caveat to carry into the writeup

`h sel` is not comparable across runs — the joint-correct doc set is filtered by
each AE's own reconstruction, so L22 and L33 evaluate on slightly different doc
subsets. Within a run, h-vs-z is a clean paired comparison; across runs, treat the
h column as a per-run reference point, not a fixed constant.
