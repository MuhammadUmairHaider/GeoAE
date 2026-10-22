---
name: gemma3-l47-mse-vs-kl
description: "Gemma 3 12B layer 47 K=2000 — MSE-only vs KL-lambda sweep vs raw k-means clustering-quality result, and the near-identity caveat"
metadata:
  type: project
---

Ran 2026-08-21, 500k-token probe, seed 0:
`clustering_quality_gemma3_l47_kl_vs_mse.json` / `logs/clustering_quality_gemma3_l47_kl_vs_mse.log`.

**MSE-only wins every scale-normalised metric among the AEs**: silhouette -0.0372
(best overall, beats raw's -0.1275), min centroid dist 10.21 (1.6x the best KL),
effective K 1995/2000 with 0 empty, effective rank 432 vs KL's 59/130/164, DB 4.67.
It did this on 4.0M train rows vs the KL control's 9.33M (`--max_train_rows 4000000`
capped it) — so the gap is if anything understated.

**The near-identity caveat is unresolved.** val_mse 0.00596 => FVE 0.994, and
latent_dim == hidden_size with no sparsity term anywhere in geoae.losses. Effective
rank 432 sits between the KL runs (59-164) and raw (1131) — not identity, but moving
that way. High FVE + high rank is NOT success here; the deciding evidence is whether
clusters carry semantics (closest_tokens mono, DB14 steering), not reconstruction.

**Raw's collapse is OCCUPANCY, not geometry** — an important distinction found by
masking dead centroids (`clustering_quality_gemma3_l47_livecentroids.json`, the
corrected numbers; the earlier `..._kl_vs_mse.json` has three contaminated rows).
Centroid-matrix metrics (inter/min centroid dist, effective rank) used to include
the 684 dead centroids. Masked to live ones, raw's min centroid dist goes
**0.0000 -> 9.2316** (2nd best, behind only mse-only's 10.21) and effective rank
**1131 -> 806**. So raw's 1316 live centroids are well-SPREAD; the failure is that
token mass concentrates on ~4 of them (balance H 0.188, normalised by survivors;
effective K 270). Well-placed-but-unoccupied centroids is the signature of a
k-means++ init that never converged — see the under-fit evidence below.
Raw's remaining "wins" on Davies-Bouldin (2.06) and separability ratio (125.9) are
K-mismatch effects (270 clusters vs ~1990) — coarser partitions score better on
both — not evidence of better geometry.

**The raw baseline is probably UNDER-FIT.** `logs/fit_baseline_kmeans_gemma3_l47.log`:
converged at step 15/1171 while still printing "Reassigning 1998 cluster centers",
and sampled only 500k tokens for K=2000 (the script's own default is 1.5M). The fit
log also claims "1994/2000 centroids win" which contradicts the 684 empty measured
at eval. `--max_no_improvement` is now exposed on fit_baseline_kmeans (0 = None =
run the full budget); refit before trusting any raw-vs-AE claim.

**KL lambda_mse trend** 0.05 -> 0.15 -> 0.45: Dunn 0.082 -> 0.121 -> 0.140 and
effective rank 59 -> 130 -> 164 both improve monotonically; separability ratio falls
54.7-62.5. More MSE pressure = better-spread, higher-rank latent.

Naming trap: the checkpoint dir `kl_k2000_vicreg_10M_mse030` actually carries
`lambda_mse: 0.45` — hence the `mse045` filenames in dbpedia/ and logs/.

See [[gemma3-12b-setup]], [[db14-eval-gotchas]].
