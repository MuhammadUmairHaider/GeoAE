---
name: init-arms-dpc-results
description: L27 b32k init arms (kmeans++ / seeded_atlas / dpc density-peaks) compared 2026-09-17; z beats base-h steering at α=1 in ALL arms, but arm-vs-arm differences are noise; dpc wins churn + token rungs, loses topic/sequence
metadata:
  type: project
---
Three L27 K=2000 b32k d6144 arms differing only in centroid init (+ reinit rule):
kmeans++ (reinit loss), seeded_atlas (reinit anchor), dpc = density peaks (reinit peaks).

- **DB14 steering: no winner.** Paired over 14 concepts, dpc−kmeans++ Δz_sel at α=1/2 is
  −0.011±0.016 / −0.018±0.017, but the h-space control moves −0.008/−0.009 the same way;
  AE-attributable Δ(z−h) ≈ −0.003±0.014 / −0.009±0.012. dpc−seeded_atlas Δ(z−h) ≈ 0 at every α.
- **But z beats base-h within every arm** (same docs, clean test). At α=1, z−h: kmeans++ −0.032
  (p=0.012), seeded_atlas −0.034 (p=0.032), dpc −0.035 (p=0.046). Target is already erased under
  both (tgt acc ~0.04); the gain is fewer other classes broken (comp acc 0.643 vs 0.615 for dpc).
  z's own-baseline ppl rise is also smaller (dpc α=2: +3.69 vs +4.41). A recipe-level win,
  replicated in 3 AEs, not something the init choice adds.
- **Harness gotcha:** steering_concept_compare rebuilds the joint-correct eval set per AE
  (keyed on ae_sha), so each arm scores different docs and h_sel (which never touches the AE)
  differs by 0.01–0.03 across arms. Treat cross-arm steering gaps < ~0.03 as noise, or pin one
  shared doc set. Mean-diff steering never uses centroids — init reaches it only via the encoder.
- **Where dpc does win:** reinit churn 260 clusters total vs 2,394 kmeans++ / 1,341 seeded_atlas
  (confounded with reinit rule); effective rank 216 vs 162; token-rung probe NMI mean 0.251 vs
  0.218 (POS coarse 0.419 vs 0.367) — matches balanced k-means (0.252), doesn't beat it.
- **Where dpc loses:** sequence-rung NMI 0.242 (worst arm); topic14 NMI 0.446 vs 0.502/0.562; Dunn worst.
- Real-run init diag: 69th density pct, only 10/2000 strict peaks → the latent has tens of modes.
- n=1 seed per arm; dpc steered from best_val (ep47), others from step_0014200 (ep50).
- **tanh arm (dpc_tanh, finished ep50 2026-09-17):** the dpc init did NOT take in the tanh
  latent — init diag read **43rd density percentile, BELOW the pool median** (gelu dpc: 69th),
  25/2000 strict peaks. Churn still low (250 clusters / 43 events vs gelu 260/34, kmeans++ 2394).
  No saturation (|z| mean 0.479, 0% at |z|>0.99) but per-dim std 0.505 sits exactly on the
  var_gamma=0.5 hinge knee. Final val_mse 0.0485 vs gelu 0.0349 at the same epoch.
  Its best_val.pt is EPOCH 5 (pre-clustering, centroids zero) because val_mse rises once
  clustering starts — always evaluate step_0014200.pt for this run, never best_val.

**Why:** user read dpc as "clear winner in steering db14"; paired analysis with h control says otherwise.
**How to apply:** for any future steering arm comparison, compare Δ(z−h) not raw z_sel, and
report the h-control gap. Next runs worth doing: seeded_peaks (atlas anchors for topic + peaks
fill for tokens), and dpc-init+loss-reinit to separate init from reinit. Related: [[zipf-balancing-null-result]], [[fineweb-atlas-concept-alignment]].
