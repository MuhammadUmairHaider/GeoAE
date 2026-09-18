---
name: tanh-encoder-arm
description: dpc_tanh (L27 K=2000, tanh encoder + var_gamma 0.5) finished ep50 2026-09-17 — best global steering of any arm, but range-restricted interventions collapse in tanh z and effective rank halves
metadata:
  type: project
---
Arm: `..._dpc_tanh` = dpc gelu parent with nonlinearity tanh + loss.var_gamma 0.5.
Evaluated at step_0014200.pt (ep50) against the gelu dpc run at the SAME epoch.
**Its best_val.pt is epoch 5** (pre-clustering, centroids zero) — never evaluate it.

- **Global mean-diff steering: the best arm so far.** z−h at α=1 = −0.075±0.021, p=0.003,
  13/14 concepts (gelu −0.035 p=.046; seeded_atlas −0.034 p=.032). Same +0.075 appears in the
  range harness's st_global row, so both harnesses agree.
- **Range-restricted interventions look reversed at fixed alpha, but NOT at matched erasure.**
  At alpha=1 z−h is −0.21 in tanh vs +0.15 in gelu, because the tanh edit is far weaker
  (z tgt drop 0.26 vs 0.72). Interpolating each curve to a fixed 60% target erasure, z still
  beats h in tanh: collateral 0.10 (z-range) / 0.11 (z-transport) vs 0.14–0.15 for h, at
  alpha ~1.9 instead of ~0.8. gelu z reaches the same point at 0.09 collateral, alpha ~0.8.
  So alpha is NOT comparable across two different latents even though r = mu_c − mu_rest is in
  each space's own class-gap units — the decoder maps the same latent step to different
  h-space magnitudes. Always quote matched-erasure collateral, never fixed-alpha selectivity.
  Range geometry: mean |mu_c − mu_rest| on salient dims 0.114 (tanh) vs 0.335 (gelu); as a
  fraction of the 4-sd range width, 0.125 vs 0.146. CORRECTED: the per-concept matched-erasure
  test shows NO z-vs-h difference for range operators in EITHER arm — the summary-mean
  interpolation that suggested tanh z 0.10 vs h 0.14 does not survive per-concept pairing.
  tanh z range/transport cannot even reach 80% erasure at alpha<=2, so it has no operating
  point there; tanh's best is base h-transport (0.156 collateral at 80% erasure).
- Zero replacement is much less destructive in tanh (signed, centred code): rm_range_zero z
  keeps comp 0.36 (drop 0.64) vs gelu's 0.12 — but h still beats z on removal in both arms.
- Geometry: kNN-10 label agreement tanh > gelu on 5/5 rungs (+0.015, p=0.009). Probe token NMI
  +0.006 (p=.50, ns), sequence NMI −0.025 (p=.10) — same token-win/sequence-loss shape as dpc.
- Clustering: silhouette 0.014 vs gelu 0.036 (worse), Dunn 0.189 vs 0.151 (better),
  **effective rank 125 vs 217** (much lower — centroids span half the directions).
  Scale-dependent metrics (inter-centroid dist, intra var) are NOT comparable: tanh is bounded.
- Costs: val_mse 0.0485 vs 0.0349; MMLU recon delta −0.033 vs −0.027 (n=2000, within noise).
- No saturation (|z| mean 0.479, 0% >0.99) but per-dim std 0.505 sits exactly on the
  var_gamma=0.5 hinge knee. Churn 250 clusters/43 cycles ≈ gelu 260/34.
- The dpc init did NOT take here: 43rd density percentile (BELOW median) vs 69th for gelu.
  So the best global-steering arm is the one whose density init failed — init quality and
  steering quality are decoupled.

**Why:** tanh was tried to fix the shifted/unbounded GELU code (see [[probe-tpp-gotchas]]).
**How to apply:** quote global steering for the tanh win, and never compare its raw
inter-centroid/intra-var numbers with a GELU arm. Related: [[init-arms-dpc-results]],
[[range-interventions-h-vs-z]].
