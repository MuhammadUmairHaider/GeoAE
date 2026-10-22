---
name: probe-tpp-gotchas
description: TPP/bias-probe metrics that are artifacts — GELU latent scale inflates localization, L1=0.1 cripples probes, selectivity is degenerate
metadata:
  type: project
---

Three measurement artifacts found 2026-08-26 while analysing
`geoae.interp.probe_perturbation` and `geoae.bias.probe` on llama3.2-3B L27
(`kl_gelu_k2000_balance_phased` ep39).

**1. The TPP "localization win" for GeoAE is entirely a measurement artifact — under a
fair protocol there is NO difference vs the base residual stream.** Three nested confounds,
each one shrinking the effect (DBpedia-14, L27, mean k@50%):
  - as-reported (zero-ablation, L1=0.1):      geoae  93 vs raw 336  -> 3.6x "win"
  - after z-scoring latents (L1=0.1, 5 seeds): geoae 309 vs raw 349  -> 1.13x
  - **mean-fill ablation + L1=1e-4 (3 seeds): geoae 693+-96 vs base 677+-137 -> nothing.**
    All four variants (base/geoae x unnorm/zscored) land at 677-753 with seed sigma ~120,
    and probe acc is 0.979-0.980 in every one.

Mechanism: zeroing a feature is neutral only if 0 is its mean. GELU latents have median
|mu|/sigma = 1.18 (73% of dims have |mu|>sigma) and 0.0% exact zeros — a dense *shifted*
code, not a sparse one. Zeroing top-100 dims injects a logit drift of 0.249 against a
top1-top2 margin of 0.540. Base residuals are also not centred (median |mu|/sigma = 0.64,
30% of dims), so zero-ablation is mildly unfair to them too — `transform_raw` z-scoring
happens to mask it.

**Zeroing probe WEIGHTS is not a fix: for a linear probe it is bit-identical to zeroing
activations** (w.0 == 0.x; verified max|logit diff| = 0.000e+00, 100% prediction
agreement). **Mean-fill == zero-weights + adding w.mu back into the bias** (verified to
1e-6) — i.e. the drift is exactly the bias-correction term that zero-ablation drops.
Symmetric treatment (normalising neither space) also fails: geoae still shows 83 vs 300,
because the offset asymmetry is structural, not procedural.

Fix: add `--ablation_value {zero,mean}` defaulting to mean, and use L1<=1e-4.
Mean-fill + low L1 is ~reparameterisation-invariant (unnorm vs zscored agree within noise).

**2. `lambda_l1=0.1` (the default) badly underfits every probe.** The penalty is an
unnormalised `W.abs().sum()` over 3072xC weights, so it dwarfs CE; val accuracy
peaks at epoch 1 and early stopping fires at epoch 11 (patience 10) every time.
BiasBios profession: 0.574 at l1=0.1 vs **0.832** at l1=1e-4 or 0. Gender is
robust (0.987 vs 0.992). Use l1<=1e-4 for accuracy claims; the l1=0.1 numbers are
comparable across spaces but are not the representation's real linear decodability.

**3. `mean_selectivity` in TPP is degenerate — never cite it.** At full ablation all
logits equal the probe bias, so argmax collapses to one fixed class c*. Every
concept != c* scores +1/n_comp (~0.08); c* itself scores exactly -1.0. The
per-space mean is just that sentinel dragging 13 near-identical values.

Related: [[db14-eval-gotchas]], [[gemma3-l47-mse-vs-kl]].
