---
name: train-diag-metrics-misleading
description: "W&B cluster/silhouette, entropy, dominant and dying from geoae/diagnostics.py cannot be read as geometry or collapse signals — do not judge a run by them."
metadata: 
  node_type: memory
  type: project
  originSessionId: 870ea3ff-b19b-4c54-a3f1-e8289a510213
  modified: 2026-08-28T17:52:03.238Z
---

The per-step diagnostics logged during training measure the Sinkhorn output on a
single batch, not the model's geometry. Verified 2026-08-26 on the llama L27
`kl_gelu_k2000_balance_phased` run (final logged silhouette −0.00187).

**`cluster/silhouette` is meaningless.** Two independent artifacts in
`geoae/diagnostics.py:76-105` (`silhouette_sample`):
1. labels are `Q.argmax` — Sinkhorn-balanced, so points are deliberately moved
   off their nearest centroid. Same trap as [[db14-eval-gotchas]].
2. it runs on one batch: B=2048 against K=2000, where ~44% of populated
   clusters are singletons. sklearn scores singletons 0, pinning the result
   near zero. The `n_labels > len(labels)-1` guard does not fire, so it returns
   a plausible-looking number instead of the `-1.0` sentinel.

Measured properly (nearest-centroid `argmin` labels, 60k tokens from
`activations_diverse_10M`), the same checkpoint scores **+0.0118**, and
`kl_gelu_k2000_vicreg_10M` scores **−0.0117** — a sign flip and a reversed
ranking. Silhouette is strongly sample-size dependent here: +0.033 at N=2048,
+0.018 at N=10k, +0.012 at N=60k. Always state N when quoting it.

**`cluster/entropy` is not a collapse signal.** It is `sinkhorn_entropy` = mean
*row* entropy of Q in nats, ceiling ln(2000)=7.60. A value of 0.0006 means
assignments are essentially one-hot, which is what tau annealing to ~0.35 does
when squared distances run ~2000. The real consequence: past that point the
cluster loss gradient reaches only the winning centroid, so it is hard k-means
with Sinkhorn reassignment, not soft clustering.

**`dominant` / `dying` / `effective_k`** all come from `cluster_usage`, which
averages Q *columns* over the last 200 batches — the balancing target, not
natural nearest-centroid usage. They cannot tell you whether a balancing target
was reached; only `closest_tokens` can. See [[zipf-balancing-null-result]].

**Why:** two separate analyses in this project were nearly derailed by reading
these numbers as geometry — the logged silhouette says a run failed when a
correct measurement says it beat its predecessor.

**How to apply:** judge cluster geometry with `geoae.interp.clustering_quality`
or an explicit `argmin`-label silhouette over ≥50k tokens. Treat the W&B cluster
panel as training-progress telemetry only.

**`dying` vs `[reinit]` are DIFFERENT sources — do not conflate them.**
Verified 2026-09-02 by reading the code:
- `cluster/dying` = `(u < 0.5/K).sum()` at `geoae/diagnostics.py:144`, where
  `u = cluster_usage(Q_history)` — Sinkhorn **Q column means**, the balancing
  target. Not a collapse signal.
- `[reinit] ... dead clusters` = `reinit_dead_clusters` at
  `geoae/train_common.py:314`, using `u = ema_cluster_size / ema_cluster_size.sum()`
  from **hard assignments**, thresholded at `0.1 * model.target_usage()`. This one
  is real.

**Cheap honest check, no GPU, from any checkpoint:** `ema_cluster_size` lives in
`ckpt["model_state"]["ema_cluster_size"]`. Load it, normalise, and report count at
`<0.1x target`, count of exactly-zero, usage perplexity `exp(-sum u log u)`, and
top-10 share. On the b32k lam1_d6144 run at epoch 22 this gave 0 exactly-dead,
37 (1.9%) below threshold, perplexity 1347/2000, top-10 share 4.73% — while the
training log simultaneously showed `dying 1003`. Threshold against
`model.target_usage()` not uniform `1/K` when `balance: zipf` is annealing, or a
large fraction will look under-used by construction.

Reference points: plain k-means on these residuals puts 87-96% of tokens in its
top 10 clusters; B=4096 runs measured 1488/1420 usage perplexity and 4/0 dead.
