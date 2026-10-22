---
name: cq-dunn-and-steer-sign-traps
description: two metric traps found 2026-09-20 — clustering_quality's dunn (and silhouette) subsample with an UNSEEDED RNG so dunn is unusable, and steering_concept_compare's selectivity sign is the negation of range_intervention_compare's
metadata:
  type: reference
---
**1. dunn_index is not usable; never compare cq numbers across runs.**
`geoae/interp/clustering_quality.py` approximates `silhouette` on 10k and `dunn_index` on 5k
rows via a bare `np.random.choice` — NOT seeded per metric, so the draw depends on how many
models were encoded earlier in the same process. Measured on the SAME d3072 checkpoint, same
1M rows, same `--seed 0`: dunn 0.1343 in the 2-arm run vs 0.0711 in the 3-arm run (-47%).
Silhouette is stable to ~0.0006 (10k, and it averages over points). Every other metric
(CH, DB, separability, centroid dists, intra-var, effective rank/K) is exactly deterministic.
Root cause was worse than the missing seed: at K=2000 a 5k subsample leaves ~2.5 points per
cluster, so most clusters got diameter 0 and max_diam came from whichever cluster drew several
points. **FIXED 2026-09-21** in `geoae/interp/clustering_quality.py`: `silhouette(..., seed=)`
seeds both draws (its own subsample AND sklearn's `sample_size` via `random_state`), and
`dunn_index` now runs on ALL sampled rows in one argsort/split pass (no `max_samples` arg any
more) reusing `inter_centroid_dist`. Verified on real data: two arms run in both orders give
byte-identical metrics. Dunn still scales with `--n_sample` (diameter estimates grow with points
per cluster), so compare only within one n_sample.
**How to apply:** results produced BEFORE 2026-09-21 still carry the broken dunn — the cq JSONs
behind [[d12288-width-arm]] do; re-run cq to refresh. Still put all arms in ONE invocation.
`eval_out/summarize_width_series.py` pins every arm to the newest all-arm file (`PREFERRED`).

**2. "selectivity" means opposite things in the two intervention tools.**
`range_intervention_compare`: selectivity = tgt_drop - comp_drop, drops positive, HIGHER better.
`steering_concept_compare`: selectivity = tgt_acc_delta - comp_acc_delta, both NEGATIVE, so
MORE NEGATIVE is the cleaner steer. A z-h of -0.0723 in a steer_db14_*.json is an AE WIN.
Mislabeling this flips the headline of any whole-concept steering comparison.

Related: [[d12288-width-arm]], [[train-diag-metrics-misleading]], [[range-interventions-h-vs-z]].
