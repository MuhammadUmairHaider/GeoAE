---
name: b32k-transient-collapse
description: "Large-batch (B=32768) GeoAE runs pass through a transient hard-assignment collapse in epochs ~15-30 that self-resolves; the training log cannot see it."
metadata:
  type: project
---

Measured 2026-09-02 on llama L27 `k2000_bnh_b32k_lam1_d6144` (B=32768, K=2000,
d=6144, BN, hinge sep, lambda_c=1.0), 50 epochs, 10.5 min/epoch.

**The run oscillates between healthy and totally collapsed during epochs 15-30,
then stabilises permanently.** Nearest-centroid usage from `ema_cluster_size`:

| epoch | 15 | 19 | 23 | 27 | 30 | 31-40 | 41-50 |
|---|---|---|---|---|---|---|---|
| usage perplexity | 28 | 11 | 59 | 241 | 5 | 898-1297 | 1251-1296 |
| top-1 cluster share | 63% | 75% | 54% | 33% | 83% | 0.9-10.8% | 0.82-0.96% |

At epoch 30 ONE cluster held 27,089 of 32,768 points. By epoch 41+ it is flat at
perplexity ~1290, top-1 0.85%, across every reinit phase.

**During epochs 11-30 the collapse tracked the reinit cycle exactly**: collapsed
checkpoints sat 10-43 steps after a reinit, healthy ones 67-124 steps after
(`step % reinit_every`, reinit_every=125). That correlation does NOT survive into
epochs 31+, so reinit was the trigger while the centroid field was unstable, not a
standing property. What actually tracks recovery is reinit churn drying up:
43/cycle at ep21 -> 25 at ep30 -> 16 at ep35 -> 10 at ep40. tau annealing
1.0 -> 0.10 over the same span sharpens assignments.

**CORRECTED 2026-09-02 by the like-for-like measurement.** The EMA figures
(perplexity 1296, top-100 23.4%) flattered the run. `rank_clusters` over the same
491k corpus with nearest-centroid assignment — the SAME method as the baselines —
gives:

| | B=4096 balance_phased | B=4096 vicreg_10M | B=32768 |
|---|---|---|---|
| fitted Zipf alpha | 0.528 | 0.503 | **0.972** |
| usage perplexity | 1488 | 1420 | **1175** |
| top-100 share | 18.2% | 20.3% | **26.3%** |
| dead clusters | 4 | 0 | **12** |

All four degrade. B=32768 DOES cost usage spread; it does not "cost nothing".
Note the config targets zipf_alpha_end 0.5 and realises 0.972 — usage finally
moved (earlier runs were a null at ~0.5) but in the wrong direction. Zipf R^2 is
only 0.76, so alpha is a rough summary. Scale: top-10 still holds just 6.11% and
1988/2000 are live, vs 87-96% top-10 for plain k-means — a regression against
B=4096, not a collapse, and it bought the best geometry recorded (silhouette
+0.0303 vs +0.0118, Dunn 0.1178, separability 194.0). val_mse 0.03215, FVE 0.969.

**Never quote ema_cluster_size against rank_clusters baselines** — different
estimators, and the EMA reads ~10% more balanced.

**How to apply:** do not judge a large-batch run before ~epoch 40, and do not
"fix" reinit_every on the strength of a mid-training collapse — this one resolved
on its own. Sample MANY checkpoints before concluding: a single epoch-22 reading
gave perplexity 1347 and looked entirely healthy while the run was in fact
oscillating.

Related: [[train-diag-metrics-misleading]] (the `dying`/`eff_K` fields in the
training log come from Sinkhorn Q and stayed flat at 787-1326 and 1874-1976
throughout, showing none of this).
