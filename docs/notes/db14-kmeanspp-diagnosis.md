---
name: db14-kmeanspp-diagnosis
description: "DBpedia-14 layer-27 clustering — why k-means++ init fails vs semisup, and how few labels semisup needs"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6a1dd344-e616-41b2-a9dd-3464117371a5
  modified: 2026-07-23T22:26:18.095Z
---

DBpedia-14 (sequence-level doc classification) experiment lives in the sibling
repo `../geosep/` but is runnable via the installed `geoae` package (activations
+ package both under GeoAE). Layer 27 = last decoder layer. 50k train / 10k test,
3072-dim float16. Ground truth 14 classes.

**Repro run cmd** (from GeoAE root):
`python -m geoae.e2e.train --config configs/e2e/llama3.2-3B/layer27/unprompted_last_gelu.yaml --no_wandb --teacher_mode onfly`
- `--teacher_mode onfly` computes teacher = head(norm(x)) in-loop; at the LAST
  layer this is mathematically identical to the cached teacher, and avoids the
  (missing) `teacher_logits_layer27.npy` cache. Works for mean pooling too.
- Only `last/` activations exist (`dbpedia/activations/llama3.2-3B/last/unprompted`);
  mean pooling needs a fresh extract: `python -m geoae.dbpedia.extract --mode unprompted --layer 27 --pooling mean`.
- All prior `.pt` checkpoints were deleted; only eval JSONs survive in
  `../geosep/dbpedia/results/llama3.2-3B/layer27/`.

**Key result: k-means++ is NOT inherently broken — pooling decides.**
- mean pooling: kmeans++ (`unprompted_mean_gelu_kl` 92.9%) == semisup
  (`unprompted_mean_semisup_kl` 92.9%). Both balanced clusters.
- last-token: kmeans++ underperforms. Two regimes by assignment:
  - Sinkhorn ON (balanced): REPRODUCED 2026-07-23 = acc 55.44% / purity 60.24%
    (val_kl 0.160, 40 epochs). This IS the user's "~55%". Clusters balanced but
    k-means++ covers only 10/14 classes: MeanOfTransportation x3, OfficeHolder x3
    centroids; STARVED Animal, Company, NaturalPlace, WrittenWork. Sinkhorn can't
    fix semantic mis-allocation — it just splits over-covered classes evenly.
  - Sinkhorn OFF (`--no_sinkhorn`): total collapse to 11.7%, one mega-cluster
    9491/10000 (the old `unprompted_last_gelu_kl_eval.json`).

**Root cause (two independent effects):**
1. k-means++ mis-allocates the 14 centroids: density/distance-driven, so it
   doubles up on spread-out classes and starves tight ones. Raw-space (no AE)
   last-token: semisup init 94.85% vs kmeans++ 73-80% (1-3 classes get 0
   centroids); random-point init 39-57% (this is likely the "~55%" figure).
   Lloyd/k-means iters do NOT repair starvation.
2. Last-token final-layer residual is dominated by one shared "next-token"
   direction (one giant mode). With unbalanced softmax assignment this collapses
   the whole batch onto one centroid; `reinit_dead_clusters` can't escape (draws
   replacements from the same blob). Mean-pooling washes out that direction ->
   survives. See [[db14-eval-gotchas]] for eval assignment details.

**Over-clustering (K>14) as a k-means++ fix (raw last-token proxy, MiniBatch):**
K=14 cov 11/14 merged-acc 0.67; K=20 cov 13/14 merged-acc 0.76; K=28 cov 14/14
merged-acc 0.85. Verdict: helps coverage + merged-accuracy, BUT judge by
over-cluster->merge (name each cluster by dominant train class -> 14), NOT
Hungarian acc (14-way bijection wastes extra clusters, drops to 0.60 at K=28).
semisup requires K==14 (asserts one class-mean/centroid) so K>14 is kmeans++-only.
Tool: `python -m geoae.dbpedia.eval_merge --checkpoint <pt> --pooling last`
(uses dist2.argmin; reports merged-acc/purity/coverage/starved/over-covered).
`--n_clusters K` and `--semisup_cap N` CLI overrides added to the e2e trainer.

**How few labels does semisup need? (last-token raw-space proxy)**
5/class(70)→88%, 10/class→90%, 20/class(280 total)→92.9% (== the 93% figure),
50/class→94%, plateau ~95% at 200/class. So ~20 labeled docs/class suffices.
