---
name: fineweb-atlas-concept-alignment
description: "guidelabs/fineweb-atlas gives 16,790 named concepts over FineWeb chunks — first measurement where GeoAE beats raw k-means, and why it is a capacity effect not a structure one."
metadata: 
  node_type: memory
  type: project
  originSessionId: 68b112df-749e-4fbe-9162-ea85a1a8c8aa
  modified: 2026-08-28T20:02:39.275Z
---

`guidelabs/fineweb-atlas` (public, ODC-BY, 46.4 GB, built on HuggingFaceFW/fineweb).
Configs: `concepts` (4 MB, 16,790 concepts: id/type/name/description/LCC path/
prevalence), `documents` (21.7 GB), `chunks` (21.9 GB, ~95-token chunks with
per-chunk multi-labels), `field_guide` (2.3 GB reverse index), plus a concept
co-occurrence matrix. Stream it; do not download.

Concept types: content 12,786 | entity 3,386 | tone 587 | **document 31**.
Per chunk: 5.5 content, 7.0 tone, 1.6 document-type, 0.66 entity labels.
The 31 document-type concepts are genre/register (News report, Recipe, Forum/Q&A,
Product listing…) with millions of chunks each — a far better steering benchmark
than db14/ag_news, and the first DIFFUSE, whole-chunk axis available (every axis
in [[morph-steering-experiment]] is decided at one token position).

## Result (llama L27, 7,742 chunks / 770k tokens, best-single-cluster F1 per concept)

| | effective clusters | doc type | tone | content |
|---|---|---|---|---|
| AE K=2000 (balance_phased) | 1616/2000 (81%) | 0.384 | 0.181 | 0.183 |
| raw k-means K=2000 | 513/2000 (26%) | 0.213 | 0.136 | 0.091 |
| AE K=256 (k256_vicreg) | 245/256 (96%) | 0.210 | 0.113 | 0.041 |
| raw k-means K=256 | 93/256 (36%) | 0.188 | 0.113 | 0.038 |

At K=2000 the AE wins everything (+0.171 / +0.045 / +0.092, p from 8e-06 to 1e-101,
74-93% of concepts). At K=256 the gaps collapse (+0.023 n.s. / -0.001 / +0.003).

**SUPERSEDED 2026-08-30 — it is the BALANCING, not the encoder.** See the
decisive control below; the text that follows is kept for the numbers only.

**The defensible claim is capacity, not structure**: raw k-means cannot use its
clusters at any K (26% at K=2000, 36% at K=256) while the AE can (81%, 96%), so
the AE pulls away as K grows. The AE at K=256 already matches raw at K=2000.
Individual clusters are sharply interpretable — cluster 460 = Technical
specification at 0.98 precision, 713 = Cricket, 521 = Cryptocurrency.

## DECISIVE CONTROL: balanced k-means with NO encoder (2026-08-30)

`geoae/interp/fit_balanced_kmeans.py` applies the AE's Sinkhorn objective to RAW
activations — identity encoder, same K, same normalised space, same k-means++ /
hard-EMA / reinit recipe as training. It decomposes the AE into its two parts:

    AE                   = learned encoder + Sinkhorn-balanced clusters
    fit_balanced_kmeans  = identity        + Sinkhorn-balanced clusters
    fit_baseline_kmeans  = identity        + plain k-means

Llama L27, 7,742 Atlas chunks, capacity matched by construction (1616 vs 1614 live):

| | live | doc type | tone | content |
|---|---|---|---|---|
| AE k2000 | 1616 | 0.384 | 0.181 | 0.183 |
| balanced k-means, NO encoder | 1614 | 0.367 | 0.184 | **0.197** |
| plain k-means (refit) | 555 | 0.185 | 0.140 | 0.097 |

AE minus balanced: doc +0.017 (p=0.093, 48% of concepts), tone -0.003 (p=0.33),
content **-0.015 (p=2e-12, AE wins only 41%)**. **The encoder contributes nothing
and on content actively hurts.** The whole AE-vs-raw gap is the balancing.

Consistent with every other result this session: tense steering ties h vs z
(cos(r_h, dec(r_z)) = 0.979), all five morphology axes tie at FVE ~0.998 with
latent_dim == hidden_size, centroid steering underperforms linear. One fact seen
four ways — the encoder is near an invertible reparameterisation.

**What is still a real finding:** balanced assignment transforms clustering of
residual streams (555 -> 1614 live, 87% -> 6% top-10 concentration, +0.18/+0.04/
+0.10 F1). Cheap, no training, reusable.

**Untested variable:** every checkpoint in this project has
latent_dim == hidden_size and no sparsity term. A bottlenecked encoder is the one
thing that could still make the AE earn its cost — fit one, run this probe,
compare against balanced k-means at the same K.

## Plain k-means is IMBALANCED, not merely collapsed

Ten clusters hold 87% of tokens on the fitter's own training distribution and 96%
on Atlas; the AE and balanced k-means hold 5.5-6.8%. Raising K does not fix it
(live fraction FALLS with K: 36% at K=256, 28% at K=2000). "Collapsed to 555"
understates the problem — the fit log's "1801/2000 centroids win" counts clusters
winning >=1 token out of 1M, and those 1801 split the leftover 13% of mass.

## Methodological traps

- **No post-hoc capacity control is neutral.** Clipping to the top-N most-used
  clusters deletes the tail (78% of content concepts lost their best cluster, and
  losses concentrate exactly there); merging centroids destroys learned
  distinctions. They damage in OPPOSITE directions, so use both — where they
  agree (doc type wins; content ~0) the conclusion is safe.
- **K=256 is floor-limited**, not a clean null: content F1 is 0.041 vs 0.038, both
  at the noise floor, so "no difference" there cannot distinguish "no intrinsic
  advantage" from "both too starved to show one".
- Tone taxonomy is redundant (`Casual`/`casual` separate ids; matter-of-fact /
  Informational / factual co-fire at 0.5-0.84 prevalence) — dedupe before using it
  as a contrast. Document labels are MULTI-label (1.6/chunk), so
  `steering_concept_compare`'s single-label joint-correct machinery needs
  adapting.
- Raw k-means baselines: llama L27 k2000 (Jul, and an Aug-29 refit — they agree,
  513 vs 555 live, so the old one was NOT under-fit), llama L27 k256, gemma3-12B
  L47, gemma3-4B L33 (Aug 29). Balanced: llama L27 k2000.
- **fp16 overflow destroyed the gemma cache.** gemma3-4B L33 residuals have
  mean |h| ~52,000 against fp16's 65,504 ceiling — 70% of cached tokens held a
  non-finite value, which makes cdist return inf and argmin return 0, faking a
  "77/2000 cluster collapse". Cache gemma activations as float32 and ASSERT
  finiteness. **AUDITED 2026-09-01: the source dumps are CLEAN.**
  `activations_gemma3_4b/layer_22.npy` and `layer_33.npy` are dtype float32
  with 0.00% non-finite over a 200k-row sample (mean row-norm 41,028 / 81,539;
  max |x| 207,872 — far above fp16's 65,504, which is exactly why any fp16
  *re-cache* of them corrupts, while the float32 originals are fine). The
  corruption was confined to the downstream fp16 atlas cache, not the training
  data. All llama results are unaffected (|h| ~59, max value 179, zero
  non-finite).

## Cached artifacts (scratchpad, regenerate with atlas_cache.py if lost)

`atlas8k.npz` (llama L27 residuals, 770k x 3072 fp16, 4.7 GB) + `atlas8k_labels.parquet`,
and the gemma L33 equivalents. Streaming the first 8000 chunks and filtering
`chunk_status == "ok"` is deterministic and yields the SAME 7,742 chunks, so the
llama and gemma passes are directly comparable.

Related: [[train-diag-metrics-misleading]], [[zipf-balancing-null-result]].
