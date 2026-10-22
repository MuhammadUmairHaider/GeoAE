---
name: report-base-vs-bypass-only
description: "When reporting GeoAE results, compare only the base (plain residual / its balanced k-means) against the bypass AE; drop the other arms unless asked"
metadata:
  node_type: memory
  type: feedback
  originSessionId: b1c5c1c3-1bd1-42d3-9c11-49132df71932
  modified: 2026-09-26T04:11:50.904Z
---

Report results as a two-way comparison: **base** (plain residual; for clustering/steering tests its balanced k-means, same K/init, no encoder) vs **bypass AE** (the token-bypass GeoAE). Do not add the parent AE, km_tokmean, km_tok64, pca controls etc. to the tables unless the user asks.

**Why:** user said (2026-09-26) "for clarity now on just compare base and bypass ae" after multi-arm tables with percentage-point deltas vs the matched control confused them (they read +0.02–0.06 as "just 2–6% better than base").

**How to apply:** two columns (base, bypass AE) plus a ratio/difference column; state absolute numbers in plain units (e.g. "rows out of 100"), and say which data the test runs on. Controls may still be RUN for rigour and mentioned in one line if they change the conclusion. Related: [[token-bypass-design-b]], [[cluster-steering-eval]].
