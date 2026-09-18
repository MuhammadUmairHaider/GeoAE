---
name: zipf-balancing-null-result
description: The balance:zipf / balance_rho / zipf_alpha config knobs did not change realized cluster usage on llama L27 — measured alpha 0.528 vs 0.503 for plain uniform Sinkhorn.
metadata: 
  node_type: memory
  type: project
  originSessionId: 870ea3ff-b19b-4c54-a3f1-e8289a510213
  modified: 2026-08-28T17:52:15.646Z
---

Measured 2026-08-26 with `geoae.interp.rank_clusters` over the 491k-token
`closest_tokens` corpus, comparing llama L27 `kl_gelu_k2000_balance_phased`
(`balance: zipf`, `zipf_alpha_end: 0.5`, `balance_rho: 0.6`) against its
predecessor `kl_gelu_k2000_vicreg_10M` (plain uniform `sinkhorn_log`):

| | balance_phased | vicreg_10M |
|---|---|---|
| fitted Zipf alpha | 0.528 | 0.503 |
| usage perplexity | 1488/2000 | 1420/2000 |
| Gini | 0.329 | 0.331 |
| top-100 share | 18.2% | 20.3% |
| dead clusters | 4 | 0 |

The two are within noise, and both sit at the alpha≈0.46 that was measured as
the *natural* usage before any balancing was added. The uniform target was never
actually winning against the data, so replacing it with a power-law target
changed nothing measurable.

What the run *did* change is attributable to VICReg finally being applied after
the `lambda_var`/`lambda_cov` call-site bug fix: latent scale roughly doubled
(`dist_p50` ≈2000 vs ≈1000), bottom-quartile monosemanticity rose 0.216→0.262,
min inter-centroid distance 6.2→9.0, and correctly-measured silhouette
−0.0117→+0.0118 (see [[train-diag-metrics-misleading]]).

**Why:** an entire config generation was built around the balancing knobs; the
evidence says the phased VICReg is carrying the difference, not the balancing.

**How to apply:** don't attribute gains in `*_balance_phased` / `*_zipf_phased`
runs to the balancing. Before building further on it, confirm on a second track
(gemma3-4B L33 `k2000_zipf_phased_mse` vs its `mse_only_k2000` parent — run
`closest_tokens` then `rank_clusters` on both and compare the fitted alpha).
