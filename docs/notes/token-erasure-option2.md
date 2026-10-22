---
name: token-erasure-option2
description: "Layer sweep of token vs context variance (L27 is NOT a bad layer), and the token-erasure balanced k-means experiment (geoae/interp/token_erasure.py) — status and held-out numbers"
metadata:
  node_type: memory
  type: project
  originSessionId: d189b627-3f1a-44db-90a5-ea0d49b7db35
  modified: 2026-09-25T07:46:08.225Z
---

**Layer sweep (2026-09-25, 60k held-out rows, OOS R² of z-scored residual, classes >=30 fit rows):** past L4 current-token identity explains only 6–11% at every layer; predicted-next grows to 0.11 at L27; same-document context rises monotonically with depth and is HIGHEST at L27 (0.17; L14 0.09). ~70% is local context. So L14 ≈ L27 for token-level clustering is expected; the limit is clustering single token rows (K=2000 carves tight token-role groups), not the layer. Also at L27 cluster NMI with current token 0.62 vs predicted-next 0.54; ~half of mid/large clusters are next-token-defined (e.g. "noun before ' is'").

**Option 2 (token erasure, no retraining):** fixed projection removing top-r eigvecs of noise-corrected between-class scatter (cur token + LM argmax next token). Needs no token ids, so it works on sequence caches (H_last/H_mean have no ids). Basis: e2e/.../layer27/token_erasure_sampled.npz. Held-out: r=64 cuts R² cur .162→.062, pred .152→.051, removes 27% variance, keeps 65% of doc variance (pca64 control keeps 52%). Doc share within the projected space does NOT rise (.151→.136) — token directions carry some doc signal — but doc/token ratio goes ~0.95→2.2.
Command sheet: eval_out/run_token_erasure.sh (km_tok64, km_tok256, km_pca64 control, probe → eval_out/probe_token_erasure.json). Probe RESULT (2026-09-25, chance-corrected NMI = (NMI-shuffled)/(1-shuffled); raw NMI is inflated because erased arms spread sequence rows over ~2.6x more clusters, e.g. atlas_doc null .20->.34):
seq mean km_new .176 -> tok64 .222 (pca64 .207, tok256 .221; AE d6144_new .147). topic14 .469 -> .597 (pca64 .571); topic4 .073 -> .210; RAVEL country .335 -> .481 (tok256 .637); language +.10; ioi_role +.14 (tok256 loses it). Losses: pos_coarse/fine -.14/-.15, surface -.05, NER -.01/-.02. Only ~1/3 of the gain over km_new is token-specific (tok64 - pca64 +.01..+.06); the rest is removing dominant directions. atlas_doc/tone ~ null after correction. Single k-means seed; the token-specific margins (+.01-.03) need a reseed check.

Only concept_probe applies `erase_U`; load_baseline_kmeans (allow_erasure=False default) and clustering_quality refuse erased codebooks. Related: [[sampled-dump-data-ablation]], [[fineweb-atlas-concept-alignment]].
