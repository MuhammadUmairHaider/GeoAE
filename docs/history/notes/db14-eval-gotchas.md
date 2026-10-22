---
name: db14-eval-gotchas
description: "DBpedia-14 eval mechanics — assignment method, Sinkhorn, and cluster-collapse signature"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 6a1dd344-e616-41b2-a9dd-3464117371a5
  modified: 2026-07-23T20:36:55.952Z
---

DBpedia-14 eval (`geosep/dbpedia/evaluate_dbpedia.py` and `geoae/dbpedia/evaluate.py`):
- Assigns test points via `out.Q.argmax(dim=1)`. If Sinkhorn is ON at eval, Q is
  per-batch balanced, so a single mega-cluster is impossible — the observed
  9491/10000 collapse in `unprompted_last_gelu_kl` means that run effectively used
  unbalanced softmax (Sinkhorn off / `--no_sinkhorn`). Per [[db14-kmeanspp-diagnosis]].
- Purity/Accuracy(Hungarian)/NMI/ARI reported; for K=14==n_classes the Hungarian
  bijection makes accuracy==purity when clusters map 1:1 to classes.
- Collapse signature to watch during training: cluster-usage entropy crashing /
  one centroid's EMA size dominating -> retrain will reproduce ~11% if unbalanced.
