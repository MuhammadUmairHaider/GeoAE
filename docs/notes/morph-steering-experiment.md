---
name: morph-steering-experiment
description: "Tense steering (performed->perform) experiment — design decisions, why the AE clusters mix inflections, and the alpha-unit / flip-vs-hit traps."
metadata: 
  node_type: memory
  type: project
  originSessionId: 870ea3ff-b19b-4c54-a3f1-e8289a510213
  modified: 2026-08-28T17:52:34.371Z
---

Built 2026-08-28: `geoae/interp/steering_morph_compare.py`. Steers "-ed" forms
toward their base forms and compares the raw residual stream (h) against the AE
latent (z) on identical positions, with a `zrecon` (encode/decode, no steer) arm.

**Motivating evidence.** Every space has 11-14 clusters that are >=50% "-ed"
tokens, but they organize by verb *semantics*, not morphology (`published
founded formed established`; `led directed followed inspired`; `defined shown
calculated represented`). Several mix inflections of one lemma — `served
serving serves serve`, `derived derives`, `located found situated runs rises`.
So the AE groups by lemma and collapses across tense.

**Early result contradicts the obvious hypothesis.** Predicted z would
underperform. Instead `cos(r_h, decode(r_z))` = +0.965 and the two arms track
each other closely: the latent carries the tense axis linearly even though it
does not cluster by it. One asymmetry: relative to its own scale the tense gap
occupies less of z (gap 25.25 / mean norm 81.4 = 0.31) than of h (20.75 / 49.7
= 0.42). Smoke-sized only (36 verbs, 75 positions) — not a result yet.

**Axis choice is probably NOT the binding constraint.** Measuring whether the two
forms of a word land in different clusters: -ly 100%/100% (llama/gemma), -ing
74%/68%, plural 71%/71%, past 71%/78%, case 62%/59%, leading-space 53%/50%. Past
tense was ALREADY as cluster-organising as anything except -ly and still tied, so
"pick a feature the AE clusters on" does not predict steerability. The structural
reason is latent_dim == hidden_size at FVE ~0.998: the AE is near an invertible
reparameterisation, so every LINEAR edit ties by construction. Test a bottlenecked
checkpoint (e.g. kl_gelu_k256_vicreg at L27) before adding more axes.

**Centroid steering: "toward" is degenerate, use "delta".** z + a*(C[k*] - z)
lands on the cluster MEAN and so discards which word the token is — the two-token
readout then cannot succeed (plural smoke: hit 0.011 vs linear-z 0.363, wiki ppl
14.3). z + a*(C[k*] - C[k_src]) applies the centroid offset while preserving the
token's offset from its own cluster: 3x better hit at half the damage, but still
far below linear. Also note the target pool is thin at small n (only 4 singular- /
2 plural-enriched clusters at smoke scale), so judge it only on a full run.

**Traps found and fixed, worth not re-learning:**
- *Alpha unit.* Scaling the step by the mean ACTIVATION norm overshoots wildly —
  alpha=0.5 destroyed the model (wiki ppl 4432, top-1 changed at 60% of
  positions). Correct unit is the MEAN-DIFFERENCE norm |mean_past − mean_base|,
  so alpha=1 moves a point by exactly the gap, and h/z become comparable.
  CORRECTED 2026-09-02: `steering_concept_compare` does NOT have this flaw.
  Its `r_h[c] = mean(h|y=c) - mean(h|y!=c)` (steering_concept_compare.py:225) is
  the mean DIFFERENCE, not a unit vector, and make_h_steer/make_z_steer apply
  `x - alpha*r` directly — so alpha is already in class-gap units in both spaces
  and the h/z columns ARE comparable. Do not "fix" it.
- *flip vs hit.* A flip (l(base) overtakes l(past) within the pair) is
  compatible with the argmax being neither form — at alpha=2 flip was 1.000
  while wiki ppl was 32. Always read `hit_rate` (full-vocab argmax == the
  counterpart). At alpha=1: flip 0.571 vs hit 0.371.
- *Role collisions.* " found" is past-of-*find* and base-of-*founded* (same for
  left/made/held). One token id cannot be both slots; `drop_role_collisions`
  removes them.
- *Spelling filters are insufficient* — "United" -> "Unite" is a legitimate verb
  pair, so it passes lexically. Only the contextual filter (clean top-1 must BE
  the target form, counterpart in top-50) removes "the United States".
- Yield is filter-bound: ~75 certified target positions per 1M mined tokens.
  Use `--n_mine_tokens 8000000` for real runs. Operating point alpha≈1.0.

**Corpus:** RedPajama is unusable — `RedPajama-Data-1T` and `-V2` are
script-based loaders that current `datasets` refuses, and the `-Sample` /
SlimPajama mirrors 404. Use wiki + C4 + pile (closest_tokens' own defaults).

**Tooling:** `--axis {past,plural,case,ing,ly}` selects the contrastive axis;
`--centroid` adds the two latent-only cluster-space arms; `--report_from <json>`
re-renders a finished run's tables without repeating it (per-example rows are
stored in the output). Paired SEs come from those rows — on the past run the
paired SE at alpha=2 was 0.011 vs 0.027 unpaired, which flipped that row's
verdict from "tie" to a small latent win (in a regime where wiki ppl is already
16, so not a clean one).

Related: [[train-diag-metrics-misleading]], [[zipf-balancing-null-result]].
