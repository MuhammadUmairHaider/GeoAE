---
name: biasbios-range-reversal
description: bias_in_bios 27 professions (2026-09-18) — AE marginally worse than base (matched-erasure +0.03, p<=.015); per-concept, the AE margin shrinks as base quality rises (user's hypothesis, directionally supported) but a residual dataset effect remains
metadata:
  type: project
---
Run: `range_intervention_compare --dataset biasbios --saliency dprime --tao 2 --alphas 1 2`,
GELU dpc best_val, 27 professions (class 26 "teacher" excluded: 0% few-shot, model always says
"professor"), 130 joint-correct docs each (3,510 total). Best-powered range run to date.

- **Matched erasure: base h wins ALL five tests**, opposite sign to DBpedia-14 and significant:
  range a1 +0.032 (p=.0095, n=15), transport a1 +0.030 (p=.015, n=16), range a2 +0.058
  (p=.0021, n=26), transport a2 +0.048 (p=.0047, n=27), rm_range_comp +0.044 (p<.0001, n=22).
  Positive = AE breaks MORE other classes. On DB14 the same five were −0.069/−0.069/−0.053/
  −0.033/−0.060. Fixed-alpha z−h also negative (range a2 −0.056 p=.002, 6/20 concepts).
- **Why (from the numbers, not speculation): the AE's collateral is dataset-invariant, the
  base's is not.** st_range a1 collateral: base 0.24 (DB14) -> 0.10 (biasbios); AE 0.17 -> 0.14.
  The base gets much cleaner on the 27-way profession task while the AE barely moves, so the
  DBpedia "AE win" was the base being unusually bad there, not the AE being good.
- **NOT a reconstruction-quality effect**: relative recon error on each dataset's own eval docs
  is 0.189 (DB14), 0.180 (biasbios), 0.171 (AG News), FVE 0.964/0.968/0.971. The AE reconstructs
  bios slightly BETTER than DBpedia.
- AG News (4 concepts) already hinted at this: 3/4 concepts favoured base; the positive average
  came entirely from Sci/Tech (+0.26), the thinnest and noisiest class.
- Full-neuron removal in z is still catastrophic here (rm_full_zero −0.367, rm_full_comp −0.174,
  2/25 concepts) — consistent everywhere.

- **User's hypothesis (2026-09-18): "where base is already good, the AE is marginally worse."**
  Pooled 45 concepts (DB14+bios+AG, d', transport a1), base quality measured from an
  INDEPENDENT run (base rm_range_comp selectivity) to avoid regression to the mean (the naive
  z−h vs own-h correlation, r=−0.34, is biased negative by construction). Terciles:
  base weakest (sel 0.47) z−h +0.090 (AE wins 10/15); middle (0.83) +0.018; strongest (0.94)
  −0.032 (p=.061, 5/15). Spearman −0.30 (p=.049) transport, −0.50 (p<.001) range.
  Within-dataset the direction holds in both (DB14 weaker half +0.17 vs stronger +0.02;
  bios −0.01 vs −0.02; range within bios rho −0.43 p=.024), BUT with dataset fixed effects
  base quality is not significant (p=.84 transport, .24 range) while dataset is (p=.026/.012):
  at similar base quality the AE margin is still ~0.11 lower on bios. So: moderation is real in
  direction, "marginal" is the right size word for the strong-base loss (−0.02 to −0.03), and
  base quality alone doesn't explain DB14 vs bios. Don't call it a "reversal".
**Why:** tests whether the DB14 AE advantage generalises. It does not.
**How to apply:** do not quote the DB14 range result as a general AE win — quote it alongside
biasbios. Any future claim needs >= 20 concepts; DB14's 14 and AG News's 4 are underpowered.
Harness support added 2026-09-18: `label_column` in _shared (bias_in_bios uses `profession`),
and build_joint_correct_set takes `classes=` so hopeless classes aren't scanned.
Related: [[range-interventions-h-vs-z]], [[tanh-encoder-arm]].
