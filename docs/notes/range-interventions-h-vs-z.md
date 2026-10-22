---
name: range-interventions-h-vs-z
description: DB14-ONLY result, REVERSED on bias_in_bios (see [[biasbios-range-reversal]]) — NeuronLens range removal + steering, base h vs AE z (L27 dpc, 2026-09-17) — range STEERING clearly better in z; removal ties; ranges are NOT more separable in z; zero-replacement invalid in z
metadata:
  type: project
---
Harness: `geoae.interp.range_intervention_compare` (built 2026-09-17; ops in neuronlens.py
make_h_edit/make_z_edit share one operator for both spaces). Selectivity sign there is
tgt_drop − comp_drop, HIGHER = better (negation of steering_concept_compare).
Run: dpc best_val, saliency=abs (mean |a|, 30% dims), tao=2, 14 concepts, one seed.
Wiring check passed: st_global_a1.0 reproduced steer_db14_dpc.json within 0.004.

- **Steering: z wins ONLY for DENSE (all-dims) steering.** Corrected 2026-09-17 by a
  matched-erasure reanalysis (interpolate each concept's collateral-vs-erasure curve, then pair
  vs h). Dense/global: z − h = −0.086 (p=.024) at 60% erasure, −0.125 (p=.003) at 80%, and the
  tanh arm agrees (−0.089 / −0.119). Range-restricted (st_range, st_transport): NO significant
  difference from h at any erasure level 0.3–0.8 in either arm; signs flip.
  The earlier "+0.125 best-vs-best, p<.001" was an artifact: it compared z-transport against
  h-GLOBAL, i.e. h's fixed-alpha best, but at matched erasure h-range/h-transport are far better
  than h-global (0.15 vs 0.36 collateral at 80% erasure). Fixed-alpha selectivity is NOT a fair
  cross-space metric because the same alpha erases very different amounts.
- **d' SALIENCY RUN (2026-09-17, GELU arm, tao=2, 14 concepts) — the matched-erasure null above
  was itself an artefact of the mean-|a| dim rule.** With d' dims the AE wins on PRECISION:
  at alpha=2 both spaces erase 100% of the target, collateral z 0.386 vs h 0.438, Δ −0.053
  p=.009 (11/14); at alpha=1 on the 8 concepts with matched erasure, Δ −0.069, p=.007 (range)
  and p=.002 (transport), 8/8 concepts. rm_range_comp likewise −0.060 p=.015.
  Fixed-alpha z−h under d': range a1 +0.105 (p=.041, 12/2), range a2 +0.053 (p=.009),
  transport a1 +0.094 (p=.104), rm_range_comp +0.089 (p=.015), rm_full_comp +0.103 (p=.012).
- **d' fixes the z zero-replacement catastrophe**: rm_range_zero z gained +0.567 (p<.001) from
  d' while h gained nothing, so z−h went −0.506 -> +0.086 (ns, a tie). So zero replacement is
  not invalid in a GELU latent per se — it was invalid for mean-|a| dims, which are the
  high-offset dims of the shifted code. rm_full_zero still destroys z (−0.481): the RANGE GATE
  is what saves it, the paper's claim seen inside the latent.
- **d' helps the BASE more on most steering ops** (+0.27 vs +0.15 at range a0.5, +0.14 vs +0.10
  at a1), which is why the fixed-alpha margins shrank vs the mean-|a| run. Gate firing on
  other-class docs drops 0.57 -> 0.40 (h) and 0.58 -> 0.41 (z).
- Sanity check that held: st_global_* rows are bit-identical between the two saliency runs,
  since dense steering never uses the salient set.
- **Ranges beat dense steering INSIDE both spaces** (this is the paper's claim and it replicates):
  at 80% erasure gelu z range−global = −0.047 (p=.039); best operating points are
  gelu z-transport 0.149 / base h-range 0.151 / base h-transport 0.157 / gelu z-range 0.158,
  all far below global (z 0.203, h 0.357). So ranges help everywhere and the AE adds nothing
  once ranges are used.
- **Removal: tie.** h rm_range with zero (the paper's reference) +0.629 vs z rm_range with
  other-class mean +0.575: −0.054±0.087, p=.55, 7/7. Range beats full in both spaces
  (h +0.184 13/1, z +0.387 14/0) — the paper's claim replicates at L27 DB14.
- **Zero replacement is invalid in z**: z rm_range_zero tgt/comp drop 1.00/0.88, rm_full_zero
  1.00/1.00. GELU latent is a shifted code. This is what killed z on AG News (Aug 7), not ranges.
- **Ranges are not more separable in z**: gate fires on 95% of salient dims for target docs
  and 57–58% for others in BOTH spaces (gap +0.378 h, +0.365 z). The z steering advantage
  must come from the decode step, not cleaner concept ranges (untested hypothesis:
  decoder projects the edit back on-manifold).
- rm_full_comp ppl +143/+161 is expected: a last-token mean forced on every position.
- Thin classes in dpc joint-correct set: MeanOfTransportation (8 eval), Village (19).

**Why:** answers "do NeuronLens ranges work better in base or the AE" — user is the NeuronLens author.
**How to apply:** never compare z removal with zero replacement; quote best-vs-best, and
matched-erasure collateral, since z has 2× the salient dims. Pending: d′ saliency run,
k-means++ parent AE, decode-step mechanism test. Related: [[probe-tpp-gotchas]], [[init-arms-dpc-results]].
