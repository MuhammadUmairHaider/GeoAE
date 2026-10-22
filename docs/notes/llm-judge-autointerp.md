---
name: llm-judge-autointerp
description: "LLM-as-judge auto-interp for GeoAE clusters — the tool, the four controls it needs, and the null result: the AE does NOT beat balanced k-means on semantic coherence."
metadata:
  type: project
---

`geoae/interp/llm_judge.py` (built 2026-09-02). Explain-then-score auto-interp over
`closest_tokens` JSONs, via OpenRouter (OpenAI-compatible; `--provider openrouter`).
Disk-caches by hash(provider, model, prompt), so re-runs are free.

## Measures
- `semantic_coherence` 1-5 and `surface_coherence` 1-5, rated SEPARATELY. Necessary:
  a cluster of one repeated token string rates a trivial 5 on a single combined score.
- **word intrusion** (Chang et al. 2009): 5 members + 1 intruder from a neighbouring
  cluster, chance 1/6. Behavioural — independent of explanation quality. Best measure.
- detection balanced-accuracy against hard negatives, minus a null floor.

## RESULT (llama L27, n=200/arm, gemini-2.5-flash-lite)

| | semantic | intrusion (chance .167) | live clusters |
|---|---|---|---|
| AE b32k lam1_d6144 | 4.48 | 0.318 | 1988 |
| balanced kmeans (no encoder) | 4.38 | 0.332 | 1999 |
| plain kmeans | 4.61 | 0.512 | 1099 |

**AE - balanced is NOT significant**: semantic +0.10 +/- 0.065 (1.5 SE) on full
sequences, intrusion -0.014 (sign flipped). A short-context run gave +0.15 (2.4 SE)
and looked significant — it did not replicate at a longer context window. Do not
claim the encoder wins on semantic coherence. Consistent with every other method here.

## Traps, all of them load-bearing
- **Judge choice changes the taxonomy wildly.** Same clusters and prompt, `level` =
  syntax: gpt-4o-mini 20%, gemini-2.5-flash-lite 48%, deepseek-v3.1 70%, gpt-5-mini 53%.
  Absolute `level` and `mono` are ONLY comparable within one judge across arms.
- **gpt-5-mini fails as a judge** at max_tokens 900: it is a reasoning model and the
  cap counts reasoning tokens, so it scored 17/40 clusters and picked 2.4/12 items.
- **Track degenerate answers** (picked all or none) — those score exactly 0.5 balanced
  accuracy and are indistinguishable from an honest coin flip without the counter.
- **Plain k-means "wins" are a capacity confound, do not quote them.** It has 1099 live
  clusters vs ~1990, so its centroid neighbours are farther apart and its hard negatives
  / intruders are actually EASY. Its surface coherence is also highest (1.58 vs 1.39).
- **The raw mono score tracks token repetition, not meaning.** Raw mono ranked
  plain 4.32 > AE 4.21 > balanced 3.96, exactly matching token-repetition 0.433 /
  0.359 / 0.241; restricted to low-repetition clusters all three collapse to ~3.75-3.86.
- `closest_tokens` stores only the RENDERED context (default `--context_window 6`,
  ~7.6 words) with no doc id or offset, so longer context CANNOT be recovered
  post-hoc — it must be regenerated. `--context_window 128` gives ~110 words.
- Per-cluster RNG seeded from (seed, cluster id); a shared RNG across worker threads
  makes model-vs-model comparisons unfair.

Related: [[fineweb-atlas-concept-alignment]], [[train-diag-metrics-misleading]].

## Multi-judge agreement (2026-09-09) — `geoae/interp/judge_agreement.py`
Follows VQLC (arXiv 2602.02726, the user's own group): >=3 judges, Kendall's W
(tie-corrected), majority-vote intrusion with unresolved items dropped.

**Chance level for W is NOT 0 — it is ~1/m.** Measured 0.344 for 3 judges on
random scores. VQLC reports W 0.30-0.78; their low end (AG News + LLaMA, 0.300)
is at chance. Always print the null alongside W (the tool does).

L27, 120 stratified clusters, judges gemini-2.5-flash-lite / gpt-4o-mini /
deepseek-v3.1:

| W | seeded_atlas | balanced km | b32k parent |
|---|---|---|---|
| intruder_acc | 0.704 | 0.690 | 0.646 |
| semantic_coh | 0.440 | 0.508 | 0.453 |
| mono_llm | 0.490 | 0.528 | 0.548 |

Intrusion is the only metric with real inter-judge agreement (~2x chance) in
every arm; rating scales sit at 1.3-1.6x chance. Absolute scores drift
systematically: deepseek highest, gemini lowest, on every metric and arm.
Judges are models sharing pretraining, so W is an upper bound on true
annotator agreement.

At n=120 seeded_atlas beat balanced on intrusion under all 3 judges
(0.378/0.358/0.417 vs 0.305/0.342/0.330), contradicting the earlier
single-judge full-sample result (0.352 vs 0.371). Full-scale 3-judge rerun
pending — do not quote the n=120 win.

## FULL-SCALE 3-JUDGE RERUN (2026-09-17) — base vs dpc GELU AE, n=200/arm, all resolved
`results/judge_agreement_base_vs_dpc.json`. Arms: ct_balanced_fullseq (balanced k-means, NO
encoder) vs ct_dpc_fullseq (dpc GELU AE best_val), both context_window 128, same corpus.

| metric | base | dpc AE | verdict |
|---|---|---|---|
| intruder_acc (chance .167) | .332/.315/.388 → **.345** | .267/.308/.365 → .313 | **base higher on 3/3 judges** |
| consensus intrusion pass | 0.580 | 0.565 | base |
| semantic_coh | 4.385/4.480/4.695 → 4.520 | 4.490/4.480/4.665 → 4.545 | tie (judges disagree in sign) |
| generality | 3.608 | 3.575 | tie |
| mono_llm | 4.270 | **4.393** | AE — but artefact, see below |

The mono_llm "win" is the documented repetition effect: token repetition in the top examples
is 0.551 (dpc) vs 0.505 (base). Do not quote it.
W: intrusion 0.673/0.637 vs null 0.334 (~2x chance, the only trustworthy metric); rating
scales 0.45-0.58 (1.3-1.7x). Effect sizes are small vs noise (intrusion gap 0.032 with ~600
trials/judge/arm, SE ~0.027), so read this as "no AE win", not "base significantly better".
**So the null result now holds for dpc too, at full scale, with 3 judges.** The n=120
seeded_atlas intrusion win remains unreplicated and still should not be quoted.

Tool bug fixed: one judge returning null on one cluster used to drop the
whole metric for that arm; filter per-metric instead.
