# Llama L27: token-subtracted k-means versus bypass GeoAE

Model/data: Llama-3.2-3B L27, K=2000, sampled-position 10M training dump;
bypass AE d6144, epoch 50. **Base-tok** is balanced k-means on normalized
`x-b[current_token]`, not an AE. Bypass AE clusters `encoder(x-b[token])`.

The saved held-out atlas covers 500,000 token positions from the same 7,857
documents in both spaces (`text_differs=[]`). Judge: OpenRouter,
`google/gemini-2.5-flash-lite`. Judged coverage is 99.7% of base-tok usage
(1,965/2,000 live clusters) and 98.9% of bypass usage (1,853/1,999).

| Usage-weighted metric | Base-tok | Bypass AE | AE minus base (95% interval) |
| --- | --- | --- | --- |
| Nameable mass | 27.8% | 17.9% | -9.9 pp [-14.3, -5.5] |
| Conceptual mass | 11.9% | 9.6% | -2.3 pp [-6.1, +1.1] |
| Token-detector mass | 3.5% | 2.3% | -1.2 pp [-2.3, -0.2] |
| Semantic coherence (1–5) | 4.12 | 4.10 | -0.02 [-0.08, +0.06] |
| Intrusion accuracy | 34.4% | 25.4% | -9.1 pp [-12.8, -5.2] |
| Detection minus null | 11.6 pp | 6.4 pp | -5.1 pp [-7.7, -2.1] |

Nameable requires intrusion accuracy >=2/3 (three trials) and semantic coherence
>=4. Conceptual additionally requires a semantics/entity/topic/discourse label
and excludes clusters dominated by one token. These shares are renormalized over
judged cluster usage. Intervals resample clusters independently and do not cover
judge variability, training seeds, or the text sample.

Verdict: **base-tok is more nameable under this protocol**. Conceptual mass has
no resolved difference; token-detector mass is lower in bypass. A higher fraction
of usage labeled "semantics" (20.4% -> 25.7%) or "topic" (1.0% -> 2.8%) does
not itself demonstrate better interpretation; the judge's level categories are
unstable. Unweighted conceptual cluster share is lower in bypass, 11.7% -> 8.6%
(-3.1 pp [-5.0, -1.3]); distinguish that from the usage-weighted result.

Earlier CoarseWSD-20 word-sense result answers a different question: base-tok
cNMI 0.144 versus bypass 0.196, paired difference +0.0515 [+0.0454, +0.0586],
better on 16/20 words. Bypass ties vanilla residual k-means (0.191). The word-sense
encoder gain does not transfer to the atlas nameability measure.

## Repeat-judge control for complementary coverage

On the existing same 400k dump-token assignment sample:

| Metric | Base seed 0 | Base seed 1 | Bypass | Same-base repeat union | Base+bypass union |
| --- | --- | --- | --- | --- | --- |
| Nameable | 27.4% | 26.5% | 17.5% | 36.8% | 39.5% |
| Conceptual | 12.0% | 13.0% | 9.3% | 16.7% | 19.5% |

Thus the apparent extra nameable coverage from two views cannot all be attributed
to complementary representations: repeating judgments of one partition adds
9.4 pp, while base+bypass adds 12.1 pp. The remaining 2.7 pp is not a tested
significant gain, and the controls have different marginal nameability rates.
This sample is distinct from the held-out atlas above.

Evidence:
- `results/delta/d6144/interp/vs_tokmean/atlas_summary.json` (copied from box).
- `results/delta/d6144/interp/view_union_repeat_control.json` (computed from
  existing seed-1 judgments and assignments, no additional API calls).
- Full passages/judgments and browsable atlases remain on `geoae-box`.

Agreed next steps: matched separate no-bypass AE, then alpha=0.5 fixed-table
subtraction, then consider multiple resolutions. Training is explicitly assigned
to Delta by the user; see `HANDOFF_REMOTE.md` and `STATUS.md`.
