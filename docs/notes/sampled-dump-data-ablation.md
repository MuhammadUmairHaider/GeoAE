---
name: sampled-dump-data-ablation
description: "d6144 dpc trained on activations_sampled_10M vs activations_diverse_10M (identical config) — what changed in data and evals, plus epoch-snapshot and number-control traps"
metadata:
  node_type: memory
  type: project
  originSessionId: e3fe5c7d-e535-4dab-b0eb-3ca472d39b5c
  modified: 2026-09-25T07:00:24.893Z
---

Data-only ablation at L27 d6144 dpc (analysed 2026-09-25). Configs differ only in activations_dir.

**Data**: old = prefix pos 4..255, 43.7k docs, equal 5-way (code 22%, math 21%, wiki = first 8.7k articles by page id — "Anarchism, Albedo, A, Alabama..."), no shuffle. New = 64 random positions/doc up to 2047 (45% rows at pos>=256), 157k docs, edu_web 40%, 7% de/fr/es wiki. Old-dump token/pos/domain sidecars can be rebuilt exactly by replaying geoae.extract stream_docs (tokenizer only; verified cos=1.000 vs stored rows). Raw-space shift is modest: Fréchet 155 vs split-half null 15.5 (trace 2945); doc-disjoint linear detector 73%; the shift is source-mix driven, not position (restricting new to pos<256 doesn't lower it).

**Evals (new vs old)**:
- Recon: compare BEST checkpoints — old ep50 (step_0014200) sits on a val spike (val_mse 0.0349 vs best 0.0299 at ep47). Best-vs-best FVU: old AE 0.0265 old-dump / 0.0319 new-dump; new AE 0.0273 / 0.0269. Non-English: old AE 0.053-0.057 vs new 0.030-0.032.
- Probe (paired bootstrap, AE diff minus km diff): new AE wins ioi_role (+0.093), formality; loses topic14 (-0.160, same live count so real), language (-0.086), ravel_language/country, token rungs slightly. RAVEL gains are mostly the DATA (km_new gains more than AE).
- Number control (d' both): new AE's z-over-rotation advantage halves at a0.5: -0.16 [-0.24,-0.08] noun bootstrap, mostly plural suppression; edit norms matched.
- DB14 steer/range: no robust difference. Bios: only rm_full_zero robustly worse (-0.16, p<=.036 vs both old ep47 and ep50).

**closest_tokens**: by default it streams the LEGACY recipe (hardcoded DEFAULT_SOURCES, pos 4..255), whatever checkpoint is passed — closest_tokens_dpc_sampled.json (2026-09-25) profiled the new AE on old-recipe text. `--sampled_from activations_sampled_10M` (added 2026-09-25) draws held-out corpus docs (offset = meta sources[i].docs + skipped_short; verified exact by length multiset) with the dump's mix/context/64 pos per doc. Needs only meta.json + corpus/, not the 61 GB npy.

**Traps**: summarize_data_ablation.py points the old number arm at number_control_review.json (abs saliency) while new uses dprime — compare against results/range_number_dpc_dprime.json instead. Old ep47 vs ep50 of the same run differ as much as old vs new on many range operators → epoch noise floor ~0.05 per operator. See [[range-interventions-h-vs-z]], [[cq-dunn-and-steer-sign-traps]].
