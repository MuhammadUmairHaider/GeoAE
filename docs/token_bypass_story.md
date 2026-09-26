# Layer 27: new data → token-bypass AE → control tests

A summary of this line of work (Sept 2026). Model: Llama-3.2-3B, layer 27 (the last block), residual stream.
GeoAE = autoencoder with a 6144-d latent and 2000 balanced clusters.
**Base** = the plain residual, or balanced k-means (K=2000) on it when clusters are needed.
Numbers are "per 100" unless marked. All results are from a single AE training seed.

---

## 1. New activation dump

**What changed.**
- **Old dump:** the first 256 tokens of 44k docs, from 5 equal sources.
- **New dump:** 64 random positions per doc, up to 2048 tokens, from 157k docs, in a pretraining-like mix (40% educational web, 7% non-English wiki).

**How different.** The shift is modest (Fréchet distance 10× a split-half null). It comes from the source mix, not from position.

**Same AE recipe retrained on it:**

| | old dump | new dump |
|---|---|---|
| reconstruction error on non-English text | 0.055 | **0.031** |
| topic clustering (topic14) | better | −0.16 |
| language clustering | better | −0.09 |
| indirect-object role (ioi_role) | | +0.09 |
| number-control edit advantage | | halved |
| DBpedia / bias_in_bios interventions | no robust difference | |

**Decision:** keep the new dump. It is more realistic, and what it loses is concept alignment, which later steps fixed.

## 2. Why clusters looked like "token dumps"

- The biggest clusters looked like junk. Part of that was the profiling tool, which sampled old-recipe text. The clusters are more interpretable than their top-token lists suggest.
- The real issue is that **20% of AE clusters were pure current-token detectors** (one token makes up ≥80% of members), against 11% for base k-means.
- **Layer sweep:**
  - Past layer 4, current-token identity explains only 6–11% of residual variance at every layer.
  - Document context is highest at layer 27.
  - So the problem is clustering single-token rows, not the choice of layer.

## 3. Token erasure without retraining

- **Method:** project out the top 64 "token identity" directions, then run k-means.
- **Result:**
  - Sequence-level concept alignment rises by 0.046; topic14 goes 0.47 → 0.60.
  - Part-of-speech falls by 0.15.
  - About 1/3 of the gain is token-specific; the rest comes from removing dominant directions in general.
- **Current-token vs predicted-next-token directions:**
  - Each alone gives most of the gain (+0.043 vs +0.038).
  - The predicted-next version keeps more entity information.
- **Decision:** bypass the **current token**. A predicted-next bypass would be computed from the unedited state and would undo latent edits.

## 4. The token-bypass AE

**Idea** (like Tokenized SAEs, but also subtracted on the way in):
- Keep a table with each token's average residual across all its contexts (shrunk toward zero for rare tokens; 49.7k tokens, 98.5% of rows).
- Subtract the current token's entry before the encoder, and add it back after the decoder.
- The encoder therefore sees only context.

| | parent AE | base k-means | **bypass AE** |
|---|---|---|---|
| pure token-detector clusters | 21% | 11% | **5%** |
| topic clustering (topic14) | 0.33 | 0.45 | **0.60** |
| mean sequence-concept alignment | 0.145 | 0.173 | **0.205** |
| part-of-speech clustering | 0.40 | 0.34 | 0.15 (syntax now lives in the token table) |
| MMLU with the AE spliced in (base 0.562) | 0.539 | — | 0.548 |
| reconstruction error | 0.031 | — | 0.034 (+11%) |
| hub clusters | 12, 2.8% of rows | — | 22, 7.7% of rows |

- **What the clusters are:** they group by document and topic (same-document NMI 0.08 → 0.23). The hubs are predictable continuations and boilerplate, which have little context left once the token is removed.
- **Matched control:** k-means on the same token-subtracted residual *ties* the bypass AE on concept alignment (+0.005). **The clustering gain is the subtraction, not the encoder.** The encoder's value shows up in editing and control (below).

## 5. Labelled interventions (bypass latent vs base residual)

- **DBpedia, 14 topics:** at equal or higher target removal, collateral is about 4–5 lower. Range removal goes from 86 removed / 20 collateral to 89 / 15; range steering from 86 / 24 to 90 / 20. Plain global steering ties.
- **bias_in_bios, 27 professions:** the earlier AE was worse than base. The bypass AE **reaches parity**.
- **Number control:** tie.
- **Matched base check:** doing the same edits on the token-subtracted residual equals plain base, so the subtraction alone explains nothing here.

## 6. Discovered clusters as control handles (each space steers its own clusters)

**One row, next token.** Move a row into a target cluster. At equal disruption, bypass clusters transfer **2.4×** as often as base clusters.

**Generated text.**
- **Setup:** push every position toward a target cluster; 40 targets each; 20 prompts; an LLM judge.
- **Scoring:** only fluent continuations count, net of the unsteered text.

| | base clusters | bypass clusters |
|---|---|---|
| usable (fluent + on-target) at best strength | 6.5 | **15** |
| blind human read: fluent on-target | 3 / 180 | 17 / 180 |

- **Why:**
  - Per fluent continuation the two are equally on-target; bypass text simply stays fluent at stronger pushes.
  - Base clusters are often token-shaped, so steering toward them makes the model repeat a word.
  - Range gating changed nothing.
- **Caveat:** the targets differed between the two arms, which led to §7.

## 7. Named concepts, same targets (DBpedia, 14 topics)

Labels are used in every handle, but differently:
- **Label direction:** labels define the direction (topic average minus the all-topics average).
- **Cluster handles:** labels only pick the 5 clusters most specific to the topic. The direction then comes from those clusters' averages over general text.

Usable per 100 at best strength:

| handle | usable |
|---|---|
| label direction (supervised) | 27.9 |
| **bypass clusters, used as a direction** | **28.6** |
| base clusters, used as a direction | 22.9 |
| bypass clusters, "move into cluster" | 18.6 |
| base clusters, "move into cluster" | 11.8 |

- **Bypass beats base as handles:** +6.8 on the same topics (p=0.04). Bypass clusters are also more topic-specific on all 14 topics (precision 0.94 vs 0.88).
- **Edit form matters most:** adding a direction beats "move into cluster" by about 10 for both codebooks (p≤0.01), because "move into cluster" also erases the position's current cluster.
- **Bypass clusters used as a direction tie the supervised label direction** (+0.7, p=0.94).
- **With labels, the AE adds nothing to additive steering.** The label direction computed in the bypass latent is the same vector (cosine 0.99) and steers the same (28.9 vs 27.9).

## 8. Removal in generated text (DBpedia)

- **Result: null.** Continuations stay on-topic 96% of the time before and after removal, in both spaces, so base and bypass can't be compared.
  - **Reason:** layer 27 is the last block. An edit only changes each step's next-token choice, and the unedited context keeps supplying the topic.
  - **So** the DBpedia removal advantage in §5 applies to single-step readouts, not to free text at this layer.
- **Side finding:** NeuronLens-style per-coordinate ranges fire on about 95% of *all* token positions. They are only selective at a classification token. A per-position gate is needed.
- **Topic separability per token is equal** in the bypass latent and the residual (AUROC 0.92 vs 0.91).

---

## Bottom line

1. Subtracting each token's average turns clusters from token detectors into context and topic concepts. That is the main win.
2. **As unsupervised control handles, bypass clusters are clearly better than base clusters**, and used as a direction they match a supervised label direction.
3. **When labels exist**, the AE adds little to steering; its labelled-edit edge is limited to coordinate-wise edits at single-step readouts.
4. **Costs:** +11% reconstruction error, more hub clusters, and syntax and entity information moved out of the latent.

## Open items

- A second training seed of the bypass AE, then rerun §6–7.
- Named-concept steering on bias_in_bios professions (only DBpedia done so far).
- Label efficiency: redo §7 with 5 and 20 labelled docs per topic. Clusters might beat label directions when labels are scarce.
- Removal strength sweep, as a last check before dropping generation-level removal at layer 27.
- Recover reconstruction (softer separation ramp, less shrinkage) and handle hub clusters.
- The evaluated bypass checkpoint is the final epoch; best validation reconstruction was around epoch 25.

## Method lessons

- Report **chance-corrected** NMI. Raw NMI inflates arms that spread rows over more clusters.
- Bootstrap CIs must be centred on the estimate. Percentile intervals of NMI can exclude the estimate itself.
- LLM-judge hit rates include chance, per-prompt priors and word loops. Score **fluent continuations only, net of the unsteered text**.
- Perplexity *falls* for repetition loops. Use text variety (distinct bigrams) as the fluency axis.
- Compare steering handles on the **same targets** and at matched edit size.
