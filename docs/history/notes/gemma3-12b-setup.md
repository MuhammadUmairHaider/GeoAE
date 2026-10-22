---
name: gemma3-12b-setup
description: "Gemma 3 12B / Gemma Scope 2 track — what the Scope repo actually is, chosen layers, disk math, and the multimodal-wrapper trap"
metadata: 
  node_type: memory
  type: project
  originSessionId: edca91a7-d6a3-43dd-9daf-193659f3c461
  modified: 2026-08-11T22:10:11.813Z
---

Second model track added 2026-08-11 alongside Llama-3.2-3B: **Gemma 3 12B PT**, chosen so
GeoAE runs have published SAE baselines from `google/gemma-scope-2-12b-pt`.

**`google/gemma-scope-2-12b-pt` is an SAE suite, not an LM.** It is a `saelens` repo of
JumpReLU SAEs/transcoders trained on `google/gemma-3-12b-pt`. Load baselines with
`SAE.from_pretrained(release="gemma-scope-2-12b-pt-resid_post", sae_id="layer_31_width_65k_l0_medium")`.
Each SAE's `config.json` names its hook point, e.g. `"hf_hook_point_in": "model.layers.31.output"`
— the same point `geoae.extract` hooks, so the spaces are directly comparable.

- Scope subset layers = **12, 24, 31, 41** of 48 (25/50/65/85% depth). `resid_post` has
  widths 16k/64k/256k/1m × L0 small/medium/big. `resid_post_all` covers every layer 0..47
  but only 16k/262k × small/big. DeepMind recommends 64k or 256k, L0 medium.
- **Layer 47** (final block) is extracted and configured too: it is the only layer where
  e2e KL keeps the cheap head-only path, matching the Llama-L27 / Qwen-L31 regime, so it
  is the cheapest run and the like-for-like continuation. `resid_post` has nothing at 47 —
  baselines come from `resid_post_all/layer_47_width_{16k,262k}_l0_{small,big}` only.
- Gemma 3 12B: 48 layers, **d=3840**, vocab **262208**, 24.4 GB bf16,
  `final_logit_softcapping: null` (so the last-layer head-only KL shortcut stays exact).
- **Gated repo** — needs accepted licence + HF token. Ungated identical mirror if ever
  needed: `unsloth/gemma-3-12b-pt`.

**Decisions:** extract layers 12/24/31/41/47. Layers 12–41 are NOT last, so e2e KL there
uses the all-position splice path in `geoae.e2e.train_stream` (teacher forward + student
forward + backward through 48−1−L frozen blocks); only layer 47 keeps the head-only
shortcut Llama L27 had.

**Disk:** at the required fp32, 3840 dims = **15.4 GB per 1M tokens per layer** → 46.1 GB
per 1M across three layers. Settled config: layers [24, 41, 47] × 5M tokens = 230 GB.
10M × 3 at fp32 (453 GB) does NOT fit. `geoae.extract` preflights free space and aborts
before loading the model, printing the largest budget that fits — trust that over any
number written here, since disk state drifts fast on this box.
The dump is NOT the e2e training corpus: `train_stream` reads it only for norm stats
(~500k rows) and k-means baselines sample ~1M, so a few M tokens suffices. Large n_tokens
only earns its disk for the MSE base pipeline (`geoae.train`), which trains on these rows.

**CONFIRMED FAMILY-WIDE 2026-08-21 — probe any new Gemma 3 size before committing disk.**
Gemma 3 4B PT layer 33: **dim 443**, median |x| = 58,496, overflows fp16 on **85.1%** of
tokens (39,050 non-finite in 45,865 tokens). Exactly one channel; every other dim peaks at
a median of ~2,240, so dim 443 is 26x the next largest. Same shape of failure as 12B's dim
2339 at a different index — so this is a Gemma 3 architecture trait, not a 12B quirk.
Assume float32 for every size, and probe with
`--n_tokens 200000 --out_dir /tmp/probe` (aborts on the first overflow) before any big run.

**float16 storage is BROKEN for Gemma 3 — use float32.** Measured 2026-08-11 on a 9.83M-token
dump of layers 24/41/47: dimension **2339** is a massive-activation channel, median |x| ≈
53,760, exceeding the fp16 ceiling of 65,504 on 15.7% of tokens at L24 and ~98% at L41/L47.
Those became `inf`, so mean/std came back inf/nan and all 226 GB was unusable. Every *other*
dim peaks at ~10,560 — it is one channel that forces fp32. bfloat16 has the range at half the
size but numpy's `.npy` cannot round-trip a third-party bf16 dtype (serialises as opaque
`|V2|`, and even the memmap write fails), so fp32 is the only safe option. `geoae.extract` now
aborts within seconds of the first overflow instead of at the end of the run.

**Do not use the naive lr rule on Gemma.** `lr ≈ 1e-4 × (56 / median_token_norm)` was
calibrated on Llama/Qwen. Dim 2339 alone sets Gemma's median token norm to ~53,900, which
would suggest lr ~1e-7. Excluding it the real scale is ~4,140 (73× Llama's 56.4). The AE input
is z-scored per dim anyway, so raw norm only matters through the KL gradient at the head —
start at 2e-5 and watch val_kl.

**Architecture trap:** `AutoModelForCausalLM` returns `Gemma3ForConditionalGeneration` for
model_type `gemma3`. `lm.model` has **no `.layers` and no `.norm`** — the decoder is at
`lm.model.language_model`. Never hard-code `lm.model.layers`; use `geoae.lm_arch`
(`decoder_layers` / `locate_lm_parts` / `hidden_size`), which resolves via `get_decoder()`
and is pinned by `tests/test_lm_arch.py` for Llama, Qwen3 and Gemma 3.
`model.config.hidden_size` is likewise wrong on the multimodal config — use
`lm_arch.hidden_size(model)`.

Key paths: configs `configs/base/gemma3-12b_extract.yaml` and
`configs/e2e/general/gemma3-12B/layer{12,24,31,41}/kl_gelu_k2000_vicreg.yaml`;
activations `activations_gemma3_12b/layer_{12,24,31,41}.npy`.

See [[geoae-training-gotchas]] for the shared loss/eval pitfalls.
