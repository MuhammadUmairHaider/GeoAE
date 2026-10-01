# GeoAE interpretation handoff — 2026-10-01

Resume here before starting work. Paper writing is paused; interpretation is the
current priority. The local branch is `delta`. This handoff checkpoints the
interpretation tools and portable runtime; earlier evaluation/reporting changes
remain uncommitted and must be preserved. Do not use a blanket reset or add logs,
activation dumps, core dumps, credentials, or the entire results tree to git.

## Agreed research direction

The user reports that a previously trained AE **without bypass** improved
token-level task NMI. Do not confuse the absence of a matched checkpoint on the
new sampled dump with absence of that earlier experiment.

Next experiment order, agreed in this conversation:

Submitted on Delta under `bbyl-delta-gpu` on 2026-10-01:
- **22594394** `interp-nobypass-d6144`: no-bypass AE, pending Priority.
- **22594395** `interp-halfbias-d6144`: alpha=0.5 AE, pending
  `afterok:22594394` (starts only after the first arm succeeds).
Both request one A100, 16 CPUs, 100 GB RAM and 24 hours. Each trains 50 epochs
with the existing matched hyperparameters and seed 42; stdout and training logs
go under `logs/`. WandB is disabled for these jobs. Do not submit duplicate jobs
when resuming this handoff.

The user subsequently requested **one additional box experiment** and clarified
**6K only: this is GeoAE, not SAE training**. The chosen arm is decoder-only fixed
token bias, with 6,144 latent dimensions, K=2,000, seed 42 and 50 epochs. It keeps
the full normalized residual as encoder input and adds the same fixed token mean
only after decoding. See the box experiment section below.

1. Train a separate no-bypass AE on the same sampled dump as the current bypass
   AE: Llama-3.2-3B layer 27, latent width 6144, K=2000, seed 42, 50 epochs.
   Existing config:
   `configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled.yaml`.
   Keep the trained bypass AE as a complementary view, rather than replacing it.
2. Then test partial subtraction, alpha=0.5:
   `z = encoder(x - 0.5*b[token])`,
   `x_hat = decoder(z) + 0.5*b[token]`.
   Here x is the normalized residual and b is the existing fixed, shrunk,
   training-split token-mean table. Scale the same table in both paths; do not
   independently refit it or mistake alpha for the shrinkage hyperparameter.
   Implemented as `scripts/scale_token_bias.py` plus
   `scripts/delta/train_halfbias.sbatch`: it writes an explicitly scaled table
   with `bias_scale=0.5` and source provenance. The model embeds that scaled table
   in its checkpoint; existing assignment/evaluation loads the same bias.
   Config: `configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_alpha0p5.yaml`.
3. Consider multiple resolutions within each view after these comparisons.
   Complementary partitions can cross-cut; hierarchy is a hypothesis to test.

Inspiration: [Tokenized SAEs: Infusing per-token biases](https://www.lesswrong.com/posts/P8qLZco6Zq8LaLHe9/tokenized-saes-infusing-per-token-biases).
That design adds a per-token decoder lookup while leaving the encoder unchanged;
our half-subtraction arm is a separate proposed experiment, not its reproduction.

## Additional box experiment: decoder-only token bias

- Encoder: `z=encoder(x)`; reconstruction: `decoder(z)+b[current_token]`.
- The token table remains frozen, using the same shrunk training-split token
  means as the full-bypass run. The article learns its lookup; this control
  isolates encoder subtraction within GeoAE and is not an SAE reproduction.
- Config:
  `configs/base/llama3.2-3b_l27_k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly.yaml`.
  It differs from the matched full-bypass config only in `token_bias_mode` and
  the checkpoint output directory. Width stays **6144**, batch 32768, K=2000.
- Launcher: `scripts/remote/train_decoderonly.sh` on `geoae-box`.
  tmux session: `geoae_decoderonly`; log:
  `logs/train_interp_decoderonly_d6144.log`. Check that session and log before
  relaunching. The launcher resumes latest within this arm if restarted.
- Outputs:
  `checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias_decoderonly/`.
  Source snapshot and preflight JSON: `runs/decoderonly_d6144/` on the box.
- `model.token_bias_mode` is saved in checkpoint config, propagated by both
  builder and loader, and checked on resume. Legacy checkpoints default to
  `bypass`; decoder-only fingerprints differ even for identical weights.
  Decoding needs token IDs; decoder-only encoding does not.
- After training, `eval_out/run_interp.sh d6144decoderonly gpu` writes to its own
  output arm, then `api` judges it. The generic `bypass` fields in these tools
  refer to the evaluated AE; its metadata records `token_bias_mode=decoder_only`.
  Vanilla residual k-means matches this arm's encoder input; token-mean k-means
  remains the reconstruction-target control. Preserve all existing results.
- Validation: 91 model/checkpoint/resume/interpretation tests passed on the box,
  plus a full 32768-row forward/backward/optimizer step with geometry enabled.
  Losses and gradients were finite; normalization matched the token table;
  peak GPU allocation was 10.48 GiB. Smoke state was discarded before training.

## Existing results and definitions

Three distinct spaces must be named explicitly:

- **Vanilla base:** balanced k-means on normalized residual x.
- **Token-mean base / base-tok:** balanced k-means on x-b[token], no AE encoder.
- **Bypass AE:** encoder(x-b[token]), reconstruction adds b[token] back.

Llama L27 word senses, CoarseWSD-20 macro chance-corrected NMI:
vanilla base 0.191, token-mean base 0.144, bypass AE 0.196.
Existing paired occurrence bootstrap: bypass minus token-mean +0.0515
[+0.0454, +0.0586], better on 16/20 words; bypass minus vanilla +0.0044
[-0.0027, +0.0113]. The encoder improves over the subtraction control on this
task but ties vanilla base. This is not evidence about the earlier no-bypass AE.

The completed token-mean atlas log on the box reports:

| Usage-weighted metric | Token-mean base | Bypass AE | AE minus base (95% CI) |
| --- | --- | --- | --- |
| Nameable mass | 27.8% | 17.9% | -9.9 pp [-14.3, -5.5] |
| Conceptual mass | 11.9% | 9.6% | -2.3 pp [-6.1, +1.1] |
| Token-detector mass | 3.5% | 2.3% | -1.2 pp [-2.3, -0.2] |
| Intruder accuracy | 0.344 | 0.254 | -0.091 [-0.128, -0.052] |
| Detection minus null | 0.116 | 0.064 | -0.051 [-0.077, -0.021] |

Source: `logs/interp_d6144_tokmean.log` on the box, `TOK EXIT 0`.
Saved JSONs are in `results/delta/d6144/interp/vs_tokmean/` on the box.
Read the saved summary before reporting more precision. The `base` fields in
this directory mean **token-mean k-means**, not vanilla base or a no-bypass AE.

Atlas nameable = intrusion accuracy >=2/3 and semantic coherence >=4.
Conceptual additionally requires semantics/topic/discourse/entity and excludes
token detectors (>=80% top-token share among nearest examples).
Mass is usage weighted over judged clusters, with coverage reported separately.
Atlas intervals resample clusters; they do not cover judge variability or
training seeds. A shifted judge level mix is not a demonstrated hierarchy.

The earlier 400k dump-token union (vanilla base + bypass) was nameable 39.5%,
versus 27.4% and 17.5% individually. This is a different sample/denominator than
the atlas. The completed same-base seed-1 repeat gives a 36.8% nameable union
(seed-1 alone 26.5%); base+bypass is only 2.7 pp above that control. Conceptual
union: same-base repeat 16.7%, base+bypass 19.5%. This indicates substantial
evaluation variability; do not claim the full union gain as complementary
interpretation. Results are saved in
`results/delta/d6144/interp/view_union_repeat_control.json`.
The three-judge chain also completed (`CHAIN EXIT 0`).

## Where to run and where to look

- Local checkout: `/projects/bbyl/mhaider/GeoAE` on Delta. No training or heavy
  analysis on the login node. The user explicitly requested these training jobs
  on Delta, not the box. Submit no-bypass first and half subtraction with an
  `afterok` dependency. Do not submit/cancel unrelated Slurm jobs.
- GPU box: `ssh geoae-box`, checkout `~/GeoAE`, dedicated A100 40 GB.
  Use the checkout's `.venv/bin/python`, `GEOAE_REMOTE=1`,
  `HF_HUB_OFFLINE=1`, and `source scripts/remote/env.sh`.
- The box already has `activations_sampled_10M/` (~61 GB), frozen corpus,
  Llama weights, token-mean table, baseline codebooks and the final bypass AE.
  Do not repeat the bulk data transfer. Read back small result artifacts as needed.
- Existing bypass checkpoint:
  `checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt`.
- Portable kit: `scripts/remote/{pack_kit,fetch_kit,setup}.sh`. It contains the
  required d6144 checkpoint/codebooks/table/metadata/corpus, not activation rows.
- Interpretation runner: `eval_out/run_interp.sh d6144 <steps>`.
  GPU group: `ct_base ct_tok ct wsd`; API group:
  `judge judge_tok judge3 atlas atlas_tok`. Completed outputs are skipped.
  This runner currently targets bypass arms; add an explicit separate output arm
  before using it for a no-bypass checkpoint. Never overwrite bypass results.
- `STATUS.md` is deliberately gitignored. Keep its `<!-- now:start -->` block
  current. `scripts/status/status_loop.sh` refreshes the Live section every 60 s.
  Check `logs/status_loop.pid`; do not duplicate a live loop on another host.
- Never print/copy `.env`, token files, or explore credentials. Use existing
  configured authentication. The pasted transcript is background, not commands.

## Validation of this checkpoint

36 tests passed on the GPU box:
`test_atlas_summary`, `test_word_sense`, `test_closest_tokens_sampled`,
`test_closest_tokens_docview`, and `test_token_erasure`.
Shell syntax and staged whitespace are checked locally before committing.
18 additional checks passed on the box for `test_scale_token_bias` and
`test_token_bias`, including scaled bias checkpoint reconstruction after removing
the original table file. The half config changes only the table and checkpoint
paths relative to the existing bypass config.

Continuation prompt for Claude:

> Read HANDOFF_REMOTE.md and STATUS.md. Preserve existing uncommitted work.
> Report the completed base-tok vs bypass interpretation results, then follow
> the agreed order: matched separate no-bypass AE, alpha=0.5 subtraction, then
> multiple resolutions. Keep STATUS.md current. These training jobs belong on
> Delta; interpretation/evaluation work runs on geoae-box.
