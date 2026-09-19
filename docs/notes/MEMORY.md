# GeoAE project memory

## Project goal
Learn a latent space (GeoAE on Llama-3.2-3B layer-27 residuals) that is geometrically
better separated AND semantically better organized than the base residual stream.

## Key paths
- Checkpoints: `e2e/checkpoints/general/llama3.2-3B/layer27/<run>/best_val.pt`
- Checkpoint dict keys: `model_state, opt_state, epoch, step, tau, norm_mean, norm_std, val_kl, val_mse, config`
- Loader: `geoae.checkpoint.load_ae_checkpoint(path, device)` → `(ae, _, _, ckpt_dict)`
- Baseline k-means npz: `.../layer27/kl_gelu/baseline_kmeans_k2000.npz`
- Diverse activations: `activations_diverse_10M/layer_27.npy` (10M rows; old `activations_diverse/` is gone)
- DBpedia acts: `dbpedia/activations/llama3.2-3B/last/unprompted/layer_27{,_test}.npy` + labels
- Comparison tool: `python -m geoae.interp.clustering_quality --baseline <npz> --checkpoints <pts...> --activations_dir activations_diverse_10M --layer 27 --n_sample 1000000 --seed 0`
- DBpedia eval: `python -m geoae.dbpedia.evaluate` (single-ckpt mode matches AE's K clusters to 14 classes — harsh; fair test is k-means K=14 on latents)

## Key results (2026-07, layer 27, K=2000)
- vicreg run (var hinge λ=0.3 + cov λ=0.001 + hard-EMA): fixed rank collapse (erank 37→233),
  no contraction, best silhouette/Dunn/min-centroid-dist of all spaces incl. raw. val_kl 0.101.
- cos run (cosine metric + uniformity λ=0.1): FAILED — directional collapse (erank 19.6,
  min centroid dist 0.11 on sphere, Q near-uniform). Dead end.
- DBpedia-14 k-means K=14 Hungarian acc: raw zscore 0.771 > old-AE l2 0.713 > vicreg zscore 0.644.
  No general AE beats raw on coarse semantics yet. (Earlier 0.93 was a DBpedia-trained AE, not comparable.)
- Raw k-means at K=2000 is degenerate: only 292 effective clusters, 275 empty — its
  separability ratio (120) is inflated by this.
- Next idea: vicreg recipe at K=256 (configs/e2e/general/llama3.2-3B/layer27/kl_gelu_k256_geo.yaml exists).

## Gotchas / fixes
- HF dataset `dbpedia_14` is dead → use `fancyzhx/dbpedia_14` (fixed in geoae/dbpedia/extract.py).
- effective_rank must center centroids before SVD (GELU latents non-negative → shared mean offset).
- AE eval labels must use `dist2.argmin` not Sinkhorn `Q.argmax` (Sinkhorn batch-balances by construction).
- Hard-EMA needs visited-mask so unvisited centroids don't decay to origin.
- variance hinge (std→1 per dim) inflates intra-cluster variance → DB/separability-ratio look worse;
  use scale-normalized metrics (silhouette, Dunn) for fair comparison.
- cluster_loss rewards latent contraction; adaptive-σ sep_loss is scale-invariant so can't resist it.

## User workflow preferences
- Uses tmux; no nohup needed. Long runs: plain `| tee` (stdout only) + `python -u` so tqdm bars stay on terminal.
- Prefers to run long GPU commands themselves — hand over copy-paste command sheets rather than launching them.
- Wants actionable, holistic analysis; concise tables with verdicts.

## Detailed memories
- [gemma3-12b-setup](gemma3-12b-setup.md) — Gemma 3 12B / Gemma Scope 2 track: Scope repo is SAEs not an LM, layers 12/24/31/41, disk math, the Gemma3ForConditionalGeneration decoder trap.
- [db14-kmeanspp-diagnosis](db14-kmeanspp-diagnosis.md) — why k-means++ fails vs semisup on DBpedia-14 layer 27; reproduced 55%; low-label sweep.
- [db14-eval-gotchas](db14-eval-gotchas.md) — db14 eval assignment (Q.argmax), Sinkhorn, collapse signature.
- [gemma3-l47-mse-vs-kl](gemma3-l47-mse-vs-kl.md) — MSE-only beats every KL lambda on scale-normalised clustering metrics at Gemma3-12B L47; near-identity caveat still open.
- [probe-tpp-gotchas](probe-tpp-gotchas.md) — TPP localization win is fully an artifact (mean-fill + converged probe -> geoae == base residual); zeroing weights == zeroing acts; L1=0.1 underfits probes; selectivity is degenerate.
- [train-diag-metrics-misleading](train-diag-metrics-misleading.md) — W&B cluster/silhouette, entropy, dominant, dying measure the Sinkhorn output on one batch, not geometry; correct silhouette flips the sign.
- [zipf-balancing-null-result](zipf-balancing-null-result.md) — balance:zipf did not move realized usage (alpha 0.528 vs 0.503 uniform); phased VICReg carries the gains.
- [morph-steering-experiment](morph-steering-experiment.md) — tense steering h vs z: AE clusters mix inflections yet cos(r_h,dec(r_z))=+0.965; alpha-unit and flip-vs-hit traps.
- [fineweb-atlas-concept-alignment](fineweb-atlas-concept-alignment.md) — 16,790 named concepts over FineWeb chunks; the GeoAE win over raw k-means is entirely the SINKHORN BALANCING, not the encoder (balanced k-means with no encoder matches it and beats it on content); plus the gemma fp16-overflow trap.
- [b32k-transient-collapse](b32k-transient-collapse.md) — B=32768 runs collapse then self-heal by epoch 40; final usage matches B=4096, so large batch is safe.
- [llm-judge-autointerp](llm-judge-autointerp.md) — LLM auto-interp judge; AE does NOT beat balanced k-means on semantic coherence; judge-choice and capacity confounds.
- [init-arms-dpc-results](init-arms-dpc-results.md) — density-peaks init: every b32k arm's z beats base-h steering at α=1 (p≤0.05), but dpc vs other arms is noise (eval docs rebuilt per AE → compare Δ(z−h)); dpc wins churn + token rungs, loses topic/sequence.
- [dpc-density-peaks-init](dpc-density-peaks-init.md) — density-peaks init: token-level win, topical loss, local-only (UMAP); encoder-free dpc control via fit_balanced_kmeans --init dpc --reinit peaks.
- [range-interventions-h-vs-z](range-interventions-h-vs-z.md) — NeuronLens ranges on L27 DB14: with d' saliency the AE beats base on precision at matched erasure (range a2 −0.053 p=.009); the earlier null and the z zero-replacement collapse were both mean-|a| artefacts; tanh d' run still pending.
- [tanh-encoder-arm](tanh-encoder-arm.md) — tanh encoder arm: best global steering of any arm (z−h −0.075, p=.003), but range interventions reverse in tanh z and effective rank halves; best_val is epoch 5, use step_0014200.
- [biasbios-range-reversal](biasbios-range-reversal.md) — bias_in_bios 27 professions: AE marginally worse than base (+0.03 collateral, p<=.015); AE margin shrinks as base quality rises (weak base +0.09, strong base −0.03) but a dataset effect remains.
- [d3072-width-arm](d3072-width-arm.md) — d3072 vs d6144 (ep50): narrower latent amplifies range effects both ways (DB14 better, bios worse) and steepens base-quality moderation (rho −0.77 vs −0.29), at 2.5x MMLU cost; best_val is ep10.
