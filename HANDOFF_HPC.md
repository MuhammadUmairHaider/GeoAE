# GeoAE — handoff for an HPC session (2026-09-03)

> **Superseded by `HANDOFF_DELTA.md` (2026-09-18).** The notes it refers to below
> are now in the repo at `docs/notes/` (index: `docs/notes/MEMORY.md`).

Llama-3.2-3B residual-stream autoencoder + Sinkhorn-balanced clustering. This
file is the state of play, the tooling built in the 2026-09-02/03 session, the
runs that are configured but not started, and the traps that cost the most time.

Read `~/.claude/projects/-home-exouser-RepresentationAE-GeoAE/memory/MEMORY.md`
too — it indexes longer notes on individual findings.

---

## 1. The one-paragraph state of play

The encoder is close to an invertible reparameterisation and does not beat an
**encoder-free control** on almost anything. The control is
`fit_balanced_kmeans` — identity encoder, same Sinkhorn balancing, same K.
Balancing is what transforms the clustering; the encoder mostly moves the
partition toward **token identity**, which helps POS-like rungs and hurts
everything contextual. Every run so far has `latent_dim >= hidden_size` and no
sparsity term, so the encoder has never been required to discard anything. The
untested variables are a real bottleneck and a stronger cluster weight.

---

## 2. Established findings (measured, replicated)

| finding | evidence |
|---|---|
| Encoder organises by **lexical identity**, control organises by **context** | top-token share per cluster 48.7% vs 39.5% on 169k tokens / 15,182 distinct strings |
| Encoder is near-identity | `W_dec@W_enc` cosine-to-I: b32k **0.985**, KL **0.974**, sq3072_lam2 0.774. FVE 0.969 |
| Concept **separation** improves, modestly | +8-10% mean over 16 rungs, 8-9/16 rungs, consistent across 5 variants and both widths |
| ...but it does not translate | corr(separation, NMI) = +0.449; corr(separation, F1) = +0.157 |
| Token vs sequence split is real | 4 of 5 L27 variants: TOKEN +12..+16%, SEQUENCE -3..+0% separation |
| **λ=2.0 at d=3072 is the best arm measured**, not the λ=1.0 one analysed most | sq3072_lam2 +10.0% sep, pos_coarse +55.7%; also the *least* identity-like (0.774) |
| Latent width is a **null** | 3072 and 6144 within noise on separation and concepts |
| Balanced k-means wins token-level semantics at BOTH depths | word intrusion, token-weighted: L27 AE 0.314 vs bal 0.363; L14 AE 0.338 vs bal 0.422 |
| Large batch (32768) is safe but costs usage spread | Zipf α 0.972 vs 0.528, perplexity 1175 vs 1488 vs the B=4096 runs |
| Depth trades grammar for entities | L27 RAVEL NMI -42/-66/-43%; L14 -3.5/-6.9/-1.8%. But L27 POS/NER gains vanish at L14 |

## 3. Refuted — do not re-derive these

- **AE beats control on semantic coherence** — retracted. The 1-5 scale is
  saturated (96% of clusters score 4 or 5 in every space). Differences live in
  the top 3% of the scale and did not survive a context-window change.
- **AE is better at word-sense disambiguation** — refuted. n=17 gave +0.020
  (1.9 SE); n=93 gave **-0.026 ± 0.007 (3.9 SE), AE wins only 30%**. Clustering
  by word identity lumps senses together; senses *are* contexts.
- **Separation gain is a √d artifact** — wrong, my error. The 3072 arms separate
  as well as or better than the 6144 arms. (The *metric* did need fixing: use a
  ratio of two d-dimensional distances so dimension cancels — see §4.)
- **Plain k-means has the most coherent clusters** — confound. It has 1099 live
  clusters vs ~1990, so its neighbours are farther apart and its "hard" negatives
  and intruders are easy. Never quote its judge numbers.
- **Structural "generality" score** — broken as written. `token_div` conflates
  "one repeated string" with "narrow concept", so it scores the `import` keyword
  cluster at 0.00. LLM-rated generality is also useless (65% of clusters get 3).
  If you need generality, use `domain_div × context_div` and drop `token_div`.

**Four results this session reversed under proper power. Do not report anything
below ~50 units without a sensitivity check at two thresholds.**

## 4. Tooling built this session

| file | what |
|---|---|
| `geoae/interp/llm_judge.py` | LLM-as-judge auto-interp over `closest_tokens` JSONs. Explain → detect vs hard negatives → null floor, plus **word intrusion** (Chang et al. 2009, chance 1/6) and split surface/semantic coherence. Disk-cached by hash(provider, model, prompt). `--report_from` re-renders saved runs with **no API calls**. |
| `geoae/seeded_init.py` | `centroid_init: "seeded"` — labelled anchors from concept caches + density×coverage fill; `AnchorPool` for `reinit_mode: "anchor"`. |
| `geoae/interp/cluster_generality.py` | Structural generality. **Known-broken, see §3.** |
| `geoae/interp/benchmark_cache.py` | Builds `ravel.npz` / `ioi.npz`. `concept_suite` does NOT build these. |

Modified: `concept_probe.py` now always prints `MEAN TOKEN` / `MEAN SEQUENCE`
rows; `fit_balanced_kmeans.py` k-means++ init is ~2× faster (was materialising a
2.4 GB temporary per iteration); `train.py`/`train_common.py`/`config.py` carry
the seeded-init and anchor-reinit wiring; `diagnostics.py`+`train.py` carry the
Q_history OOM fix.

### The separation metric (not yet in a module — inline it)

```python
# between-class distance / within-class RMS radius, BOTH d-dimensional so
# dimensionality cancels. Do NOT divide a d-dim norm by a per-dim scalar:
# that scales with sqrt(d) and makes a 6144-dim latent look 1.41x better.
Ms, ws = [], []
for c in classes_with_at_least_30:
    P = Z[y == c]; m = P.mean(0)
    Ms.append(m); ws.append((P - m).norm(dim=1).pow(2).mean().sqrt().item())
M = torch.stack(Ms); D = torch.cdist(M, M)
sep = D[~torch.eye(len(M), dtype=bool)].mean().item() / mean(ws)
```

## 5. Runs configured but NOT started

All six have distinct `checkpoints_dir` and validated headers stating their own
falsification criterion.

| config | one variable vs parent | why |
|---|---|---|
| `configs/base/..._lam1_d768.yaml` | latent 6144 → **768** | first sub-1.0 ratio ever trained. FVE must drop well below 0.9 or 768 is not binding (try 384) |
| `configs/base/..._lam1_d6144_cov01.yaml` | λ_cov 0.01 → **0.1** | cov measured 9.84 vs an achievable floor of 0.19 — 52× too weak. cov should fall below ~3 by epoch 20 |
| `configs/base/..._lam1_d3072_cov01.yaml` | as above at d=3072 | width control |
| `configs/base/..._lam1_d6144_seeded.yaml` | `centroid_init: seeded`, `reinit_mode: anchor` | k-means++ samples ∝ D², i.e. chases outliers; DBpedia-14 K=14 seeded from 5-10 labelled examples/class hit ~93% purity vs ~55% |
| `configs/e2e/.../kl_k8000_balance_phased.yaml` | K 2000 → **8000**, B scaled 4× so B/K stays 1.02 | head clusters are the grab-bags and hold 27-38% of tokens |
| `configs/e2e/.../kl_d768_balance_phased.yaml` | latent 3072 → **768** on the KL track | |

**Priority if you can only run some: `lam1_d768`, then `lam1_d6144_seeded`.**
Both test untested variables; the cov01 arms tune a knob on a losing arm.

**Consider λ=2.0 rather than λ=1.0 for the bottleneck** — `sq3072_lam2` is the
best-separating and least identity-like arm measured, and the d768 config
currently inherits λ=1.0 from its parent.

## 6. Eval pipeline

Dependencies run top to bottom. Nothing here needs the LM except steps 1 and 3.

```bash
# 0. ENCODER-FREE CONTROL — needed by every comparison. ~2 min.
#    NOTE: --activations takes the .npy FILE, not the directory (silent
#    IsADirectoryError after 3s otherwise), and --out is REQUIRED.
python -m geoae.interp.fit_balanced_kmeans \
  --checkpoint <ckpt.pt> --activations activations_diverse_10M/layer_27.npy \
  --n_clusters 2000 --out checkpoints/.../balanced_kmeans_k2000.npz

# 1. CONCEPT CACHE — per LAYER. cache/ is L27, cache_l14/ is L14.
python -m geoae.interp.concept_suite --checkpoint <ckpt.pt> --out cache_lXX
python -m geoae.interp.benchmark_cache --bench ravel --checkpoint <ckpt.pt> --out cache_lXX/ravel.npz
python -m geoae.interp.benchmark_cache --bench ioi   --checkpoint <ckpt.pt> --out cache_lXX/ioi.npz

# 2. CONCEPT LADDER — NMI + F1, auto-prints MEAN TOKEN / MEAN SEQUENCE
python -m geoae.interp.concept_probe --cache cache_lXX \
  --models "ae=<ckpt.pt>" --baselines "bal=<balanced.npz>" --out results/probe.json

# 3. CLUSTER CONTENT (needs the LM, ~20 min per arm)
python -m geoae.interp.closest_tokens --checkpoint <ckpt.pt> --out results/ct_ae.json
python -m geoae.interp.closest_tokens --baseline_kmeans <balanced.npz> --out results/ct_bal.json
python -m geoae.interp.rank_clusters results/ct_ae.json      # Zipf α, usage perplexity

# 4. GEOMETRY
python -m geoae.interp.clustering_quality --baseline <raw_kmeans.npz> \
  --checkpoints <ckpt.pt> --activations_dir activations_diverse_10M \
  --layer 27 --n_sample 1000000 --seed 0

# 5. LLM JUDGE (needs OPENROUTER_API_KEY in the environment)
python -m geoae.interp.llm_judge results/ct_ae.json results/ct_bal.json \
  --provider openrouter --model google/gemini-2.5-flash-lite \
  --n_clusters 2000 --min_assigned 0 --concurrency 16 --out results/judge.json
python -m geoae.interp.llm_judge --report_from results/judge.json   # free re-render
```

Full census ≈ 23k calls ≈ **$2-3**. Use `--dry_run` to cost it with no key.

### Which numbers to trust

1. **Word intrusion, token-weighted** — behavioural, chance 1/6, observed
   0.31-0.47, real headroom. This is the headline.
2. **NMI, split token vs sequence.**
3. **Separation** (§4) — real but correlates only +0.449 with NMI.
4. **F1** — best-single-cluster; at K=2000 over 35 classes recall is capped
   ~0.17, so it mostly measures K vs n_classes. Favours imbalanced partitions.
5. **Semantic coherence 1-5** — saturated, do not use as a headline.
6. **`cluster/dying`, `eff_K`, `cluster/silhouette` in the training log** — these
   come from the Sinkhorn `Q`, not from geometry. `dying 1326` was logged while
   real usage was healthy. For true usage read `ema_cluster_size` from a
   checkpoint, or run `rank_clusters`.

## 7. Environment traps that cost real time

- **Page cache is load-bearing.** `activations_*/layer_N.npy` is 56 GB. Random
  4 KB reads run at ~6.5 MB/s vs ~900 MB/s sequential — a 130× penalty that took
  one epoch from 8 min to **107 min**. Warm it first:
  `dd if=activations_diverse_10M/layer_27.npy of=/dev/null bs=8M`.
  Bulk `du`/`find` over other large trees evicts it; do not run them during
  training. On a shared HPC filesystem this is likely worse — stage to node-local
  scratch if you can.
- **Do NOT reclaim cgroup memory after warming** — it drops the pages you just
  warmed. (Learned the hard way.)
- **`Q_history` OOM is fixed** but check it survived any merge: `train.py` must
  append `out.Q.detach().mean(dim=0)`, not `out.Q.detach()`. The full form is
  262 MB/entry at B=32768 × 200 entries = **52 GB** and kills the run ~step 200.
- **`best_val.pt` is wrong for phased runs** — it selects on val_mse, which peaks
  before clustering starts. Use the last `step_*.pt`. (Exception: the KL
  `balance_phased` best_val is epoch 39 and *is* valid — check `epoch` and
  `ema_cluster_size` before assuming either way.)
- **Do not judge cluster health before epoch ~40.** The b32k run oscillated
  between healthy and 83%-of-tokens-in-one-cluster over epochs 15-30 and
  self-resolved by epoch 31 with no intervention.
- **Disk.** Checkpoints are 502 MB (d6144) / 240 MB (d3072) / ~76 MB (d768) and
  `save_every: 1`. A 50-epoch d6144 run is 25 GB. This box repeatedly hit 95%.
- Effective rank uses **squared** singular values; the unsquared version reports
  605.8 where the real figure is 165.0.
