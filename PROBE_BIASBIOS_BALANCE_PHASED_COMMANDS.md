# Probe (TPP) + BiasBios — llama3.2-3B L27, `kl_gelu_k2000_balance_phased`

Checkpoint: `e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/best_val.pt`
(epoch 39, step 177723, val_kl 0.0301, val_mse 0.0661, K=2000, GELU, latent 3072,
zipf balance ρ=0.6, semisup_cap 500)

Every run reports `raw` (z-scored layer-27 residuals) and `geoae` (encoder latents)
in the same pass, so the baseline comparison is built in.

Inputs verified present:
- `dbpedia/activations/llama3.2-3B/last/unprompted/{layer_27.npy,layer_27_test.npy,labels_*}` (50k/10k)
- `biasbios/activations/llama3.2-3B/last/{layer_27.npy,layer_27_test.npy,labels_profession_*,labels_gender_*}` (50k/20k, 28 professions)
- `dbpedia/joint_correct_db14_l27_balance_phased_ep39.json` (already built for this exact ckpt)

---

## 1. TPP probe — DBpedia-14

```bash
uv run python -u -m geoae.interp.probe_perturbation \
  --act_dir dbpedia/activations/llama3.2-3B/last/unprompted \
  --layer 27 \
  --ae_checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/best_val.pt \
  --spaces raw geoae \
  --out results/tpp_dbpedia_balance_phased_ep39.json \
  --plot_dir plots/tpp_balance_phased_ep39 \
  | tee logs/tpp_dbpedia_balance_phased_ep39.log
```

### 1b. Same, restricted to joint-correct docs (LLM + AE splice both right)

Uses the cache already built for this checkpoint — no rebuild, but it does re-download/
stream `fancyzhx/dbpedia_14` test to recover the index mapping.

```bash
uv run python -u -m geoae.interp.probe_perturbation \
  --act_dir dbpedia/activations/llama3.2-3B/last/unprompted \
  --layer 27 \
  --ae_checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/best_val.pt \
  --spaces raw geoae \
  --joint_correct dbpedia/joint_correct_db14_l27_balance_phased_ep39.json \
  --out results/tpp_dbpedia_balance_phased_ep39_jc.json \
  --plot_dir plots/tpp_balance_phased_ep39_jc \
  | tee logs/tpp_dbpedia_balance_phased_ep39_jc.log
```

Defaults in effect: `--lambda_l1 0.1 --lambda_l2 0.01 --probe_lr 1e-3 --probe_epochs 100
--patience 10 --n_steps 200 --n_random_trials 5 --seed 42`.
Prior comparable run for reference: `results/tpp_dbpedia_semisup.json` (same λ, same splits).

---

## 2. BiasBios dual probe (gender ⟂ profession)

Recommended settings — z-score the GELU latents (they're non-negative, so an unscaled
comparison against z-scored raw is unfair) and drop the probe bias term (otherwise the
class prior alone keeps gender above chance and `k@chance` is meaningless):

```bash
uv run python -u -m geoae.bias.probe \
  --act_dir biasbios/activations/llama3.2-3B/last \
  --layer 27 \
  --ae_checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/best_val.pt \
  --spaces raw geoae \
  --zscore_latents \
  --no_probe_bias \
  --out results/bias_probe_balance_phased_ep39.json \
  --plot_dir plots/bias_probe_balance_phased_ep39 \
  | tee logs/bias_probe_balance_phased_ep39.log
```

### 2b. Sparser gender probe (matches the `gl1_1_zscore` sweep point)

Higher L1 on the gender probe concentrates gender into fewer dims, which is the
setting where `gender k@chance` separates spaces most cleanly:

```bash
uv run python -u -m geoae.bias.probe \
  --act_dir biasbios/activations/llama3.2-3B/last \
  --layer 27 \
  --ae_checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/best_val.pt \
  --spaces raw geoae \
  --gender_lambda_l1 1.0 --prof_lambda_l1 0.1 --lambda_l2 0.01 \
  --zscore_latents \
  --out results/bias_probe_balance_phased_ep39_gl1_1.json \
  --plot_dir plots/bias_probe_balance_phased_ep39_gl1_1 \
  | tee logs/bias_probe_balance_phased_ep39_gl1_1.log
```

Comparable earlier runs: `results/bias_probe_zscore_nobias.json` (2a-style),
`results/bias_probe_gl1_1_zscore.json` (2b-style).

---

## 3. Smoke tests first (~1 min each, 1000 train / 20 steps / 20 epochs)

```bash
uv run python -m geoae.interp.probe_perturbation \
  --act_dir dbpedia/activations/llama3.2-3B/last/unprompted --layer 27 \
  --ae_checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/best_val.pt \
  --spaces raw geoae --smoke --out results/tpp_smoke_balance_phased.json

uv run python -m geoae.bias.probe \
  --act_dir biasbios/activations/llama3.2-3B/last --layer 27 \
  --ae_checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/best_val.pt \
  --spaces raw geoae --zscore_latents --no_probe_bias --smoke \
  --out results/bias_probe_smoke_balance_phased.json
```

---

## What to read out

| metric | where | good direction |
|---|---|---|
| probe accuracy (raw vs geoae) | `spaces.*.{profession_probe_accuracy,probe_accuracy}` | geoae ≥ raw = no info lost |
| `k@50%` / `k@chance` | `summary` | **lower** for geoae = concept more localized |
| `gender_k_at_chance` | bias `summary` | lower = gender concentrated in fewer latents |
| `profession_delta_at_gender_kch` | bias `summary` | ≥ 0 = debiasing is surgical, profession survives |
| `dim_overlap_at_100` | bias `summary` | lower = gender/profession disentangled |
| `debiasing_efficiency` | bias `summary` | higher = more gender removed per unit profession lost |

Caveat carried over from the geometry work: raw is z-scored per-dim while geoae latents
are GELU-sparse, so always compare `k` as a **fraction of dims** (both are 3072-d here,
so the raw counts happen to be directly comparable too).
