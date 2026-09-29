# DB14 steering — Gemma 3 12B, layer 47: MSE-only vs KL vs base residual stream

Every run reports BOTH `h sel` (steering the raw residual stream = base representation)
and `z sel` (steering the AE latent), on the same docs, in the same pass. So the base
comparison is built in — there is no separate baseline run to launch.

## State of play (checked 2026-08-21)

| variant | checkpoint | λ_mse | prior steering run | usable? |
|---|---|---|---|---|
| kl-lm0.05 | `e2e/.../kl_k2000_vicreg_10M/best_val.pt` (ep38) | 0.05 | ep26 ckpt, Aug 13 | NO — different ckpt (ae_sha b7bca9f36344 ≠ 38ab9952b5d4) |
| kl-lm0.15 | `e2e/.../kl_k2000_vic
reg_10M_mse015/best_val.pt` | 0.15 | none | — |
| kl-lm0.45 | `e2e/.../kl_k2000_vicreg_10M_mse030/best_val.pt` (ep50) | 0.45 | **died after 6/14 concepts**, no json | NO — rerun |
| mse-only  | `checkpoints/gemma3-12B/layer47/mse_only_k2000/best_val.pt` (ep47) | — (MSE is the recon term) | ep36 snapshot, Aug 19 | NO — stale ckpt (31fd89d4dd21 ≠ 284c37b02eb1) |

Note the dir named `_mse030` actually has `lambda_mse: 0.45` — hence the `mse045` filenames.

Only kl-lm0.45's joint-correct cache is still valid, so that run skips the ~15 min
rebuild. The other three rebuild theirs (the set is filtered by each AE's own
reconstruction, so it is per-checkpoint by design).

## Core pair — run these two

```bash
# 1. MSE-only (ep47, the new run)
uv run python -u -m geoae.interp.steering_concept_compare \
  --checkpoint checkpoints/gemma3-12B/layer47/mse_only_k2000/best_val.pt \
  --dataset db14 --layer 47 \
  --correct_json dbpedia/joint_correct_db14_gemma3_l47_mse_only_ep47.json \
  --out results_steering_db14_gemma3_l47_mse_only_ep47.json \
  2>&1 | tee logs/steering_db14_gemma3_l47_mse_only_ep47.log

# 2. KL control, lambda_mse 0.45 (ep50) — joint-correct cache already valid
uv run python -u -m geoae.interp.steering_concept_compare \
  --checkpoint e2e/checkpoints/general/gemma3-12B/layer47/kl_k2000_vicreg_10M_mse030/best_val.pt \
  --dataset db14 --layer 47 \
  --correct_json dbpedia/joint_correct_db14_gemma3_l47_mse045_ep50.json \
  --out results_steering_db14_gemma3_l47_kl_lm045_ep50.json \
  2>&1 | tee logs/steering_db14_gemma3_l47_kl_lm045_ep50_rerun.log
```

## Optional — the other two KL lambdas, if you want the sweep

```bash
uv run python -u -m geoae.interp.steering_concept_compare \
  --checkpoint e2e/checkpoints/general/gemma3-12B/layer47/kl_k2000_vicreg_10M/best_val.pt \
  --dataset db14 --layer 47 \
  --correct_json dbpedia/joint_correct_db14_gemma3_l47_kl_lm005_ep38.json \
  --out results_steering_db14_gemma3_l47_kl_lm005_ep38.json \
  2>&1 | tee logs/steering_db14_gemma3_l47_kl_lm005_ep38.log

uv run python -u -m geoae.interp.steering_concept_compare \
  --checkpoint e2e/checkpoints/general/gemma3-12B/layer47/kl_k2000_vicreg_10M_mse015/best_val.pt \
  --dataset db14 --layer 47 \
  --correct_json dbpedia/joint_correct_db14_gemma3_l47_kl_lm015.json \
  --out results_steering_db14_gemma3_l47_kl_lm015.json \
  2>&1 | tee logs/steering_db14_gemma3_l47_kl_lm015.log
```

Defaults used (match the Aug 17/19 runs): `--n_fit 80 --n_eval 50 --n_comp 80
--alphas 0.5 1.0 2.0 4.0 8.0 --seed 42` → 1120 fit docs, 700 eval docs, 14 concepts.

## Caveat to carry into the writeup

`h sel` is not identical across runs: the joint-correct doc set is filtered by each
AE's own reconstruction, so each run evaluates on a slightly different doc subset.
Within a run, h-vs-z is a clean paired comparison; across runs, treat the h column as
a per-run reference point rather than a fixed constant.
