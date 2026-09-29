# Delta round 2 — eval results

Llama-3.2-3B layer 27, token-bypass AE vs **base** (balanced k-means on the plain residual,
same K, no encoder). Every run was trained on the new sampled dump (`activations_sampled_10M`),
50 epochs, seed 42, single seed. Old-box results are in `archive/oldbox/`
and are not comparable run-for-run.

**Start with `summary.txt`** (the base vs bypass table for every run, plain units).
`summary.json` holds the same numbers for scripts.

## Layout

```
results/delta/
  summary.txt / summary.json      base vs bypass, all runs
  joint_correct/                  documents both the plain model and the AE-spliced model classify
                                  correctly (the pool the interventions edit), per run
  <run>/                          d768  d3072  d6144 (main arm)  d12288  d6144k4000 (d6144, K = 4000)
    evals/
      token_structure.json        reconstruction (FVE) and share of "pure current-token" clusters
      probe.json                  chance-corrected NMI of clusters vs labels (topic, sentiment, POS, ...)
      mmlu.json                   MMLU accuracy with the AE spliced into the model vs without
      cq.json                     clustering quality: silhouette, balance, effective K (ignore dunn)
    interventions/
      range_intervention_db14_dprime.json       remove / steer a DBpedia topic: target removed vs collateral
      range_intervention_biasbios_dprime.json   same on 27 bias_in_bios professions
      steer_db14.json                           plain steering on DBpedia topics
      range_number_dprime(_hb).json             singular/plural number control (hb = token-subtracted base)
      number_review.json                        paired summary of the number control
    steering/
      cluster_steering.json (+ _allhubs)        next-token transfer when a row is moved into a cluster
      cluster_steer_generate.json               steered generations (40 target clusters x 20 prompts)
      cluster_steer_judge.json                  LLM-judged: usable (fluent, on-target) generations per 100
    concept/
      concept_steer_generate.json               steering toward named DBpedia / bias_in_bios concepts
      concept_steer_judge.json                  LLM-judged usable per 100, per handle
                                                (label direction, base clusters, bypass clusters, ...)
    figures/                                    number-control review plots
    logs/<group>/                               one log per step
```

`h` in the intervention files = the base residual, `z` = the bypass latent.
d768 has no DBpedia / bias_in_bios intervention results: with it spliced in, the model gets
no documents right, so there is nothing to edit (MMLU 0.0 for the same reason).

## Commands

```bash
cd /projects/bbyl/mhaider/GeoAE
scripts/delta/py eval_out/summarize_delta.py > results/delta/summary.txt   # rebuild the summary
bash scripts/delta/organize_delta_results.sh                              # pull stray outputs into this tree
bash eval_out/run_delta_evals.sh <run> <group>                            # (re)run a group; FORCE=1 redoes
```
