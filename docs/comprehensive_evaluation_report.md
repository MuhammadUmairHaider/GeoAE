# GeoAE: comprehensive evaluation report

**Results available as of 23 September 2026**  
**Primary comparison:** Llama-3.2-3B, layer 27; original residual stream versus dense autoencoders with 3,072, 6,144 and 12,288 latent coordinates.

## Executive assessment

The experiments establish that GeoAE can substantially improve reconstruction and organize the representation into better-separated, well-used clusters. They do **not** establish a general improvement in semantic disentanglement, downstream decoding, or selective causal control over the original residual stream.

The strongest positive editing result is grammatical number control: selecting the top 30% of coordinates and applying range-gated edits increases singular/plural forced-choice flips from **80.6% in the base representation to 95.1% in the 12,288-wide AE**. The 6,144-wide AE achieves 95.8%, so this is a replication at greater width, not evidence of a width-dependent improvement. These are grammatical-preference flips, not demonstrated factual-knowledge edits; full-vocabulary counterpart generation is much less successful.

| Question | Best-supported answer | Key evidence |
|---|---|---|
| Does width preserve the model better? | Yes, within the current width series. | Validation MSE falls from 0.09489 to 0.01927; reconstructed MMLU rises from 49.60% to 54.90%, versus 56.20% base. |
| Does width improve clustering geometry? | Yes on several major metrics. | Silhouette 0.0254 → 0.0502; Davies–Bouldin 3.181 → 2.702; centroid effective rank 180 → 290. |
| Does that yield better semantic clusters than a strong base control? | Not overall. | Mean 22-task NMI: d12k AE 0.2503, balanced base k-means 0.2557. Mean F1 is essentially tied, 0.2882 versus 0.2880. |
| Does the AE improve selective editing? | Dataset-dependent. | At matched suppression, range editing reduces DB14 collateral by 5.9 percentage points at d12k, but increases BiasBios collateral by 7.3 points. |
| Is information localized in many fewer AE coordinates? | Not supported by the corrected TPP experiment. | Half-effect budget: 44.49% of base coordinates versus 43.71% of AE coordinates; the AE uses more coordinates in absolute count. |
| Does supervised continuation recover useful semantics? | Mixed, with major design caveats. | POS/NER NMI improves, but mean NMI falls to 0.2234/0.2326 and IOI/RAVEL deteriorate. Actual resumed LR was 9.33× the configured LR. |

**Bottom line:** the defensible contribution is improved geometric organization and some task-specific control, with reconstruction improving strongly with width. A broad claim that AE coordinates are more semantic, more localized, or uniformly more causally selective than the original residual coordinates would overstate the evidence.

## 1. Scope, provenance and reading conventions

This report reconciles saved evaluation JSONs, checkpoint metadata, evaluation scripts, audit documents and Claude's local notes. It does not pool experiments with incompatible protocols. Corrected audits take precedence over preliminary claims, and saved current results take precedence over historical statements that a run was unfinished.

The primary models use a 3,072-dimensional residual stream, a dense GELU/BatchNorm AE, K = 2,000 clusters, DPC initialization/reseeding, batch size 32,768 and training seed 42. The current width checkpoints are epoch 50, step 14,200. “d12k” means **12,288**, not exactly 12,000. The 3,072-wide AE is square, while d6k and d12k are overcomplete. None is a sparse AE, and reconstruction uses the full dense latent vector, not a discrete cluster ID.

| Evaluation family | Base reference | d3k | d6k | d12k | Comparison constraint |
|---|---:|---:|---:|---:|---:|
| Reconstruction, MMLU, main geometry/concept probes | Unmodified h / balanced raw codebook | Epoch 50 | Epoch 50 | Epoch 50 | Main width series. |
| DB14 dense/range editing and BiasBios range editing | Within-run paired h interventions | Epoch 50 | Best-validation checkpoint, epoch 47 | Epoch 50 | Not a perfectly checkpoint-matched width ablation. |
| Top-30% number control | Same h arms for both widths | No completed result identified | Epoch 47 | Epoch 50 | Same held-out templates/nouns and reproduced base arms. |
| Audited TPP/BiasBios probes | Audited raw-h fixed probes | — | Audited DPC arm | — | Distinct fixed-readout experiments. |
| Supervised continuation | Unmodified model / balanced raw codebook | — | — | Epoch 55, step 15,620 | Two resumed runs; not a clean freeze-policy comparison. |
| Legacy linear/MLP probe suite | Corresponding raw-h probe | Separate d3k λ=2 PD arm | Some historical arms | — | Not the primary λ=1 width series. |

Main width checkpoints follow `checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d{3072,6144,12288}_dpc/step_0014200.pt`. The d6k editing checkpoint is `best_val.pt`, epoch 47, hash prefix `a6569378d383`. A generic `best_val.pt` is not necessarily a valid clustering checkpoint: d3k's historical best-validation checkpoint was epoch 10, before the clustering phase, and tanh's was epoch 5.

### Metrics and signs

Every model-result table includes an explicit base column. For clustering, the primary base is the **balanced raw-space k-means codebook**, not an AE; for predictive/causal metrics, base means the original model or a matched intervention on h. Parent-AE columns are not substitutes for base. **NR** means not reported in the saved artifact; **N/A** means no meaningful equivalent. Base reconstruction MSE is 0 by identity, not a separately trained reconstruction result. Geometry baselines repeated in initialization, tanh and continuation tables come from the current raw-space control in §3; they are reference measurements, not newly rerun historical controls.

- Accuracies and flip rates are shown as percentages where marked; **pp** means percentage-point difference. NMI, F1, MSE and geometric quantities otherwise remain on their native scales.
- “Concept probe” NMI/F1 measures agreement between cluster assignments and labels. It is **not** a trained linear classifier's accuracy. Cluster/class matching is descriptive, not an independently learned downstream readout.
- For editing, target drop is desirable suppression and complement drop is collateral damage. Define selectivity as **S = target drop − complement drop**, and report **ΔS = S(AE) − S(base)**, so positive is better for AE.
- Matched-suppression tables instead report **Δcollateral = collateral(AE) − collateral(base)**, so negative is better for AE.
- Stars reproduce exploratory, concept-paired tests: * p < .05; ** p < .01; *** p < .001. They are not multiple-comparison-corrected and do not represent independent AE training-seed replication.

## 2. Reconstruction and model preservation

| Metric | Base | d3k AE | d6k AE | d12k AE |
|---|---:|---:|---:|---:|
| Validation reconstruction MSE ↓ | 0 (identity) | 0.094892 | 0.034862 | 0.019269 |
| MMLU accuracy, 2,000 examples ↑ | 56.20% | 49.60% | 53.55% | 54.90% |
| MMLU change from base | 0 pp | −6.60 pp | −2.65 pp | −1.30 pp |

Width substantially improves reconstruction and downstream preservation. However, d12k still loses 1.30 pp on this MMLU evaluation. These results support “less destructive reconstruction,” not a demonstrated reasoning improvement over the unmodified model. The small continuation differences reported later should not be interpreted as established MMLU gains without paired uncertainty estimates.

Sources: checkpoint validation metadata; [d3k MMLU](../eval_out/mmlu_d3072_50.json), [d6k MMLU](../eval_out/mmlu_gelu50.json), [d12k MMLU](../eval_out/mmlu_d12288_50.json).

## 3. Clustering geometry and occupancy

The main geometry evaluation samples one million diverse token states (data-sampling seed 0). Silhouette and Dunn use smaller subsamples, approximately 10,000 and 5,000 states respectively.

| Metric | Base balanced KM | d3k | d6k | d12k |
|---|---:|---:|---:|---:|
| Silhouette ↑ | -0.02647 | 0.02538 | 0.03607 | 0.05022 |
| Davies–Bouldin ↓ | 4.4116 | 3.1810 | 2.9706 | 2.7019 |
| Calinski–Harabasz ↑ | 225.42 | 699.48 | 856.47 | 1,072.10 |
| Dunn index ↑, unstable subsampling | 0.03915 | 0.07109 | 0.15122 | 0.12119 |
| Separation ratio ↑ | 54.19 | 108.75 | 175.66 | 293.93 |
| Minimum centroid distance | 7.351 | 8.442 | 9.489 | 11.045 |
| Mean centroid distance | 45.668 | 65.949 | 82.538 | 106.933 |
| Intra-cluster variance | 0.71838 | 0.41820 | 0.27448 | 0.18580 |
| Centroid-matrix effective rank | 145.23 | 180.23 | 216.67 | 290.33 |
| Effective K | 2,000 | 1,995 | 1,997 | 1,997 |
| Empty clusters | 0 | 0 | 0 | 0 |
| Normalized balance entropy | 0.96929 | 0.96463 | 0.96617 | 0.96428 |
| Assignment entropy | NR | 0.000693 | 0.000495 | 0.000289 |

The geometry trend is strong across silhouette, Davies–Bouldin, Calinski–Harabasz and separation ratio. Effective rank here describes the **centroid matrix**, not the rank of all latent representations or the number of independent semantic features.

The canonical values above come from [the joint width geometry artifact](../eval_out/cq_d12288_50.json). An older d3k summary reports Dunn = 0.1343 rather than 0.0711. Silhouette/Dunn evaluation contains unseeded subsampling; the same d12k parent also has silhouette 0.04704 in the continuation comparison, versus 0.05022 here. Small changes, especially Dunn rankings, should not be overinterpreted. Raw distances are also sensitive to representation scale.

### Encoder-free base controls

| Metric | Base balanced KM | Base balanced DPC | Base plain KM |
|---|---:|---:|---:|
| Silhouette | −0.02647 | −0.03350 | −0.10292 |
| Davies–Bouldin | 4.4116 | 4.3334 | 2.1702 |
| Calinski–Harabasz | 225.42 | 225.75 | 106.13 |
| Centroid rank | 145.23 | 159.91 | 614.74 |
| Effective K | 2,000 | 1,999 | 309 |
| Empty | 0 | 0 | 342 |
| Balance entropy | 0.96929 | 0.96673 | 0.38840 |

Balanced base controls are essential: plain k-means has severely concentrated occupancy, making it an inadequate sole baseline. Its lower Davies–Bouldin score does not override the occupancy mismatch or imply better semantic organization. AE geometry is better separated than the balanced controls, but the next section shows that this does not automatically translate to better semantic labels.

Sources: [balanced k-means](../eval_out/cq_base_balanced_kmeans.json), [balanced DPC](../eval_out/cq_base_balanced_dpc.json), [plain k-means](../eval_out/cq_base_plain_kmeans.json).

## 4. Semantic cluster agreement: all 22 tasks

| Metric | Base balanced k-means | Base balanced DPC | Base plain k-means | d3k AE | d6k AE | d12k AE |
|---|---:|---:|---:|---:|---:|---:|
| Mean NMI ↑ | **0.2557** | 0.2394 | 0.1316 | 0.2441 | 0.2445 | 0.2503 |
| Mean cluster-label F1 ↑ | 0.2880 | 0.2828 | 0.2779 | 0.2780 | 0.2771 | 0.2882 |

These are unweighted averages over 22 heterogeneous tasks, not a single pooled predictive score. Width yields a modest mean NMI increase of 0.0062 from d3k to d12k. Balanced raw k-means remains ahead by 0.0054. Similar average F1 can coexist with very different NMI because the metrics measure different aspects of agreement and cluster/class matching.

### Per-task NMI

| Task | Base balanced KM | Base balanced DPC | Base plain KM | d3k | d6k | d12k |
|---|---:|---:|---:|---:|---:|---:|
| Surface | .2363 | .2371 | .1894 | .2716 | .2691 | .2400 |
| POS coarse | .3464 | .3453 | .2342 | .4276 | .4180 | .3620 |
| POS fine | .3954 | .3921 | .2668 | .4734 | .4654 | .4069 |
| NER coarse | .1314 | .1326 | .0976 | .1540 | .1525 | .1416 |
| NER fine | .1581 | .1606 | .0918 | .1799 | .1799 | .1726 |
| Sentiment | .0069 | .0030 | .0062 | .0034 | .0052 | .0038 |
| Sentiment, long | .0426 | .0449 | .0045 | .0562 | .0536 | .0542 |
| Subjectivity | .0352 | .0333 | .0198 | .0732 | .0618 | .0460 |
| Formality | .1197 | .1302 | .0384 | .1177 | .1193 | .1206 |
| Atlas document | .3770 | .3384 | .1002 | .3063 | .3124 | .3213 |
| Atlas tone | .2949 | .2705 | .0694 | .2464 | .2507 | .2746 |
| Atlas content | .3274 | .3048 | .0863 | .2761 | .2790 | .2958 |
| Language | .6234 | .6331 | .2866 | .6939 | .6700 | .7399 |
| Domain | .3410 | .3397 | .4393 | .3271 | .3229 | .3243 |
| Topic 4 | .0927 | .0910 | .0423 | .0756 | .1004 | .1169 |
| Topic 14 | .5688 | .4183 | .1407 | .4278 | .4430 | .4716 |
| Topic 20 | .2731 | .2705 | .0529 | .2722 | .2705 | .2761 |
| RAVEL country | .2901 | .2524 | .0370 | .2194 | .1956 | .2547 |
| RAVEL continent | .1501 | .1317 | .0078 | .0984 | .0922 | .1362 |
| RAVEL language | .2551 | .2028 | .0255 | .1835 | .1597 | .1994 |
| IOI role | .4154 | .3994 | .5822 | .3078 | .3585 | .3578 |
| IOI name | .1444 | .1341 | .0769 | .1785 | .1995 | .1897 |

Increasing width trades away some surface/POS/NER agreement while improving language, topic and several RAVEL scores. This is not a uniform semantic improvement. In particular, the balanced base exceeds d12k on Atlas document/content/tone, Topic 14 and all three RAVEL tasks, while d12k is stronger on language and several token-level tasks.

Sources: [width concept probes](../eval_out/probe_d12288_50.json), [base codebook probes](../eval_out/probe_d12288_50_base3.json).

### Local neighborhood label purity

| Task | Examples | Base h | d3k z | d6k z | d12k z |
|---|---:|---:|---:|---:|---:|
| POS coarse | 6,000 | .801 | .880 | .873 | .815 |
| POS fine | 6,000 | .817 | .860 | .849 | .806 |
| RAVEL country | 3,305 | .922 | .905 | .905 | .908 |
| Topic 14 | 3,529 | .967 | .950 | .940 | .930 |
| Language | 1,283 | .994 | .994 | .990 | .988 |

These k = 10 neighborhood diagnostics use eight classes per task. Better cluster geometry does not imply better local semantic neighborhoods: Topic 14 purity worsens with width, and the base is already strong on RAVEL and language. Base codebook variants share the same underlying h points, so their raw-space neighborhood scores are identical by construction. PCA/t-SNE figures are illustrative, not independent validation. Sources: [base/width geometry log](../logs/geom_d12288_50_base.log), [raw baseline log](../logs/geom_raw4.log).

## 5. Downstream linear and nonlinear decoding

These saved probe experiments primarily compare base h to a **separate d3k λ=2 PD checkpoint**, not the current d3k/d6k/d12k λ=1 series. Each aggregate averages probe seeds 42, 43 and 44; these are not three independently trained AEs. Features use train-only normalization; model selection uses validation loss. MLPs have a 512-unit GELU hidden layer and dropout 0.2. Multilabel thresholds are selected on validation micro-F1.

| Dataset / probe | Metric | Base | AE |
|---|---|---:|---:|
| DBpedia14 linear, 50k train / 10k test | Accuracy | .97777 | .97750 |
| | Macro-F1 | .97782 | .97752 |
| BiasBios linear, 50k / 20k | Accuracy | .82820 | .82377 |
| | Balanced accuracy | .75329 | .74979 |
| | Macro-F1 | .76898 | .76528 |
| BiasBios MLP | Accuracy | .83335 | .82827 |
| | Balanced accuracy | .75853 | .75003 |
| | Macro-F1 | .77519 | .76604 |
| FineWeb Atlas linear, 5,962 / 890, 92 labels | Micro-F1 | .72564 | .72529 |
| | Macro-F1 | .51294 | .50287 |
| | Micro average precision | .79383 | .80109 |
| | Macro average precision | .55265 | .56095 |
| FineWeb Atlas MLP | Micro-F1 | .73088 | .72626 |
| | Macro-F1 | .50786 | .50104 |
| | Micro average precision | .81735 | .81133 |
| | Macro average precision | .57206 | .55715 |
| GoEmotions linear, 43,410 / 5,427, 28 labels | Micro-F1 | .49464 | .48973 |
| | Macro-F1 | .35263 | .34027 |
| | Micro average precision | .49167 | .48272 |
| GoEmotions MLP | Micro-F1 | .49837 | .48512 |
| | Macro-F1 | .34869 | .32568 |
| | Micro average precision | .49906 | .48267 |

The AE generally preserves much of the decodable signal but provides no systematic gain. Atlas linear average precision is an exception, without a corresponding F1 improvement; the MLP does not preserve that advantage. GoEmotions is consistently worse in AE space.

Historical d6k controls likewise do not show a decoding advantage:

| Historical d6k probe | Base | Unsupervised AE | Atlas-seeded AE |
|---|---:|---:|---:|
| BiasBios linear accuracy, initial comparison | .82820 | .81628 | .81368 |
| BiasBios linear accuracy, best-val comparison | .82820 | .81635 | .81525 |
| BiasBios MLP accuracy, best-val comparison | .83335 | .82470 | .82148 |
| Atlas linear micro-F1 | .72564 | — | .71187 |
| Atlas linear macro-F1 | .51294 | — | .50890 |
| Atlas linear micro-AP | .79383 | — | .77071 |
| Atlas MLP micro-F1 | .73088 | — | .71500 |
| Atlas MLP macro-F1 | .50786 | — | .50065 |
| Atlas MLP micro-AP | .81735 | — | .79624 |

These are historical utility measurements, not substitutes for the later deduplicated BiasBios audit. Sources: [DB14 linear](../results/linear_probe_dbpedia14_l27_base_vs_k2000_d3072_pd.json), [BiasBios linear](../results/linear_probe_biasbios_profession_l27_base_vs_k2000_d3072_pd.json), [BiasBios MLP](../results/nonlinear_probe_biasbios_profession_l27_base_vs_k2000_d3072_pd.json), [Atlas linear](../results/linear_probe_fineweb_atlas_l27_base_vs_k2000_d3072_pd.json), [Atlas MLP](../results/nonlinear_probe_fineweb_atlas_l27_base_vs_k2000_d3072_pd.json), [GoEmotions linear](../results/linear_probe_goemotions_l27_base_vs_k2000_d3072_pd.json), [GoEmotions MLP](../results/nonlinear_probe_goemotions_l27_base_vs_k2000_d3072_pd.json).

## 6. Concept suppression and editing selectivity

### Protocol and fair interpretation

The current range experiments use d′ coordinate saliency, the top 30% of coordinates and τ = 2 range gating. They fit on 80 examples per concept and evaluate up to 50 target examples plus 80 complement examples. DB14 sweeps α = 0.5, 1, 2; BiasBios sweeps α = 1, 2. The classifier harness edits all token positions, although direction/range fitting uses final-position states. BiasBios excludes teacher because the model emits “professor,” leaving 27 concepts rather than the 28 used in the fixed-probe audit.

Each AE/base comparison is paired on examples where both corresponding baselines are correct. Those populations can differ across checkpoints. Consequently, within-run AE/base contrasts are stronger evidence than an apparent ordering of widths.

To compare collateral at similar suppression, the current summarizer selects, per concept, the smaller of the maximum target drops attained by h and z over the tested α grid, and interpolates collateral at that target drop. This is a **descriptive, adaptively selected operating point**, not a prespecified 80% suppression threshold. Sparse grids, non-monotone responses, duplicate target drops and weak overlap limit the interpolation. Earlier Claude tables sometimes use different matched subsets and should not be numerically merged with this table.

### Matched-suppression collateral difference

Base columns show actual matched collateral drop in percent; AE columns show Δcollateral in pp, so **negative favors AE**. Add the AE difference to its adjacent base value to obtain AE collateral. Each width retains its own paired base population and interpolation point; there is no single shared absolute baseline. All 14 DB14 and 27 BiasBios concepts are included in the current summary.

| Dataset | Edit mode | Base h, d3k run | d3k Δ | Base h, d6k run | d6k Δ | Base h, d12k run | d12k Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DB14 | Global | 61.0 | −2.9 | 59.2 | −1.8 | 61.6 | −3.9* |
|  | Salient coordinates | 52.4 | −2.4 | 50.3 | +0.3 | 52.1 | −0.7 |
|  | Range-gated | 41.5 | −5.0 | 43.8 | −5.3** | 44.6 | −5.9* |
|  | Transport | 45.8 | −2.8 | 46.6 | −3.4 | 47.7 | −4.5* |
| BiasBios | Global | 46.5 | +5.7** | 46.2 | +2.5** | 46.9 | +2.5** |
|  | Salient coordinates | 34.4 | +11.9*** | 33.7 | +4.5** | 36.7 | +5.9*** |
|  | Range-gated | 28.4 | +16.8*** | 31.4 | +5.7** | 31.9 | +7.3*** |
|  | Transport | 33.5 | +17.1*** | 37.5 | +4.7** | 38.3 | +4.7** |

The reversal is decisive for interpretation: a DB14 editing advantage does not generalize to occupations. d12k does not remove it. Increasing width from d3k substantially reduces some BiasBios disadvantages, but does not consistently outperform d6k.

### Fixed operating points and the value of range gating

At α = 2, range-gated mean target suppression is approximately 100% in both spaces:

| Dataset | Width | Base target / collateral drop | AE target / collateral drop |
|---|---|---:|---:|
| DB14 | d3k | 1.00 / .42 | 1.00 / .37 |
| | d6k | 1.00 / .44 | 1.00 / .39 |
| | d12k | 1.00 / .45 | 1.00 / .39 |
| BiasBios | d3k | 1.00 / .28 | 1.00 / .45 |
| | d6k | 1.00 / .31 | 1.00 / .37 |
| | d12k | 1.00 / .32 | 1.00 / .39 |

Range gating itself often helps **both** representations. In the d6k mean-replacement experiment, DB14 base collateral falls from .39 with full replacement to .17 with range-restricted replacement, while AE collateral falls from .33 to .12. Target drops remain .87→.85 for base and .91→.89 for AE. On BiasBios, collateral falls from .25 to .08 for base and .44 to .13 for AE; target drops are .93→.89 and .95→.90. Therefore, “range gating is useful” and “the AE is more selective” are separate claims.

### Dense DB14 steering across α

Base columns show actual selectivity S; adjacent AE columns show ΔS = AE − base. Thus AE S = base S + ΔS. Both use the consistent favorable sign:

| α | Base S, d3k run | d3k ΔS | Base S, d6k run | d6k ΔS | Base S, d12k run | d12k ΔS |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.4960 | +.1100 | 0.5094 | +.0221 | 0.5267 | +.0209 |
| 1 | 0.5941 | +.0195 | 0.5738 | +.0348* | 0.5637 | +.0723*** |
| 2 | 0.3893 | +.0330* | 0.4089 | +.0214 | 0.3830 | +.0393* |
| 4 | 0.1768 | +.0625* | 0.1938 | +.0339 | 0.1848 | +.0214 |
| 8 | 0.0027 | +.0196 | 0.0054 | +.0045 | 0.0054 | +.0080 |

The ordering depends on α. Fixed α is not a matched-effect comparison, and these numbers should not be used alone to infer a causal width benefit. The legacy dense-steering field has the opposite sign: the old log label “higher is better” was incorrect for that field.

### AG News: a small, mixed external check

For d6k, there are only four concepts, so uncertainty is large.

| Edit | Base target / collateral | AE target / collateral | ΔS ± SE |
|---|---:|---:|---:|
| Full complement-mean replacement | .750 / .5531 | .750 / .5844 | −.0312 ± .0081 |
| Range complement-mean replacement | .755 / .1688 | .750 / .1625 | +.0012 ± .0528 |
| Global α=1 | .785 / .1344 | .780 / .1500 | −.0206 ± .0361 |
| Range α=1 | .760 / .1437 | .755 / .1062 | +.0325 ± .0768 |
| Transport α=1 | .760 / .1531 | .765 / .1000 | +.0581 ± .0849 |
| Range α=2 | .830 / .1906 | .850 / .2188 | −.0081 ± .0306 |
| Transport α=2 | .840 / .2031 | .855 / .2125 | +.0056 ± .0324 |

Three of four concepts favor base for range/transport at α=1; the positive means are driven by Sci/Tech. This is not robust evidence of a general AE advantage.

Claude's exploratory cross-concept analysis found larger AE gains where base selectivity was weak: weakest/middle/strongest terciles approximately +.090 / +.018 / −.032. An independently computed base metric gave ρ≈−.30, p≈.049, but base-strength terms were not significant with dataset fixed effects (p≈.84/.24), whereas dataset effects were (p≈.026/.012). Dataset dependence is supported; a specific mechanism is not established.

### Reliability across widths and updated moderation check

The moderation table reports correlations of an AE-minus-base effect, so base ΔS is zero by definition; no separate meaningful base correlation or p-value exists.

The current width summarizer separately tests transport α=1 against base range-replacement selectivity estimated from the other width runs. This is not the same analysis as the historical Claude regression above.

| Width | Concepts | Base ΔS reference | Spearman ρ with base strength | p | Mean ΔS, weakest / middle / strongest tercile |
|---|---:|---:|---:|---:|---:|
| d3k | 41 | 0 (by definition) | −.69 | <.0001 | +.158 / −.031 / −.111 |
| d6k | 41 | 0 (by definition) | −.20 | .2000 | +.075 / −.004 / −.008 |
| d12k | 41 | 0 (by definition) | −.32 | .0386 | +.046 / +.024 / −.022 |

Within-dataset correlations for d12k are only −.20 on DB14 (p≈.49) and −.01 on BiasBios (p≈.96); the pooled association should not be interpreted as a demonstrated within-dataset mechanism. d3k shows a stronger within-dataset association, but also has the worst reconstruction and substantial BiasBios collateral.

The identity of concepts that benefit is not stable across widths. For transport α=1, correlations of per-concept AE-minus-base effects across d3k/d6k, d3k/d12k and d6k/d12k are .34/.47/.00 on DB14 and .12/−.04/.21 on BiasBios. Corresponding base-only selectivity correlations are much stronger: .93/.82/.80 and .78/.90/.95. With up to 50 target and 80 complement examples per concept, per-concept differences are noisy; dataset averages are more defensible than narratives about individual concepts.

Sources: [current width summarizer](../eval_out/summarize_width_series.py), [d12k DB14](../results/range_intervention_db14_d12288_50_dprime.json), [d12k BiasBios](../results/range_intervention_biasbios_d12288_50_dprime.json), [d6k AG News](../results/range_intervention_agnews_b32k_dpc_dprime.json), [range notes and corrections](notes/range-interventions-h-vs-z.md), [BiasBios reversal notes](notes/biasbios-range-reversal.md).

## 7. Completed top-30% number-control experiment

This is the experiment previously described conversationally as “knowledge flip.” Its actual test is whether an edit flips the model's grammatical preference between **“is” and “are.”** It does not test changing a factual association such as a country's capital.

The completed d12k run contains **144 intervention arms** and evaluates **144 held-out prompts**. The fit set uses 16 nouns and three templates, totaling 192 prompts; testing uses 12 different nouns and three different templates. Distractor number is balanced. Edits affect only the last prompt position. Both raw and reconstructed baselines get all 144 forced-choice cases correct.

Arms span h, z, three structured orthogonal rotations and shuffled z; four edit modes; α = .5, 1, 2; and singular/plural suppression directions. Strict flips require the original gold choice to fall below 50% pair probability; ties are excluded.

### Main result at α = 1

| Metric | Base h | d6k AE | d12k AE |
|---|---:|---:|---:|
| Range: strict pair flip rate | 80.56% | 95.83% | 95.14% |
| Range: AE improvement over base | — | +15.28 pp | +14.58 pp |
| Range: bootstrap 95% CI for improvement | — | [10.42, 19.44] pp | [10.42, 18.75] pp |
| Transport: strict pair flip rate | 81.94% | 95.83% | 95.83% |
| Transport: AE improvement over base | — | +13.89 pp | +13.89 pp |
| Transport: bootstrap 95% CI for improvement | — | [9.03, 18.75] pp | [10.42, 17.36] pp |
| Coordinates selected per direction | 922 | 1,843 | 3,686 |

The d12k raw-space arms reproduce the d6k raw-space results exactly. AE beats the mean of the three rotated controls by 8.80 pp for range editing (95% CI 5.56–12.04) and 8.33 pp for transport (5.09–11.57). Shuffled d12k controls produce zero strict flips across their arms. These controls support a structured, task-specific editing effect rather than arbitrary perturbation.

### Forced-choice success is not equivalent to successful generation

| Collateral / generation metric, α=1 | Base h | d6k AE | d12k AE |
|---|---:|---:|---:|
| Range: intended counterpart is vocabulary top-1 | 2.78% | 22.92% | 11.11% |
| Transport: intended counterpart is vocabulary top-1 | 2.78% | 20.83% | 13.19% |
| Range: neutral-prompt KL | .04595 | .08835 | .09272 |
| Transport: neutral-prompt KL | .04870 | .08444 | .09108 |
| Range: neutral top-1 change | 6.25% | 15.625% | 12.50% |
| Transport: neutral top-1 change | 12.50% | 21.875% | 12.50% |

Pair-restricted complement drop is zero for h/z, but that does **not** imply zero full-vocabulary collateral. The AE causes larger neutral-distribution KL than base. Neutral checks use only 16 prompts. Confidence intervals resample the 12 test nouns, not model seeds or a broad population of templates.

The coordinate budget is matched by **fraction**, not absolute number or induced residual-space perturbation norm. The d12k AE edits four times as many coordinates as base. Its full-vocabulary counterpart success is lower than d6k despite similar forced-choice flips, so increased width is not an unqualified improvement.

### d12k sweep: strict pair flips across all edit modes

| Mode | α | Base flip rate | d12k AE flip rate |
|---|---:|---:|---:|
| Global | .5 | 59.03% | 55.56% |
| | 1 | 96.53% | 96.53% |
| | 2 | 100.00% | 100.00% |
| Salient | .5 | 23.61% | 52.78% |
| | 1 | 86.11% | 96.53% |
| | 2 | 100.00% | 100.00% |
| Range | .5 | 18.06% | 47.92% |
| | 1 | 80.56% | 95.14% |
| | 2 | 100.00% | 100.00% |
| Transport | .5 | 18.75% | 52.08% |
| | 1 | 81.94% | 95.83% |
| | 2 | 100.00% | 100.00% |

The range/transport advantage is not a global-steering advantage: unrestricted base steering already reaches 96.53% at α=1, matching AE, and every listed mode reaches 100% at α=2. Stronger edits increase collateral; for example, range α=2 neutral KL is .22835 for base and .55590 for d12k AE. The appropriate claim concerns the behavior of a restricted editing method at a specified operating point, not an inability of base representations to flip number.

An earlier mean-absolute-difference selector produced a much larger apparent d6k range gap, 88.9% versus 13.9%. Replacing it with the corrected d′ selector raises the base result and reduces the AE advantage to about 15 pp. The selector is therefore a major experimental variable; the older 75 pp gap should not headline the finding.

Sources: [d12k raw results](../results/range_number_d12288_50_dprime.json), [d12k review and intervals](../eval_out/number_d12288_50_review.json), [d6k d′ results](../results/range_number_dpc_dprime.json), [canonical selector review](../eval_out/number_selector_review.json), [number-control protocol](number_control.md). The older `number_control_review.json` summarizes the legacy mean-absolute selector and is not the current comparison.

## 8. Corrected TPP: localization under a fixed readout

The audited DBpedia experiment deduplicates the 50k training cache, removes three duplicate training entries, uses a 10k test set with no train/test overlap, standardizes both representations using training statistics, ranks weight contrasts, and mean-replaces coordinates. The linear readout is held fixed; this is neither an LM intervention nor a retraining-based information-erasure test.

| Audited quantity | Base h | d6k AE z |
|---|---:|---:|
| Baseline test accuracy | .9787 | .9709 |
| Mean k50 coordinate count | 1,366.64 | 2,685.36 |
| Mean k50 fraction of width | .44487 | .43707 |
| Target accuracy at k50 | .47236 | .46889 |
| Complement drop at k50 | .003389 | .004017 |
| Validation-selected mean fraction | .44057 | .43634 |
| Test relative target drop at selected budget | .50252 | .50882 |
| Test complement drop at selected budget | .003096 | .004063 |
| Random-order trials reaching criterion | 69/70 | 68/70 |
| Median random fraction among reached trials | .98991 | .99495 |

The AE uses a slightly smaller fraction in 10 of 14 classes, but the mean advantage is only about **0.78 pp of width**, while the absolute number of coordinates is nearly twice as large. Random-order failures are censored, so reached-trial medians are not a universal “times more localized” statistic.

The original approximately 3.6× localization claim, based on phased counts around 336 versus 93, is invalidated by normalization, cohort and regularization issues. Intermediate corrected counts around 693 versus 677 were from another protocol and must not be mixed with this final k50 table. Full-ablation legacy selectivity can be mathematically degenerate; the current operating-point measures are more informative.

**Conclusion:** these data show a small difference in susceptibility of a fixed classifier, not a large gain in localization and not proof that class information has been removed from the representation.

Sources: [audited TPP result](../results/tpp_dbpedia_dpc_audited.json), [audit summary](../eval_out/tpp_audit/full_run_review.json), [audit explanation](tpp_audit.md).

## 9. Corrected BiasBios gender/profession intervention

The audit removes train/test overlap and duplicate cache rows: 49,977 training examples, 19,981 test examples, and 13 shared hashes removed. This fixed-probe experiment includes all 28 professions.

| Baseline metric | Base h | d6k AE z |
|---|---:|---:|
| Profession accuracy | .82353 | .80702 |
| Profession balanced accuracy | .75440 | .73373 |
| Gender accuracy | .99074 | .98874 |
| Gender balanced accuracy | .99071 | .98870 |

Budgets below are selected on validation data. “Half” halves gender balanced accuracy's excess over chance; “chance” targets chance-level decoding.

| Selector | Target | Metric | Base h | d6k AE z |
|---|---:|---:|---:|---:|
| Weight | Half | Count | 2,611 | 4,669 |
| Weight | Half | Fraction | .84993 | .75993 |
| Weight | Half | Test gender balanced acc. | .74127 | .74743 |
| Weight | Half | Test profession acc. | .58390 | .60007 |
| Weight | Half | Profession change | −.23963 | −.20695 |
| Weight | Chance | Count | 2,903 | 5,744 |
| Weight | Chance | Fraction | .94499 | .93490 |
| Weight | Chance | Test gender balanced acc. | .50702 | .50802 |
| Weight | Chance | Test profession acc. | .39683 | .38401 |
| Weight | Chance | Profession change | −.42671 | −.42300 |
| d′ | Half | Count | 1,720 | 2,795 |
| d′ | Half | Fraction | .55990 | .45492 |
| d′ | Half | Test gender balanced acc. | .73335 | .74915 |
| d′ | Half | Test profession acc. | .74966 | .72784 |
| d′ | Half | Profession change | −.07387 | −.07918 |
| d′ | Chance | Count | 2,703 | 5,283 |
| d′ | Chance | Fraction | .87988 | .85986 |
| d′ | Chance | Test gender balanced acc. | .50583 | .51023 |
| d′ | Chance | Test profession acc. | .51919 | .42130 |
| d′ | Chance | Profession change | −.30434 | −.38572 |

AE sometimes needs a smaller fraction, but a larger absolute count, and does not consistently retain profession performance better. With d′ at the chance operating point, profession loss is worse in AE space. Selector choice matters substantially in both spaces. These results do not establish fairness, demographic-information removal or a general disentanglement advantage.

Sources: [audited BiasBios result](../results/biasbios_dpc_audited.json), [cache audit](../eval_out/biasbios_cache_audit.json), [audit explanation](biasbios_audit.md).

## 10. Initialization and nonlinearity ablations

### Initialization/reseeding arms, d6k

| Metric | Base h / balanced KM | k-means++ | Atlas-seeded | DPC |
|---|---:|---:|---:|---:|
| Mean 22-task NMI | .25570 | .23710 | .25506 | .24591 |
| Mean cluster-label F1 | .28803 | .27269 | .27676 | .28238 |
| Silhouette | -0.02647 | .03017 | .03276 | .03360 |
| Centroid effective rank | 145.23 | 161.80 | 201.11 | 216.51 |
| Effective K | 2,000 | 1,970 | 1,992 | 1,997 |
| Balance entropy | 0.96929 | .94181 | .95121 | .96629 |
| Total recorded reinitializations | N/A | 2,394 | 1,341 | 260 |
| MMLU, best-validation checkpoint | 56.20% | 54.05% | 53.15% | 53.80% |
| DB14 dense ΔS, α=1 | 0 (ΔS reference) | +.032 | +.034 | +.035 |
| Corresponding paired p | N/A | .012 | .032 | .046 |

The initialization table’s base ΔS is zero by definition, not an absolute selectivity score. Actual base S at α=1 is .5657 / .5868 / .5738 for the k-means++ / seeded / DPC paired runs respectively (legacy summary precision).

DPC improves occupancy stability and geometric rank, but its dense-editing advantage over k-means++ is only about .003 ± .014 under the historical comparison. Initialization and reseeding both changed, so this is not an isolated initialization effect. Atlas seeding has the highest mean NMI among these AE arms, yet still does not outperform the balanced raw baseline's .2557 mean NMI or .2880 F1.

A coarse K=14 DB14 diagnostic reported accuracies base .6823, seeded .5575, k-means++ .4997 and DPC .4946. It is a different diagnostic, not the K=2,000 concept-probe score.

Sources: [initialization probes](../eval_out/probe_init_arms.json), [initialization geometry](../eval_out/cq_init_arms.json), [Claude's initialization analysis](notes/init-arms-dpc-results.md), [DPC notes](notes/dpc-density-peaks-init.md).

### Tanh versus GELU at epoch 50, d6k

| Metric | Base h / balanced KM | GELU | Tanh |
|---|---:|---:|---:|
| Validation MSE | 0 (identity) | .0349 | .0485 |
| MMLU | 56.20% | 53.55% | 52.90% |
| Mean concept NMI | .25570 | .24451 | .23334 |
| Mean cluster-label F1 | .28803 | .27712 | .26935 |
| Silhouette | -0.02647 | .03607 | .01406 |
| Davies–Bouldin | 4.41158 | 2.97055 | 3.62533 |
| Dunn, unstable | 0.03915 | .15122 | .18877 |
| Centroid effective rank | 145.23 | 216.67 | 124.72 |
| Effective K | 2,000 | 1,997 | 1,991 |

Tanh worsens reconstruction, semantic agreement and most geometry metrics. Dense DB14 steering nevertheless gives ΔS≈+.075 ± .021 at α=1, p≈.003, with 13/14 concepts favoring AE. This does not translate into a robust matched-suppression range advantage in the historical mean-absolute-saliency protocol. A corrected d′ tanh range run was not identified.

The ablation also changes `var_gamma` to .5, so it is not activation-only. Tanh was not simply saturated: mean |z|≈.479 and no measured activations above .99 magnitude in the cited diagnostic. Claude's later matched-operating-point correction supersedes an earlier “collapse” interpretation.

Sources: [tanh geometry](../eval_out/cq_tanh50.json), [tanh concept probe](../eval_out/probe_tanh50.json), [tanh MMLU](../eval_out/mmlu_tanh50.json), [corrected tanh notes](notes/tanh-encoder-arm.md).

## 11. Human-readable concepts and LLM judges

The main judge comparison samples 200 clusters per representation and uses Gemini Flash Lite, GPT-4o mini and DeepSeek 3.1. Judge scores are useful descriptive evidence, but do not directly establish causal semantics.

| Metric | Base balanced | DPC AE |
|---|---:|---:|
| Semantic coherence, Gemini | 4.385 | 4.490 |
| Semantic coherence, GPT | 4.480 | 4.480 |
| Semantic coherence, DeepSeek | 4.695 | 4.665 |
| Mean coherence | 4.520 | 4.545 |
| Intrusion accuracy, Gemini | .3317 | .2667 |
| Intrusion accuracy, GPT | .3150 | .3083 |
| Intrusion accuracy, DeepSeek | .3883 | .3650 |
| Mean intrusion accuracy | .3450 | .3133 |
| Mean monosemanticity score | 4.270 | 4.393 |
| Consensus intrusion pass rate | .580 | .565 |
| Kendall W, coherence | .4707 | .4479 |
| Kendall W, intrusion | .6726 | .6372 |

Coherence is close to the scale ceiling in both spaces, while intrusion is worse for AE. The higher monosemanticity score can be inflated by repeated token strings. Intrusion chance is 1/6; the reported agreement null W≈.3343 provides additional context. An older 120-cluster comparison has different valid sample counts and is not pooled here.

The historical full single-judge evaluation similarly reports token-weighted intrusion around .314 AE versus .363 balanced at layer 27, and .338 versus .422 at layer 14. A later word-sense diagnostic in Claude's handoff reports n=93, AE−base≈−.026 ± .007 and AE wins on about 30%; it is secondary, historical evidence rather than a current-width experiment.

Nearest-token inspection contains striking lexical groups: cluster 352 has 95 “is” tokens among 103 assigned states; cluster 1603 has 85 “are” tokens among 181; cluster 1482 has 134 “have” and 52 “has” among 201. These are states observed **after** tokens, not proof that the same clusters causally encode a pre-answer grammatical decision. Lexical coherence and pre-answer causal control are distinct pieces of evidence.

Sources: [three-judge results](../results/judge_agreement_base_vs_dpc.json), [full judge results](../results/llm_judge_full_ae_vs_balanced.json), [nearest tokens](../results/closest_tokens_dpc_by_freq.json), [judge notes](notes/llm-judge-autointerp.md), [historical HPC handoff](../HANDOFF_HPC.md).

## 12. Morphological steering: completed legacy experiments

These are older phased-checkpoint experiments, not the current DPC width series. Full saved evaluations supersede the early morphology smoke-test note.

### Past-tense steering

The fit/test split uses 105/106 verbs, 523 held-out positions and 219 control positions. The decoded AE direction has cosine .9788 with the corresponding base direction.

| α | Base pair flip | AE pair flip | Base target hit | AE target hit |
|---|---:|---:|---:|---:|
| 1 | .3384 | .3442 | .2543 | .2734 |
| 2 | .9694 | .9751 | .7419 | .7667 |

Perplexity is 5.518 base and 5.692 reconstructed before steering. At α=1 it is 6.913 for base edits and 7.174 for AE edits; at α=2, 16.097 and 16.622. At α=4 it rises to roughly 758 for both.

### Plural steering

There are 99 fit and 99 held-out words, with 1,477 held-out positions. Direction cosine is .9725.

| α | Base pair flip | AE pair flip | Base target hit | AE target hit |
|---|---:|---:|---:|---:|
| 1 | .2119 | .2133 | .1503 | .1550 |
| 2 | .6865 | .6987 | .4665 | .4895 |

Perplexity at α=1 is 6.135/6.302 for base/AE, and at α=2 is 8.199/8.392. These are small target-success gains accompanied by language-model disruption. Nearly collinear decoded directions are consistent with a largely reparameterized base direction, not a uniquely AE-specific control mechanism. Position-level standard errors do not account fully for shared lemmas/documents, and sparse target-position interventions differ from the global interventions used in the perplexity check.

Sources: [past-tense full results](../results_morph_l27_balance_phased_ep39.json), [plural full results](../results_morph_plural_l27_balance_phased.json), [earlier morphology notes](notes/morph-steering-experiment.md).

## 13. Reconstruction perplexity, hard quantization and exploratory causal diagnostics

### Historical perplexity and quantization

| Artifact / condition | Base PPL | Reconstructed PPL | AE hard-centroid PPL | k-means hard-centroid PPL |
|---|---:|---:|---:|---:|
| L27 b32k | 6.5365 | 6.7886 | 32,371.97 | 90,971.61 |
| L14 seeded | 6.5365 | 23.3296 | 939.36 | 1,062.78 |
| Separate L27 reconstruction evaluation | 6.6728 | 6.9362 | — | — |
| Separate L14 k-means++ evaluation | 6.6728 | 28.6432 | — | — |
| Separate L14 seeded-v2 evaluation | 6.6728 | 22.4084 | — | — |

Continuous reconstruction and hard cluster replacement behave very differently. Lower AE quantization PPL than k-means does not make quantization usable: both can severely damage the model. Layer 14 reconstruction is much more damaging in these historical runs than layer 27.

The historical `inference_benchmark` sliding-window scorer has a known target-coverage issue that can skip/repeat scored targets. These numbers should be treated as implementation-limited historical diagnostics, not a gold-standard PPL benchmark or a directly matched comparison to current MMLU.

Sources: [L27 quantization](../results/ppl_l27_b32k.json), [L14 quantization](../results/ppl_l14_seeded.json), [separate L27 reconstruction](../results/ppl_recon_L27_b32k.json), [independent code review](independent-code-review.md).

### Direct logit attribution and cluster distinctness

| Metric, 400k sampled states | Base balanced KM | Base plain KM | KL | b32k k-means++ | Atlas-seeded |
|---|---:|---:|---:|---:|---:|
| Live clusters | 2,000 | 1,275 | 2,000 | 2,000 | 2,000 |
| Distinct at cosine .95 | 1,977 | 1,273 | 1,969 | 1,803 | 1,815 |

The raw plain-codebook run has 725 dead clusters in this sample. This is a different sample/protocol from the current geometry table, so its dead count is not expected to match. Balanced raw k-means is already highly distinct by this criterion. The DLA implementation's “exact” designation is too strong because the state-dependent RMSNorm denominator is omitted.

### Exploratory faithfulness ablation

| Metric | Base balanced raw | KL | b32k k-means++ | Atlas-seeded |
|---|---:|---:|---:|---:|
| Baseline accuracy | .4825 | .4825 | .4825 | .4825 |
| Accuracy after edit | .3150 | .3475 | .3225 | .3350 |
| Confidence drop | .40522 | .40581 | .40542 | .40520 |
| Active fraction | .9975 | 1.0000 | 1.0000 | 1.0000 |
| Selected concepts | 6 | 5 | 5 | 4 |

These 400-example evaluations use different selected concepts across arms, nearly always-active interventions and first-token class confidence. They do not yield a clean comparative causal-faithfulness claim. Sources are the saved `results/dla_*.json` and `results/faith_*.json` families; implementation caveats are documented in the [code review](independent-code-review.md).

## 14. d12k recovery experiments: codebook refits and supervised continuation

### Tier 0: refit the codebook without changing the AE

| Metric | Balanced base k-means | Original d12k | Refit latent DPC | Refit latent seeded |
|---|---:|---:|---:|---:|
| Mean NMI | .25570 | .25027 | .25077 | .27212 |
| Mean cluster-label F1 | .28803 | .28824 | .27590 | .26083 |

Seeded latent refitting raises NMI above the balanced base but lowers F1. Topic 14 NMI is .4971, versus .4716 for the parent and .5688 for balanced base. Atlas content rises to .4067, versus .2958 parent and .3274 balanced base. Surface is nearly unchanged at .2405 versus .2400. This shows that codebook fitting substantially affects the semantic diagnostic, without establishing a better representation or better editing. The unchanged AE/decoder implies no reconstruction-behavior improvement from the codebook refit alone.

Source: [Tier-0 latent probe](../eval_out/probe_tier0_latent.json).

### Five-epoch supervised continuation

| Metric | Base h / balanced KM | Parent d12k, ep50 | Encoder-only continuation, ep55 | Full continuation, ep55 |
|---|---:|---:|---:|---:|
| Validation MSE | 0 (identity) | .019269 | .017492 | .018347 |
| MMLU | 56.20% | 54.90% | 55.20% | 54.55% |
| MMLU difference from base | 0 pp | −1.30 pp | −1.00 pp | −1.65 pp |
| Mean concept NMI | .25570 | .25027 | .22337 | .23256 |
| Mean cluster-label F1 | .28803 | .28824 | .28478 | .29109 |
| Silhouette, continuation comparison | -0.02647 | .04704 | .04934 | .05108 |
| Davies–Bouldin | 4.41158 | 2.70192 | 2.69927 | 2.69256 |
| Calinski–Harabasz | 225.42 | 1,072.10 | 1,082.84 | 1,084.91 |
| Centroid effective rank | 145.23 | 290.33 | 298.99 | 299.08 |
| Effective K | 2,000 | 1,997 | 1,992 | 1,996 |

The supervised runs improve several token-label scores while reducing average NMI and substantially damaging some relational scores:

| Task NMI | Base balanced KM | Parent | Encoder-only | Full |
|---|---:|---:|---:|---:|
| Surface | 0.2363 | .2400 | .2691 | .2654 |
| POS coarse | 0.3464 | .3620 | .4152 | .4095 |
| POS fine | 0.3954 | .4069 | .4528 | .4483 |
| NER coarse | 0.1314 | .1416 | .1525 | .1524 |
| NER fine | 0.1581 | .1726 | .1859 | .1855 |
| Sentiment | 0.0069 | .0038 | .0039 | .0069 |
| Sentiment, long | 0.0426 | .0542 | .0515 | .0552 |
| Subjectivity | 0.0352 | .0460 | .0529 | .0752 |
| Formality | 0.1197 | .1206 | .1175 | .1216 |
| Atlas document | 0.3770 | .3213 | .3209 | .3282 |
| Atlas tone | 0.2949 | .2746 | .2638 | .2644 |
| Atlas content | 0.3274 | .2958 | .2907 | .2891 |
| Language | 0.6234 | .7399 | .7007 | .7071 |
| Domain | 0.3410 | .3243 | .3260 | .3259 |
| Topic 4 | 0.0927 | .1169 | .0863 | .0985 |
| Topic 14 | 0.5688 | .4716 | .4696 | .4725 |
| Topic 20 | 0.2731 | .2761 | .2774 | .2755 |
| RAVEL country | 0.2901 | .2547 | .1783 | .1565 |
| RAVEL continent | 0.1501 | .1362 | .0794 | .0716 |
| RAVEL language | 0.2551 | .1994 | .1410 | .1180 |
| IOI role | 0.4154 | .3578 | .0297 | .1010 |
| IOI name | 0.1444 | .1897 | .0491 | .1880 |

### Important continuation caveats

1. **Actual learning rate differs from the requested configuration.** Both configs specify 3×10⁻⁵, but checkpoint optimizer state contains 2.8×10⁻⁴, inherited on resume: **9.33× higher**. Conclusions about a carefully controlled low-LR continuation are therefore unwarranted.
2. **Supervised labels differ between arms.** Encoder-only includes surface, coarse/fine POS, coarse/fine NER, IOI role and IOI name; full continuation omits IOI role. Freeze policy is not the only treatment difference.
3. **Training overlap limits interpretation.** The evaluation runner explicitly notes that supervised rungs use training data. `--exclude_anchors` removes initial anchor rows but does not implement every held-out class/row from the fine-tuning supervision manifest. POS/NER improvements are fit diagnostics, not proven held-out generalization.
4. Both use λ_sup=.02, supervised fraction .1 and configured .2 class/.2 row holdouts. The encoder-only log records 132,287 supervision rows and 150 classes, with 38 held out, but those declarations do not by themselves prove evaluation disjointness.
5. There is no matched unsupervised five-epoch continuation control, and the separation schedule is recomputed. Extra training, scheduling, supervision and optimizer-state effects are not disentangled.
6. No completed continuation-specific number-control or causal-editing results were identified. The separate enriched-data fine-tuning document is a proposal, not another completed result.

Sources: [encoder-only probe](../eval_out/probe_ft_enc.json), [full probe](../eval_out/probe_ft_full.json), [encoder-only geometry](../eval_out/cq_ft_enc.json), [full geometry](../eval_out/cq_ft_full.json), [encoder-only MMLU](../eval_out/mmlu_ft_enc.json), [full MMLU](../eval_out/mmlu_ft_full.json), [runner](../eval_out/run_supervised_tiers.sh), [encoder-only log](../logs/ft_sup_enc_d12288.log), [full log](../logs/ft_sup_full_d12288.log), embedded checkpoint optimizer/config state. See also the distinct [enriched-data proposal](enriched-finetuning-plan.md).

## 15. Earlier layers and cross-model evidence

These results broaden coverage but are historical and heterogeneous. They must not be presented as a single controlled scaling curve.

### Llama layer 14

| Metric | Balanced base | Seeded-v2 | k-means++ AE |
|---|---:|---:|---:|
| Token-level mean NMI | .2617 | .2841 | .2665 |
| Token-level mean F1 | .2442 | .2226 | .2190 |
| Sequence-level mean NMI | .1609 | .1543 | .1518 |
| Token-weighted intrusion | .422 | .349 | .338 |
| Cluster-usage perplexity | 1,585 | 1,243 | 1,492 |
| Validation MSE | 0 (identity) | .02878 | .02990 |

Seeding improves token NMI without improving sequence NMI or judge intrusion over balanced base. Strong layer-14 reconstruction PPL damage further limits usefulness. No completed sub-hidden-width d768 experiment was identified, so these runs do not establish a bottleneck benefit. Source: [scaling handoff, section 5.6](../HANDOFF_SCALING_GEMMA_QWEN.md) and its linked layer-14 artifacts.

### Gemma-3 12B, layer 47: corrected live-centroid geometry

| Metric | Base raw k-means | KL, MSE weight .05 | KL, MSE weight .15 | KL, MSE weight .45 | MSE-only |
|---|---:|---:|---:|---:|---:|
| Silhouette | −.12750 | −.05350 | −.05392 | −.04890 | −.03718 |
| Davies–Bouldin | 2.0564 | 5.8606 | 5.7250 | 5.8402 | 4.6747 |
| Calinski–Harabasz | 20.36 | 119.60 | 98.58 | 85.28 | 98.94 |
| Dunn | .07980 | .08162 | .12097 | .14038 | .10768 |
| Centroid rank | 806.43 | 58.86 | 127.65 | 161.70 | 432.12 |
| Effective K | 270 | 1,985 | 1,920 | 1,932 | 1,995 |
| Empty | 684 | 0 | 14 | 10 | 0 |

MSE-only has the best silhouette and Davies–Bouldin among these AEs, but does **not** win every normalized metric: KL .05 has higher Calinski–Harabasz and KL .45 higher Dunn. This narrows an overbroad claim in the historical note. MSE-only validation MSE≈.00596 and fraction of variance explained≈.994 are promising reconstruction measurements, but it trained on approximately 4M rows versus 9.33M for KL. The raw k-means run was underfit, with an early 15/1,171 iteration status, so it is not a strong cross-method benchmark.

Use the live-centroid artifact rather than the older dead-centroid calculation. A directory labeled `mse030` actually used λ=.45. Gemma activations reached 158,720, above float16's maximum 65,504; valid activation extraction requires avoiding that overflow, using float32 in these runs.

Source: [corrected Gemma geometry](../clustering_quality_gemma3_l47_livecentroids.json), [Gemma MSE/KL notes](notes/gemma3-l47-mse-vs-kl.md), [setup caveats](notes/gemma3-12b-setup.md).

### Historical Gemma dense steering

The ΔS column uses the favorable AE-minus-base convention at α=1. These are separate datasets/checkpoints, not a common operating-point benchmark.

| Model / layer | Dataset / arm | Base S | AE S | ΔS | Base PPL | Reconstruction PPL |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4B / L22 | DB14, MSE ep39 | 0.7555 | 0.7778 | +.0223 | 5.316 | 5.886 |
| Gemma 4B / L33 | DB14, Zipf ep48 | 0.6871 | 0.6815 | −.0056 | 5.316 | 5.339 |
| Gemma 4B / L33 | AG News, Zipf ep48 | 0.8625 | 0.8556 | −.0069 | — | — |
| Gemma 4B / L33 | Emotions, MSE ep42 | 0.7417 | 0.7401 | −.0016 | 5.316 | 5.362 |
| Gemma 12B / L47 | DB14, MSE ep36 | 0.6904 | 0.6953 | +.0049 | 3.437 | 3.446 |
| Gemma 12B / L47 | DB14, KL/VICReg | 0.7075 | 0.7170 | +.0095 | 3.437 | 3.600 |
| Gemma 12B / L47 | AG News, KL MSE .45 ep50 | 0.7269 | 0.7669 | +.0400 | 3.437 | 3.507 |

There is no consistent advantage across these settings. Sources are the root `results_steering_*.json` family and the [cross-model handoff](../HANDOFF_SCALING_GEMMA_QWEN.md). No saved aggregate Qwen evaluation supporting a quantitative conclusion was identified. Configured Qwen pipelines and a proposed GemmaScope SAE comparison are not completed evidence.

### Other legacy aggregate concept probes

The older `concept_probe_llama.json` uses 14 tasks, not the current 22-task suite:

| Historical comparison arm | Base balanced NMI | Arm NMI | Base balanced F1 | Arm F1 |
|---|---:|---:|---:|---:|
| AE BN-hinge | .23966 | .23043 | .30851 | .28905 |
| AE wide | .23966 | .23794 | .30851 | .28823 |
| AE K256 | .23966 | .17361 | .30851 | .29702 |
| E2E KL λ=.05 | .23966 | .25243 | .30851 | .34146 |
| MSE λ=2 median | .23966 | .22480 | .30851 | .26594 |
| MSE λ=2 wide | .23966 | .22438 | .30851 | .27046 |
| Plain base | .23966 | .13636 | .30851 | .32404 |

Do not directly rank these averages against the current width table. Likewise, the open root file [clustering_quality_comparison.json](../clustering_quality_comparison.json) is an older comparison with insufficient embedded provenance to identify it as the current width series; its positive AE silhouettes are not the current canonical values. Source: [legacy concept probe](../concept_probe_llama.json).

## 16. Reconciliation with Claude's notes

Consulted sources include [MEMORY](notes/MEMORY.md), all detailed notes under `docs/notes/`, [HANDOFF_DELTA](../HANDOFF_DELTA.md), [HANDOFF_HPC](../HANDOFF_HPC.md), [the scaling handoff](../HANDOFF_SCALING_GEMMA_QWEN.md), and the independent code/audit documents. The notes provide valuable protocol history but contain successive corrections. This report follows the latest validated result rather than averaging contradictory claims.

| Historical statement or pitfall | Treatment in this report |
|---|---|
| d12k has not run | Superseded: width, editing, number control and two continuation result sets exist. |
| TPP demonstrates a ≈3.6× localization win | Invalidated by audit; final fractional budgets are approximately .445 versus .437. |
| Number-control AE advantage is ≈75 pp | Legacy selector; corrected d′ comparison is ≈14–15 pp at α=1. |
| Range editing is invalid / AE adds nothing | Earlier blanket language is superseded by corrected d′ runs; DB14 benefit and BiasBios reversal are both retained. |
| Tanh range behavior proves collapse | Superseded by matched-effect correction; also a different saliency protocol. |
| Morphology consists only of a small smoke test | Superseded by full past-tense/plural result artifacts. |
| DPC alone causes the initialization gain | Initialization and reseeding changed together. |
| `best_val.pt` is always a cluster-trained checkpoint | False for some arms; checkpoint epoch/phase must be checked. |
| Large training-time silhouette/rank implies held-out cluster quality | Training Sinkhorn assignments and pointwise evaluation use different objects. |
| Dense-steering z−h selectivity is higher-is-better | Legacy field sign is opposite; converted here to a consistent favorable ΔS. |
| Gemma MSE-only wins all normalized geometry metrics | Too broad; silhouette/DB improve among AEs, while some KL arms win CH/Dunn. |
| Supervised continuation used LR=3e−5 | Configured intent only; saved optimizer LR is 2.8e−4. |

The notes also document transient large-batch collapse followed by recovery, and a Zipf-balancing null result whose apparent phase improvement was entangled with VICReg. These are training diagnostics, not extra independent evaluation wins. See [large-batch notes](notes/b32k-transient-collapse.md), [Zipf notes](notes/zipf-balancing-null-result.md) and [training-metric caveats](notes/train-diag-metrics-misleading.md).

## 17. What is established, what remains open

### Supported by the current evidence

- Wider dense AEs reconstruct better and yield better-separated, well-used codebooks in the main layer-27 series.
- Balancing the raw-space codebook explains a substantial part of the apparent semantic improvement over plain k-means.
- Range restriction can improve editing specificity in both raw and latent coordinates.
- AE editing is better on some DB14 and number-agreement measures, but worse on important BiasBios measures.
- The completed d12k number-control run reproduces the strong d6k forced-choice effect without a convincing width improvement.
- Supervised/codebook interventions change diagnostic cluster alignment, but gains in one metric or task can accompany losses elsewhere.

### Not established

- Uniformly more semantic or monosemantic AE features than a balanced base codebook.
- General factual-knowledge editing, information erasure, demographic fairness or mechanistic disentanglement.
- A large reduction in the fraction of coordinates needed for the corrected TPP effect.
- A causal width benefit for selectivity under matched examples, checkpoint epoch, norm and absolute coordinate budget.
- Out-of-sample generalization from the supervised continuation's token-task gains.
- Robustness across independent AE training seeds, or broad transfer to Qwen / GemmaScope SAEs.

There is a basic mechanistic reason for caution: with a linear decoder, a latent edit decodes into a residual-space direction. Overcomplete AEs can also contain decoder-null directions. Better coordinate selection can still be useful, but it does not by itself show that the AE has discovered a new causal variable. A residual-preserving edit, `h + decode(z_edited) − decode(z)`, would separate the edit from baseline reconstruction error; that control is proposed, not established by the current results.

Additional limitations include row-level rather than document-level activation splits, potentially shared adjacent prefixes, unresolved source-corpus overlap for some historical evaluations, free-form classifier output/label-format mismatches, and first-token confidence approximations. The unit of uncertainty differs across analyses: concepts for steering, nouns for number control, probe seeds for decoding, and usually only one AE training seed. None can silently substitute for another.

### Highest-value next comparisons, not run for this report

1. Finish a common epoch-50 editing comparison on the same example intersection, including d6k, with prespecified suppression targets and denser α sweeps.
2. Separate range-gating gains from AE gains using matched absolute coordinate counts, fractions and induced residual perturbation norms; include residual-preserving and rotation controls.
3. Extend number control to more independently held-out nouns/templates and neutral prompts, with full-vocabulary success and collateral as co-primary metrics. Add an actual factual-editing task before using “knowledge editing” terminology.
4. Repair and rerun supervised continuation with verified optimizer LR, identical supervised tasks, disjoint evaluation manifests, and an unsupervised continuation control.
5. Replicate central claims across AE seeds and report paired confidence intervals/multiplicity handling before making broad statistical claims.

## Appendix A. Full fixed-operating-point range selectivity summary

Base columns show actual S on the paired evaluation population for each width. AE columns show ΔS = AE − base; positive favors AE, and AE S = base S + ΔS. These include target suppression and collateral simultaneously and therefore differ from the matched-collateral table. `comp` is complement-mean replacement; `zero` is zero replacement. Values are proportions, not percentages.

### DB14

| Operation | Base S, d3k run | d3k ΔS | Base S, d6k run | d6k ΔS | Base S, d12k run | d12k ΔS |
|---|---:|---:|---:|---:|---:|---:|
| Full replacement, comp | 0.461 | +.128 | 0.478 | +.103* | 0.497 | +.089** |
| Full replacement, zero | 0.559 | −.480*** | 0.566 | −.481*** | 0.590 | −.347** |
| Range replacement, comp | 0.646 | +.150 | 0.676 | +.089* | 0.678 | +.066* |
| Range replacement, zero | 0.605 | +.090 | 0.604 | +.086 | 0.634 | +.104 |
| Global α=.5 | 0.498 | +.105 | 0.508 | +.019 | 0.528 | +.017 |
| Global α=1 | 0.590 | +.026 | 0.570 | +.036* | 0.565 | +.069*** |
| Global α=2 | 0.390 | +.029 | 0.408 | +.018 | 0.384 | +.039* |
| Range α=.5 | 0.491 | +.017 | 0.526 | −.045 | 0.512 | +.032 |
| Range α=1 | 0.592 | +.142 | 0.593 | +.105* | 0.619 | +.071** |
| Range α=2 | 0.578 | +.053* | 0.562 | +.053** | 0.554 | +.059* |
| Salient α=.5 | 0.475 | +.010 | 0.494 | −.059 | 0.500 | +.034 |
| Salient α=1 | 0.553 | +.099 | 0.544 | +.055 | 0.579 | +.061* |
| Salient α=2 | 0.476 | +.024 | 0.497 | −.003 | 0.479 | +.007 |
| Transport α=.5 | 0.466 | +.035 | 0.488 | −.022 | 0.488 | +.045 |
| Transport α=1 | 0.554 | +.146 | 0.555 | +.094 | 0.581 | +.092** |
| Transport α=2 | 0.539 | +.026 | 0.530 | +.032 | 0.521 | +.043* |

### BiasBios

| Operation | Base S, d3k run | d3k ΔS | Base S, d6k run | d6k ΔS | Base S, d12k run | d12k ΔS |
|---|---:|---:|---:|---:|---:|---:|
| Full replacement, comp | 0.676 | −.378*** | 0.679 | −.174*** | 0.661 | −.116*** |
| Full replacement, zero | 0.675 | −.589*** | 0.652 | −.367*** | 0.640 | −.321*** |
| Range replacement, comp | 0.805 | −.058 | 0.804 | −.033* | 0.805 | −.026* |
| Range replacement, zero | 0.734 | −.152* | 0.738 | −.102 | 0.722 | −.003 |
| Global α=1 | 0.808 | −.114*** | 0.780 | −.025* | 0.766 | −.022*** |
| Global α=2 | 0.535 | −.057** | 0.538 | −.025** | 0.531 | −.025** |
| Range α=1 | 0.772 | −.060* | 0.766 | −.014 | 0.754 | −.014 |
| Range α=2 | 0.714 | −.168*** | 0.681 | −.056** | 0.678 | −.072*** |
| Salient α=1 | 0.769 | −.045 | 0.756 | +.012 | 0.749 | −.013 |
| Salient α=2 | 0.655 | −.119*** | 0.660 | −.045** | 0.631 | −.059*** |
| Transport α=1 | 0.746 | −.063** | 0.734 | −.015 | 0.729 | −.022 |
| Transport α=2 | 0.662 | −.170*** | 0.620 | −.046** | 0.614 | −.047** |

## Appendix B. Artifact map and coverage boundaries

This report covers the substantive evaluation families found in the repository, prioritizing full/current runs over smoke tests and duplicates. It is a synthesis of aggregate findings, not a reproduction of every prompt, cluster or intermediate checkpoint row. Machine-readable files retain the per-example/per-concept details.

| Family | Authoritative artifacts / location |
|---|---|
| Main widths | [Geometry](../eval_out/cq_d12288_50.json), [concept probes](../eval_out/probe_d12288_50.json), `eval_out/mmlu_{d3072_50,gelu50,d12288_50}.json` |
| Raw controls | `eval_out/cq_base_*.json`, [three-base concept comparison](../eval_out/probe_d12288_50_base3.json) |
| DB14 range, d3k / d6k / d12k | [d3k](../results/range_intervention_db14_d3072_50_dprime.json), [d6k](../results/range_intervention_db14_b32k_dpc_dprime.json), [d12k](../results/range_intervention_db14_d12288_50_dprime.json) |
| BiasBios range, d3k / d6k / d12k | [d3k](../results/range_intervention_biasbios_d3072_50_dprime.json), [d6k](../results/range_intervention_biasbios_b32k_dpc_dprime.json), [d12k](../results/range_intervention_biasbios_d12288_50_dprime.json) |
| Dense DB14 steering | [d3k](../results/steer_db14_d3072_50.json), [d6k](../results/steer_db14_dpc.json), [d12k](../results/steer_db14_d12288_50.json) |
| Number agreement | [d6k review](../eval_out/number_selector_review.json), [d12k review](../eval_out/number_d12288_50_review.json), [protocol](number_control.md) |
| Corrected TPP / BiasBios | [TPP audit](tpp_audit.md), [BiasBios audit](biasbios_audit.md), corresponding `results/*_audited.json` |
| Linear/MLP utility | `results/{linear,nonlinear}_probe_*.json`; main legacy d3k files linked in §5 |
| Initializers / tanh | `eval_out/{cq,probe}_init_arms.json`, `eval_out/{cq,probe,mmlu}_tanh50.json` |
| Judges / lexical examples | [three-judge comparison](../results/judge_agreement_base_vs_dpc.json), [token exemplars](../results/closest_tokens_dpc_by_freq.json) |
| Morphology | [past tense](../results_morph_l27_balance_phased_ep39.json), [plural](../results_morph_plural_l27_balance_phased.json) |
| DLA / faithfulness / PPL | `results/dla_*.json`, `results/faith_*.json`, `results/ppl_*.json` |
| d12k recovery | [Tier 0](../eval_out/probe_tier0_latent.json), `eval_out/{cq,probe,mmlu}_ft_{enc,full}.json` |
| Cross-model history | [scaling handoff](../HANDOFF_SCALING_GEMMA_QWEN.md), [Gemma corrected geometry](../clustering_quality_gemma3_l47_livecentroids.json), root `results_steering_*.json` |
| Audit / historical interpretation | [code review](independent-code-review.md), [Claude note index](notes/MEMORY.md), root handoffs |

Incomplete reruns, smoke tests, proposed enriched-data training, Qwen configurations, proposed SAE comparisons, broken legacy generality scores and invalidated localization estimates are not promoted to completed findings. No new training or evaluation run was launched to prepare this report.
