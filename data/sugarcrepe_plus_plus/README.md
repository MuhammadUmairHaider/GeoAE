# SugarCrepe++ text evaluation data

Downloaded from the authors' [official repository](https://github.com/Sri-Harsha/scpp),
at commit `c083bbf039a5ca215acaf25e7ee736d2ea6f3864`. This is the human-validated `data` directory,
not the automatically generated examples in `generated_data`.

The five original JSON files contain 4,757 triplets in total:

| Category | Triplets |
| --- | ---: |
| replace_att | 788 |
| replace_obj | 1,652 |
| replace_rel | 1,406 |
| swap_att | 666 |
| swap_obj | 245 |

Each record contains `id`, `filename`, `caption`, `caption2`, and
`negative_caption`. Use `caption` as the anchor, `caption2` as the paraphrase,
and `negative_caption` as the distractor. Images are unnecessary for the
text-only comparison and were not downloaded.

For the requested evaluation, L2-normalize each representation separately
and count a pass when `d(anchor, paraphrase) <= d(anchor, distractor)`.
This accepts exact distance ties. The paper's stricter text-only metric
requires the positive pair to be closer than **both** positive-negative
pairs, with ties failing; it is a separate metric.

Source URLs, immutable revision, record counts, and file SHA-256 checksums
are recorded in [provenance/manifest.json](provenance/manifest.json).

The authors publish the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Citation: Sri Harsha Dumpala, Aman Jaiswal, Chandramouli Sastry, Evangelos
Milios, Sageev Oore, and Hassan Sajjad (2024),
[SUGARCREPE++ Dataset: Vision-Language Model Sensitivity to Semantic and Lexical Alterations](https://arxiv.org/abs/2406.11171).
