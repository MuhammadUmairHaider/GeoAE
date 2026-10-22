"""Raw homogeneity / completeness / NMI per probe for chosen subsets of an geoae/evaluation/alignment.py file
(no chance correction; reporting rule: the three scores together, on rows in clusters with more than 5 rows in
every space = subset common_fp6), with rows and clusters kept.

    python experiments/paper/analysis/summarize_probe_raw_hc.py <probe_dense_subsets.json> common_fp6 all > probe_raw_hc.md
"""

from geoae.paths import resolve_input_path
import json
import sys

import numpy as np

NAMES = {"res_balanced": "Residual, balanced", "res_plain": "Residual k-means (plain)",
         "tok_plain": "Token-removed k-means (plain)", "geoae": "Bypass AE (GeoAE)", "nobypass": "No-bypass AE",
         "geoae_lplain": "Bypass latent, plain k-means", "nobypass_lplain": "No-bypass latent, plain k-means"}
TOK = ["surface", "pos_coarse", "pos_fine", "ner_coarse", "ner_fine"]
SEQ = ["sentiment", "sentiment_long", "subjectivity", "formality", "language", "domain", "topic4", "topic14", "topic20"]
ENT = ["ravel_country", "ravel_continent", "ravel_language", "ioi_role", "ioi_name"]
GROUPS = [("token-level (5)", TOK), ("sequence (9)", SEQ), ("sequence without language (8)", [r for r in SEQ if r != "language"]),
          ("entity/relation (5)", ENT)]
path, subs = sys.argv[1], sys.argv[2:] or ["common_fp6"]
R = json.load(open(resolve_input_path(path)))["rungs"]
ARMS = list(next(iter(R.values()))["arms"])
for sub in subs:
    print(f"\n## Subset `{sub}`: raw homogeneity / completeness / **NMI**\n")
    print("| Probe | rows kept | " + " | ".join(NAMES.get(a, a) for a in ARMS) + " |")
    print("|---|---|" + "---|" * len(ARMS))
    for grp, rungs in [("Token-level", TOK), ("Sequence-level", SEQ), ("Entity and relation", ENT)]:
        print(f"| **{grp}** | |" + " |" * len(ARMS))
        for r in rungs:
            if r not in R:
                continue
            x0 = R[r]["arms"][ARMS[0]][sub]
            cells = [f"{x['homogeneity']:.3f} / {x['completeness']:.3f} / **{x['nmi']:.3f}** ({x['clusters']} cl)"
                     for x in (R[r]["arms"][a][sub] for a in ARMS)]
            print(f"| {r} | {x0['n']:,} of {R[r]['n']:,} | " + " | ".join(cells) + " |")
    for lab, rungs in GROUPS:
        rungs = [r for r in rungs if r in R]
        cells = [" / ".join(f"{np.mean([R[r]['arms'][a][sub][k] for r in rungs]):.3f}"
                            for k in ("homogeneity", "completeness", "nmi")) for a in ARMS]
        print(f"| mean, {lab} | | " + " | ".join(cells) + " |")
