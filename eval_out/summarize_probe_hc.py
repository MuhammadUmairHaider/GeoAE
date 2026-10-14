"""Homogeneity / completeness / NMI per probe for one subset of results/delta/nosinkhorn/probe_dense_subsets.json.

    python eval_out/summarize_probe_hc.py [file.json] min10 common_min10 > results/delta/nosinkhorn/probe_homogeneity_min10.md
"""
import json
import sys

import numpy as np

NAMES = {"res_balanced": "Residual, balanced", "res_plain": "Residual, plain", "tok_plain": "Token-removed, plain",
         "geoae": "GeoAE", "A_nosk_reinit": "A", "B_nosk_lloyd": "B", "nobypass": "No-bypass AE",
         "nobypass_partial": "No-bypass AE (partial)"}
argv = sys.argv[1:]
path = "results/delta/nosinkhorn/probe_dense_subsets.json"
if argv and argv[0].endswith(".json"):
    path, argv = argv[0], argv[1:]
d = json.load(open(path))
R = d["rungs"]
ARMS = [(a, NAMES.get(a, a)) for a in next(iter(R.values()))["arms"]]
TOK = ["surface", "pos_coarse", "pos_fine", "ner_coarse", "ner_fine"]
SEQ = ["sentiment", "sentiment_long", "subjectivity", "formality", "language", "domain", "topic4", "topic14", "topic20"]
ENT = ["ravel_country", "ravel_continent", "ravel_language", "ioi_role", "ioi_name"]
GROUPS = [("mean of 5 token-level", TOK), ("mean of 9 sequence", SEQ),
          ("mean of 8 sequence (no language)", [r for r in SEQ if r != "language"]), ("mean of 5 entity/relation", ENT)]
for sub in argv or ["min10"]:
    print(f"\n## Subset `{sub}`: chance-corrected homogeneity / completeness / NMI (raw NMI in brackets; rows kept)\n")
    print("| Probe | " + " | ".join(n for _, n in ARMS) + " |")
    print("|---|" + "---|" * len(ARMS))
    for grp, rungs in [("Token-level", TOK), ("Sequence-level", SEQ), ("Entity and relation", ENT)]:
        print(f"| **{grp}** |" + " |" * len(ARMS))
        for r in rungs:
            cells = []
            for a, _ in ARMS:
                x = R[r]["arms"][a][sub]
                cells.append(f"{x['homogeneity_c']:.3f} / {x['completeness_c']:.3f} / **{x['cnmi']:.3f}** "
                             f"({x['nmi']:.3f}; {100 * x['n'] / R[r]['n']:.0f}%)")
            print(f"| {r} | " + " | ".join(cells) + " |")
    for lab, rungs in GROUPS:
        cells = [" / ".join(f"{np.mean([R[r]['arms'][a][sub][k] for r in rungs]):.3f}"
                            for k in ("homogeneity_c", "completeness_c", "cnmi")) for a, _ in ARMS]
        print(f"| {lab} | " + " | ".join(cells) + " |")
