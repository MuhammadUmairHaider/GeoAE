
from geoae.paths import resolve_input_path
import os
from pathlib import Path
ROOT = Path(os.environ.get("GEOAE_ROOT", Path(__file__).resolve().parents[3]))
os.chdir(ROOT)
import json, numpy as np
d = json.load(open(resolve_input_path("results/delta/nosinkhorn/probe_dense_subsets.json")))
R = d["rungs"]; arms = ["res_balanced", "res_plain", "tok_plain", "geoae", "A_nosk_reinit", "B_nosk_lloyd"]
SEQ = ["sentiment", "sentiment_long", "subjectivity", "formality", "language", "domain", "topic4", "topic14", "topic20"]
ENT = ["ravel_country", "ravel_continent", "ravel_language", "ioi_role", "ioi_name"]
TOK = ["surface", "pos_coarse", "pos_fine", "ner_coarse", "ner_fine"]
subs = ["all", "min5", "min10", "top90", "top75", "top50", "common_min5"]
t = R["topic14"]
print("topic14 detail: kept% / clusters / cNMI / purity (shuffled)")
for s in subs:
    print(s, " | ".join(f"{a}: {100*t['arms'][a][s]['n']/t['n']:.0f}% {t['arms'][a][s]['clusters']} {t['arms'][a][s]['cnmi']:.3f} "
                        f"{t['arms'][a][s]['purity']:.2f}({t['arms'][a][s]['purity_null']:.2f})" for a in arms))
print()
for grp, name in [(SEQ, "seq9"), ([r for r in SEQ if r != "language"], "seq8"), (ENT, "ent5"), (TOK, "tok5")]:
    for s in subs:
        print(name, s, " ".join(f"{a}={np.mean([R[r]['arms'][a][s]['cnmi'] for r in grp]):.3f}" for a in arms),
              " kept% " + " ".join(f"{np.mean([100*R[r]['arms'][a][s]['n']/R[r]['n'] for r in grp]):.0f}" for a in arms))
print()
for s in ["all", "min5", "top50", "common_min5"]:
    print("== ", s)
    for r in SEQ + ENT:
        print(f"{r:<16}", " ".join(f"{R[r]['arms'][a][s]['cnmi']:.3f}({100*R[r]['arms'][a][s]['n']/R[r]['n']:.0f}%)" for a in arms))
