"""Raw confidence levels behind the control tables: the model's probability of the correct label (first token, %),
before and after each edit, on the target concept's documents and on the other-class documents, in both spaces.

base  = the edit applied in the model's own residual stream; "before" is the unedited model.
GeoAE = the edit applied in the GeoAE latent and decoded back; "before" is the model with the GeoAE splice, unedited.
Per concept: mean over its target docs / its other-class docs, then a 10%-trimmed mean over concepts (each column
trimmed separately). Reads range_intervention_<ds>_dprime_conf.json (per-doc records).
Usage: python eval_out/paper_control_conf_levels.py [--trim 0.1]"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--trim", type=float, default=0.10)
args = ap.parse_args()
ARMS = [("d6144", "Llama-3.2-3B"), ("g4b_l33_d5120", "Gemma 3 4B"), ("qwen25_l35_d4096", "Qwen2.5-3B")]
DSN = [("db14", "DBpedia-14"), ("biasbios", "Bias in Bios"), ("ag_news", "AG News")]
OPS = [("st_range_a0.5", "Range steering, α 0.5"), ("st_global_a0.5", "Plain steering, α 0.5"), ("rm_range_comp", "Range removal")]


def tmean(x):
    x = np.sort(np.asarray(x, float)); n = len(x); k = int(args.trim * n)
    return x[k:n - k].mean()


def load(arm, ds):
    p = Path(f"results/delta/{arm}/interventions/range_intervention_{ds}_dprime_conf.json")
    return list(json.loads(p.read_text())["concepts"].values()) if p.exists() else []


def levels(C, op):
    """{(space, group): (before, after)} in %, trimmed over concepts; group = target / other."""
    out = {}
    for s in ("h", "z"):
        rows = []
        for c in C:
            nt = len(c["eval_idx"]["tgt"])
            r, e = np.array(c["ref"][s]["doc_conf"]), np.array(c[f"{s}_{op}"]["doc_conf"])
            rows.append((r[:nt].mean(), e[:nt].mean(), r[nt:].mean(), e[nt:].mean(), np.maximum(0, r[nt:] - e[nt:]).mean()))
        a = 100 * np.array(rows)
        out[(s, "target")] = (tmean(a[:, 0]), tmean(a[:, 1]))
        out[(s, "other")] = (tmean(a[:, 2]), tmean(a[:, 3]), tmean(a[:, 4]))
    return out


data = {arm: {ds: load(arm, ds) for ds, _ in DSN} for arm, _ in ARMS}
groups = [(name, [ds]) for ds, name in DSN] + [("**All three**", [ds for ds, _ in DSN])]
for op, opname in OPS:
    print(f"\n## {opname}\n")
    print("### Target documents (the concept being edited): confidence in the correct label, before → after (%)\n")
    print("| Dataset | " + " | ".join(f"{m}: base | {m}: GeoAE" for _, m in ARMS) + " |")
    print("|---|" + "---|---|" * len(ARMS))
    for gname, dss in groups:
        cells = []
        for arm, _ in ARMS:
            C = sum((data[arm][d] for d in dss), [])
            if not C:
                cells += ["–", "–"]; continue
            L = levels(C, op)
            cells += [f"{L[('h', 'target')][0]:.1f} → {L[('h', 'target')][1]:.1f}", f"{L[('z', 'target')][0]:.1f} → {L[('z', 'target')][1]:.1f}"]
        print(f"| {gname} | " + " | ".join(cells) + " |")
    print("\n### Other-class documents (should not change): confidence in their own correct label, before → after (%); "
          "[average loss, counting only documents whose confidence fell]\n")
    print("| Dataset | " + " | ".join(f"{m}: base | {m}: GeoAE" for _, m in ARMS) + " |")
    print("|---|" + "---|---|" * len(ARMS))
    for gname, dss in groups:
        cells = []
        for arm, _ in ARMS:
            C = sum((data[arm][d] for d in dss), [])
            if not C:
                cells += ["–", "–"]; continue
            L = levels(C, op)
            for s in ("h", "z"):
                b, a, loss = L[(s, "other")]
                cells.append(f"{b:.1f} → {a:.1f} [{loss:.1f}]")
        print(f"| {gname} | " + " | ".join(cells) + " |")
