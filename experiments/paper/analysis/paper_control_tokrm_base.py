"""Does the base for plain steering change when the token means are removed first?

Reads a range_intervention_compare.py run with --substrates h,z,hb (scripts/{argo,delta}/control_tokrm_base.sbatch):
h = raw residual stream, hb = token-removed residual stream (same per-token means as the bypass, no encoder), z = GeoAE
latent; plain steering (st_global) at --alpha, two-way true-positive documents. Metric as in
experiments/paper/analysis/paper_control_plain_table.py: accuracy points, per concept, each space against its own unedited run;
target drop and collateral are accuracy lost on the concept's and on the other classes' documents,
selectivity = target drop - collateral. Differences are plain means over concepts with a 95% bootstrap interval over
concepts (10k resamples, seed 0). If the paper's h,z run on the same pool exists, checks that both runs scored the same
documents and reports how far the repeated h and z numbers are from it.

    python3 experiments/paper/analysis/paper_control_tokrm_base.py qwen25_l35_d4096 db14
"""

from geoae.paths import resolve_input_path
import json
import sys
from pathlib import Path

import numpy as np

arm, ds = sys.argv[1], sys.argv[2]
alpha = sys.argv[3] if len(sys.argv) > 3 else "0.5"
KEY = f"st_global_a{float(alpha):.1f}"
root = Path("results/delta") / arm / "interventions" / "tp2"
D = json.loads((resolve_input_path(root / f"range_intervention_{ds}_dprime_hzhb_tp2.json")).read_text())
C = D["concepts"]
names = D["meta"]["classes"]


def metrics(c, s):
    ref, rec = c["ref"][s], c[f"{s}_{KEY}"]
    nt = len(c["eval_idx"]["tgt"])
    r, e = np.array(ref["doc_correct"], float), np.array(rec["doc_correct"], float)
    drop, coll = r[:nt].mean() - e[:nt].mean(), r[nt:].mean() - e[nt:].mean()
    return 100 * drop, 100 * coll, 100 * (drop - coll)


def ci(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    return tuple(np.quantile(d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1), [.025, .975]))


keys = [k for k in C if all(f"{s}_{KEY}" in C[k] for s in ("h", "hb", "z"))]
M = {s: np.array([metrics(C[k], s) for k in keys]) for s in ("h", "hb", "z")}
LAB = {"h": "Residual stream", "hb": "Token-removed residual stream", "z": "GeoAE latent"}
print(f"# {arm} {ds}: plain steering, alpha {alpha}, {len(keys)} concepts, "
      f"{sum(D['meta']['n_eval_by_class'][k] for k in keys)} evaluation documents{'' if 'summary' in D else ' (RUNNING)'}\n")
print("| Space | Target drop | Collateral | Selectivity |\n|---|---|---|---|")
for s in ("h", "hb", "z"):
    print(f"| {LAB[s]} | {M[s][:, 0].mean():.1f} | {M[s][:, 1].mean():.1f} | {M[s][:, 2].mean():.1f} |")
print("\n| Selectivity difference | Mean [95% CI] | Concepts better / worse / equal |\n|---|---|---|")
for a, b in (("hb", "h"), ("z", "h"), ("z", "hb")):
    d = M[a][:, 2] - M[b][:, 2]
    lo, hi = ci(d)
    print(f"| {LAB[a]} − {LAB[b]} | {d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] | {int((d > 1e-9).sum())} / {int((d < -1e-9).sum())} / {int((np.abs(d) <= 1e-9).sum())} |")

# how different are the two base edits, document by document?
same = tot = 0
for k in keys:
    a, b = np.array(C[k][f"h_{KEY}"]["doc_correct"]), np.array(C[k][f"hb_{KEY}"]["doc_correct"])
    same += int((a == b).sum()); tot += len(a)
print(f"\nEdited predictions, residual vs token-removed base: same outcome on {same} of {tot} document evaluations ({100 * same / tot:.1f}%).")
print("\n| Concept | Selectivity: residual | token-removed | GeoAE |\n|---|---|---|---|")
for i, k in enumerate(keys):
    print(f"| {names[int(k)]} | {M['h'][i, 2]:.1f} | {M['hb'][i, 2]:.1f} | {M['z'][i, 2]:.1f} |")

p0 = root / f"range_intervention_{ds}_dprime_hz_tp2.json"
if resolve_input_path(p0).exists():
    P = json.loads(resolve_input_path(p0).read_text())
    same_docs = all(P["concepts"][k]["eval_idx"] == C[k]["eval_idx"] for k in keys if k in P["concepts"]) and P["meta"].get("eval_sha") == D["meta"].get("eval_sha")
    d = {s: np.array([metrics(P["concepts"][k], s)[2] for k in keys]) - M[s][:, 2] for s in ("h", "z")}
    print(f"\nPaper run on the same pool: same evaluation documents: {same_docs}; repeated selectivity differs by "
          f"{np.abs(d['h']).max():.1f} (residual) and {np.abs(d['z']).max():.1f} (GeoAE) points at most per concept.")
