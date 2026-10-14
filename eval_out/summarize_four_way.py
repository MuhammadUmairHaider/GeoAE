"""Four spaces side by side, per model and layer: plain residual k-means, plain token-removed k-means, the token-bypass
GeoAE and the no-bypass AE (all plain fits: Lloyd, density-peaks seeds, seed 42). Probes from eval_out/probe_dense_subsets.py
under the reporting rule: raw homogeneity / completeness / NMI together, on the fixed-point subset common_fp6 (every kept
row's cluster holds > 5 kept rows in EVERY space), with the share of rows kept; geometry, token structure and fidelity
from the per-space result files where they exist (last layers only).

    python eval_out/summarize_four_way.py > results/delta/four_way_tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

L27, G33 = Path("results/delta/d6144/evals"), Path("results/delta/g4b_l33_d5120/evals")
NB = Path("results/delta/d6144_nobypass/evals")
GNB = Path("results/delta/g4b_l33_nobypass/evals")
MID = Path("results/midlast_ablation_20261006")
COLS = ["Residual k-means (plain)", "Token-removed k-means (plain)", "Bypass AE (GeoAE)", "No-bypass AE"]
SETTINGS = [
    dict(title="Llama-3.2-3B, layer 27 (last)", probes=NB / "probe_dense_subsets.json",
         arms=["res_plain", "tok_plain", "geoae", "nobypass"], seed="plain k-means seed 42",
         geom=[(L27 / "plain_kmeans_grid_seed42.json", ("geometry", "plain_residual")),
               (L27 / "plain_kmeans_grid_seed42.json", ("geometry", "plain_tokmean")),
               (L27 / "cq.json", ("bypass",)), (NB / "cq.json", ("nobypass",))],
         ts=[(L27 / "token_structure_plain_kmeans_seed42.json", "plain_seed42"),
             (L27 / "token_structure_plain_kmeans_seed42.json", "plain_tokmean_seed42"),
             (L27 / "token_structure.json", "bypass"), (NB / "token_structure.json", "nobypass")],
         mmlu=[None, None, L27 / "mmlu.json", NB / "mmlu.json"]),
    dict(title="Gemma 3 4B, layer 33 (last)", probes=GNB / "probe_dense_subsets.json",
         arms=["res_plain", "tok_plain", "geoae", "nobypass"],
         seed="no-bypass AE = the midlast campaign's gemma4b_last_plain, trained on its own 10M-row dump of the same corpus",
         geom=[(G33 / "plain_kmeans_grid_seed42.json", ("geometry", "plain_residual")),
               (G33 / "plain_kmeans_grid_seed42.json", ("geometry", "plain_tokmean")), (G33 / "cq.json", ("bypass",)),
               (GNB / "cq.json", ("nobypass",))],
         ts=[(GNB / "token_structure.json", "plain_seed42"), (GNB / "token_structure.json", "plain_tokmean_seed42"),
             (G33 / "token_structure.json", "bypass"), (GNB / "token_structure.json", "nobypass")],
         mmlu=[None, None, G33 / "mmlu.json", GNB / "mmlu.json"]),
    dict(title="Llama-3.2-3B, layer 14 (middle)", probes=MID / "llama3.2-3B/layer14/final_probes/probe_dense_subsets.json",
         arms=["res_plain", "tok_plain", "geoae", "nobypass"], seed="midlast campaign dumps and AEs (50 epochs)",
         geom=None, ts=None, mmlu=None),
    dict(title="Gemma 3 4B, layer 17 (middle)", probes=MID / "gemma3-4B/layer17/final_probes/probe_dense_subsets.json",
         arms=["res_plain", "tok_plain", "geoae", "nobypass"], seed="midlast campaign dumps and AEs (50 epochs)",
         geom=None, ts=None, mmlu=None),
]
TOK = ["surface", "pos_coarse", "pos_fine", "ner_coarse", "ner_fine"]
SEQ = ["sentiment", "sentiment_long", "subjectivity", "formality", "language", "domain", "topic4", "topic14", "topic20"]
ENT = ["ravel_country", "ravel_continent", "ravel_language", "ioi_role", "ioi_name"]
SUB = "common_fp6"


def get(path, keys):
    d = json.loads(Path(path).read_text())
    for k in keys:
        d = d[k]
    return d


def row(label, cells):
    print(f"| {label} | " + " | ".join(cells) + " |")


for S in SETTINGS:
    R = json.loads(S["probes"].read_text())["rungs"]
    print(f"\n## {S['title']}\n\n({S['seed']}; probes: rows whose cluster holds > 5 kept rows in every space; cells = raw "
          f"homogeneity / completeness / **NMI**)\n")
    print("| | rows kept | " + " | ".join(COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for grp, rungs in [("Token-level probes", TOK), ("Sequence-level probes", SEQ), ("Entity and relation probes", ENT)]:
        print(f"| **{grp}** | |" + " |" * len(COLS))
        for r in rungs:
            a0 = next(a for a in S["arms"] if a)
            cells = [f"{100 * R[r]['arms'][a0][SUB]['n'] / R[r]['n']:.0f}%"]
            for a in S["arms"]:
                if a is None:
                    cells.append("—")
                    continue
                x = R[r]["arms"][a][SUB]
                cells.append(f"{x['homogeneity']:.3f} / {x['completeness']:.3f} / **{x['nmi']:.3f}**")
            row(r, cells)
    print("| **Means** | |" + " |" * len(COLS))
    for lab, rungs in [("token-level (5)", TOK), ("sequence (9)", SEQ),
                       ("sequence without language (8)", [r for r in SEQ if r != "language"]), ("entity/relation (5)", ENT)]:
        cells = [""]
        for a in S["arms"]:
            if a is None:
                cells.append("—")
                continue
            m = [np.mean([R[r]["arms"][a][SUB][k] for r in rungs]) for k in ("homogeneity", "completeness", "nmi")]
            cells.append(f"{m[0]:.3f} / {m[1]:.3f} / **{m[2]:.3f}**")
        row(lab, cells)
    if S["geom"]:
        print("| **Geometry (1M rows)** | |" + " |" * len(COLS))
        G = [None if g is None else get(*g) for g in S["geom"]]
        for k, lab, f in [("calinski_harabasz", "Calinski–Harabasz ↑", "{:.0f}"), ("separability_ratio", "Separability ↑", "{:.1f}"),
                          ("davies_bouldin", "Davies–Bouldin ↓", "{:.2f}"), ("silhouette", "Silhouette ↑", "{:+.3f}")]:
            row(lab, [""] + ["—" if g is None else f.format(g[k]) for g in G])
    if S["ts"]:
        print("| **Token structure (held-out rows)** | |" + " |" * len(COLS))
        T = [None if t is None else get(t[0], (t[1],)) for t in S["ts"]]
        row("Current-token cNMI ↓", [""] + ["—" if t is None else f"{t['cnmi_cur']:.3f}" for t in T])
        row("Clusters ≥80% one token", [""] + ["—" if t is None else f"{100 * t['trivial_cur_80']:.1f}%" for t in T])
        row("Share of 10 largest clusters", [""] + ["—" if t is None else f"{100 * t['top10']:.1f}%" for t in T])
        row("Variance explained", [""] + ["—" if (t is None or "fvu" not in t) else f"{1 - t['fvu']:.3f}" for t in T])
    if S["mmlu"]:
        M = [None if m is None else get(m, ("meta",)) for m in S["mmlu"]]
        row("MMLU with splice (change, points)", [""] + ["—" if m is None else f"{100 * m['recon_acc']:.1f} ({100 * m['delta']:+.1f})"
                                                         for m in M])
