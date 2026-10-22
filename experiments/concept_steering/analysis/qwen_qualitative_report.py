"""Summarize exact Qwen memberships with inspectable, exploratory text labels."""
from __future__ import annotations

from geoae.paths import resolve_input_path

import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure

OUT = Path("results/qwen_artist_qualitative_20261009")
ARMS = ["tok_plain", "geoae"]


def main():
    rows = json.loads((resolve_input_path(OUT / "rows.json")).read_text())
    summary = json.loads((resolve_input_path(OUT / "summary.json")).read_text())
    profiles = json.loads((resolve_input_path(OUT / "cluster_profiles.json")).read_text())
    paired = [r for r in rows if r["paired_retained"]]
    # Explicit vehicle types read from all 167 retained transportation descriptions.
    # Cluster IDs choose the review ordering; labels describe the source text, not a
    # putative meaning assigned to the entire cluster. This is not a blinded study.
    transport = {}
    for cl, default, exceptions in [
        (555, "ship", {1209: "aircraft", 1562: "aircraft"}),
        (1215, "aircraft", {}), (847, "ship", {}),
        (442, "ship", {241: "aircraft", 1903: "aircraft", 1916: "aircraft", 2691: "aircraft"}),
        (950, "rail", {61: "road", 2475: "aircraft"}),
        (1825, "ship", {253: "rail", 749: "rail", 2647: "rail", 523: "aircraft",
                       1025: "aircraft", 1362: "aircraft", 1418: "aircraft", 2025: "aircraft"}),
        (1221, "road", {}), (502, "ship", {1373: "company", 2935: "company"}),
        (1814, "ship", {}), (1499, "ship", {}), (1144, "ship", {}),
    ]:
        for r in paired:
            if r["class_name"] == "MeanOfTransportation" and r["geoae"] == cl:
                transport[r["row"]] = exceptions.get(r["row"], default)
    tr = [r for r in paired if r["class_name"] == "MeanOfTransportation"]
    assert len(transport) == len(tr) == 167
    assert Counter(transport.values()) == Counter(ship=92, aircraft=45, rail=15, road=13, company=2)
    annotations = {
        "method": "Exploratory assistant text review of all 167 retained descriptions. Review grouped by AE cluster; unblinded. Categories use explicitly described object types. No external facts or judge API.",
        "taxonomy": {"ship": "ships, boats, submarines", "aircraft": "powered aircraft, helicopters, gliders, parachutes",
                     "rail": "locomotives and railcars", "road": "cars, motorcycles, buses",
                     "company": "description primarily identifies a vehicle company, despite DBpedia transportation label"},
        "rows": [{"row": r["row"], "title": r["title"], "type": transport[r["row"]],
                  "evidence_prefix": r["model_text"][:240]} for r in tr],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "transport_annotations.json").write_text(json.dumps(annotations, indent=2, ensure_ascii=False))
    # All 28 Artist descriptions in these four clusters were individually read.
    music = [r for r in paired if r["class_name"] == "Artist" and r["geoae"] in [139, 1601, 1103, 1056]]
    assert len(music) == 28
    music_annotations = [{"row": r["row"], "title": r["title"],
        "music_occupation": r["row"] != 2780, "evidence": r["model_text"][:300]} for r in music]
    (OUT / "artist_music_review.json").write_text(json.dumps({"method": "Assistant read all 28 retained Artist descriptions in clusters 139, 1601, 1103, 1056; unblinded exploratory text annotations. This is not musician recall over the entire Artist class.",
        "rows": music_annotations}, indent=2, ensure_ascii=False))
    village = [r for r in paired if r["class_name"] == "Village"]
    def village_tag(r):
        t = r["model_text"]
        if re.search(r"\bIran\b", t): return "Iran"
        if re.search(r"\bPoland\b", t): return "Poland"
        if re.search(r"\b(India|Nepal|Sri Lanka)\b", t): return "South Asia"
        return "Other"
    # Literal context tags for a coarse within-Village diagnostic. They are not
    # independent semantic ground truth: country words and templates covary.
    vg = [village_tag(r) for r in village]
    metrics = {}
    for name, rr, labels in [("transport_types", tr, [transport[r["row"]] for r in tr]),
                             ("village_explicit_region_tags", village, vg)]:
        metrics[name] = {}
        for arm in ARMS:
            vals = homogeneity_completeness_v_measure(labels, [r[arm] for r in rr])
            metrics[name][arm] = dict(zip(["homogeneity", "completeness", "nmi"], map(float, vals)))
    (OUT / "qualitative_metrics.json").write_text(json.dumps(metrics, indent=2))
    for name in ["Artist", "MeanOfTransportation", "Village"]:
        rr = [r for r in rows if r["class_name"] == name]
        with open(OUT / (name.lower() + "_ledger.csv"), "w", newline="") as f:
            columns = ["row", "source_index", "title", "anchor_excluded", "paired_retained", "tok_plain", "geoae",
                       "last_token", "token_count", "truncated", "model_text", "text"]
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader(); w.writerows(rr)
        lines = [f"# {name}: complete source ledger", "", "Every original cached row is included. Flags identify the exact paired table subset; the full model-visible prefix is shown.", ""]
        for r in rr:
            status = "paired table" if r["paired_retained"] else "excluded from paired table"
            if r["anchor_excluded"]: status += "; legacy anchor exclusion"
            lines += [f"## Row {r['row']}: {r['title']}", "",
                      f"Source DBpedia train row {r['source_index']}; {status}. K-means {r['tok_plain']}; AE {r['geoae']}. Final token {r['last_token']!r}; {r['token_count']} source tokens; truncated={r['truncated']}.", "", r["model_text"].strip(), ""]
        (OUT / (name.lower() + "_examples.md")).write_text("\n".join(lines))
    # Distribution and contamination profiles: two different quantities.
    cluster_tables = {}
    for name in ["Artist", "MeanOfTransportation", "Village"]:
        rr = [r for r in paired if r["class_name"] == name]
        cluster_tables[name] = {}
        for arm in ARMS:
            counts = Counter(r[arm] for r in rr)
            cs = []
            for cl, n in counts.most_common():
                members = [r for r in rr if r[arm] == cl]
                p = profiles[arm][str(cl)]
                item = dict(cluster=cl, target_rows=n, all_dbpedia_rows=p["n"],
                    class_precision=n / p["n"], class_coverage=n / len(rr), class_mix=p["classes"],
                    row_ids=[r["row"] for r in members],
                    last_tokens=dict(Counter(r["last_token"] for r in members)),
                    median_length=float(np.median([r["token_count"] for r in members])),
                    truncated=sum(r["truncated"] for r in members))
                if name == "MeanOfTransportation": item["text_types"] = dict(Counter(transport[r["row"]] for r in members))
                if name == "Village":
                    item["region_mentions"] = dict(Counter(village_tag(r) for r in members))
                    item["templates"] = {s: sum(s in r["model_text"] for r in members)
                        for s in ["At the 2006 census", "population was", "administrative district", "Gmina", "not reported"]}
                cs.append(item)
            cluster_tables[name][arm] = cs
    (OUT / "class_cluster_tables.json").write_text(json.dumps(cluster_tables, indent=2, ensure_ascii=False))
    # Exact membership transitions; geographic names are descriptive annotations.
    columns = [798, 450, 269, 1006, 1144, 502, 1814]
    base_ids = [1730, 349]
    matrix = np.array([[sum(r["tok_plain"] == b and r["geoae"] == c for r in village)
                        for c in columns] for b in base_ids])
    fig, ax = plt.subplots(figsize=(10, 2.7))
    ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(columns)), ["798\nIran/census", "450\nPoland/Balkans", "269\nMixed geography",
        "1006\nSouth Asia", "1144\nMixed context", "502\nBroad prose", "1814\nSummary ending"])
    ax.set_yticks(range(2), ["K-means 1730", "K-means 349"])
    for i in range(2):
        for j in range(len(columns)):
            ax.text(j, i, str(matrix[i,j]), ha="center", va="center", color="white" if matrix[i,j] > 40 else "black")
    ax.set_title("Same 173 Village descriptions: one broad partition becomes several context groups")
    ax.set_xlabel("Qwen bypass AE cluster (exploratory text descriptions)")
    fig.tight_layout()
    fig.savefig(OUT / "village_split.png", dpi=180)
    fig.savefig(OUT / "village_split.pdf")
    plt.close(fig)
    with open(OUT / "all_class_screen.csv", "w", newline="") as f:
        keys = ["class", "n_paired", "km_clusters", "ae_clusters", "km_effective_clusters", "ae_effective_clusters",
                "km_weighted_class_purity", "ae_weighted_class_purity", "km_top_cluster_coverage", "ae_top_cluster_coverage"]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for name, x in summary["classes"].items():
            a, b = x["arms"]["tok_plain"], x["arms"]["geoae"]
            w.writerow(dict(zip(keys, [name, x["n_paired"], a["clusters"], b["clusters"], a["effective_clusters"],
                b["effective_clusters"], a["weighted_class_purity"], b["weighted_class_purity"],
                a["top_cluster_coverage"], b["top_cluster_coverage"]])))
    print(json.dumps(metrics, indent=2))
    print("Wrote complete ledgers, manual text annotations, within-class metrics, and Village transition plot.")


if __name__ == "__main__":
    main()
