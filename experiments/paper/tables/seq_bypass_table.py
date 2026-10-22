"""The sequence-level bypass table (tab:seq-bypass) for any set of models: plain token-removed k-means vs bypass AE.

Cells: raw homogeneity / completeness / NMI on the paired subset (rows whose cluster holds >= 10 kept rows in BOTH spaces,
fixed point: probe_dense_subsets.py --pairs tok_plain:geoae --pair_min 10; atlas_nmi_filtered.py --arms
plain42_tokmean,bypass --min_rows 10). Atlas = mean of the three 14-topic draws; * = the document-bootstrap 95% CI of the
homogeneity difference (bypass - k-means) excludes zero in every draw.

    python experiments/paper/tables/seq_bypass_table.py                        # Llama L27, Gemma L33, Qwen L35
    python experiments/paper/tables/seq_bypass_table.py --probe_file probe_dense_subsets_allrows.json   (Qwen pass without anchor exclusion)
"""

from geoae.paths import resolve_input_path
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--probe_file", default="probe_dense_subsets.json", help="Qwen probe file name inside seq_table/")
ap.add_argument("--set", default="last", choices=["last", "p85"], help="last layers (paper) or the 85%%-depth bypass arms")
ap.add_argument("--latex", action="store_true", help="also print the table* in the paper's tab:seq-bypass layout")
args = ap.parse_args()

Q = "results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/seq_table/"
MODELS = [("Llama-3.2-3B L27", "results/delta/d6144_nobypass/evals/probe_dense_subsets.json",
           "results/delta/d6144/evals/atlas_lastword/nmi_pair10_bypass.json"),
          ("Gemma 3 4B L33", "results/delta/g4b_l33_nobypass/evals/probe_dense_subsets.json",
           "results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/nmi_pair10_bypass.json"),
          ("Qwen2.5-3B L35", Q + args.probe_file, Q + "atlas_lastword/nmi_pair10_bypass.json")]
if args.set == "p85":
    P = "results/p85_bypass_20261009/"
    MODELS = [(n, P + d + "/seq_table/" + args.probe_file, P + d + "/seq_table/atlas_lastword/nmi_pair10_bypass.json")
              for n, d in (("Llama-3.2-3B L24", "llama3.2-3B/layer24"), ("Gemma 3 4B L29", "gemma3-4B/layer29"),
                           ("Qwen2.5-3B L31", "qwen2.5-3B/layer31"))]
RUNGS = [("domain", "Domain (3)"), ("topic4", "Topic, AG News (4)"), ("topic14", "Topic, DBpedia-14 (14)"),
         ("topic20", "Topic, 20 Newsgroups (20)"), ("toxicity", "Toxicity, Jigsaw (2)")]
SUB = "pair_fp10_tok_plain+geoae"
KEYS = ("homogeneity", "completeness", "nmi")


def s3(x):
    return f"{x:.3f}".replace("0.", ".", 1) if x >= 0 else "-" + f"{-x:.3f}".replace("0.", ".", 1)


def cell(v, star=""):
    return f"{s3(v[0])}{star}/{s3(v[1])}/{s3(v[2])}"


have = [(n, p, a) for n, p, a in MODELS if resolve_input_path(Path(p)).exists()]
cols = {}
for name, probe, atlas in have:
    R = json.load(open(resolve_input_path(probe)))["rungs"]
    col = {}
    for rung, _ in RUNGS:
        if rung not in R:
            continue
        x = R[rung]
        col[rung] = {arm: [x["arms"][arm][SUB][k] for k in KEYS] for arm in ("tok_plain", "geoae")}
        col[rung]["kept"] = (x["common_rows"][SUB], x["n"])
    present = [r for r, _ in RUNGS if r in col]
    col["mean"] = {arm: [sum(col[r][arm][i] for r in present) / len(present) for i in range(3)]
                   for arm in ("tok_plain", "geoae")}
    if resolve_input_path(Path(atlas)).exists():
        S = json.load(open(resolve_input_path(atlas)))["sets"]
        draws = [S[f"atlas14_s{i}"] for i in range(3)]
        col["atlas"] = {arm: [sum(d["point"][a][k] for d in draws) / 3 for k in KEYS]
                        for arm, a in (("tok_plain", "plain42_tokmean"), ("geoae", "bypass"))}
        cis = [d["diff_ref_minus_arm"]["plain42_tokmean"]["homogeneity"]["ci"] for d in draws]
        col["atlas"]["star"] = "*" if all(lo > 0 for lo, _ in cis) or all(hi < 0 for _, hi in cis) else ""
        col["atlas"]["kept"] = [round(d["kept_share"], 2) for d in draws]
        col["atlas"]["h_ci"] = cis
    cols[name] = col

print("| Task | " + " | ".join(f"{n}: tok-rm k-means | {n}: bypass AE | ΔNMI" for n in cols) + " |")
print("|---|" + "---|---|---|" * len(cols))
for key, label in RUNGS + [("mean", "*Mean*"), ("atlas", "FineWeb-Atlas 14 topics (3 draws)")]:
    cells = []
    for n, col in cols.items():
        if key not in col:
            cells += ["—", "—", "—"]
            continue
        c = col[key]
        star = c.get("star", "") if key == "atlas" else ""
        cells += [cell(c["tok_plain"]), cell(c["geoae"], star), f"{c['geoae'][2] - c['tok_plain'][2]:+.3f}"]
    print(f"| {label} | " + " | ".join(cells) + " |")
print()
for n, col in cols.items():
    kept = ", ".join(f"{r} {col[r]['kept'][0]}/{col[r]['kept'][1]}" for r, _ in RUNGS if r in col)
    extra = ""
    if "atlas" in col:
        extra = f"; Atlas kept share {col['atlas']['kept']}, homogeneity-diff CIs " + \
                ", ".join(f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in col["atlas"]["h_ci"])
    print(f"- {n}: rows kept {kept}{extra}")

if args.latex:
    TEX = {"domain": "Domain (3)", "topic4": "AG News (4)", "topic14": "DBpedia-14 (14)", "topic20": "20 Newsgroups (20)",
           "toxicity": "Jigsaw toxicity (2)", "mean": "\\emph{Mean}", "atlas": "14 topics (3 draws)"}

    def tex_pair(c, star=""):
        out = []
        for arm, other in (("tok_plain", "geoae"), ("geoae", "tok_plain")):
            h, cc, n = (s3(v) for v in c[arm])
            st = f"$^{{{star}}}$" if star and arm == "geoae" else ""
            h = f"\\textbf{{{h}}}" if round(c[arm][0], 3) > round(c[other][0], 3) else h
            out.append(f"{h}{st}/{cc}/{n}")
        return out

    def tex_row(key):
        cells = []
        for col in cols.values():
            cells.append(" & ".join(tex_pair(col[key], col[key].get("star", "") if key == "atlas" else ""))
                         if key in col else "--- & ---")
        return f"    {TEX[key]} & " + " &  & ".join(cells) + " \\\\"

    heads = [n.replace(" L", ", layer ").replace("Gemma 3", "Gemma~3") for n in cols]
    print()
    print("    & " + " && ".join(f"\\multicolumn{{2}}{{c}}{{{h}}}" for h in heads) + " \\\\")
    print("    \\midrule")
    print("    \\multicolumn{9}{@{}l}{\\emph{Probes (last token of each text)}} \\\\")
    for key in [r for r, _ in RUNGS] + ["mean"]:
        print(tex_row(key))
    print("    \\midrule")
    print("    \\multicolumn{9}{@{}l}{\\emph{FineWeb-Atlas (last word of each chunk)}} \\\\")
    print(tex_row("atlas"))
