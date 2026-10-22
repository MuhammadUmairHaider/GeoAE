"""Paper figures as pgfplots code, straight from the result files (no number is typed by hand).

  fig:alignment  dumbbell plot of cluster-label homogeneity, k-means -> GeoAE, per task and model (+ the Gemma Scope SAE).
                 Same cells as tab:token / tab:seq (experiments/paper/tables/latex_paper_tables.py): token-level tasks = plain k-means on
                 the residual stream vs the no-bypass GeoAE, each on its own min10 rows; sequence-level tasks and Atlas =
                 plain k-means on the token-removed residual stream vs the bypass GeoAE on the pair_fp10 rows; SAE = plain
                 k-means on Gemma Scope 16k codes (min10 rows / the rows it shares with GeoAE).
  fig:control    dot-and-interval plot of the selectivity difference to the residual stream with its 95% bootstrap
                 interval, from results/delta/control_plain_tp_a05.json (experiments/paper/analysis/paper_control_plain_table.py --json).

Form and color follow the dataviz method: before/after per item -> dumbbell; difference to a baseline -> dot and
interval around a zero line. GeoAE is the accent (blue #2a78d6), the SAE the second categorical slot (orange #eb6834;
the pair passes the all-pairs colour-vision checks on white), baselines are neutral gray. Marker shape and fill repeat the
identity for grayscale print; the full values stay in the appendix tables.

    python experiments/paper/analysis/paper_figures.py [--check docs/paper/main.tex] > results/delta/paper_figures.tex

--check: compare every homogeneity value of fig:alignment with the cells of tab:token / tab:seq in that LaTeX file.
The preamble needs: \\usepackage{pgfplots} \\pgfplotsset{compat=1.16} \\usepgfplotslibrary{groupplots} and the colors
geoblue (2A78D6), saeorange (EB6834), inkgray (52514E), axisgray (C3C2B7), gridgray (E1E0D9).
"""

from geoae.paths import resolve_input_path
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--check", default="", help="LaTeX file whose tab:token / tab:seq cells must match the plotted values")
ap.add_argument("--control_json", default="results/delta/control_plain_tp_a05.json")
args = ap.parse_args()

R = Path("results/delta")
Q = Path("results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/seq_table")
G = R / "g4b_l33_d5120/evals/sae_suite"
J = lambda p: json.loads(resolve_input_path(Path(p)).read_text())
MODELS = [("Llama-3.2-3B", R / "d6144_nobypass/evals/probe_dense_subsets.json", R / "d6144/evals/atlas_lastword/nmi_pair10_bypass.json"),
          ("Gemma 3 4B", R / "g4b_l33_nobypass/evals/probe_dense_subsets.json", G / "atlas_lastword/nmi_pair10_bypass.json"),
          ("Qwen2.5-3B", Q / "probe_dense_subsets_fourway.json", Q / "atlas_lastword/nmi_pair10_bypass.json")]
TOKEN = [("surface", "Surface form"), ("pos_fine", "Part of speech"), ("ner_coarse", "Entity type")]
SEQ = [("domain", "Domain"), ("topic4", "AG News"), ("topic14", "DBpedia-14"), ("topic20", "20 Newsgroups"), ("toxicity", "Jigsaw toxicity")]
ATLAS = ("atlas", "FineWeb-Atlas topics")
SETS = ["atlas14_s0", "atlas14_s1", "atlas14_s2"]
H = "homogeneity"

# ---------------------------------------------------------------- fig:alignment data
rows = []          # (task key, label, block) top to bottom
rows += [(k, n, "token") for k, n in TOKEN] + [(k, n, "seq") for k, n in SEQ] + [(ATLAS[0], ATLAS[1], "atlas")]
val = {}           # (model index, task key) -> {"km": h, "ae": h, "sae": h or absent}
for mi, (_, probe, atlas) in enumerate(MODELS):
    P, A = J(probe)["rungs"], J(atlas)["sets"]
    for k, _, block in rows:
        if block == "token":
            v = {"km": P[k]["arms"]["res_plain"]["min10"][H], "ae": P[k]["arms"]["nobypass"]["min10"][H]}
        elif block == "seq":
            s = "pair_fp10_tok_plain+geoae"
            v = {"km": P[k]["arms"]["tok_plain"][s][H], "ae": P[k]["arms"]["geoae"][s][H]}
        else:
            v = {"km": float(np.mean([A[s]["point"]["plain42_tokmean"][H] for s in SETS])),
                 "ae": float(np.mean([A[s]["point"]["bypass"][H] for s in SETS]))}
        val[mi, k] = v
S = J(G / "probe_dense_subsets_gs16k_plain.json")["rungs"]
S["toxicity"] = J(G / "probe_dense_subsets_gs16k_plain_toxicity.json")["rungs"]["toxicity"]
SA = J(G / "atlas_lastword/nmi_pair10_gs16k_plain.json")["sets"]
for k, _, block in rows:
    val[1, k]["sae"] = (S[k]["arms"]["sae"]["min10"][H] if block == "token" else
                        S[k]["arms"]["sae"]["pair_fp10_sae+geoae"][H] if block == "seq" else
                        float(np.mean([SA[s]["point"]["gs16k_plain"][H] for s in SETS])))
assert all(v["ae"] > v["km"] and v["ae"] > v.get("sae", -1) for v in val.values()), "GeoAE is not the highest everywhere"

if args.check:     # every plotted value must equal the corresponding table cell (first number of each h/c/NMI triple)
    tex = resolve_input_path(Path(args.check)).read_text()
    names = {"surface": "Surface form (7)", "pos_fine": "POS, Penn (44)", "ner_coarse": "Entity type (9)", "domain": "Domain (3)",
             "topic4": "AG News (4)", "topic14": "DBpedia-14 (14)", "topic20": "20 Newsgroups (20)", "toxicity": "Jigsaw toxicity (2)",
             "atlas": "14 topics (3 draws)"}
    n = 0
    for k, _, _ in rows:
        line = next(l for l in tex.split("\n") if l.strip().startswith(names[k] + " &") and l.count("&") == 9)
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")[1:] if c.strip()]
        got = [float(re.search(r"(\.\d{3})", c).group(1)) for c in cells]
        want = [val[0, k]["km"], val[0, k]["ae"], val[1, k]["km"], val[1, k]["sae"], val[1, k]["ae"], val[2, k]["km"], val[2, k]["ae"]]
        assert len(got) == 7 and all(abs(a - b) < 6e-4 for a, b in zip(got, want)), (k, got, want)
        n += 7
    print(f"% checked {n} plotted values against the table cells in {args.check}: all equal to three decimals", file=sys.stderr)

# ---------------------------------------------------------------- shared style
STYLE = r"""  scale only axis, clip=false,
  axis x line*=bottom, axis y line*=left, y axis line style={draw=none},
  axis line style={draw=axisgray, line width=0.4pt},
  tick style={draw=axisgray, line width=0.4pt}, ytick style={draw=none},
  xmajorgrids, grid style={draw=gridgray, line width=0.3pt},
  tick label style={font=\scriptsize}, title style={font=\footnotesize, yshift=-1.5mm},
  label style={font=\scriptsize},
  legend style={draw=none, fill=none, font=\scriptsize, legend columns=-1, /tikz/every even column/.append style={column sep=4mm}},
  legend cell align=left,"""
KM = r"only marks, mark=o, mark size=1.9pt, draw=inkgray, line width=0.7pt"
AE = r"only marks, mark=*, mark size=2.2pt, mark options={draw=white, fill=geoblue, line width=0.5pt}"
SAE = r"only marks, mark=triangle*, mark size=2.6pt, mark options={draw=white, fill=saeorange, line width=0.5pt}"
f3 = lambda x: f"{x:.3f}"

# ---------------------------------------------------------------- fig:alignment
# y positions, top to bottom: header, 3 token tasks, header, 5 sequence tasks + Atlas
ypos, ylab, y = {}, {}, 11
ylab[y] = r"\textit{Token level}"; y -= 1
for k, n, b in rows:
    if b == "seq" and k == SEQ[0][0]:
        ylab[y] = r"\textit{Sequence level}"; y -= 1
    ypos[k], ylab[y] = y, n
    y -= 1
assert y == 0
out = [r"\begin{figure*}[t]", r"\centering", r"\begin{tikzpicture}", r"\begin{groupplot}[",
       r"  group style={group size=3 by 1, horizontal sep=5mm, y descriptions at=edge left},",
       r"  width=4.1cm, height=3.75cm,", STYLE,
       r"  xmin=0, xmax=1.02, xtick={0,0.25,0.5,0.75,1}, xticklabels={0,.25,.5,.75,1},",
       r"  ymin=0.4, ymax=11.6, ytick={1,...,11},",
       "  yticklabels={" + ", ".join("{" + ylab[i] + "}" for i in range(1, 12)) + "},", "]"]
for mi, (name, _, _) in enumerate(MODELS):
    mid = mi == 1
    opt = f"title={{{name}}}" + (r", xlabel={Homogeneity}, legend style={at={(0.5,-0.33)}, anchor=north}" if mid else "")
    out.append(f"\\nextgroupplot[{opt}]")
    for k, _, _ in rows:
        v = val[mi, k]
        lo = min(v["km"], v.get("sae", v["km"]))
        out.append(f"\\draw[axisgray, line width=1pt] (axis cs:{f3(lo)},{ypos[k]}) -- (axis cs:{f3(v['ae'])},{ypos[k]});")
    out.append(f"\\addplot[{KM}] coordinates {{" + " ".join(f"({f3(val[mi, k]['km'])},{ypos[k]})" for k, _, _ in rows) + "};")
    if mid:
        out.append(r"\addlegendentry{$k$-means}")
        out.append(f"\\addplot[{SAE}] coordinates {{" + " ".join(f"({f3(val[mi, k]['sae'])},{ypos[k]})" for k, _, _ in rows) + "};")
        out.append(r"\addlegendentry{SAE}")
    out.append(f"\\addplot[{AE}] coordinates {{" + " ".join(f"({f3(val[mi, k]['ae'])},{ypos[k]})" for k, _, _ in rows) + "};")
    if mid:
        out.append(r"\addlegendentry{GeoAE}")
out += [r"\end{groupplot}", r"\end{tikzpicture}",
        r"\caption{Homogeneity of the clusters with respect to task labels ($K = 2{,}000$ clusters in every space). Token-level tasks compare the GeoAE without the token bypass with $k$-means on the residual stream; sequence-level tasks compare the token-bypass GeoAE with $k$-means on the token-removed residual stream. SAE: $k$-means on the codes of the pretrained Gemma Scope SAE. GeoAE is the most homogeneous space in all 27 comparisons with $k$-means and in all nine with the SAE. Tables~\ref{tab:token} and~\ref{tab:seq} give every value together with completeness and NMI.}",
        r"\label{fig:alignment}", r"\end{figure*}", ""]

# ---------------------------------------------------------------- fig:control
C = J(args.control_json)
st = {(r["model"], r["rows"], r["difference"]): r for r in C["rows"]}
DS = ["DBpedia-14", "Bias in Bios", "AG News", "Emotion", "20NG", "FineWeb-Atlas", "All concepts"]
DSLAB = {"20NG": "20 Newsgroups", "All concepts": r"\textbf{All concepts}"}
yy = {d: 7 - i for i, d in enumerate(DS)}
lim = (-22.0, 42.0)
for r in C["rows"]:
    if r["difference"] in ("GeoAE − Base", "SAE − Base"):
        assert lim[0] < r["lo"] and r["hi"] < lim[1], ("interval outside the x range", r)
def series(model, diff, dy):
    pts = []
    for d in DS:
        r = st[model, d, diff]
        pts.append(f"({r['mean']:.2f},{yy[d] + dy:.2f}) += ({r['hi'] - r['mean']:.2f},0) -= ({r['mean'] - r['lo']:.2f},0)")
    return " ".join(pts)
ERR = r"error bars/.cd, x dir=both, x explicit, error mark=none, error bar style={line width=0.8pt"
out += [r"\begin{figure*}[t]", r"\centering", r"\begin{tikzpicture}", r"\begin{groupplot}[",
        r"  group style={group size=3 by 1, horizontal sep=5mm, y descriptions at=edge left},",
        r"  width=4.1cm, height=2.75cm,", STYLE,
        f"  xmin={lim[0]:.0f}, xmax={lim[1]:.0f}, xtick={{-20,0,20,40}},",
        r"  ymin=0.4, ymax=7.6, ytick={1,...,7},",
        "  yticklabels={" + ", ".join("{" + DSLAB.get(d, d) + "}" for d in reversed(DS)) + "},", "]"]
for name in ("Llama-3.2-3B", "Gemma 3 4B", "Qwen2.5-3B"):
    mid = name == "Gemma 3 4B"
    opt = f"title={{{name}}}" + (r", xlabel={Selectivity relative to the residual stream (accuracy points)}, legend style={at={(0.5,-0.47)}, anchor=north}" if mid else "")
    out.append(f"\\nextgroupplot[{opt}]")
    out.append(f"\\draw[axisgray, line width=0.4pt] (axis cs:{lim[0]:.0f},1.5) -- (axis cs:{lim[1]:.0f},1.5);")
    out.append(r"\draw[inkgray, line width=0.5pt] (axis cs:0,0.4) -- (axis cs:0,7.6);")
    dy = 0.17 if mid else 0.0
    if mid:
        out.append(f"\\addplot[{SAE}, {ERR}, draw=saeorange}}] coordinates {{{series(name, 'SAE − Base', -dy)}}};")
        out.append(r"\addlegendentry{SAE codes}")
    out.append(f"\\addplot[{AE}, {ERR}, draw=geoblue}}] coordinates {{{series(name, 'GeoAE − Base', dy)}}};")
    if mid:
        out.append(r"\addlegendentry{GeoAE latent}")
    r = st[name, "All concepts", "GeoAE − Base"]      # the pooled value is the one direct label
    out.append(f"\\node[font=\\scriptsize, anchor=west, inner sep=2pt] at (axis cs:{r['hi']:.2f},{1 + dy:.2f}) {{${r['mean']:+.1f}$}};")
out += [r"\end{groupplot}", r"\end{tikzpicture}",
        r"\caption{Selectivity of steering away from a concept ($\alpha = 0.5$) in the GeoAE latent and, on Gemma, in the codes of the pretrained Gemma Scope SAE, relative to the same edit in the residual stream (vertical line). Accuracy points on true-positive documents, means over concepts with 95\% bootstrap intervals; values to the right of the line favor the edited space. The pooled GeoAE value is printed. Table~\ref{tab:control} gives target drop and collateral for every space.}",
        r"\label{fig:control}", r"\end{figure*}", ""]
print("\n".join(out))
