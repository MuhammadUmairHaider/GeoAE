"""Two paired comparison tables (last layers): each autoencoder against its encoder-free control.

  Table A, no bypass:  plain residual k-means      vs  no-bypass AE   (both see the full residual stream)
  Table B, bypass:     plain token-removed k-means vs  bypass AE      (both see the residual minus the token's mean)

Cells: raw homogeneity / completeness / NMI; Delta = NMI(AE) - NMI(k-means). Tasks: the user's selection (8 Oct) of the
probe ladder (incl. Jigsaw toxicity, toxic vs clean, 1,500 each), plus FineWeb-Atlas read at the last word of each
chunk (14 topics = mean of the three 14-concept draws; document type dropped on request;
* = the doc-bootstrap 95% CI of the NMI difference excludes 0 in every draw / the set). Row filter (--filter):
  pair10 (default, user 8 Oct): token-level tasks as min10; sequence-level and Atlas tasks filtered over the TWO spaces
        of the table only: rows whose cluster holds >= 10 kept rows in both spaces, iterated to a fixed point
        (probe_dense_subsets.py --pairs ... --pair_min 10 -> pair_fp10_<km>+<ae>; atlas_nmi_filtered.py --arms <km>,<ae>
        --min_rows 10 -> nmi_pair10_<ae>.json);
  min10 the original probe setting: each space keeps the rows in its OWN clusters holding >= 10 of the probe's
        (Atlas set's) rows -- probe_dense_subsets.py subset min10, atlas_nmi_filtered.py --per_space --min_rows 10;
  fp6   the reporting-rule fixed point: rows whose cluster holds > 5 kept rows in all four spaces (common_fp6).

    python eval_out/latex_pair_tables.py md  > results/delta/pair_tables.md
    python eval_out/latex_pair_tables.py tex > docs/paper/drafts/table_pairs.tex
    python eval_out/latex_pair_tables.py md --all --filter fp6
    python eval_out/latex_pair_tables.py md --metric h      # homogeneity only (user, 9 Oct): cells, Delta, bold and * all on homogeneity
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["md", "tex"])
ap.add_argument("--all", action="store_true", help="every probe instead of the selection")
ap.add_argument("--filter", choices=["pair10", "min10", "fp6"], default="pair10")
ap.add_argument("--metric", choices=["hcn", "h"], default="hcn",
                help="hcn: cells homogeneity / completeness / NMI, Delta on NMI (default); h: homogeneity only")
args = ap.parse_args()

MODELS = [("Llama-3.2-3B, L27", "results/delta/d6144_nobypass/evals/probe_dense_subsets.json"),
          ("Gemma 3 4B, L33", "results/delta/g4b_l33_nobypass/evals/probe_dense_subsets.json")]
ATLAS_TAG = {"pair10": "nmi_pair10_{}.json", "min10": "nmi_perspace10_fourway_ref_{}.json",
             "fp6": "nmi_filtered_fourway_ref_{}.json"}[args.filter]
ATLAS = ["results/delta/d6144/evals/atlas_lastword/" + ATLAS_TAG,
         "results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/" + ATLAS_TAG]
# GEOAE_NE100=1: the 12 Oct no-balancing / 100-epoch models (scripts/argo/ne100_evals.sbatch). Same probe file layout;
# the Atlas pair files sit in each AE's own folder and name the AEs bypass_ne100 / nobypass_ne100. Qwen is added as a
# third model (its four-space file has the same layout). Only the pair10 filter exists for these runs.
import os
SFX = os.environ.get("GEOAE_SFX") or ("_ne100" if os.environ.get("GEOAE_NE100") == "1" else "")     # GEOAE_SFX=_ns100: the no-staging models
NE = bool(SFX)
if NE:
    _B = {"Llama-3.2-3B, L27": ("d6144" + SFX, "d6144_nobypass" + SFX), "Gemma 3 4B, L33": ("g4b_l33_d5120" + SFX, "g4b_l33_nobypass" + SFX),
          "Qwen2.5-3B, L35": ("qwen25_l35_d4096" + SFX, "qwen25_l35_nobypass" + SFX)}
    _B = {k: v for k, v in _B.items() if os.path.exists(f"results/delta/{v[1]}/evals/probe_dense_subsets.json")}
    MODELS = [(k, f"results/delta/{v[1]}/evals/probe_dense_subsets.json") for k, v in _B.items()]
    ATLAS = [{"bypass": f"results/delta/{v[0]}/evals/atlas_lastword/nmi_pair10_bypass.json",
              "nobypass": f"results/delta/{v[1]}/evals/atlas_lastword/nmi_pair10_nobypass.json"} for v in _B.values()]
SUB = {"pair10": "min10", "min10": "min10", "fp6": "common_fp6"}[args.filter]     # token-level rows (all rows unless pair10)


def sub_for(grp, km, ae):
    return f"pair_fp10_{km}+{ae}" if args.filter == "pair10" and grp != "Token-level" else SUB
PAIRS = [("nobypass", "No token bypass: residual $k$-means vs. no-bypass AE", "res_plain", "nobypass",
          "Resid. $k$-means", "No-bypass AE"),
         ("bypass", "Token bypass: token-removed $k$-means vs. bypass AE (GeoAE)", "tok_plain", "geoae",
          "Tok.-rem. $k$-means", "Bypass AE")]
ALL_GROUPS = [
    ("Token-level", [("surface", "Surface form"), ("pos_coarse", "POS, coarse"), ("pos_fine", "POS, Penn"),
                     ("ner_coarse", "Entity type, coarse"), ("ner_fine", "Entity type, fine")]),
    ("Sequence-level", [("sentiment", "Sentiment (SST-2)"), ("sentiment_long", "Sentiment (IMDB)"),
                        ("subjectivity", "Subjectivity"), ("formality", "Formality"), ("language", "Language"),
                        ("domain", "Domain"), ("topic4", "Topic (AG News)"), ("topic14", "Topic (DBpedia-14)"),
                        ("topic20", "Topic (20 Newsgroups)"), ("toxicity", "Toxicity (Jigsaw)"),
                        ("toxicity_pattern", "Toxicity flags (Jigsaw)")]),
    ("Entity and relation", [("ravel_country", "RAVEL country"), ("ravel_continent", "RAVEL continent"),
                             ("ravel_language", "RAVEL language"), ("ioi_role", "IOI role"), ("ioi_name", "IOI name")]),
]
SELECT = {"surface", "pos_fine", "ner_coarse", "domain", "topic4", "topic14", "topic20", "toxicity"}
GROUPS = [(g, [(r, n) for r, n in t if args.all or r in SELECT]) for g, t in ALL_GROUPS]
GROUPS = [(g, t) for g, t in GROUPS if t]
if not MODELS:
    raise SystemExit("no four-space probe file yet (needs both the bypass and the no-bypass model of a pair)")
_avail = set.intersection(*[set(json.load(open(p))["rungs"]) for _, p in MODELS])
GROUPS = [(g, [(r, n) for r, n in t if r in _avail]) for g, t in GROUPS]
ATLAS_ARM = {"res_plain": "plain42", "tok_plain": "plain42_tokmean", "geoae": "bypass", "nobypass": "nobypass"}
ATLAS_FILE = dict(ATLAS_ARM)                       # which pair file holds the AE
if NE:
    ATLAS_ARM.update(geoae="bypass" + SFX, nobypass="nobypass" + SFX)
ATLAS_ROWS = [("14 topics (3 draws)", ["atlas14_s0", "atlas14_s1", "atlas14_s2"])]   # document type dropped (user, 8 Oct)
KEYS = ("homogeneity", "completeness", "nmi")
HONLY = args.metric == "h"
MI, MKEY, MNAME = (0, "homogeneity", "homogeneity") if HONLY else (2, "nmi", "NMI")   # the compared score
tex = args.mode == "tex"
R = [json.load(open(p))["rungs"] for _, p in MODELS]
kept_all = []   # every (space, task) kept share, for the caption


def s3(x):
    return f"{x:.3f}".replace("0.", ".", 1) if x >= 0 else "−" + f"{-x:.3f}".replace("0.", ".", 1)


def triple(v, bold):
    t = [s3(x) for x in v]
    if HONLY:
        h = t[0].replace("−", "$-$") if tex else t[0]
        return (f"\\textbf{{{h}}}" if bold else h) if tex else (f"**{h}**" if bold else h)
    if tex:
        t = [x.replace("−", "$-$") for x in t]
        return f"{t[0]}/{t[1]}/" + (f"\\textbf{{{t[2]}}}" if bold else t[2])
    return f"{t[0]} / {t[1]} / " + (f"**{t[2]}**" if bold else t[2])


def delta(a, b, star=""):
    d = b - a
    s = ("+" if d >= 0 else "−") + f"{abs(d):.3f}".replace("0.", ".", 1) + star
    return s.replace("−", "$-$") if tex else s


def probe_cells(Rm, rungs, km, ae, grp):
    S = sub_for(grp, km, ae)
    a = [float(np.mean([Rm[r]["arms"][km][S][k] for r in rungs])) for k in KEYS]
    b = [float(np.mean([Rm[r]["arms"][ae][S][k] for r in rungs])) for k in KEYS]
    ka = float(np.mean([Rm[r]["arms"][km][S]["n"] / Rm[r]["n"] for r in rungs]))
    kb = float(np.mean([Rm[r]["arms"][ae][S]["n"] / Rm[r]["n"] for r in rungs]))
    return a, b, "", ka, kb


def atlas_cells(mi, sets, km, ae):
    A = json.load(open(ATLAS[mi][ATLAS_FILE[ae]] if NE else ATLAS[mi].format(ATLAS_ARM[ae])))["sets"]
    a = [float(np.mean([A[s]["point"][ATLAS_ARM[km]][k] for s in sets])) for k in KEYS]
    b = [float(np.mean([A[s]["point"][ATLAS_ARM[ae]][k] for s in sets])) for k in KEYS]
    ci = [A[s]["diff_ref_minus_arm"][ATLAS_ARM[km]][MKEY]["ci"] for s in sets]
    star = "*" if all(c[0] > 0 or c[1] < 0 for c in ci) else ""
    kk = lambda arm: float(np.mean([A[s]["point"][arm].get("kept_share", A[s]["kept_share"]) for s in sets]))
    return a, b, star, kk(ATLAS_ARM[km]), kk(ATLAS_ARM[ae])


def emit(name, per_model):
    cells = []
    for mi, (a, b, star, ka, kb) in enumerate(per_model):
        kept_all.extend([ka, kb])
        cells += [triple(a, a[MI] > b[MI]), triple(b, b[MI] > a[MI]), delta(a[MI], b[MI], star)]
        if tex:
            if mi == 0:
                cells.append("")
        else:
            cells.append(f"{100 * ka:.0f}% / {100 * kb:.0f}%")
    if tex:
        print(f"    \\quad {name} & " + " & ".join(cells) + " \\\\")
    else:
        print(f"| {name} | " + " | ".join(cells) + " |")


FILTER_TXT = {"pair10": "token-level tasks: each space keeps the rows in its own clusters holding at least ten of the "
                        "task's rows; sequence-level and Atlas tasks: rows whose cluster holds at least ten kept rows in "
                        "both spaces of the table",
              "min10": "each space keeps the rows that fall in its own clusters holding at least ten of the task's rows",
              "fp6": "rows whose cluster holds more than five rows in all four spaces"}[args.filter]
for key, title, km, ae, hk, ha in PAIRS:
    kept_all.clear()
    if tex:
        print(f"% {title} (eval_out/latex_pair_tables.py --filter {args.filter}; probes {', '.join(p for _, p in MODELS)} "
              f"subset {SUB}; Atlas {', '.join(a.format('*') for a in ATLAS)})")
        print("\\begin{table*}[t]\n  \\centering\n  \\scriptsize\n  \\setlength{\\tabcolsep}{2.5pt}")
        print("  \\begin{tabular}{@{}l ccr c ccr@{}}\n    \\toprule")
        print(f"    & \\multicolumn{{3}}{{c}}{{{MODELS[0][0]}}} && \\multicolumn{{3}}{{c}}{{{MODELS[1][0].replace('Gemma 3', 'Gemma~3')}}} \\\\")
        print("    \\cmidrule(lr){2-4} \\cmidrule(l){6-8}")
        print(f"    & {hk} & {ha} & $\\Delta$ {MNAME} && {hk} & {ha} & $\\Delta$ {MNAME} \\\\\n    \\midrule")
    else:
        h1 = hk.replace("$", "")
        print(f"\n### {title.replace('$', '')}\n")
        cells_txt = ("Cells: homogeneity (raw); Δ = homogeneity(AE) − homogeneity(k-means); bold = higher homogeneity of the pair; "
                     if HONLY else
                     "Cells: homogeneity / completeness / **NMI** (raw); Δ = NMI(AE) − NMI(k-means); bold = higher NMI of the pair; ")
        print(cells_txt + f"kept = share of rows scored (k-means / AE): {FILTER_TXT}.\n")
        print(f"| Task | Llama {h1} | Llama {ha} | Llama Δ | Llama kept | Gemma {h1} | Gemma {ha} | Gemma Δ | Gemma kept |")
        print("|---|---|---|---|---|---|---|---|---|")
    blocks = [(grp, [(name, [r]) for r, name in tasks] + [("*Mean*" if not tex else "\\emph{Mean}", [r for r, _ in tasks])],
               "probe") for grp, tasks in GROUPS]
    blocks.append(("FineWeb-Atlas, last word of each chunk", ATLAS_ROWS, "atlas"))
    for bi, (grp, rows, kind) in enumerate(blocks):
        if tex:
            if bi:
                print("    \\midrule")
            print(f"    \\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{grp}}}}} \\\\")
        else:
            print(f"| **{grp}** |" + " |" * 8)
        for name, rungs in rows:
            per_model = [probe_cells(Rm, rungs, km, ae, grp) if kind == "probe" else atlas_cells(mi, rungs, km, ae)
                         for mi, Rm in enumerate(R)]
            emit(name, per_model)
    if tex:
        lo, hi = 100 * min(kept_all), 100 * max(kept_all)
        print("    \\bottomrule\n  \\end{tabular}")
        print(f"  \\caption{{{title}. " + ("Each cell gives homogeneity (raw); " if HONLY else
              "Each cell gives homogeneity / completeness / NMI (raw, no chance correction); ") +
              f"{FILTER_TXT} ({lo:.0f}--{hi:.0f}\\% of rows); bold marks the higher {MNAME} of the pair; $\\Delta$ {MNAME} = "
              f"autoencoder minus $k$-means. Toxicity: Jigsaw comments, toxic (any flag) vs.\\ clean, 1{{,}}500 each. FineWeb-Atlas: "
              f"one row per chunk, read at its last word, chunks carrying exactly one of 14 topic concepts, mean over three "
              f"draws; $^{{*}}$the document-bootstrap 95\\% interval of $\\Delta$ {MNAME} excludes zero in every draw. "
              f"$K = 2{{,}}000$ clusters throughout.}}")
        print(f"  \\label{{tab:pair-{key}}}\n\\end{{table*}}\n")
