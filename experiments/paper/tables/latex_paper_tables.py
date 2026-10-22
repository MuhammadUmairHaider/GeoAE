"""The paper's three clustering tables (user, 8 Oct), straight from the result JSONs, as LaTeX:

  1. tab:token-nobypass  token-level tasks: plain residual k-means vs the no-bypass AE
  2. tab:seq-bypass      sequence-level tasks + FineWeb-Atlas topics: plain token-removed k-means vs the bypass AE (GeoAE)
  3. tab:full-{llama,gemma,qwen}  appendix: both pairs on every task, one table per model

Cells: raw homogeneity / completeness / NMI (bold = the higher homogeneity of the pair; no Delta column, user 8 Oct).
Rows scored (the user's setting): token-level and entity/relation tasks -- each space keeps the rows in its own clusters
holding >= 10 of the task's rows (probe_dense_subsets.py subset min10); sequence-level and Atlas tasks -- rows whose
cluster holds >= 10 kept rows in BOTH spaces of the pair, iterated to a fixed point (pair_fp10_<km>+<ae>;
atlas_nmi_filtered.py --arms <km>,<ae> --min_rows 10 -> atlas_lastword/nmi_pair10_<ae>.json). Atlas: one row per chunk at
its last word, chunks with exactly one of 14 topic concepts, mean of three draws; * = the document-bootstrap 95% CI of
the homogeneity difference excludes 0 in every draw.

    python experiments/paper/tables/latex_paper_tables.py > docs/paper/drafts/tables_final.tex
"""

from geoae.paths import resolve_input_path
import json

import numpy as np

MODELS = [("Llama-3.2-3B, layer 27", "results/delta/d6144_nobypass/evals/probe_dense_subsets.json",
           "results/delta/d6144/evals/atlas_lastword/nmi_pair10_{}.json"),
          ("Gemma~3 4B, layer 33", "results/delta/g4b_l33_nobypass/evals/probe_dense_subsets.json",
           "results/delta/g4b_l33_d5120/evals/sae_suite/atlas_lastword/nmi_pair10_{}.json"),
          ("Qwen2.5-3B, layer 35", "results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/seq_table/probe_dense_subsets_fourway.json",
           "results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/seq_table/atlas_lastword/nmi_pair10_{}.json")]
NOBYPASS = ("res_plain", "nobypass", "Resid.\\ $k$-means", "No-bypass AE")
BYPASS = ("tok_plain", "geoae", "Tok.-rem.\\ $k$-means", "Bypass AE")
ATLAS_ARM = {"res_plain": "plain42", "tok_plain": "plain42_tokmean", "geoae": "bypass", "nobypass": "nobypass"}
TOKEN = [("surface", "Surface form (7)"), ("pos_coarse", "POS, coarse (12)"),
         ("pos_fine", "POS, Penn (44)"), ("ner_coarse", "Entity type, coarse (9)"),
         ("ner_fine", "Entity type, fine (67)")]
SEQUENCE = [("sentiment", "Sentiment, SST-2 (2)"), ("sentiment_long", "Sentiment, IMDB (2)"),
            ("subjectivity", "Subjectivity (2)"), ("formality", "Formality (3)"), ("language", "Language (20)"),
            ("domain", "Domain (3)"), ("topic4", "AG News (4)"),
            ("topic14", "DBpedia-14 (14)"), ("topic20", "20 Newsgroups (20)"),
            ("toxicity", "Jigsaw toxicity (2)"), ("toxicity_pattern", "Jigsaw flag set ($\\sim$30)")]
ENTITY = [("ravel_country", "RAVEL country"), ("ravel_continent", "RAVEL continent"),
          ("ravel_language", "RAVEL language"), ("ioi_role", "IOI role (3)"), ("ioi_name", "IOI name")]
MAIN_TOKEN = {"surface", "pos_fine", "ner_coarse"}
MAIN_SEQ = {"domain", "topic4", "topic14", "topic20", "toxicity"}
KEYS = ("homogeneity", "completeness", "nmi")
R = [json.load(open(resolve_input_path(p)))["rungs"] for _, p, _ in MODELS]
kept = []


def s3(x):
    s = f"{abs(x):.3f}".replace("0.", ".", 1)
    return s if x >= 0 else "$-$" + s


def triple(v, bold, star=""):
    """homogeneity (bold if the higher of the pair, * if its difference is significant) / completeness / NMI."""
    h = f"\\textbf{{{s3(v[0])}}}" if bold else s3(v[0])
    return f"{h}{star}/{s3(v[1])}/{s3(v[2])}"


def sub(kind, km, ae):
    return f"pair_fp10_{km}+{ae}" if kind == "sequence" else "min10"


def probe_pair(Rm, rungs, km, ae, kind):
    S = sub(kind, km, ae)
    a = [float(np.mean([Rm[r]["arms"][km][S][k] for r in rungs])) for k in KEYS]
    b = [float(np.mean([Rm[r]["arms"][ae][S][k] for r in rungs])) for k in KEYS]
    kept.extend(Rm[r]["arms"][x][S]["n"] / Rm[r]["n"] for r in rungs for x in (km, ae))
    return a, b, ""


def atlas_pair(mi, km, ae):
    A = json.load(open(resolve_input_path(MODELS[mi][2].format(ATLAS_ARM[ae]))))["sets"]
    sets = ["atlas14_s0", "atlas14_s1", "atlas14_s2"]
    a = [float(np.mean([A[s]["point"][ATLAS_ARM[km]][k] for s in sets])) for k in KEYS]
    b = [float(np.mean([A[s]["point"][ATLAS_ARM[ae]][k] for s in sets])) for k in KEYS]
    ci = [A[s]["diff_ref_minus_arm"][ATLAS_ARM[km]]["homogeneity"]["ci"] for s in sets]
    kept.extend(A[s]["kept_share"] for s in sets)
    return a, b, "$^{*}$" if all(c[0] > 0 or c[1] < 0 for c in ci) else ""


def cells(a, b, star):
    """Bold the higher homogeneity as printed (both on a printed tie); the star goes to the higher one."""
    ra, rb = round(a[0], 3), round(b[0], 3)
    sa, sb = (star, "") if a[0] > b[0] else ("", star)
    return [triple(a, ra >= rb, sa), triple(b, rb >= ra, sb)]


def present(rows):
    return [(r, n) for r, n in rows if all(r in Rm for Rm in R)]


def pair_table(label, caption, pair, blocks):
    """Columns: model 1 (k-means, AE) | model 2 (k-means, AE)."""
    km, ae, hk, ha = pair
    kept.clear()
    nm = len(MODELS)
    ncol = 3 * nm
    spec = " c ".join(["cc"] * nm)
    rules = " ".join(f"\\cmidrule({'lr' if i < nm - 1 else 'l'}){{{2 + 3 * i}-{3 + 3 * i}}}" for i in range(nm))
    out = ["\\begin{table*}[t]", "  \\centering", "  \\scriptsize", "  \\setlength{\\tabcolsep}{2.5pt}",
           f"  \\begin{{tabular}}{{@{{}}l {spec}@{{}}}}", "    \\toprule",
           "    & " + " && ".join(f"\\multicolumn{{2}}{{c}}{{{m[0]}}}" for m in MODELS) + " \\\\",
           "    " + rules,
           "    Task & " + " && ".join(["$k$-means & AE"] * nm) + " \\\\", "    \\midrule"]
    for bi, (title, rows, kind) in enumerate(blocks):
        if bi:
            out.append("    \\midrule")
        if title:
            out.append(f"    \\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for name, rungs in rows:
            c = []
            for mi, Rm in enumerate(R):
                a, b, star = atlas_pair(mi, km, ae) if kind == "atlas" else probe_pair(Rm, rungs, km, ae, kind)
                c += cells(a, b, star) + ([""] if mi < nm - 1 else [])
            out.append(f"    {name} & " + " & ".join(c) + " \\\\")
    out += ["    \\bottomrule", "  \\end{tabular}",
            f"  \\caption{{{caption} ({100 * min(kept):.0f}--{100 * max(kept):.0f}\\% of rows scored).}}",
            f"  \\label{{{label}}}", "\\end{table*}", ""]
    print("\n".join(out))


def with_mean(rows):
    return [(n, [r]) for r, n in rows] + [("\\emph{Mean}", [r for r, _ in rows])]


CELLS = ("Each cell gives homogeneity / completeness / NMI (raw, no chance correction) of the clusters against the "
         "task's labels; bold marks the higher homogeneity of the pair.")

print("% Generated by experiments/paper/tables/latex_paper_tables.py from " + ", ".join(p for _, p, _ in MODELS) +
      " and the atlas_lastword/nmi_pair10_*.json files. Do not edit by hand.\n")
pair_table("tab:token-nobypass",
           "Token-level tasks: the autoencoder without the token bypass against plain $k$-means on the same residual "
           "stream ($K = 2{,}000$ clusters each; $k$-means: plain $k$-means on the residual stream; AE: no token bypass). " + CELLS + " Each space is scored on the rows that fall in its own "
           "clusters holding at least ten of the task's rows; class counts in brackets; POS = part of speech",
           NOBYPASS, [(None, with_mean([t for t in TOKEN if t[0] in MAIN_TOKEN]), "token")])
pair_table("tab:seq-bypass",
           "Sequence-level tasks: the token-bypass autoencoder (GeoAE) against plain $k$-means after the same "
           "subtraction of each token's mean vector, with no encoder ($K = 2{,}000$ clusters each; $k$-means: plain $k$-means "
           "after the subtraction; AE: GeoAE). Cells as in "
           "Table~\\ref{tab:token-nobypass}. Both spaces are scored on the same rows: those whose cluster holds at least "
           "ten scored rows in both. Probes read the last token of each text (domain = code / math / prose); FineWeb-Atlas "
           "reads the last word of each chunk, on chunks carrying exactly one of 14 topic concepts, averaged over three "
           "draws; $^{*}$the document-bootstrap 95\\% interval of the homogeneity difference excludes zero in every draw",
           BYPASS, [("Probes (last token of each text)", with_mean([t for t in SEQUENCE if t[0] in MAIN_SEQ]), "sequence"),
                    ("FineWeb-Atlas (last word of each chunk)", [("14 topics (3 draws)", None)], "atlas")])

# Appendix: both pairs, every task; one table per model (three models do not fit one page).
KEYS_SHORT = ["llama", "gemma", "qwen"]
for mi, Rm in enumerate(R):
    kept.clear()
    out = ["\\begin{table*}[p]", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}",
           "  \\begin{tabular}{@{}l cc c cc@{}}", "    \\toprule",
           "    & \\multicolumn{2}{c}{No token bypass} && \\multicolumn{2}{c}{Token bypass} \\\\",
           "    \\cmidrule(lr){2-3} \\cmidrule(l){5-6}",
           f"    Task & {NOBYPASS[2]} & {NOBYPASS[3]} && {BYPASS[2]} & {BYPASS[3]} \\\\"]
    for title, rows, kind in [("Token-level", present(TOKEN), "token"), ("Sequence-level", present(SEQUENCE), "sequence"),
                              ("Entity and relation", present(ENTITY), "token"),
                              ("FineWeb-Atlas, last word", [("x", "14 topics (3 draws)")], "atlas")]:
        out += ["    \\midrule", f"    \\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\"]
        entries = [(n, [r]) for r, n in rows] if kind != "atlas" else [(rows[0][1], None)]
        if kind != "atlas":
            entries.append(("\\emph{Mean}", [r for r, _ in rows]))
        for name, rungs in entries:
            c = []
            for pi, (km, ae, _, _) in enumerate((NOBYPASS, BYPASS)):
                a, b, star = atlas_pair(mi, km, ae) if kind == "atlas" else probe_pair(Rm, rungs, km, ae, kind)
                c += cells(a, b, star) + ([""] if pi == 0 else [])
            out.append(f"    \\quad {name} & " + " & ".join(c) + " \\\\")
    if mi == 0:
        cap = (f"All tasks, both pairs, {MODELS[mi][0]} ($K = 2{{,}}000$ clusters each). Resid.\\ / Tok.-rem.\\ $k$-means: "
               "plain $k$-means on the residual stream / after subtracting each token's mean vector, with no encoder. "
               "Cells and scored rows as in Table~\\ref{tab:token-nobypass} for token-level and entity/relation tasks and as "
               "in Table~\\ref{tab:seq-bypass} for sequence-level and Atlas tasks; class counts in brackets")
    else:
        cap = f"All tasks, both pairs, {MODELS[mi][0]}; as Table~\\ref{{tab:full-{KEYS_SHORT[0]}}}"
    out += ["    \\bottomrule", "  \\end{tabular}",
            f"  \\caption{{{cap} ({100 * min(kept):.0f}--{100 * max(kept):.0f}\\% of rows scored).}}",
            f"  \\label{{tab:full-{KEYS_SHORT[mi]}}}", "\\end{table*}", ""]
    print("\n".join(out))
