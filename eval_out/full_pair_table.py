"""One full four-space table for one model: every probe rung + FineWeb-Atlas, raw homogeneity / completeness / NMI.

Pairs (as eval_out/latex_pair_tables.py): plain residual k-means vs no-bypass AE, plain token-removed k-means vs bypass AE.
Rows filter = latex_pair_tables.py --filter pair10: token-level rungs keep, per space, the rows in its own clusters
holding >= 10 of the rung's rows (min10); sequence-level, entity/relation and Atlas rows keep the fixed point over the
two spaces of the pair (>= 10 kept rows in both: pair_fp10_<km>+<ae>; Atlas nmi_pair10_<ae>.json). Atlas 14 topics =
mean of the three draws; * = the document-bootstrap 95% CI of the difference excludes zero in every draw (on the Δ's
metric).

    python eval_out/full_pair_table.py        # Qwen2.5-3B L35
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
Q = "results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/seq_table/"
ap.add_argument("--probe", default=Q + "probe_dense_subsets_fourway.json")
ap.add_argument("--atlas_dir", default=Q + "atlas_lastword/")
ap.add_argument("--title", default="Qwen2.5-3B, layer 35 (K = 2,000)")
args = ap.parse_args()

GROUPS = [
    ("Token-level", [("surface", "Surface form"), ("pos_coarse", "POS, coarse (12)"), ("pos_fine", "POS, Penn (44)"),
                     ("ner_coarse", "Entity type, coarse (9)"), ("ner_fine", "Entity type, fine (67)")]),
    ("Sequence-level", [("sentiment", "Sentiment, SST-2 (2)"), ("sentiment_long", "Sentiment, IMDB (2)"),
                        ("subjectivity", "Subjectivity (2)"), ("formality", "Formality (3)"), ("language", "Language (20)"),
                        ("domain", "Domain (3)"), ("topic4", "Topic, AG News (4)"), ("topic14", "Topic, DBpedia-14 (14)"),
                        ("topic20", "Topic, 20 Newsgroups (20)"), ("toxicity", "Toxicity, Jigsaw (2)"),
                        ("toxicity_pattern", "Toxicity flags, Jigsaw (~30)")]),
    ("Entity and relation", [("ravel_country", "RAVEL country"), ("ravel_continent", "RAVEL continent"),
                             ("ravel_language", "RAVEL language"), ("ioi_role", "IOI role"), ("ioi_name", "IOI name")]),
]
PAIRS = [("res_plain", "nobypass", "plain42"), ("tok_plain", "geoae", "plain42_tokmean")]
AE_ATLAS = {"nobypass": "nobypass", "geoae": "bypass"}
KEYS = ("homogeneity", "completeness", "nmi")
R = json.load(open(args.probe))["rungs"]


def s3(x):
    return (f"{x:.3f}" if x >= 0 else f"−{-x:.3f}").replace("0.", ".", 1)


def tri(v):
    return "/".join(s3(x) for x in v)


def dl(a, b, star=""):
    d = b - a
    return ("+" if d >= 0 else "−") + f"{abs(d):.3f}".replace("0.", ".", 1) + star


def probe_pair(rungs, km, ae, grp):
    sub = "min10" if grp == "Token-level" else f"pair_fp10_{km}+{ae}"
    a = [sum(R[r]["arms"][km][sub][k] for r in rungs) / len(rungs) for k in KEYS]
    b = [sum(R[r]["arms"][ae][sub][k] for r in rungs) / len(rungs) for k in KEYS]
    kept = sum(min(R[r]["arms"][km][sub]["n"], R[r]["arms"][ae][sub]["n"]) / R[r]["n"] for r in rungs) / len(rungs)
    return a, b, "", "", kept


def atlas_pair(sets, km, ae):
    A = json.load(open(Path(args.atlas_dir) / f"nmi_pair10_{AE_ATLAS[ae]}.json"))["sets"]
    kmn, aen = km, AE_ATLAS[ae]
    a = [sum(A[s]["point"][kmn][k] for s in sets) / len(sets) for k in KEYS]
    b = [sum(A[s]["point"][aen][k] for s in sets) / len(sets) for k in KEYS]
    star = lambda m: "*" if all(c[0] > 0 or c[1] < 0 for c in
                                (A[s]["diff_ref_minus_arm"][kmn][m]["ci"] for s in sets)) else ""
    kept = sum(A[s]["kept_share"] for s in sets) / len(sets)
    return a, b, star("homogeneity"), star("nmi"), kept


print(f"### {args.title}: every task, raw homogeneity / completeness / NMI\n")
print("| Task | Resid. k-means | No-bypass AE | Δh | ΔNMI | kept | Tok.-rem. k-means | Bypass AE | Δh | ΔNMI | kept |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
blocks = [(g, [(n, [r]) for r, n in t if r in R] + [("*Mean*", [r for r, _ in t if r in R])], "probe") for g, t in GROUPS]
atl = [("14 topics (3 draws)", ["atlas14_s0", "atlas14_s1", "atlas14_s2"]), ("Document type (29)", ["doctype"])]
if all((Path(args.atlas_dir) / f"nmi_pair10_{v}.json").exists() for v in AE_ATLAS.values()):
    blocks.append(("FineWeb-Atlas, last word of each chunk", atl, "atlas"))
for grp, rows, kind in blocks:
    print(f"| **{grp}** |" + " |" * 10)
    for name, rungs in rows:
        cells = []
        for km, ae, kma in PAIRS:
            a, b, sh, sn, kept = probe_pair(rungs, km, ae, grp) if kind == "probe" else atlas_pair(rungs, kma, ae)
            cells += [tri(a), tri(b), dl(a[0], b[0], sh), dl(a[2], b[2], sn), f"{100 * kept:.0f}%"]
        print(f"| {name} | " + " | ".join(cells) + " |")
