"""
Score a set of clusterings against the whole concept ladder at once.

Answers, for any AE checkpoint or raw-centroid baseline: WHICH KINDS of structure
does this partition carry, from single-token surface form up to document genre?

Two metrics per rung, because they answer different questions and this project
has seen them disagree:

  NMI  normalised mutual information between the cluster partition and the label.
       "Does the partition know about this property at all." Chance-corrected, so
       it is comparable across rungs with 2 classes and rungs with 12,786.
  F1   mean over classes of the best SINGLE cluster for that class. "Is there one
       readable cluster that means this." A partition can score high NMI while no
       individual cluster is interpretable — that gap is itself the finding.

Baselines (`--baselines`) matter more than usual here: Sinkhorn-balanced k-means
with NO encoder has matched or beaten every AE configuration tried so far, so a
rung where the AE does not beat it is a rung where the encoder is not earning
its cost.

Usage:
    python -m geoae.interp.concept_probe --models auto --baselines auto
    python -m geoae.interp.concept_probe --models ckpt_a=path/a.pt,ckpt_b=path/b.pt
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score as nmi_score

from geoae.checkpoint import load_ae_checkpoint
from geoae.interp.closest_tokens import load_baseline_kmeans

SP = "/tmp/claude-1001/-home-exouser-RepresentationAE-GeoAE/870ea3ff-b19b-4c54-a3f1-e8289a510213/scratchpad/"

# rung -> (cache file, label key, grain, human description)
LADDER = [
    ("surface",       "pos.npz",            "surface",     "token",    "orthographic class"),
    ("pos_coarse",    "pos.npz",            "_coarse",     "token",    "12 POS classes"),
    ("pos_fine",      "pos.npz",            "label",       "token",    "44 Penn tags"),
    ("ner_coarse",    "ner.npz",            "_nercoarse",  "token",    "9 entity types"),
    ("ner_fine",      "ner.npz",            "label",       "token",    "67 entity types"),
    ("sentiment",     "sentiment.npz",      "label",       "sequence", "SST-2"),
    ("sentiment_long","sentiment_long.npz", "label",       "sequence", "IMDB"),
    ("subjectivity",  "subjectivity.npz",   "label",       "sequence", "subj/obj"),
    ("formality",     "formality.npz",      "label",       "sequence", "3 buckets"),
    # FineWeb-Atlas, CHUNK LAST-TOKEN activations (the chunk's final token has
    # attended over the whole chunk at a late layer, and is still a single token
    # of the kind the AE trains on; a mean-pooled chunk vector is off-distribution).
    # atlas_doc uses only chunks with EXACTLY ONE document label — unambiguous.
    # tone and content have ZERO single-label chunks (~6-7 labels each), so they
    # need a multi-label -> single reduction; treat them as derived, not ground
    # truth. tone takes the RAREST-in-corpus label (most specific). Content CANNOT
    # use that rule — with 6,461 concepts over 7,742 chunks the rarest label is
    # near-unique per chunk and NO class reaches 25 positives — so it takes the
    # MOST COMMON label instead (65 classes, 6,012 chunks).
    ("atlas_doc",     "atlas_doc.npz",      "label",       "sequence", "Atlas document type, single-label"),
    ("atlas_tone",    "atlas_tone.npz",     "label",       "sequence", "Atlas tone, rarest-label reduction"),
    ("atlas_content", "atlas_content.npz",  "label",       "sequence", "Atlas content, rarest-label reduction"),
    ("language",      "language.npz",       "label",       "sequence", "20 languages"),
    ("domain",        "domain.npz",         "label",       "sequence", "code/math/prose"),
    ("topic4",        "topic4.npz",         "label",       "sequence", "AG News"),
    ("topic14",       "topic14.npz",        "label",       "sequence", "DBpedia-14"),
    ("topic20",       "topic20.npz",        "label",       "sequence", "20 Newsgroups"),
    # --- MIB benchmark rungs (geoae.interp.benchmark_cache) ------------------
    # RAVEL: one entity token, three INDEPENDENT attributes. Scoring high on one
    # while low on another is disentanglement, which single-label purity cannot see.
    ("ravel_country",   "ravel.npz", "Country",   "token", "RAVEL entity attribute"),
    ("ravel_continent", "ravel.npz", "Continent", "token", "RAVEL entity attribute"),
    ("ravel_language",  "ravel.npz", "Language",  "token", "RAVEL entity attribute"),
    # IOI: role is positional, and every NAME appears in every role, so token
    # identity carries no information about it. ioi_name is the paired control —
    # it IS token identity, so a merely lexical clustering scores high there and
    # low on ioi_role. Read the two together, never ioi_role alone.
    ("ioi_role",        "ioi.npz",   "role",      "token", "IOI S1/IO/S2 position"),
    ("ioi_name",        "ioi.npz",   "name",      "token", "IOI name identity (lexical control)"),
]

POS_COARSE = {
    "NN": "NOUN", "NNS": "NOUN", "NNP": "PROPN", "NNPS": "PROPN",
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBP": "VERB",
    "VBZ": "VERB", "MD": "VERB", "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "PRP": "PRON", "PRP$": "PRON",
    "WP": "PRON", "WP$": "PRON", "EX": "PRON", "DT": "DET", "PDT": "DET",
    "WDT": "DET", "IN": "ADP", "TO": "PART", "RP": "PART", "POS": "PART",
    "CC": "CONJ", "CD": "NUM", "UH": "INTJ", "FW": "X",
}


def load_rung(cache_dir, fname, key, n_max):
    d = np.load(Path(cache_dir) / fname, allow_pickle=True)
    H = d["H"] if "H" in d.files else d["H_last"]
    if key == "_coarse":
        y = np.array([POS_COARSE.get(t, "OTHER") for t in d["label"]], dtype=object)
    elif key == "_nercoarse":
        y = np.load(Path(cache_dir) / "ner_coarse.npy", allow_pickle=True)[: len(H)]
    else:
        y = d[key]
    if n_max and len(H) > n_max:
        idx = np.random.RandomState(0).choice(len(H), n_max, replace=False)
        H, y = H[idx], np.asarray(y)[idx]
    return np.asarray(H, dtype=np.float32), np.asarray(y)


def best_f1(lab, y, min_support=20):
    kc = Counter(lab.tolist())
    scores = []
    for cls in np.unique(y):
        m = y == cls
        n = int(m.sum())
        if n < min_support:
            continue
        best = 0.0
        for k, h in Counter(lab[m].tolist()).items():
            if kc[k] < min_support:
                continue
            p, r = h / kc[k], h / n
            f = 2 * p * r / (p + r)
            best = max(best, f)
        scores.append(best)
    return float(np.mean(scores)) if scores else float("nan")


@torch.no_grad()
def assign(H, model, dev, chunk=16384):
    kind, obj = model
    out = []
    for i in range(0, len(H), chunk):
        h = torch.from_numpy(H[i : i + chunk]).to(dev).float()
        if kind == "ae":
            ae, mean, std = obj
            v, C = ae.encoder((h - mean) / std), ae.centroids
        else:
            C, mean, std = obj
            v = (h - mean) / std
        out.append(torch.cdist(v, C).argmin(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--models", default="auto")
    ap.add_argument("--baselines", default="auto")
    ap.add_argument("--n_token", type=int, default=80000, help="subsample for token rungs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude_anchors", action="store_true",
                    help="drop the rows that seeded init consumed as anchors. REQUIRED when "
                         "scoring a centroid_init=seeded model: anchors come from these same "
                         "caches, so without this the seeded rungs report a fake win (37%% of "
                         "ravel_country was anchors on the first seeded run).")
    ap.add_argument("--anchor_per_class", type=int, default=25)
    ap.add_argument("--anchor_min_examples", type=int, default=5)
    ap.add_argument("--anchor_seed", type=int, default=0)
    ap.add_argument("--atlas_last", default="",
                    help="atlas last-token anchor file, to hold Atlas anchor CHUNKS out. "
                         "REQUIRED when scoring a model trained with atlas_last, or the "
                         "atlas rungs are ~78%% contaminated.")
    ap.add_argument("--atlas_min_examples", type=int, default=25)
    args = ap.parse_args()

    excl = None
    if args.exclude_anchors:
        from geoae.seeded_init import anchor_row_indices
        excl = anchor_row_indices(args.cache, None, args.anchor_per_class,
                                  args.anchor_min_examples, args.anchor_seed,
                                  args.atlas_last, args.atlas_min_examples)
        print(f"[probe] excluding anchor rows from {len(excl)} rungs "
              f"(per_class={args.anchor_per_class}, seed={args.anchor_seed})")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = []
    if args.models == "auto":
        cands = [
            ("AE e2e-KL lam0.05", "e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/best_val.pt"),
            ("AE mse lam2 median", "checkpoints/llama3.2-3B/layer27/k2000_sq3072_strong_mse/step_0113900.pt"),
            ("AE mse lam2 wide", "checkpoints/llama3.2-3B/layer27/k2000_wide6144_mse/step_0113900.pt"),
            ("AE BN+hinge", "checkpoints/llama3.2-3B/layer27/k2000_sq3072_bnhinge_mse/step_0113900.pt"),
            ("AE BN+hinge wide", "checkpoints/llama3.2-3B/layer27/k2000_wide6144_bnhinge_mse/step_0113900.pt"),
            ("AE K=256", "e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k256_vicreg/best_val.pt"),
        ]
        specs += [(n, p, "ae") for n, p in cands if os.path.exists(p)]
    else:
        specs += [(s.split("=")[0], s.split("=")[1], "ae") for s in args.models.split(",") if s]
    if args.baselines == "auto":
        cands = [("balanced kmeans (no enc)", "e2e/checkpoints/general/llama3.2-3B/layer27/balanced_kmeans_k2000.npz"),
                 ("plain kmeans", "e2e/checkpoints/general/llama3.2-3B/layer27/baseline_kmeans_k2000_refit.npz")]
        specs += [(n, p, "km") for n, p in cands if os.path.exists(p)]
    elif args.baselines:
        # Explicit name=path.npz list. Without this branch a non-"auto" value
        # silently added NO baseline and the run reported the AE alone — which
        # is the one comparison that cannot answer anything, since the whole
        # question is AE vs encoder-free control.
        for s in args.baselines.split(","):
            if not s:
                continue
            n, pth = s.split("=", 1)
            if not os.path.exists(pth):
                raise SystemExit(f"[probe] baseline not found: {pth}")
            specs.append((n, pth, "km"))

    models = {}
    for name, path, kind in specs:
        if kind == "ae":
            ae, mean, std, _ = load_ae_checkpoint(path, dev)
            ae.eval()
            models[name] = ("ae", (ae, mean, std))
        else:
            C, mean, std, K, *_ = load_baseline_kmeans(Path(path), dev)
            models[name] = ("km", (C, mean, std))
        print(f"[probe] loaded {name}")

    res = defaultdict(dict)
    for rung, fname, key, grain, desc in LADDER:
        fp = Path(args.cache) / fname
        if not fp.exists():
            continue
        # Load FULL, drop anchor rows, THEN subsample — excluding after the
        # subsample would not line the indices up.
        H, y = load_rung(args.cache, fname, key, 0)
        if excl and rung in excl:
            keep = np.ones(len(y), dtype=bool)
            keep[excl[rung][excl[rung] < len(y)]] = False
            n_drop = int((~keep).sum())
            H, y = H[keep], y[keep]
            print(f"[probe] {rung:<15} excluded {n_drop:,} anchor rows "
                  f"({100 * n_drop / (n_drop + len(y)):.1f}% of the rung)")
        cap = args.n_token if grain == "token" else 0
        if cap and len(H) > cap:
            sub = np.random.RandomState(0).choice(len(H), cap, replace=False)
            H, y = H[sub], y[sub]
        for name, m in models.items():
            lab = assign(H, m, dev)
            res[rung][name] = dict(nmi=round(float(nmi_score(y, lab)), 4),
                                   f1=round(best_f1(lab, y), 4),
                                   live=int(len(set(lab.tolist()))))
        print(f"[probe] {rung:<15} {grain:<9} n={len(H):>7,} classes={len(np.unique(y)):>3}")

    names = list(models)
    for metric in ("nmi", "f1"):
        print(f"\n{'=' * (26 + 14 * len(names))}")
        print(f"{metric.upper()}  (higher = better)")
        print(f"{'rung':<16}{'grain':<10}" + "".join(f"{n[:13]:>14}" for n in names))
        for rung, _f, _k, grain, desc in LADDER:
            if rung not in res:
                continue
            row = "".join(f"{res[rung][n][metric]:>14.4f}" for n in names)
            print(f"{rung:<16}{grain:<10}{row}")

        # TOKEN vs SEQUENCE aggregate. These two groups move independently and
        # often in OPPOSITE directions -- the AE is fit on token activations, so
        # token rungs are where the training signal lives and sequence rungs are
        # extrapolation. A single grand mean over the ladder hides that entirely,
        # so it is never reported on its own.
        print(f"{'-' * (26 + 14 * len(names))}")
        for grp in ("token", "sequence"):
            vals = [[res[r][n][metric] for r in res
                     if dict((x[0], x[3]) for x in LADDER).get(r) == grp] for n in names]
            if not vals[0]:
                continue
            row = "".join(f"{sum(v) / len(v):>14.4f}" for v in vals)
            print(f"{'MEAN ' + grp.upper():<16}{'(' + str(len(vals[0])) + ' rungs)':<10}{row}")

    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"\n[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
