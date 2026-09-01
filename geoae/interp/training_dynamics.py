"""
Across-epoch autopsy of a GeoAE run: what the latent is actually doing, epoch by
epoch, at every scale we can measure.

Motivation. Training logs report loss terms, which say whether an objective is
being minimised but not what the representation is becoming. Several conclusions
in this project have been overturned by looking at the representation instead
(the logged silhouette is computed on Sinkhorn labels over a 2048-token batch and
is meaningless; a concept-alignment win turned out to be the balancing, not the
encoder). This module reconstructs the trajectory from saved checkpoints so the
run can be debugged rather than guessed at.

REQUIRES the trajectory to exist: train with save_every=1 and keep_checkpoints
larger than n_epochs. The default keep_checkpoints=3 rotates it away.

What it measures per checkpoint
-------------------------------
LATENT SCALE / SHAPE      |z|, per-dim std (min/mean/frac below the VICReg
                          hinge), effective rank, variance concentration.
CLUSTER GEOMETRY          live clusters, usage top-10 share and perplexity,
                          fitted Zipf alpha, mean intra radius, min/mean
                          inter-centroid distance, inter/intra separation,
                          silhouette on argmin labels (NOT Sinkhorn labels).
MOVEMENT (vs prev ckpt)   centroid drift, assignment churn (tokens changing
                          cluster), and latent CKA — separates "the space
                          rotated" from "the space restructured".
RECONSTRUCTION            FVE.
SINGLE-TOKEN SEMANTICS    per-cluster token entropy -> monosemanticity, and the
                          share of clusters dominated by one token type.
CONCEPT ALIGNMENT         Atlas concepts, split three ways so coarse and fine
                          structure are visible separately:
                            by TYPE   document (genre, sequence-level, abstract)
                                      tone (register, sequence-level, abstract)
                                      content (topical)
                                      entity (concrete, often name-like)
                            by SIZE   frequent vs rare concepts (big vs small)
                            by READOUT sequence-level presence, sequence-level
                                      token-FRACTION (a cluster covering 40% of a
                                      chunk should not score like one covering 1%),
                                      and token-level precision (does a SINGLE
                                      token identify the concept?)
TRACKED CONCEPTS          a curated ladder from abstract/large to concrete/small,
                          each reported with its best cluster ID per epoch so
                          cluster IDENTITY stability is visible, not just score.

Usage:
    python -m geoae.interp.training_dynamics \
      --checkpoints checkpoints/llama3.2-3B/layer27/k2000_sq3072_bnhinge_bigbatch_mse \
      --out dynamics_bigbatch.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from geoae.checkpoint import load_ae_checkpoint

# A ladder of concepts spanning abstract->concrete and large->small. Tracked
# individually so we can watch WHICH cluster owns each one and whether that
# assignment is stable across epochs.
TRACKED = [
    # (name, why it is here)
    ("Short announcement/bulletin", "largest genre, very abstract"),
    ("News report",                 "large genre"),
    ("Recipe/cooking instructions", "small genre, highly distinctive"),
    ("Technical specification/project proposal", "small genre, best-aligned cluster historically"),
    ("Formal",                      "abstract register"),
    ("Casual",                      "abstract register"),
    ("Cricket",                     "mid-size topic, distinctive vocabulary"),
    ("Cryptocurrency",              "mid-size topic"),
    ("Classical music",             "mid-size topic"),
    ("Version control",             "tiny topic, raw k-means won this one"),
    ("Radioactive decay",           "tiny topic"),
]


def zipf_alpha(counts: np.ndarray) -> tuple[float, float]:
    nz = np.sort(counts[counts > 0])[::-1]
    if len(nz) < 10:
        return float("nan"), float("nan")
    r = np.arange(1, len(nz) + 1)
    slope, icept = np.polyfit(np.log(r), np.log(nz), 1)
    pred = np.exp(icept + slope * np.log(r))
    ss_res = ((np.log(nz) - np.log(pred)) ** 2).sum()
    ss_tot = ((np.log(nz) - np.log(nz).mean()) ** 2).sum()
    return float(-slope), float(1 - ss_res / ss_tot)


def effective_rank(z: torch.Tensor, n: int = 20000) -> float:
    zc = (z[:n] - z[:n].mean(0)).float()
    sv = torch.linalg.svdvals(zc)
    p = sv / sv.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def linear_cka(a: torch.Tensor, b: torch.Tensor, n: int = 8000) -> float:
    """How similar are two latent spaces on the same inputs, up to rotation+scale."""
    x = (a[:n] - a[:n].mean(0)).float()
    y = (b[:n] - b[:n].mean(0)).float()
    xty = (x.T @ y).norm() ** 2
    xx = (x.T @ x).norm()
    yy = (y.T @ y).norm()
    return float(xty / (xx * yy + 1e-12))


# ---------------------------------------------------------------------------
# Concept alignment, three readouts
# ---------------------------------------------------------------------------

def concept_scores(lab: np.ndarray, owner: np.ndarray, chunk_concepts: list[set],
                   n_clusters: int, min_support: int = 8, frac_thresh: float = 0.10):
    """
    Returns {concept_id: dict(f1_presence, f1_frac, tok_prec, best_cluster, n)}.

    presence : chunk is positive for cluster k if ANY token lands in k. This is
               the historical readout, and it is generous — a ~100-token chunk
               touches ~60 of 2000 clusters and the largest covers only ~10% of
               it, so one token counts the same as fifty.
    frac     : chunk counts for k only if >= frac_thresh of its tokens are in k.
               Asks whether a cluster actually CHARACTERISES the chunk.
    tok_prec : token-level. Of all TOKENS in cluster k, what fraction sit in a
               chunk carrying the concept? Answers whether a single token
               identifies a sequence-level property at all.
    """
    n_chunks = int(owner.max()) + 1
    # per (chunk, cluster) token counts
    tok_tot = np.bincount(owner, minlength=n_chunks).astype(np.float64)
    pair = defaultdict(int)
    for k, c in zip(lab, owner):
        pair[(int(c), int(k))] += 1

    present = defaultdict(list)   # cluster -> chunks (any token)
    strong = defaultdict(list)    # cluster -> chunks (>= frac_thresh)
    ktok = Counter()              # cluster -> total tokens
    for (c, k), n in pair.items():
        present[k].append(c)
        ktok[k] += n
        if n / max(tok_tot[c], 1) >= frac_thresh:
            strong[k].append(c)

    ccount = Counter()
    for cs in chunk_concepts:
        for c in cs:
            ccount[c] += 1

    # concept -> chunk membership as a boolean per chunk, for fast intersect
    out = {}
    for concept, nc in ccount.items():
        if nc < min_support:
            continue
        member = np.zeros(n_chunks, dtype=bool)
        for ci, cs in enumerate(chunk_concepts):
            if concept in cs:
                member[ci] = True
        best = (0.0, None, 0.0, 0.0)
        for k, chunks in present.items():
            if len(chunks) < min_support:
                continue
            arr = np.fromiter(chunks, int, len(chunks))
            hit = int(member[arr].sum())
            if hit == 0:
                continue
            p, r = hit / len(arr), hit / nc
            f1 = 2 * p * r / (p + r)
            if f1 > best[0]:
                # token-level precision for this cluster
                tp = sum(n for (c2, k2), n in pair.items() if k2 == k and member[c2])
                # fraction readout for the same cluster
                sc = strong.get(k, [])
                if sc:
                    sarr = np.fromiter(sc, int, len(sc))
                    sh = int(member[sarr].sum())
                    sp, sr = sh / len(sarr), sh / nc
                    f1f = 2 * sp * sr / (sp + sr) if sp + sr else 0.0
                else:
                    f1f = 0.0
                best = (f1, k, f1f, tp / max(ktok[k], 1))
        out[concept] = dict(f1_presence=best[0], best_cluster=best[1],
                            f1_frac=best[2], tok_prec=best[3], n=nc)
    return out


# ---------------------------------------------------------------------------

@torch.no_grad()
def analyse(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.atlas)
    H, owner = d["H"], d["owner"]
    if args.n_tokens and args.n_tokens < len(H):
        H, owner = H[: args.n_tokens], owner[: args.n_tokens]
    if not np.isfinite(H[:1000]).all():
        raise SystemExit("non-finite activations in the cache — see the fp16 overflow note")
    lab_df = pd.read_parquet(args.labels)
    concepts = pd.read_parquet(args.concepts)
    cname = dict(zip(concepts.concept_id, concepts.name))
    name2id = {v: k for k, v in cname.items()}
    n_chunks = int(owner.max()) + 1
    fields = ["document_ids", "tone_ids", "content_ids", "entity_ids"]
    chunk_sets = {f: [set(lab_df.iloc[c][f]) for c in range(n_chunks)] for f in fields}
    tok_ids = np.load(args.token_ids)["ids"][: len(H)] if args.token_ids else None

    ckpts = sorted(Path(args.checkpoints).glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise SystemExit(f"no step_*.pt in {args.checkpoints} — was keep_checkpoints too small?")
    print(f"[dyn] {len(ckpts)} checkpoints, {len(H):,} tokens, {n_chunks:,} chunks")

    Ht = torch.from_numpy(H).to(dev)
    rows, prev_z, prev_lab, prev_C = [], None, None, None
    for cp in ckpts:
        ae, mean, std, ck = load_ae_checkpoint(cp, dev)
        ae.eval()
        xn = (Ht - mean) / std
        z = torch.cat([ae.encoder(xn[i:i + 16384]) for i in range(0, len(xn), 16384)])
        xh = torch.cat([ae.decoder(z[i:i + 16384]) for i in range(0, len(z), 16384)])
        fve = float(1 - ((xn - xh) ** 2).sum() / ((xn - xn.mean(0)) ** 2).sum())
        C = ae.centroids
        lab = torch.cat([torch.cdist(z[i:i + 16384], C).argmin(1)
                         for i in range(0, len(z), 16384)])
        labn = lab.cpu().numpy()

        cnt = Counter(labn.tolist())
        counts = np.zeros(ae.n_clusters); counts[list(cnt)] = list(cnt.values())
        p = counts / counts.sum()
        nzp = p[p > 0]
        used = torch.tensor(sorted(cnt), device=dev)
        dd = torch.cdist(C[used], C[used]); dd.fill_diagonal_(float("inf"))
        intra = float(np.mean([float((z[lab == k] - C[k]).norm(dim=1).mean())
                               for k in list(cnt)[:300]]))
        std_dims = z.std(0)
        alpha, r2 = zipf_alpha(counts)

        r = dict(
            step=int(cp.stem.split("_")[1]), epoch=int(ck["epoch"]),
            fve=round(fve, 4),
            z_norm=round(float(z.norm(dim=1).mean()), 2),
            dim_std_mean=round(float(std_dims.mean()), 3),
            dim_std_min=round(float(std_dims.min()), 3),
            dim_below_hinge=round(float((std_dims < 1.0).float().mean()), 3),
            erank=round(effective_rank(z), 1),
            live=len(cnt),
            top10=round(float(np.sort(p)[::-1][:10].sum()), 4),
            usage_ppl=round(float(np.exp(-(nzp * np.log(nzp)).sum())), 1),
            zipf_alpha=round(alpha, 3),
            intra=round(intra, 2),
            inter_min=round(float(dd.min()), 2),
            inter_mean=round(float(dd[dd.isfinite()].mean()), 2),
            sep_ratio=round(float(dd.min()) / max(intra, 1e-9), 4),
        )
        if args.silhouette:
            from sklearn.metrics import silhouette_score
            n = min(args.silhouette, len(z))
            r["silhouette"] = round(float(silhouette_score(
                z[:n].cpu().numpy(), labn[:n], sample_size=min(10000, n), random_state=0)), 4)
        # movement vs previous checkpoint
        if prev_z is not None:
            r["churn"] = round(float((lab != prev_lab).float().mean()), 4)
            r["centroid_drift"] = round(float((C - prev_C).norm(dim=1).mean() / C.norm(dim=1).mean()), 4)
            r["cka_prev"] = round(linear_cka(z, prev_z), 4)
        # single-token semantics
        if tok_ids is not None:
            ents, doms = [], []
            for k in list(cnt)[:400]:
                t = Counter(tok_ids[labn == k].tolist())
                tot = sum(t.values())
                if tot < 5: continue
                pr = np.array(list(t.values())) / tot
                ents.append(1 - (-(pr * np.log(pr)).sum() / math.log(max(len(pr), 2))))
                doms.append(pr.max())
            r["monosemanticity"] = round(float(np.mean(ents)), 3)
            r["top_token_share"] = round(float(np.mean(doms)), 3)
        # concept alignment
        for f in fields:
            sc = concept_scores(labn, owner, chunk_sets[f], ae.n_clusters,
                                frac_thresh=args.frac_thresh)
            if not sc: continue
            arr = np.array([v["f1_presence"] for v in sc.values()])
            frac = np.array([v["f1_frac"] for v in sc.values()])
            tp = np.array([v["tok_prec"] for v in sc.values()])
            ns = np.array([v["n"] for v in sc.values()])
            tag = f.replace("_ids", "")
            r[f"{tag}_f1"] = round(float(arr.mean()), 4)
            r[f"{tag}_f1frac"] = round(float(frac.mean()), 4)
            r[f"{tag}_tokprec"] = round(float(tp.mean()), 4)
            if len(arr) > 20:  # big vs small concepts
                med = np.median(ns)
                r[f"{tag}_f1_big"] = round(float(arr[ns >= med].mean()), 4)
                r[f"{tag}_f1_small"] = round(float(arr[ns < med].mean()), 4)
            for nm, _why in TRACKED:
                cid = name2id.get(nm)
                if cid in sc:
                    r[f"T::{nm}"] = round(sc[cid]["f1_presence"], 3)
                    r[f"K::{nm}"] = sc[cid]["best_cluster"]
        rows.append(r)
        print(f"  epoch {r['epoch']:>3}  fve {r['fve']:.3f}  sep {r['sep_ratio']:.3f}  "
              f"live {r['live']:>4}  doc {r.get('document_f1', float('nan')):.3f}  "
              f"churn {r.get('churn', float('nan')):.3f}", flush=True)
        prev_z, prev_lab, prev_C = z, lab, C.clone()
    return rows


def report(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    def block(title, cols, fmt="{:>9.4f}"):
        cols = [c for c in cols if c in df.columns]
        if not cols: return
        print(f"\n--- {title} " + "-" * max(0, 84 - len(title)))
        print(f"{'epoch':>6}" + "".join(f"{c[:12]:>13}" for c in cols))
        for _, r in df.iterrows():
            cells = "".join(
                (f"{r[c]:>13.4f}" if isinstance(r[c], float) else f"{str(r[c]):>13}")
                for c in cols)
            print(f"{int(r.epoch):>6}" + cells)

    print("\n" + "=" * 92)
    print("TRAINING DYNAMICS — what the latent is doing, epoch by epoch")
    print("=" * 92)
    block("RECONSTRUCTION & LATENT SHAPE", ["fve", "z_norm", "dim_std_mean", "dim_std_min",
                                            "dim_below_hinge", "erank"])
    block("CLUSTER GEOMETRY", ["live", "top10", "usage_ppl", "zipf_alpha", "intra",
                               "inter_min", "sep_ratio", "silhouette"])
    block("MOVEMENT vs PREVIOUS EPOCH", ["churn", "centroid_drift", "cka_prev"])
    block("SINGLE-TOKEN SEMANTICS", ["monosemanticity", "top_token_share"])
    block("CONCEPT F1 by TYPE (sequence-level, presence)",
          ["document_f1", "tone_f1", "content_f1", "entity_f1"])
    block("CONCEPT F1 by READOUT (document type)",
          ["document_f1", "document_f1frac", "document_tokprec"])
    block("CONCEPT F1 by READOUT (content)",
          ["content_f1", "content_f1frac", "content_tokprec"])
    block("BIG vs SMALL concepts (content)", ["content_f1_big", "content_f1_small"])
    block("BIG vs SMALL concepts (tone)", ["tone_f1_big", "tone_f1_small"])

    tcols = [c for c in df.columns if c.startswith("T::")]
    if tcols:
        print("\n--- TRACKED CONCEPTS: F1 (and owning cluster id) " + "-" * 40)
        for c in tcols:
            nm = c[3:]
            kc = f"K::{nm}"
            first, last = df[c].iloc[0], df[c].iloc[-1]
            ids = df[kc].dropna().tolist() if kc in df.columns else []
            stable = (len(set(ids)) if ids else 0)
            print(f"  {nm[:44]:<46} {first:.3f} -> {last:.3f}   "
                  f"owning cluster changed {stable} time(s), final {ids[-1] if ids else '-'}")

    print("\n--- HOW TO READ " + "-" * 76)
    print("  sep_ratio   min inter-centroid distance / mean intra radius. <1 means the")
    print("              nearest two clusters overlap. This is the separation number.")
    print("  churn       fraction of tokens that changed cluster since the previous epoch.")
    print("              High late = the partition never settles; ~0 early = frozen.")
    print("  cka_prev    latent similarity to the previous epoch up to rotation/scale.")
    print("              High churn WITH high CKA = re-labelling, not restructuring.")
    print("  f1frac      a cluster only counts for a chunk if it covers >=frac_thresh of")
    print("              its tokens. Much stricter than presence, which a 1-token hit passes.")
    print("  tokprec     given ONE token in the cluster, P(its chunk carries the concept).")
    print("              This is the token-level vs sequence-level gap.")
    print("  dim_below_hinge  fraction of latent dims with std < 1 (the VICReg hinge).")
    print("              Rising = the variance term is losing to cluster contraction.")


def main():
    SP = "/tmp/claude-1001/-home-exouser-RepresentationAE-GeoAE/870ea3ff-b19b-4c54-a3f1-e8289a510213/scratchpad/"
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, help="dir containing step_*.pt")
    ap.add_argument("--atlas", default=SP + "atlas8k.npz")
    ap.add_argument("--labels", default=SP + "atlas8k_labels.parquet")
    ap.add_argument("--concepts", default=SP + "concepts.parquet")
    ap.add_argument("--token_ids", default=None, help="npz with 'ids' aligned to atlas rows")
    ap.add_argument("--n_tokens", type=int, default=300_000, help="0 = all (slower)")
    ap.add_argument("--frac_thresh", type=float, default=0.10)
    ap.add_argument("--silhouette", type=int, default=20000, help="0 to skip")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = analyse(args)
    report(rows)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2, default=str)
        print(f"\n[dyn] wrote {args.out}")


if __name__ == "__main__":
    main()
