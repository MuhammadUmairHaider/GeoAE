"""Token-level Atlas in detail: dumps every space's per-row cluster label on the whole Atlas cache, and scores each
continuous space by neighbourhoods and a linear probe instead of a hard partition.

The headline token-grain Atlas cNMI (eval_out/atlas_nmi.py) is one number over ~9 correlated rows per chunk (8 sampled
positions + the final token). The label dump lets CPU-side analyses split it by position, by token type, keep one row
per chunk, and separate same-chunk co-clustering from cross-document concept grouping. The space scores ask the same
question without a codebook:
  knn   label agreement of the k=10 Euclidean nearest neighbours (in the space the clustering uses), excluding rows of
        the query's own chunk ("chunk") or own document ("doc"); chance = sum_c p_c^2, kappa = (agree - chance)/(1 - chance)
  probe multinomial logistic regression (LBFGS, standardised features, fixed weight decay) on a document-disjoint
        80/20 split; test accuracy and balanced accuracy
Spaces: base (h - mean)/std, base_tokmean (minus b[tok]), bypass z (AE encoder output), and SAE codes.

    python -u eval_out/atlas_token_detail.py --cache C --rows_dir SHM --models bypass=<ckpt> \
        --baselines base=<npz>,base_tokmean=<npz> --saes "mat10k=matched:<last.pt>,bt10k=matched:<last.pt>" \
        --sae_kmeans "mat10k_km=matched:<last.pt>@<kmeans_mat10k.npz>" --out_dir OUT
Writes OUT/assign.npz (int32 label per cache row per arm), OUT/knn_rows.npz (per-row chunk-excluded agreement, for
position / token-type splits) and OUT/space_scores.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geoae.interp import atlas_siblings as AS  # noqa: E402

GRAINS = ("token", "last")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def embed(H, arm, dev, ids, chunk=8192):
    """Continuous coordinates of rows H in an arm's space: ('ae', ...) -> z, ('km', ...) -> the standardised (and
    token-mean-subtracted / erased) residual the codebook lives in, ('sae', space) -> dense SAE codes."""
    import torch
    kind, obj = arm
    out = []
    with torch.no_grad():
        for i in range(0, len(H), chunk):
            h = torch.from_numpy(np.asarray(H[i:i + chunk])).to(dev).float()
            t = torch.from_numpy(ids[i:i + chunk]).to(dev)
            if kind == "ae":
                ae, mean, std = obj
                v = ae.encode((h - mean) / std, t if ae.has_token_bias else None)
            elif kind == "km":
                C, mean, std, U, tb = obj
                v = (h - mean) / std
                if tb is not None:
                    v = v - tb(t)
                if U is not None:
                    v = v - (v @ U) @ U.T
            else:
                v = obj.encode_raw(h, t if obj.needs_tokens else None)
            out.append(v.float())
    return torch.cat(out)


def knn_scores(V, y, group, k=10, block=1024, tok=None, mode="all"):
    """Per-row share of the k nearest neighbours (Euclidean, rows of the same group excluded) carrying the row's label,
    and the majority-vote prediction. With `tok`: mode "same_tok" keeps only neighbours with the query's current token
    (rows with < k such candidates get NaN), "diff_tok" only neighbours with another token; mode "all" also returns
    the share of neighbours that share the query's token."""
    import torch
    n = len(V)
    sq = (V * V).sum(1)
    agree = torch.empty(n, device=V.device)
    pred = torch.empty(n, dtype=torch.long, device=V.device)
    same = torch.empty(n, device=V.device)
    n_cls = int(y.max()) + 1
    for i in range(0, n, block):
        j = slice(i, min(i + block, n))
        d = sq[j, None] + sq[None, :] - 2 * (V[j] @ V.T)
        d[group[j, None] == group[None, :]] = float("inf")
        if mode == "same_tok":
            d[tok[j, None] != tok[None, :]] = float("inf")
        elif mode == "diff_tok":
            d[tok[j, None] == tok[None, :]] = float("inf")
        val, idx = d.topk(k, largest=False)
        ny = y[idx]
        a = (ny == y[j, None]).float().mean(1)
        agree[j] = torch.where(torch.isfinite(val).all(1), a, torch.full_like(a, float("nan")))
        pred[j] = torch.nn.functional.one_hot(ny, n_cls).sum(1).argmax(1)
        if tok is not None:
            same[j] = (tok[idx] == tok[j, None]).float().mean(1)
    return (agree, pred, same) if tok is not None else (agree, pred)


def balanced_acc(y, p):
    cls = np.unique(y)
    return float(np.mean([(p[y == c] == c).mean() for c in cls]))


def linear_probe(V, y, doc, seed=0, wd=1e-3, iters=300):
    """Document-disjoint 80/20 split; standardised features; multinomial logistic regression (LBFGS)."""
    import torch
    rng = np.random.default_rng(seed)
    docs = np.unique(doc)
    test_docs = rng.choice(docs, max(1, len(docs) // 5), replace=False)
    te = np.isin(doc, test_docs)
    tr = ~te
    Vt = V[torch.from_numpy(tr).to(V.device)]
    mu, sd = Vt.mean(0), Vt.std(0).clamp_min(1e-6)
    X = (V - mu) / sd
    yt = torch.from_numpy(y).to(V.device)
    n_cls = int(y.max()) + 1
    W = torch.zeros(X.shape[1], n_cls, device=V.device, requires_grad=True)
    b = torch.zeros(n_cls, device=V.device, requires_grad=True)
    trm = torch.from_numpy(tr).to(V.device)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X[trm] @ W + b, yt[trm]) + wd * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        p = (X @ W + b).argmax(1).cpu().numpy()
    return dict(train_acc=float((p[tr] == y[tr]).mean()), test_acc=float((p[te] == y[te]).mean()),
                test_bacc=balanced_acc(y[te], p[te]), n_train=int(tr.sum()), n_test=int(te.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rows_dir", default=None)
    ap.add_argument("--draws", default="results/delta/d6144/interventions/atlas_classes/draws.json")
    ap.add_argument("--atlas_dir", default="/scratch/dogar/geoae/atlas")
    ap.add_argument("--models", default="")
    ap.add_argument("--baselines", default="")
    ap.add_argument("--saes", default="")
    ap.add_argument("--sae_kmeans", default="")
    ap.add_argument("--space_arms", default="base,base_tokmean,bypass,mat10k",
                    help="arms scored by knn / probe (continuous coordinates)")
    ap.add_argument("--n_token", type=int, default=40000, help="token-grain rows per label set for knn / probe")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tok_knn_only", action="store_true",
                    help="skip the label dump and the plain knn / probe; write OUT/knn_tok.json: chunk-excluded knn "
                         "agreement restricted to same-token / other-token neighbours, and the same-token share")
    args = ap.parse_args()

    import pandas as pd
    import torch

    import probe_chance_corrected as PCC
    from atlas_nmi import build_label_sets, select_rows, subsample
    from geoae.interp import concept_probe as CP

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = AS.AtlasCache(args.cache, args.rows_dir)
    row_pos = np.load(Path(args.rows_dir or args.cache) / "rows_pos.npy", mmap_mode="r")
    tok_all = np.asarray(cache.tok, dtype=np.int64)
    chunk_all = np.asarray(cache.row_chunk, dtype=np.int64)
    log(f"cache {cache.n_rows:,} rows, {cache.n_chunks:,} chunks, d={cache.rows.shape[1]}")

    arms, saes = PCC.load_arms(args.models, args.baselines, args.saes, dev, sae_kmeans=args.sae_kmeans)
    log(f"arms {list(arms)}  saes {list(saes)}")
    if args.tok_knn_only:
        return tok_knn(args, cache, row_pos, tok_all, chunk_all, arms, saes, dev, out)

    # ---- 1. labels of every arm on every cache row
    lab = {}
    B = 65536
    for name in list(arms) + list(saes):
        parts = []
        t0 = time.time()
        for i in range(0, cache.n_rows, B):
            H = np.asarray(cache.rows[i:i + B])
            ids = tok_all[i:i + B]
            if name in arms:
                parts.append(CP.assign(H, arms[name], dev, ids=ids))
            else:
                parts.append(PCC.sae_assign(H, saes[name], dev, ids))
        lab[name] = np.concatenate(parts).astype(np.int32)
        log(f"  labels {name}: {len(np.unique(lab[name])):,} distinct  ({time.time() - t0:.0f}s)")
    np.savez_compressed(out / "assign.npz", **lab, row_chunk=chunk_all.astype(np.int32),
                        row_pos=np.asarray(row_pos, dtype=np.int32), row_tok=tok_all.astype(np.int32),
                        chunk_doc=cache.doc.astype(np.int64), chunk_ntok=np.asarray(cache.n_tok, dtype=np.int32))
    log(f"wrote {out / 'assign.npz'}")

    # ---- 2. knn + linear probe in the continuous spaces
    draws = json.load(open(args.draws))
    conc = pd.read_parquet(Path(args.atlas_dir) / "fineweb-concept-atlas.parquet")
    sets = build_label_sets(cache, draws, conc)
    space = [a for a in args.space_arms.split(",") if a]
    res = {"meta": vars(args), "sets": {}}
    rows_out = {}
    for sname, ls in sets.items():
        res["sets"][sname] = {}
        for grain in GRAINS:
            idx, y = select_rows(cache.row_start, ls["chunks"], ls["labels"], grain)
            if grain == "last":
                ok = row_pos[idx] == (cache.n_tok[cache.row_chunk[idx]] - 1)
                idx, y = idx[ok], y[ok]
            if grain == "token" and len(idx) > args.n_token:
                s = subsample(len(idx), args.n_token)
                idx, y = idx[s], y[s]
            y = np.unique(y, return_inverse=True)[1]
            ch = chunk_all[idx]
            doc = cache.doc[ch]
            H = np.asarray(cache.rows[idx])
            ids = tok_all[idx]
            p = np.bincount(y) / len(y)
            g = dict(n_rows=int(len(y)), n_chunks=int(len(np.unique(ch))), n_docs=int(len(np.unique(doc))),
                     chance=float((p ** 2).sum()), arms={})
            rows_out[f"{sname}/{grain}/idx"] = idx.astype(np.int64)
            yt = torch.from_numpy(y).to(dev)
            for a in space:
                arm = arms[a] if a in arms else ("sae", saes[a])
                V = embed(H, arm, dev, ids)
                r = dict(dim=int(V.shape[1]))
                for gname, grp in (("chunk", ch), ("doc", doc)):
                    ag, pr = knn_scores(V, yt, torch.from_numpy(grp).to(dev), k=args.k)
                    ag, pr = ag.cpu().numpy(), pr.cpu().numpy()
                    r[f"knn_{gname}"] = dict(agree=float(ag.mean()),
                                             kappa=float((ag.mean() - g["chance"]) / (1 - g["chance"])),
                                             maj_acc=float((pr == y).mean()), maj_bacc=balanced_acc(y, pr))
                    if gname == "chunk":
                        rows_out[f"{sname}/{grain}/{a}"] = ag.astype(np.float32)
                r["probe"] = linear_probe(V, y, doc)
                g["arms"][a] = r
                del V
                torch.cuda.empty_cache()
                log(f"  {sname:<11} {grain:<5} {a:<13} knn(chunk) kappa {r['knn_chunk']['kappa']:.3f}  "
                    f"knn(doc) kappa {r['knn_doc']['kappa']:.3f}  probe bacc {r['probe']['test_bacc']:.3f}")
            res["sets"][sname][grain] = g
            (out / "space_scores.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out / "knn_rows.npz", **rows_out)
    log(f"wrote {out / 'space_scores.json'} and knn_rows.npz")


def tok_knn(args, cache, row_pos, tok_all, chunk_all, arms, saes, dev, out):
    """Token-grain knn split by whether neighbours share the query's current token: same_tok asks whether the space
    separates one word's occurrences by topic, diff_tok whether it groups different words by topic."""
    import pandas as pd
    import torch

    from atlas_nmi import build_label_sets, select_rows, subsample
    draws = json.load(open(args.draws))
    conc = pd.read_parquet(Path(args.atlas_dir) / "fineweb-concept-atlas.parquet")
    sets = build_label_sets(cache, draws, conc)
    res = {"meta": vars(args), "sets": {}}
    for sname, ls in sets.items():
        idx, y = select_rows(cache.row_start, ls["chunks"], ls["labels"], "token")
        if len(idx) > args.n_token:
            s = subsample(len(idx), args.n_token)
            idx, y = idx[s], y[s]
        y = np.unique(y, return_inverse=True)[1]
        ch = torch.from_numpy(chunk_all[idx]).to(dev)
        ids = tok_all[idx]
        tk = torch.from_numpy(ids).to(dev)
        yt = torch.from_numpy(y).to(dev)
        H = np.asarray(cache.rows[idx])
        g = {}
        for a in [a for a in args.space_arms.split(",") if a]:
            V = embed(H, arms[a] if a in arms else ("sae", saes[a]), dev, ids)
            r = {}
            ag, _, same = knn_scores(V, yt, ch, k=args.k, tok=tk, mode="all")
            r["all"] = dict(agree=float(ag.mean()), same_tok_share=float(same.mean()))
            for mode, k in (("same_tok", 5), ("diff_tok", args.k)):
                ag, _, _ = knn_scores(V, yt, ch, k=k, tok=tk, mode=mode)
                ok = torch.isfinite(ag)
                r[mode] = dict(agree=float(ag[ok].mean()), n_valid=int(ok.sum()), k=k)
            # chance for same_tok: label agreement of random same-token, other-chunk pairs is what a space with no
            # within-word topic separation would score; estimate it from the same-token candidate pool
            g[a] = r
            del V
            torch.cuda.empty_cache()
            log(f"  {sname:<11} {a:<13} all {r['all']['agree']:.3f} (same-token share {r['all']['same_tok_share']:.2f})"
                f"  same_tok {r['same_tok']['agree']:.3f} (n {r['same_tok']['n_valid']})  diff_tok {r['diff_tok']['agree']:.3f}")
        # same-token chance: mean over valid queries of the label agreement with ALL same-token other-chunk rows
        yv, tv, cv = y, ids, chunk_all[idx]
        key = tv.astype(np.int64)
        order = np.argsort(key, kind="stable")
        bounds = np.r_[0, np.flatnonzero(np.diff(key[order])) + 1, len(key)]
        num = den = 0.0
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            m = order[lo:hi]
            if len(m) < 6:
                continue
            yy, cc = yv[m], cv[m]
            same_y = (yy[:, None] == yy[None, :]) & (cc[:, None] != cc[None, :])
            cand = cc[:, None] != cc[None, :]
            ok = cand.sum(1) >= 5
            num += (same_y.sum(1)[ok] / cand.sum(1)[ok]).sum()
            den += ok.sum()
        g["same_tok_chance"] = float(num / max(den, 1))
        log(f"  {sname:<11} same-token chance agreement {g['same_tok_chance']:.3f}")
        res["sets"][sname] = g
        (out / "knn_tok.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
