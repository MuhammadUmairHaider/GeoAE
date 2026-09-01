"""
Is a concept more LINEARLY SEPARABLE in the AE latent than in the raw residual?

The Atlas probe (best-single-cluster F1) asks whether some cluster happens to
coincide with a concept. That is a question about the PARTITION, and by it the
encoder contributes nothing — balanced k-means with an identity encoder matches
the AE. But two spaces can induce equally good partitions while differing in how
separable the concepts are as REPRESENTATIONS, which is the property that matters
for probing, steering and transfer. This measures that instead.

Per concept, on chunk-level mean-pooled features:
  auc   : test AUC of a ridge ("least-squares") linear probe, positives vs rest
  fisher: (between-class scatter) / (within-class scatter) along the class-mean
          difference — scale-free, no fitting

THE CONTROL THAT MAKES IT MEAN SOMETHING.  z = GELU(Wh + b) is a nonlinear
feature map, and ANY random nonlinear expansion improves linear separability
(the random-features / kernel effect). So a trained encoder beating raw h proves
nothing on its own. `--random_control` adds an UNTRAINED encoder of identical
shape and init: the trained AE has to beat THAT to have learned anything.

Fairness notes:
  * every space is standardised per-dimension before probing, so the comparison
    is not decided by scale (h and z norms differ severalfold here);
  * the ridge Gram matrix is shared across concepts, so hundreds of probes cost
    one Cholesky per space rather than one fit per concept;
  * train/test split is over CHUNKS, so no chunk contributes to both.

Usage:
    python -m geoae.interp.concept_separability \
      --acts <cached atlas .npz> --labels <cached labels .parquet> \
      --checkpoints <ae1.pt> <ae2.pt> --random_control
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@torch.no_grad()   # without this the encoder's autograd graph is retained for
                   # EVERY chunk and the accumulation OOMs a 40 GB card
def pooled(H: np.ndarray, owner: np.ndarray, n_chunks: int, enc=None,
           mean=None, std=None, dev="cuda", chunk=8192, how="mean") -> torch.Tensor:
    """
    Pool token features per chunk, optionally through an encoder first.

    THE AE IS TRAINED PER TOKEN. Every pooled representation is something the
    encoder never saw in training, so "last" and "token" below are the faithful
    evaluations and the pooled ones are conveniences:

    how="last": the chunk's final token. In a causal LM at layer 27/28 that
        position has attended over the whole chunk, so it carries sequence-level
        content while still being a SINGLE token of the kind the AE was trained
        on. This is the same point `_shared.capture_h` reads for the
        classification work elsewhere in this repo.
    how="token": no pooling at all — every token is a sample, inheriting its
        chunk's labels. Noisiest (most tokens carry no concept signal) but it is
        exactly the regime the AE operates in.
    how="mean": conventional chunk vector, but DILUTES localised concepts.
    how="max": per-dimension max; matches the F1 probe's any-token aggregation.
        Note raw h is ~zero-centred while GELU latents are non-negative, so max
        is not quite the same operation across the two spaces.
    """
    if how == "last":
        # index of the final token of each chunk
        idx = np.zeros(n_chunks, dtype=np.int64)
        np.maximum.at(idx, owner.astype(np.int64), np.arange(len(owner)))
        rows = []
        for i in range(0, len(idx), chunk):
            h = torch.from_numpy(H[idx[i:i + chunk]]).to(dev).float()
            rows.append(enc((h - mean) / std) if enc is not None else h)
        return torch.cat(rows)

    out = None
    counts = torch.zeros(n_chunks, device=dev)
    own = torch.from_numpy(owner.astype(np.int64)).to(dev)
    for i in range(0, len(H), chunk):
        h = torch.from_numpy(H[i:i + chunk]).to(dev).float()
        v = enc((h - mean) / std) if enc is not None else h
        if out is None:
            out = torch.zeros(n_chunks, v.shape[1], device=dev)
            if how == "max":
                out.fill_(float("-inf"))
        o = own[i:i + chunk]
        if how == "mean":
            out.index_add_(0, o, v)
            counts.index_add_(0, o, torch.ones_like(o, dtype=torch.float))
        else:
            out.index_reduce_(0, o, v, "amax", include_self=True)
    if how == "max":
        return torch.where(torch.isinf(out), torch.zeros_like(out), out)
    return out / counts.clamp_min(1).unsqueeze(1)


@torch.no_grad()
def separability(X: torch.Tensor, ys: dict[int, torch.Tensor], tr: torch.Tensor,
                 te: torch.Tensor, lam: float = 1.0) -> dict[int, tuple[float, float]]:
    """Ridge-probe test AUC and Fisher ratio per concept. Gram is shared."""
    X = (X - X[tr].mean(0)) / X[tr].std(0).clamp_min(1e-6)      # standardise on train
    Xtr, Xte = X[tr], X[te]
    d = X.shape[1]
    A = Xtr.T @ Xtr + lam * torch.eye(d, device=X.device)
    Lc = torch.linalg.cholesky(A)
    res = {}
    for cid, y in ys.items():
        ytr = y[tr].float() * 2 - 1
        w = torch.cholesky_solve((Xtr.T @ ytr).unsqueeze(1), Lc).squeeze(1)
        s = Xte @ w
        yte = y[te].bool()
        npos, nneg = int(yte.sum()), int((~yte).sum())
        if npos < 3 or nneg < 3:
            continue
        # AUC via rank statistic
        r = torch.argsort(torch.argsort(s)).float() + 1
        auc = float((r[yte].sum() - npos * (npos + 1) / 2) / (npos * nneg))
        mp, mn = X[y.bool()].mean(0), X[~y.bool()].mean(0)
        vp, vn = X[y.bool()].var(0).mean(), X[~y.bool()].var(0).mean()
        fisher = float(((mp - mn) ** 2).sum() / (vp + vn).clamp_min(1e-8))
        res[cid] = (auc, fisher)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, help="cached .npz with H and owner")
    ap.add_argument("--labels", required=True, help="cached labels .parquet")
    ap.add_argument("--checkpoints", nargs="+", default=[])
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--random_control", action="store_true",
                    help="Add an UNTRAINED encoder of the same shape (random-features control)")
    ap.add_argument("--min_pos", type=int, default=25)
    ap.add_argument("--max_concepts", type=int, default=80, help="Per concept type")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default="last", choices=["last", "token", "mean", "max"],
                    help="last/token are token-level (what the AE is trained on); "
                         "mean/max are pooled chunk vectors")
    ap.add_argument("--max_tokens", type=int, default=250_000,
                    help="--pool token: tokens subsampled to this many (memory)")
    args = ap.parse_args()

    from geoae.checkpoint import load_ae_checkpoint
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.acts)
    H, owner = d["H"], d["owner"]
    if not np.isfinite(H).all():
        raise SystemExit("non-finite cached activations — re-cache in float32")
    lab = pd.read_parquet(args.labels)
    n = len(lab)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    nte = int(n * args.test_frac)
    te, tr = perm[:nte].to(dev), perm[nte:].to(dev)
    print(f"[sep] {H.shape[0]:,} tokens -> {n} chunks | train {len(tr)} / test {len(te)}")

    fields = [("document_ids", "DOC TYPE"), ("tone_ids", "TONE"), ("content_ids", "CONTENT")]
    ys_by_field = {}
    for f, _ in fields:
        cnt = Counter(c for row in lab[f] for c in row)
        keep = [c for c, k in cnt.most_common(args.max_concepts) if k >= args.min_pos]
        ys = {}
        for c in keep:
            v = torch.tensor([1 if c in row else 0 for row in lab[f]], device=dev)
            ys[c] = v
        ys_by_field[f] = ys
        print(f"[sep] {f}: {len(ys)} concepts with >={args.min_pos} positives")

    print(f"[sep] pooling: {args.pool}")
    # Token-level mode: every token is a sample inheriting its chunk's labels.
    # The train/test split MUST stay by chunk — tokens from one chunk landing in
    # both splits would leak the label.
    tok_sub = None
    if args.pool == "token":
        rng = np.random.RandomState(args.seed)
        tok_sub = np.sort(rng.choice(len(H), min(args.max_tokens, len(H)), replace=False))
        own_sub = torch.from_numpy(owner[tok_sub].astype(np.int64)).to(dev)
        chunk_is_test = torch.zeros(n, dtype=torch.bool, device=dev)
        chunk_is_test[te] = True
        tok_te = torch.nonzero(chunk_is_test[own_sub]).squeeze(1)
        tok_tr = torch.nonzero(~chunk_is_test[own_sub]).squeeze(1)
        print(f"[sep] token mode: {len(tok_sub):,} tokens "
              f"(train {len(tok_tr):,} / test {len(tok_te):,})")

    def build(enc=None, mean=None, std=None):
        if args.pool != "token":
            return pooled(H, owner, n, enc, mean, std, dev, how=args.pool)
        rows = []
        for i in range(0, len(tok_sub), 8192):
            h = torch.from_numpy(H[tok_sub[i:i + 8192]]).to(dev).float()
            rows.append(enc((h - mean) / std) if enc is not None else h)
        return torch.cat(rows)

    specs = [("raw h", None, None, None)]
    names = args.names or [Path(pp).parent.name for pp in args.checkpoints]
    from geoae.model import GeoAE
    for name, pp in zip(names, args.checkpoints):
        ae, m, sd, _ = load_ae_checkpoint(pp, dev)
        ae.eval()
        specs.append((name, ae.encoder, m, sd))
        if args.random_control:
            cfg = torch.load(pp, map_location="cpu", weights_only=False)["config"]["model"]
            torch.manual_seed(args.seed)
            rnd = GeoAE(hidden_size=cfg["hidden_size"], latent_dim=cfg["latent_dim"],
                        n_clusters=cfg["n_clusters"], nonlinearity=cfg.get("nonlinearity", "gelu"),
                        metric=cfg.get("metric", "euclidean"),
                        latent_norm=cfg.get("latent_norm", "none")).to(dev).eval()
            specs.append((f"RANDOM enc ({name})", rnd.encoder, m, sd))
            args.random_control = False

    # One arm at a time: holding every arm's features at token scale would OOM.
    results = {f: {} for f, _ in fields}
    for nm, enc, m, sd in specs:
        X = build(enc, m, sd)
        for f, _ in fields:
            ys = ys_by_field[f]
            if not ys:
                continue
            if args.pool == "token":
                ys_t = {c: v[own_sub] for c, v in ys.items()}
                results[f][nm] = separability(X, ys_t, tok_tr, tok_te, args.lam)
            else:
                results[f][nm] = separability(X, ys, tr, te, args.lam)
        del X
        torch.cuda.empty_cache()

    from scipy import stats
    for f, title in fields:
        res = results[f]
        if not res or not res.get("raw h"):
            continue
        common = set.intersection(*[set(r) for r in res.values()])
        if not common:
            continue
        print(f"\n=== {title}  ({len(common)} concepts, pool={args.pool}) ===")
        print(f"{'space':<30}{'mean AUC':>10}{'median':>9}{'mean Fisher':>13}{'vs raw h':>11}")
        base = np.mean([res["raw h"][c][0] for c in common])
        for nm, r in res.items():
            a = np.array([r[c][0] for c in common])
            fi = np.array([r[c][1] for c in common])
            print(f"{nm:<30}{a.mean():>10.4f}{np.median(a):>9.4f}{fi.mean():>13.4f}"
                  f"{a.mean() - base:>+11.4f}")
        for nm, r in list(res.items())[1:]:
            dlt = np.array([r[c][0] - res["raw h"][c][0] for c in common])
            t, pv = stats.ttest_1samp(dlt, 0)
            print(f"   {nm} - raw h: {dlt.mean():+.4f}  p={pv:.2g}  wins {100*(dlt>0).mean():.0f}%")


if __name__ == "__main__":
    main()
