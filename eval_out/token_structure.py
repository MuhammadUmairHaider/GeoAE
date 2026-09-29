"""Does a codebook organise by token identity or by context? Held-out dump rows.

For each arm, on the val tail of a sampled dump (rows never trained on):
  fvu          reconstruction, full normalised space (AEs only; bypass included)
  cNMI cur     cluster vs CURRENT token, chance-corrected (label-shuffle null)
  cNMI next    cluster vs the LM's predicted NEXT token (final norm + lm_head; L27 only)
  trivial cur  share of clusters (>= 20 members) whose single most common current
               token is >= 80% of the cluster: pure token detectors, the Tokenized-SAE
               notion of a trivial feature. Also at >= 50%, and the same for next token.
  live / top10 / H  codebook usage: live clusters, share of the 10 biggest, usage entropy / log K

A token-bypass AE should drop cNMI cur and trivial cur without losing fvu.

    scripts/delta/py -u eval_out/token_structure.py \
        --arms d6144_new=<ckpt>,tokbias=<ckpt>,km_new=<npz> --out eval_out/token_structure_tokbias.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoae.checkpoint import load_ae_checkpoint, load_lm  # noqa: E402
from geoae.interp.closest_tokens import load_baseline_kmeans  # noqa: E402
from probe_chance_corrected import nmi_weighted  # noqa: E402


def cnmi(y, c, rng):
    ones = np.ones(len(y))
    nm = nmi_weighted(y, c, ones)
    null = np.mean([nmi_weighted(rng.permutation(y), c, ones) for _ in range(3)])
    return (nm - null) / (1 - null)


def trivial_share(key, lab, thr, min_size=20):
    order = np.lexsort((key, lab))
    k, l = key[order], lab[order]
    bounds = np.flatnonzero(np.diff(l)) + 1
    n_big = n_triv = 0
    for seg_k in np.split(k, bounds):
        if len(seg_k) < min_size:
            continue
        n_big += 1
        n_triv += np.bincount(np.unique(seg_k, return_inverse=True)[1]).max() / len(seg_k) >= thr
    return n_triv / max(n_big, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="name=ae_ckpt_or_kmeans_npz,...")
    ap.add_argument("--activations_dir", default="activations_sampled_10M")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--n_eval", type=int, default=500_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    act = Path(args.activations_dir)
    X = np.load(str(act / f"layer_{args.layer}.npy"), mmap_mode="r")
    tok_all = np.load(act / "rows_tok.npy")
    N = len(X)
    ev = np.arange(int(N * (1 - args.val_frac)), N)[: args.n_eval]
    cur = tok_all[ev].astype(np.int64)
    meta = json.load(open(act / "meta.json"))
    lm = load_lm(meta["model"], device=dev)
    head, norm = lm.get_output_embeddings(), lm.model.norm
    nxt = np.empty(len(ev), np.int64)
    with torch.no_grad():
        for i in range(0, len(ev), 8192):
            h = torch.from_numpy(np.array(X[ev[i:i + 8192]])).to(dev, head.weight.dtype)
            nxt[i:i + 8192] = head(norm(h)).argmax(-1).cpu().numpy()
    del lm, head, norm
    torch.cuda.empty_cache()
    print(f"[tokstruct] {len(ev):,} held-out rows of {act}")

    out = {}
    for spec in args.arms.split(","):
        name, path = spec.split("=", 1)
        if path.endswith(".npz"):
            C, mean, std, *_ = load_baseline_kmeans(Path(path), dev, allow_erasure=True)
            meta_k = np.load(path, allow_pickle=True)
            U = (torch.as_tensor(meta_k["erase_U"], device=dev) if "erase_U" in meta_k.files else None)
            from geoae.token_bias import TokenBiasLookup
            tbl = TokenBiasLookup(str(meta_k["token_bias"]), dev) if "token_bias" in meta_k.files else None
            ae = None
        else:
            ae, mean, std, _ = load_ae_checkpoint(path, dev, allow_token_bias=True)
            ae.eval()
            C = ae.centroids
        lab = np.empty(len(ev), np.int64)
        se = tot = 0.0
        with torch.no_grad():
            xs = []
            for i in range(0, len(ev), 16384):
                x = (torch.from_numpy(np.array(X[ev[i:i + 16384]])).to(dev).float() - mean) / std
                t = torch.from_numpy(cur[i:i + 16384]).to(dev)
                if ae is None:
                    v = x if tbl is None else x - tbl(t)
                    v = v if U is None else v - (v @ U) @ U.T
                else:
                    tt = t if ae.has_token_bias else None
                    v = ae.encode(x, tt)
                    se += float((ae.decode(v, tt) - x).pow(2).sum())
                    xs.append(x.sum(0)); tot += float(x.pow(2).sum())
                lab[i:i + 16384] = torch.cdist(v, C).argmin(1).cpu().numpy()
        r = {}
        if ae is not None:
            mu = torch.stack(xs).sum(0) / len(ev)
            r["fvu"] = se / (tot - len(ev) * float(mu.pow(2).sum()))
        K = C.shape[0]
        use = np.bincount(lab, minlength=K) / len(lab)
        p = use[use > 0]
        r.update(live=int((use > 0).sum()), top10=float(np.sort(use)[-10:].sum()),
                 usage_H=float(-(p * np.log(p)).sum() / np.log(K)),
                 cnmi_cur=cnmi(cur, lab, rng), cnmi_next=cnmi(nxt, lab, rng),
                 trivial_cur_80=trivial_share(cur, lab, .8), trivial_cur_50=trivial_share(cur, lab, .5),
                 trivial_next_80=trivial_share(nxt, lab, .8), trivial_next_50=trivial_share(nxt, lab, .5))
        out[name] = r
        print(f"[tokstruct] {name:12s} " + "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[tokstruct] wrote {args.out}")


if __name__ == "__main__":
    main()
