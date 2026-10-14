"""Concept probe with CHANCE-CORRECTED NMI and paired bootstrap resamples.

Why. concept_probe's NMI rises with the number of clusters a rung's rows spread
over, even for random labels. Arms that change that spread (token-erased codebooks
put sequence rungs over ~2.6x more clusters) are then not comparable on raw NMI:
on atlas_doc the shuffled-label NMI moved 0.20 -> 0.34. Here each arm is scored as

    cNMI = (NMI - NMI_shuffled) / (1 - NMI_shuffled)

with NMI_shuffled the mean over 5 label permutations, and every arm is scored on
the SAME bootstrap resamples of each rung, so arm differences are paired.

Same rungs, anchor holdout and 80k token-rung subsample as concept_probe.

Token-bypass AEs (model.token_bias) are scored only on rungs whose cache stores
token ids (token_id, or last_token_id beside H_last). Each rung also reports
`leak` = cNMI(label; current token id): the bypass is a function of the token id
alone, so this bounds how much of the concept the bypass can carry around the latent.

    scripts/delta/py -u eval_out/probe_chance_corrected.py \
        --models d6144_new=<ckpt> --baselines km_a=<npz>,km_b=<npz> --out eval_out/x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geoae.checkpoint import load_ae_checkpoint  # noqa: E402
from geoae.interp import concept_probe as CP  # noqa: E402
from geoae.interp.closest_tokens import load_baseline_kmeans  # noqa: E402
from geoae.interp.sae_space import load_sae_kmeans, load_sae_space, parse_saes  # noqa: E402
from geoae.seeded_init import anchor_row_indices  # noqa: E402


def nmi_weighted(y: np.ndarray, c: np.ndarray, w: np.ndarray) -> float:
    """Arithmetic-mean NMI of integer labels y vs clusters c, rows weighted by w."""
    ny, nc = y.max() + 1, c.max() + 1
    J = np.bincount(y * nc + c, weights=w, minlength=ny * nc).reshape(ny, nc)
    N = J.sum()
    py, pc, P = J.sum(1) / N, J.sum(0) / N, J / N
    nz = P > 0
    mi = (P[nz] * np.log(P[nz] / (py[:, None] * pc[None, :])[nz])).sum()
    h = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    return float(mi / max((h(py) + h(pc)) / 2, 1e-12))


def sae_assign(H, sae, dev, ids=None):
    """Top-1 SAE feature per row (-1 = no active feature, its own label). None if a token-bypass SAE has no ids."""
    if sae.needs_tokens and ids is None:
        return None
    out = []
    for i in range(0, len(H), 16384):
        h = torch.from_numpy(H[i:i + 16384]).to(dev).float()
        t = None if ids is None else torch.from_numpy(ids[i:i + 16384]).to(dev)
        out.append(sae.assign_labels(h, t).cpu().numpy())
    return np.concatenate(out)


def load_arms(models: str, baselines: str, saes: str, dev, sae_kmeans: str = ""):
    """Same arm loading as main() (kept separate so main stays byte-for-byte as it was): returns (models, saes) dicts
    ready for CP.assign / sae_assign. Used by eval_out/atlas_nmi.py."""
    arms = {}
    for spec in filter(None, models.split(",")):
        n, p = spec.split("=", 1)
        ae, mu, sd, _ = load_ae_checkpoint(p, dev, allow_token_bias=True)
        ae.eval()
        arms[n] = ("ae", (ae, mu, sd))
    for spec in filter(None, baselines.split(",")):
        n, p = spec.split("=", 1)
        C, m, s, *_ = load_baseline_kmeans(Path(p), dev, allow_erasure=True)
        meta = np.load(p, allow_pickle=True)
        if "space" in meta.files and str(meta["space"]) == "latent":
            raise SystemExit(f"{p}: latent-space codebooks are not supported here")
        U = (torch.as_tensor(meta["erase_U"], dtype=torch.float32, device=dev)
             if "erase_U" in meta.files else None)
        from geoae.token_bias import TokenBiasLookup
        tbl = TokenBiasLookup(str(meta["token_bias"]), dev) if "token_bias" in meta.files else None
        arms[n] = ("km", (C, m, s, U, tbl))
    sae_d = {n: load_sae_space(sp, dev, n) for n, sp in parse_saes(saes).items()}
    for n, sp in parse_saes(sae_kmeans).items():
        sae_d[n] = load_sae_kmeans(sp, dev, n)
    return arms, sae_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--models", default="", help="name=ae_checkpoint,...")
    ap.add_argument("--baselines", default="", help="name=kmeans.npz,... (erased codebooks allowed)")
    ap.add_argument("--saes", default="", help="name=spec,... (matched:<last.pt> | scope:<id|dir>); "
                    "cluster = top-1 active feature, unassigned rows = one extra label")
    ap.add_argument("--sae_kmeans", default="", help="name=<sae spec>@<centroids.npz>,...: nearest centroid in that "
                    "SAE's code space (control C1; centroids from eval_out/sae_controls_fit.py)")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--rungs", default="", help="comma list of rung names; empty = all")
    ap.add_argument("--n_token", type=int, default=80000)
    ap.add_argument("--anchor_seed", type=int, default=42)
    ap.add_argument("--atlas_last", default="cache/atlas8k_last.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_labels", default="", help="also write <npz>: per rung the class ids (<rung>__y), current-token "
                    "ids (<rung>__ids, when the rung has them) and every arm's cluster ids (<rung>__<arm>), for scoring "
                    "outside this script (eval_out/probe_hc_filtered.py)")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = {}

    models = {}
    for spec in filter(None, args.models.split(",")):
        n, p = spec.split("=", 1)
        ae, mu, sd, _ = load_ae_checkpoint(p, dev, allow_token_bias=True)
        ae.eval()
        models[n] = ("ae", (ae, mu, sd))
    for spec in filter(None, args.baselines.split(",")):
        n, p = spec.split("=", 1)
        C, m, s, *_ = load_baseline_kmeans(Path(p), dev, allow_erasure=True)
        meta = np.load(p, allow_pickle=True)
        if "space" in meta.files and str(meta["space"]) == "latent":
            raise SystemExit(f"{p}: latent-space codebooks are not supported here")
        U = (torch.as_tensor(meta["erase_U"], dtype=torch.float32, device=dev)
             if "erase_U" in meta.files else None)
        from geoae.token_bias import TokenBiasLookup
        tbl = TokenBiasLookup(str(meta["token_bias"]), dev) if "token_bias" in meta.files else None
        models[n] = ("km", (C, m, s, U, tbl))
    saes = {n: load_sae_space(sp, dev, n) for n, sp in parse_saes(args.saes).items()}
    for n, sp in parse_saes(args.sae_kmeans).items():
        saes[n] = load_sae_kmeans(sp, dev, n)
    print(f"[cprobe] arms: {list(models)}  saes: {list(saes)}")

    excl = anchor_row_indices(args.cache, None, 25, 5, args.anchor_seed, args.atlas_last, 25)
    enc = lambda v: np.unique(v, return_inverse=True)[1]
    rng = np.random.default_rng(0)
    out = {}
    only = set(filter(None, args.rungs.split(",")))
    for rung, fname, key, grain, _desc in CP.LADDER:
        if only and rung not in only:
            continue
        if not (Path(args.cache) / fname).exists():
            continue
        H, y = CP.load_rung(args.cache, fname, key, 0)
        ids = CP.load_rung_ids(args.cache, fname)
        if excl and rung in excl:
            keep = np.ones(len(y), bool)
            keep[excl[rung][excl[rung] < len(y)]] = False
            H, y = H[keep], y[keep]
            ids = None if ids is None else ids[keep]
        if grain == "token" and len(H) > args.n_token:
            sub = np.random.RandomState(0).choice(len(H), args.n_token, replace=False)
            H, y = H[sub], y[sub]
            ids = None if ids is None else ids[sub]
        yi = enc(np.asarray(y).astype(str))
        if args.save_labels:
            saved[f"{rung}__y"] = yi.astype(np.int32)
            if ids is not None:
                saved[f"{rung}__ids"] = np.asarray(ids).astype(np.int64)
        ones = np.ones(len(yi))
        W = rng.multinomial(len(yi), ones / len(yi), size=args.n_boot).astype(float)
        perms = [rng.permutation(yi) for _ in range(5)]
        def score(r, n, lab, **extra):
            c = enc(lab)   # a -1 (no active feature) label is just one more cluster id
            nm = nmi_weighted(yi, c, ones)
            null = float(np.mean([nmi_weighted(p, c, ones) for p in perms]))
            boots = [(nmi_weighted(yi, c, W[b]) - null) / (1 - null) for b in range(args.n_boot)]
            r["arms"][n] = dict(nmi=nm, null=null, cnmi=(nm - null) / (1 - null),
                                live=int(c.max() + 1), boots=boots, **extra)

        r = {"grain": grain, "n": int(len(yi)), "arms": {}, "leak": None}
        if ids is not None:
            c = enc(ids)
            nm = nmi_weighted(yi, c, ones)
            null = float(np.mean([nmi_weighted(p, c, ones) for p in perms]))
            r["leak"] = (nm - null) / (1 - null)
        for n, mdl in models.items():
            lab = CP.assign(H, mdl, dev, ids=ids)
            if lab is None:
                continue
            if args.save_labels:
                saved[f"{rung}__{n}"] = np.asarray(lab).astype(np.int32)
            score(r, n, lab)
        for n, sae in saes.items():
            lab = sae_assign(H, sae, dev, ids)
            if lab is None:
                continue
            score(r, n, lab, unassigned=float((lab < 0).mean()))
        out[rung] = r
        leak = "   —" if r["leak"] is None else f"{r['leak']:.3f}"
        print(f"[cprobe] {rung:<16} n={len(yi):>6}  leak={leak}  " + "  ".join(
            f"{n}={a['cnmi']:.3f}" for n, a in r["arms"].items()), flush=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"[cprobe] wrote {args.out}")
    if args.save_labels:
        np.savez_compressed(args.save_labels, **saved)
        print(f"[cprobe] wrote {args.save_labels}")


if __name__ == "__main__":
    main()
