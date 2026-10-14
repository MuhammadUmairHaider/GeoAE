"""
Per-token bias table for the token-bypass GeoAE (design B).

The AE subtracts b[t] from the normalised residual before encoding and adds it
back after decoding (model.token_bias), so token identity is carried AROUND the
latent: the clusters organise by context, while the reconstruction — and hence
the splice — stays faithful. See geoae/model.py (GeoAE.token_bias).

    b[t] = n_t / (n_t + k) * mean_{rows with current token t}( (h - mu) / sigma )

computed over the TRAIN split only, in exactly the normalisation training uses
(the dump's norm cache). The shrinkage keeps a token seen 3,000 times at 99% of
its mean and one seen 30 times at 50%, and pulls noisy means of rare tokens to
zero, i.e. to the plain AE's behaviour for that token; no hard cutoff where
behaviour jumps. Tokens with fewer than `min_count` rows get no row at all
(zero bias), which only drops entries that are already mostly shrunk away.

    python -u -m geoae.token_bias --activations_dir activations_sampled_10M \
        --out e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz

Held-out report (val tail, never used for the table): coverage, and the share of
variance the bias removes, which is what the encoder no longer has to explain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def shrink_means(sums: torch.Tensor, counts: torch.Tensor, shrink_k: float, min_count: int):
    """Shrunk per-token means. Returns (token_ids (M,), table (M, D) float32, counts (M,)).

    sums: (V, D) per-token sums of normalised rows; counts: (V,) rows per token.
    """
    keep = counts >= max(min_count, 1)
    ids = keep.nonzero(as_tuple=True)[0]
    n = counts[ids].to(sums.dtype)
    table = sums[ids] / n[:, None] * (n / (n + shrink_k))[:, None]
    return ids, table.float(), counts[ids]


def load_table(path) -> dict:
    """Read a table written by main(); keys token_ids, table, counts, vocab_size, shrink_k, min_count, ..."""
    d = np.load(str(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations_dir", required=True, help="dump dir: layer_L.npy + rows_tok.npy + meta.json")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--val_frac", type=float, default=0.05, help="must match training's split")
    ap.add_argument("--shrink_k", type=float, default=30.0)
    ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument("--n_train_rows", type=int, default=0, help="0 = the whole train split (smoke tests use less)")
    ap.add_argument("--n_eval", type=int, default=200_000)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from geoae.data import ActivationBuffer
    from transformers import AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act_dir = Path(args.activations_dir)
    meta = json.load(open(act_dir / "meta.json"))
    vocab = len(AutoTokenizer.from_pretrained(meta["model"]))
    # The exact mean/std training uses (the dump's cached norm params, computed on the train split).
    buf = ActivationBuffer(act_dir, args.layer, val_frac=args.val_frac, split="train")
    mean = torch.as_tensor(buf.mean, device=dev)
    std = torch.as_tensor(buf.std, device=dev)
    X = np.load(str(act_dir / f"layer_{args.layer}.npy"), mmap_mode="r")
    tok = np.load(act_dir / "rows_tok.npy")
    N = len(X)
    if len(tok) != N:
        raise SystemExit(f"rows_tok has {len(tok):,} rows, layer file {N:,}")
    if int(tok.max()) >= vocab:
        raise SystemExit(f"token id {int(tok.max())} >= tokenizer size {vocab}")
    val_start = int(N * (1.0 - args.val_frac))
    n_train = val_start if args.n_train_rows <= 0 else min(args.n_train_rows, val_start)
    print(f"[token-bias] {act_dir}  vocab {vocab:,}  train rows [0, {n_train:,}) of {val_start:,}")

    sums = torch.zeros(vocab, X.shape[1], device=dev)
    counts = torch.zeros(vocab, dtype=torch.long, device=dev)
    for s in range(0, n_train, args.chunk):
        e = min(s + args.chunk, n_train)
        z = (torch.from_numpy(np.array(X[s:e])).to(dev).float() - mean) / std
        t = torch.from_numpy(tok[s:e].astype(np.int64)).to(dev)
        sums.index_add_(0, t, z)
        counts += torch.bincount(t, minlength=vocab)
        if (s // args.chunk) % 10 == 0:
            print(f"[token-bias]   {e:,}/{n_train:,}", flush=True)
    ids, table, cnt = shrink_means(sums, counts, args.shrink_k, args.min_count)
    covered = float(cnt.sum() / n_train)
    print(f"[token-bias] {len(ids):,} tokens with >= {args.min_count} rows cover {covered:.1%} of train rows")

    # ---- held-out: variance the bias removes on val rows ------------------------
    ev = np.arange(val_start, min(N, val_start + args.n_eval))
    z = (torch.from_numpy(np.array(X[ev])).to(dev).float() - mean) / std
    t = torch.from_numpy(tok[ev].astype(np.int64)).to(dev)
    index = torch.full((vocab,), -1, dtype=torch.long, device=dev)
    index[ids] = torch.arange(len(ids), device=dev)
    report = {"train_rows": n_train, "tokens": len(ids), "train_coverage": covered}
    tot = (z - z.mean(0)).pow(2).sum()
    for name, tab in (("shrunk", table), ("unshrunk", sums[ids] / cnt[:, None].float())):
        idx = index[t]
        b = tab[idx.clamp_min(0)] * (idx >= 0).unsqueeze(1)
        report[f"var_removed_{name}"] = float(1 - (z - b).pow(2).sum() / tot)
    report["val_coverage"] = float((index[t] >= 0).float().mean())
    print(f"[token-bias] held-out {len(ev):,} val rows: coverage {report['val_coverage']:.1%}, "
          f"variance removed {report['var_removed_shrunk']:.3f} (unshrunk {report['var_removed_unshrunk']:.3f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), token_ids=ids.cpu().numpy(), table=table.cpu().numpy().astype(np.float16),
             counts=cnt.cpu().numpy(), vocab_size=vocab, shrink_k=args.shrink_k,
             min_count=args.min_count, norm_mean=buf.mean, norm_std=buf.std, layer=args.layer,
             activations=str(act_dir), model_name=meta["model"], report=json.dumps(report))
    print(f"[token-bias] Saved -> {out}  ({table.numel() * 2 / 1e6:.0f} MB fp16)")


if __name__ == "__main__":
    main()


class TokenBiasLookup:
    """b[tok] from a table file, for ENCODER-FREE arms (a GeoAE keeps its own copy in buffers).

    Lets a balanced k-means control cluster x - b[tok] with the same table the
    token-bypass AE subtracts, so the encoder is the only difference between them.
    """

    def __init__(self, path, device):
        t = load_table(path)
        ids = torch.as_tensor(t["token_ids"], dtype=torch.long, device=device)
        self.index = torch.full((int(t["vocab_size"]),), -1, dtype=torch.long, device=device)
        self.index[ids] = torch.arange(len(ids), device=device)
        self.table = torch.as_tensor(t["table"], device=device)
        self.norm_mean, self.norm_std = t["norm_mean"], t["norm_std"]
        self.path = str(path)

    def __call__(self, tok: torch.Tensor) -> torch.Tensor:
        idx = self.index[tok.to(self.index.device).long()]
        return self.table[idx.clamp_min(0)].float() * (idx >= 0).unsqueeze(-1)
