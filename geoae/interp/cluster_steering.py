"""
Cluster steering: are the UNSUPERVISED clusters control handles?

Every earlier intervention fits its edit from LABELLED concepts inside the latent.
This one uses the codebook itself: move a held-out row from its own cluster s toward
a target cluster t and ask whether the model's behaviour moves toward t's.

Layer 27 is Llama-3.2-3B's LAST block, so a row's next-token distribution is exactly
softmax(lm_head(norm(h))) — no forward pass, the test runs on dump rows directly
(asserted: --layer must be the last block).

  signature P_k   mean next-token distribution of cluster k's members, from the REFERENCE
                  half of the held-out rows (documents with even id) — the cluster's behaviour
  source rows     the other half (odd documents); targets are uniform over non-hub clusters
                  with >= --min_ref reference members

Edits (translation by the centroid gap, so a row keeps its offset inside its cluster):
  ae     z' = z + a (c_t - c_s)     h' = decode(z', tok)        [token-bypass AE: + b[tok]]
  ae_h   x' = x + a (m_t - m_s)     m_k = mean normalised residual of the AE cluster's
                                     reference members: the AE's PARTITION, edited in base space
  km     x' = x + a (C_t - C_s)     balanced k-means codebook (token-mean / erased codebooks
                                     translate the same way: b and the projection cancel in a gap)
Each arm is read against its own unedited prediction (AE: its reconstruction).

Per (row, alpha):
  transfer  JS(p', P_t) < JS(p', P_s)   behaviour now closer to the target than to the source
  specific  JS(p', P_t) < JS(p', P_r)   ... than to a random third cluster (chance 0.5)
  gain      JS(p_ref, P_t) - JS(p', P_t)
  disrupt   KL(p' || p_ref)
  move      ||h' - h_ref|| / ||h_ref||  edit size in the residual
Arms differ in how far a = 1 moves a row, so the summary also compares them at MATCHED move
(interpolating each arm's alpha curve). CIs: 95% bootstrap over source documents.

    python -u -m geoae.interp.cluster_steering \
        --arms d6144_new=<ckpt>,tokbias=<ckpt>,km_new=<npz>,km_tokmean=<npz> \
        --out eval_out/cluster_steering.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geoae.checkpoint import load_ae_checkpoint, load_lm
from geoae.interp.closest_tokens import load_baseline_kmeans


def js_rows(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise Jensen-Shannon divergence (nats) between probability rows p and q."""
    m = 0.5 * (p + q)
    lp, lq, lm = (p + eps).log(), (q + eps).log(), (m + eps).log()
    return 0.5 * (p * (lp - lm)).sum(-1) + 0.5 * (q * (lq - lm)).sum(-1)


def kl_rows(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (p * ((p + eps).log() - (q + eps).log())).sum(-1)


class Arm:
    """One codebook: assign rows, map (normalised x, tok) -> edit space and back."""

    def __init__(self, name, path, dev):
        self.name, self.ae, self.U, self.tb = name, None, None, None
        if path.endswith(".npz"):
            C, mean, std, *_ = load_baseline_kmeans(Path(path), dev, allow_erasure=True)
            meta = np.load(path, allow_pickle=True)
            if "erase_U" in meta.files:
                self.U = torch.as_tensor(meta["erase_U"], dtype=torch.float32, device=dev)
            if "token_bias" in meta.files:
                from geoae.token_bias import TokenBiasLookup
                self.tb = TokenBiasLookup(str(meta["token_bias"]), dev)
            self.C, self.kind = C.float(), "km"
        else:
            ae, mean, std, _ = load_ae_checkpoint(path, dev, allow_token_bias=True)
            self.ae, self.C, self.kind = ae.eval(), ae.centroids.float(), "ae"
        self.mean, self.std = mean.float(), std.float()

    @torch.no_grad()
    def coords(self, x, tok):
        """Coordinates the codebook lives in (for assignment)."""
        if self.kind == "ae":
            return self.ae.encode(x, tok if self.ae.has_token_bias else None)
        v = x if self.tb is None else x - self.tb(tok)
        return v if self.U is None else v - (v @ self.U) @ self.U.T

    @torch.no_grad()
    def assign(self, x, tok):
        return torch.cdist(self.coords(x, tok), self.C).argmin(1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True, help="name=ae_ckpt_or_kmeans_npz,... (all on the same dump/norm)")
    ap.add_argument("--activations_dir", default="activations_sampled_10M")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--n_eval", type=int, default=500_000, help="held-out tail rows used (both halves)")
    ap.add_argument("--n_src", type=int, default=20_000, help="source rows steered per arm")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0, 1.5])
    ap.add_argument("--min_ref", type=int, default=20, help="min reference members for a target signature")
    ap.add_argument("--hub_x", type=float, default=3.0, help="clusters above hub_x x uniform usage are hubs (excluded)")
    ap.add_argument("--moves", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5],
                    help="relative edit sizes at which arms are compared (matched move)")
    ap.add_argument("--disrupts", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0],
                    help="KL(p'||p_ref) levels at which arms are compared (matched disruption: how far "
                         "the output moves toward the target for the same total change)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    act = Path(args.activations_dir)
    meta = json.load(open(act / "meta.json"))
    lm = load_lm(meta["model"], device=dev)
    from geoae.lm_arch import decoder_layers
    if args.layer != len(decoder_layers(lm)) - 1:
        raise SystemExit(f"layer {args.layer} is not the last block: next-token behaviour needs a forward pass")
    head, fnorm = lm.get_output_embeddings(), lm.model.norm
    for p_ in lm.parameters():
        p_.requires_grad_(False)

    X = np.load(str(act / f"layer_{args.layer}.npy"), mmap_mode="r")
    N = len(X)
    ev = np.arange(int(N * (1 - args.val_frac)), N)[: args.n_eval]
    H = torch.from_numpy(np.array(X[ev])).to(dev)                       # fp16 raw residuals
    tok = torch.from_numpy(np.load(act / "rows_tok.npy")[ev].astype(np.int64)).to(dev)
    doc = np.load(act / "rows_doc.npy")[ev]
    ref_rows = np.flatnonzero(doc % 2 == 0)
    src_pool = np.flatnonzero(doc % 2 == 1)
    print(f"[csteer] {len(ev):,} held-out rows: {len(ref_rows):,} reference (even docs), "
          f"{len(src_pool):,} source pool (odd docs)")

    @torch.no_grad()
    def probs(h_raw):
        return torch.softmax(head(fnorm(h_raw.to(head.weight.dtype))).float(), -1)

    arms = [Arm(*spec.split("=", 1), dev) for spec in args.arms.split(",")]
    m0, s0 = arms[0].mean, arms[0].std
    for a in arms[1:]:
        if not (torch.allclose(a.mean, m0) and torch.allclose(a.std, s0)):
            raise SystemExit(f"arm {a.name} uses different norm stats; all arms must share one dump")

    def xnorm(idx):
        return (H[idx].float() - m0) / s0

    V = head.weight.shape[0]
    results = {"meta": {**vars(args), "n_ref": int(len(ref_rows)), "n_src_pool": int(len(src_pool))}, "arms": {}}
    per_row = {}
    src_all = rng.choice(src_pool, size=min(args.n_src * 3, len(src_pool)), replace=False)

    for arm in arms:
        K = arm.C.shape[0]
        # ---- reference pass: labels, signatures, AE-partition member means -------------
        lab_ref = torch.empty(len(ref_rows), dtype=torch.long, device=dev)
        P = torch.zeros(K, V, device=dev)
        M = torch.zeros(K, H.shape[1], device=dev)
        for i in range(0, len(ref_rows), args.chunk):
            j = torch.from_numpy(ref_rows[i:i + args.chunk]).to(dev)
            x = xnorm(j)
            l = arm.assign(x, tok[j])
            lab_ref[i:i + len(j)] = l
            P.index_add_(0, l, probs(H[j]))
            M.index_add_(0, l, x)
        cnt = torch.bincount(lab_ref, minlength=K).float()
        P /= cnt.clamp_min(1)[:, None]
        M /= cnt.clamp_min(1)[:, None]
        usage = cnt / cnt.sum()
        hub = usage > args.hub_x / K
        ok_t = (cnt >= args.min_ref) & ~hub
        targets = ok_t.nonzero(as_tuple=True)[0]
        print(f"[csteer] {arm.name}: {int(hub.sum())} hubs ({float(usage[hub].sum()):.1%} of rows), "
              f"{len(targets)} usable targets of {K}")

        # ---- source rows: non-hub, from a usable cluster ---------------------------------
        lab_src = torch.cat([arm.assign(xnorm(torch.from_numpy(src_all[i:i + args.chunk]).to(dev)),
                                        tok[torch.from_numpy(src_all[i:i + args.chunk]).to(dev)])
                             for i in range(0, len(src_all), args.chunk)])
        keep = ok_t[lab_src].cpu().numpy()
        src = src_all[keep][: args.n_src]
        s_lab = lab_src[torch.from_numpy(np.flatnonzero(keep)[: args.n_src]).to(dev)]
        tg = targets[torch.randint(len(targets), (len(src),), device=dev)]
        rd = targets[torch.randint(len(targets), (len(src),), device=dev)]
        clash = (tg == s_lab) | (rd == s_lab) | (rd == tg)
        while bool(clash.any()):                           # redraw until t, r, s are distinct
            n = int(clash.sum())
            tg[clash] = targets[torch.randint(len(targets), (n,), device=dev)]
            rd[clash] = targets[torch.randint(len(targets), (n,), device=dev)]
            clash = (tg == s_lab) | (rd == s_lab) | (rd == tg)

        methods = ["ae", "ae_h"] if arm.kind == "ae" else ["km"]
        for meth in methods:
            key = f"{arm.name}:{meth}"
            rows = {f: np.zeros((len(args.alphas), len(src)), np.float32)
                    for f in ("transfer", "specific", "gain", "disrupt", "move")}
            for i in range(0, len(src), args.chunk):
                j = torch.from_numpy(src[i:i + args.chunk]).to(dev)
                sl = slice(i, i + len(j))
                x, t = xnorm(j), tok[j]
                s_, t_, r_ = s_lab[sl], tg[sl], rd[sl]
                if meth == "ae":
                    tt = t if arm.ae.has_token_bias else None
                    z = arm.ae.encode(x, tt)
                    x_ref = arm.ae.decode(z, tt)
                    gap_z = arm.C[t_] - arm.C[s_]
                elif meth == "ae_h":
                    x_ref, gap = x, M[t_] - M[s_]
                else:
                    x_ref, gap = x, arm.C[t_] - arm.C[s_]
                h_ref = x_ref * s0 + m0
                p_ref = probs(h_ref)
                Pt, Ps, Pr = P[t_], P[s_], P[r_]
                js_ref_t = js_rows(p_ref, Pt)
                for ai, a in enumerate(args.alphas):
                    if meth == "ae":
                        x2 = arm.ae.decode(z + a * gap_z, tt)
                    else:
                        x2 = x_ref + a * gap
                    h2 = x2 * s0 + m0
                    p2 = probs(h2)
                    jt, js_, jr = js_rows(p2, Pt), js_rows(p2, Ps), js_rows(p2, Pr)
                    rows["transfer"][ai, sl] = (jt < js_).float().cpu().numpy()
                    rows["specific"][ai, sl] = (jt < jr).float().cpu().numpy()
                    rows["gain"][ai, sl] = (js_ref_t - jt).cpu().numpy()
                    rows["disrupt"][ai, sl] = kl_rows(p2, p_ref).cpu().numpy()
                    rows["move"][ai, sl] = ((h2 - h_ref).norm(dim=1) / h_ref.norm(dim=1)).cpu().numpy()
            per_row[key] = (rows, doc[src])
            summ = {f: [float(v.mean()) for v in rows[f]] for f in rows}
            results["arms"][key] = {"alphas": args.alphas, "n_src": int(len(src)), "n_targets": int(len(targets)),
                                    "hubs": int(hub.sum()), "hub_share": float(usage[hub].sum()), **summ}
            print(f"[csteer]   {key:22s} " + "  ".join(
                f"a{a}: transfer {summ['transfer'][k]:.3f} specific {summ['specific'][k]:.3f} "
                f"move {summ['move'][k]:.3f}" for k, a in enumerate(args.alphas)), flush=True)
        del P, M
        torch.cuda.empty_cache()

    # ---- matched-move comparison with a document bootstrap -----------------------------
    def at_level(rows, lev, idx=None, axis="move"):
        mv = rows[axis] if idx is None else rows[axis][:, idx]
        out = {}
        for f in ("transfer", "specific", "gain", "disrupt", "move"):
            if f == axis:
                continue
            v = rows[f] if idx is None else rows[f][:, idx]
            xs, ys = mv.mean(1), v.mean(1)
            o = np.argsort(xs)
            out[f] = float(np.interp(lev, xs[o], ys[o], left=np.nan, right=np.nan))
        return out

    def matched_table(axis, levels):
        out = {}
        for key, (rows, d) in per_row.items():
            docs = np.unique(d)
            by_doc = {k: np.flatnonzero(d == k) for k in docs}
            out[key] = {}
            for lev in levels:
                pt = at_level(rows, lev, axis=axis)
                boots = {f: [] for f in pt}
                for _ in range(args.n_boot):
                    pick = rng.choice(docs, size=len(docs), replace=True)
                    idx = np.concatenate([by_doc[k] for k in pick])
                    for f, v in at_level(rows, lev, idx, axis).items():
                        boots[f].append(v)
                out[key][str(lev)] = {f: {"est": pt[f], "lo": float(np.nanpercentile(boots[f], 2.5)),
                                          "hi": float(np.nanpercentile(boots[f], 97.5))} for f in pt}
        return out

    for axis, levels, title, other in (
            ("move", args.moves, "MATCHED EDIT SIZE (move = ||dh|| / ||h||)", "disrupt"),
            ("disrupt", args.disrupts, "MATCHED DISRUPTION (KL(p'||p_ref), nats)", "move")):
        tab = matched_table(axis, levels)
        results[f"matched_{axis}"] = tab
        print(f"\n[csteer] at {title}: transfer / specificity [95% doc CI]")
        for lev in levels:
            print(f"  {axis} {lev:.2f}")
            for key, m in tab.items():
                t, s_ = m[str(lev)]["transfer"], m[str(lev)]["specific"]
                if np.isnan(t["est"]):
                    print(f"    {key:22s} (outside this arm's alpha range)")
                    continue
                print(f"    {key:22s} transfer {t['est']:.3f} [{t['lo']:.3f},{t['hi']:.3f}]   "
                      f"specific {s_['est']:.3f} [{s_['lo']:.3f},{s_['hi']:.3f}]   "
                      f"{other} {m[str(lev)][other]['est']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    np.savez_compressed(Path(args.out).with_suffix(".rows.npz"),
                        **{f"{k}|{f}": v for k, (rows, d) in per_row.items() for f, v in rows.items()},
                        **{f"{k}|doc": d for k, (rows, d) in per_row.items()})
    print(f"[csteer] wrote {args.out}")


if __name__ == "__main__":
    main()
