"""Parameterized paper appendix range diagnostics, including nonlinear decoder interventions.

Adapted from experiments/paper/analysis/paper_ranges_along_directions.py and paper_same_direction_removal.py.
Counts use the activation dump's held-out tail. Causal removal uses NEW source documents,
with fit/eval split by document parity, and exact state-conditioned decoder differences.
Six paper controls and all strength curves are retained; no fixed decoder matrix is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from geoae.checkpoint import load_ae_checkpoint, load_lm  # noqa: E402
from geoae.analysis.closest_tokens import load_baseline_kmeans, sampled_sources, stream_sampled_docs  # noqa: E402
from geoae.evaluation.interventions import dprime_saliency, fit_ranges, gate_edit, select_top  # noqa: E402
from geoae.lm_arch import decoder_layers, locate_lm_parts  # noqa: E402
from geoae.losses import pairwise_sq_dist  # noqa: E402
from geoae.token_bias import TokenBiasLookup  # noqa: E402

# Runtime paths are supplied explicitly; the running training snapshot is untouched.
CKPT = TBFILE = RANGES = KM = KMT = DUMP = LM_NAME = CORPUS_DIR = None
LAYER = 35
SHOWCASE = ()

ARMS = ["nl_h", "nl_hb", "nl_z", "dir_range_z", "dir_all_z", "dir_range_h"]
NL_P = [0.01, 0.03, 0.1, 0.3]
DIR_ALPHA = [0.25, 0.5, 1.0, 2.0]
GRID = {a: (NL_P if a.startswith("nl_") else DIR_ALPHA) for a in ARMS}
STD_STRENGTH = {a: (0.3 if a.startswith("nl_") else 1.0) for a in ARMS}
ABS_LEVELS = [0.25, 0.5, 1.0]
EVAL_CHUNK = 2048

dev = torch.device("cpu")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(checkpoint, args):
    config = checkpoint["config"]
    actual = (config["extraction"]["model_name"], config["data"]["target_layer"])
    expected = (args.model_name, args.layer)
    if actual != expected:
        raise ValueError(f"Checkpoint model/layer {actual!r} does not match requested {expected!r}")


@torch.no_grad()
def apply_latent_delta(h, z, edited_z, tokens, ae, std, chunk=2048):
    """Preserve the original residual and bypass; only add the exact decoded edit."""
    result = h.clone()
    changed = (edited_z != z).any(1)
    indices = changed.nonzero(as_tuple=True)[0]
    for start in range(0, len(indices), chunk):
        at = indices[start:start + chunk]
        delta = ae.decode(edited_z[at], tokens[at]) - ae.decode(z[at], tokens[at])
        result[at] += delta * std
    return result, changed


@torch.no_grad()
def count_range_pairs(batches, n_clusters, min_n=50, min_cos=0.9, min_dprime=2.0):
    """Two streaming passes over (vectors, labels) in each space, matching the paper."""
    total, sums, counts = {}, {}, {}
    n_rows = 0
    for spaces in batches():
        n_rows += len(next(iter(spaces.values()))[0])
        for name, (vectors, labels) in spaces.items():
            if name not in sums:
                sums[name] = torch.zeros(n_clusters, vectors.shape[1], dtype=torch.float64,
                                         device=vectors.device)
                counts[name] = torch.zeros(n_clusters, dtype=torch.float64, device=vectors.device)
                total[name] = torch.zeros(vectors.shape[1], dtype=torch.float64, device=vectors.device)
            sums[name].index_add_(0, labels, vectors.double())
            counts[name].index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
            total[name] += vectors.double().sum(0)
    if not n_rows:
        raise ValueError("No held-out activation rows are available")

    pairs = {}
    for name in sums:
        mean = total[name] / n_rows
        centered = sums[name] / counts[name].clamp_min(1)[:, None] - mean
        eligible = counts[name] >= min_n
        unit = torch.nn.functional.normalize(centered, dim=1)
        cosine = unit @ unit.T
        cosine[~eligible] = -2
        cosine[:, ~eligible] = -2
        cosine.fill_diagonal_(-2)
        best_cos, best = cosine.max(1)
        a = eligible.nonzero(as_tuple=True)[0] if eligible.sum() >= 2 else best[:0]
        b = best[a]
        keys = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], 1).unique(dim=0)
        a, b = keys[:, 0], keys[:, 1]
        direction = torch.nn.functional.normalize(unit[a] + unit[b], dim=1)
        gap = centered[a] - centered[b]
        pairs[name] = {"mean": mean.float(), "a": a, "b": b, "u": direction.float(),
                       "cos": (unit[a] * unit[b]).sum(1), "best_cos": best_cos[eligible],
                       "radial_share": (gap * direction).sum(1).square() / gap.square().sum(1).clamp_min(1e-20),
                       "norm_ratio": torch.maximum(centered[a].norm(dim=1), centered[b].norm(dim=1)) /
                                     torch.minimum(centered[a].norm(dim=1), centered[b].norm(dim=1)).clamp_min(1e-20)}

    moments = {name: torch.zeros(len(pair["a"]), 6, dtype=torch.float64,
                                device=pair["u"].device) for name, pair in pairs.items()}
    for spaces in batches():
        for name, (vectors, labels) in spaces.items():
            pair = pairs[name]
            projection = ((vectors - pair["mean"]) @ pair["u"].T).double()
            in_a = (labels[:, None] == pair["a"][None]).double()
            in_b = (labels[:, None] == pair["b"][None]).double()
            moments[name] += torch.stack([in_a.sum(0), (in_a * projection).sum(0),
                (in_a * projection.square()).sum(0), in_b.sum(0),
                (in_b * projection).sum(0), (in_b * projection.square()).sum(0)], 1)

    summaries, tables = {}, {}
    for name, pair in pairs.items():
        moment = moments[name]
        mean_a = moment[:, 1] / moment[:, 0]
        mean_b = moment[:, 4] / moment[:, 3]
        var_a = (moment[:, 2] / moment[:, 0] - mean_a.square()).clamp_min(0)
        var_b = (moment[:, 5] / moment[:, 3] - mean_b.square()).clamp_min(0)
        separation = (mean_a - mean_b).abs() / ((var_a + var_b) / 2).sqrt().clamp_min(1e-12)
        parallel = pair["cos"] >= min_cos
        keep = parallel & (separation >= min_dprime)
        count = len(pair["a"])
        summaries[name] = {"clusters_with_partner": int((counts[name] >= min_n).sum()),
            "unique_pairs": count, "pairs_cos_ge": int(parallel.sum()), "range_pairs": int(keep.sum()),
            "range_pairs_share_of_pairs": float(keep.sum() / count) if count else 0.0,
            "median_best_partner_cos": float(pair["best_cos"].median()) if count else None,
            "median_dprime_among_parallel": float(separation[parallel].median()) if parallel.any() else None,
            "median_radial_share_among_parallel": float(pair["radial_share"][parallel].median())
                                                  if parallel.any() else None}
        order = torch.argsort(separation, descending=True)
        tables[name] = [{"a": int(pair["a"][i]), "b": int(pair["b"][i]),
            "cos": float(pair["cos"][i]), "dprime": float(separation[i]),
            "auroc": float(0.5 * (1 + torch.erf(separation[i] / 2))),
            "radial_share": float(pair["radial_share"][i]), "norm_ratio": float(pair["norm_ratio"][i]),
            "n_a": int(moment[i, 0]), "n_b": int(moment[i, 3]),
            "mean_a": float(mean_a[i]), "mean_b": float(mean_b[i])}
            for i in order.tolist() if bool(keep[i])]
    return {"status": "complete", "meta": {"min_n": min_n, "cos": min_cos,
            "dprime": min_dprime, "rows": n_rows}, "summary": summaries, "range_pairs": tables}


@torch.no_grad()
def counts_main(args):
    ae, mean, std, checkpoint = load_ae_checkpoint(args.checkpoint, dev, allow_token_bias=True)
    validate_checkpoint(checkpoint, args)
    ae.eval()
    raw_centers, raw_mean, raw_std, k, base_layer, base_model = load_baseline_kmeans(Path(args.km), dev)
    tok_centers, tok_mean, tok_std, kt, tok_layer, tok_model = load_baseline_kmeans(
        Path(args.kmt), dev, allow_token_bias=True)
    if (base_layer, base_model, tok_layer, tok_model) != (args.layer, args.model_name, args.layer, args.model_name):
        raise ValueError("Baseline codebooks must use the checkpoint model and layer")
    if k != kt or k != ae.n_clusters:
        raise ValueError("Matched range comparisons require equal cluster counts")
    table = TokenBiasLookup(args.token_bias_table, dev)
    dump = Path(args.activations_dir)
    activations = np.load(dump / f"layer_{args.layer}.npy", mmap_mode="r")
    tokens = np.load(dump / "rows_tok.npy", mmap_mode="r")
    start = int(len(activations) * (1 - args.val_frac))

    def batches():
        for offset in range(start, len(activations), args.chunk):
            h = torch.from_numpy(np.array(activations[offset:offset + args.chunk])).to(dev).float()
            tok = torch.from_numpy(np.array(tokens[offset:offset + args.chunk], dtype=np.int64)).to(dev)
            raw = (h - raw_mean) / raw_std
            removed = (h - tok_mean) / tok_std - table(tok)
            z = ae.encode((h - mean) / std, tok)
            yield {"residual": (raw, pairwise_sq_dist(raw, raw_centers, metric="euclidean").argmin(1)),
                   "token_centred": (removed, pairwise_sq_dist(removed, tok_centers, metric="euclidean").argmin(1)),
                   "geoae": (z, pairwise_sq_dist(z, ae.centroids, metric=ae.metric).argmin(1))}

    result = count_range_pairs(batches, k, args.count_min_n, args.min_cos, args.count_dprime)
    result["meta"].update({"checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint),
        "activations_dir": args.activations_dir, "layer": args.layer, "model_name": args.model_name,
        "heldout_start_row": start, "val_frac": args.val_frac, "km": args.km, "kmt": args.kmt,
        "token_bias_table": args.token_bias_table, "decoder_type": ae.decoder_type})
    Path(args.ranges).write_text(json.dumps(result, indent=1))
    log("wrote", args.ranges, json.dumps(result["summary"]))


def skey(s: float) -> str:
    return f"{s:g}"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------------------------------------------------------------------------------------
# 1. pair selection
# ---------------------------------------------------------------------------------------------
def select_pairs(args):
    R = json.load(open(RANGES))
    saved_hash = R.get("meta", {}).get("checkpoint_sha256")
    if saved_hash != file_sha256(CKPT):
        raise ValueError("Range discovery output belongs to a different checkpoint; rerun --task counts")
    selection_space = getattr(args, "selection_space", "geoae")
    use_showcase = selection_space == "geoae" and not getattr(args, "no_showcase", False)
    rp = R["range_pairs"][selection_space]
    uniq = {}
    for p in rp:                                   # mutual best partners appear twice: keep one
        uniq.setdefault(frozenset((p["a"], p["b"])), p)
    qual = [p for p in uniq.values() if p["cos"] >= args.min_cos and p["dprime"] >= args.min_dprime
            and p["n_a"] >= args.min_n and p["n_b"] >= args.min_n]
    rng = np.random.default_rng(args.seed)
    if len(qual) > args.n_pairs:
        idx = np.sort(rng.choice(len(qual), args.n_pairs, replace=False))
        chosen = [qual[i] for i in idx]
    else:
        chosen = list(qual)
    chosen = [dict(p, showcase=False) for p in chosen]
    key_show = frozenset(SHOWCASE)
    show = uniq.get(key_show)
    n_show_added = 0
    if use_showcase and show is not None:
        hit = [p for p in chosen if frozenset((p["a"], p["b"])) == key_show]
        if hit:
            hit[0]["showcase"] = True
        else:
            chosen.append(dict(show, showcase=True))
            n_show_added = 1
    log(f"range pairs: {len(rp)} rows, {len(uniq)} unique; {len(qual)} qualify "
        f"(cos>={args.min_cos}, d'>={args.min_dprime}, n>={args.min_n}); chose {len(chosen) - n_show_added} "
        f"+ {n_show_added} showcase = {len(chosen)}")
    if use_showcase and show is None:
        log(f"  showcase pair {SHOWCASE} is not among the range pairs")
    return chosen, len(rp), len(uniq), len(qual)


# ---------------------------------------------------------------------------------------------
# 2. collection
# ---------------------------------------------------------------------------------------------
def lm_head_fp32(lm):
    _, _, final_norm, lm_head = locate_lm_parts(lm)
    norm32 = copy.deepcopy(final_norm).float().eval()
    head32 = copy.deepcopy(lm_head).float().eval()
    for p in list(norm32.parameters()) + list(head32.parameters()):
        p.requires_grad_(False)
    return norm32, head32


def collect(args, C, lm, tokenizer, ae, mean, std, norm32, head32, assignment=None, n_clusters=None):
    """Collect rows with fixed labels from the selection space's native codebook.

    The optional assignment callback receives raw residual rows and their token IDs.
    Omitting it preserves the original GeoAE assignment path.
    """
    K = ae.centroids.shape[0] if n_clusters is None else n_clusters
    isC = np.zeros(K, dtype=bool)
    isC[C] = True
    layers = decoder_layers(lm)
    if LAYER != len(layers) - 1:
        raise ValueError(f"Causal range scoring requires the last block; layer {LAYER}, blocks {len(layers)}")
    captured: list[torch.Tensor] = []

    def hook(_m, _i, out):
        captured.append(out[0] if isinstance(out, tuple) else out)

    handle = layers[LAYER].register_forward_hook(hook)
    meta, sources = sampled_sources(Path(DUMP), Path(CORPUS_DIR) if CORPUS_DIR else None)
    skip, ctx = meta["skip_leading"], meta["context_len"]
    stream = stream_sampled_docs(sources, meta, meta["positions_per_doc"] * args.max_docs, tokenizer, 32, 7)
    rng = np.random.default_rng(args.seed + 1000)

    H, TOK, NXT, CL, DOC, CTRL = [], [], [], [], [], []
    counts = np.zeros(K, dtype=np.int64)
    sanity = None
    n_docs = 0
    t0 = time.time()
    for di, (ids, _domain, _pos) in enumerate(stream):
        if di >= args.max_docs:
            break
        ids = ids[:, :ctx].to(dev)
        T = ids.shape[1]
        captured.clear()
        kw = {} if di == 0 else {"logits_to_keep": 1}
        out = lm(input_ids=ids, use_cache=False, **kw)
        h = captured[0][0].float()                                  # (T, D) decoder-layer output
        P = torch.arange(skip, T - 1, device=dev)                   # every position that has a next token
        idsr = ids[0]
        tok = idsr[P]
        if assignment is None:
            x = (h[P] - mean) / std
            z = ae.encode(x, tok)
            cl = pairwise_sq_dist(z, ae.centroids, metric=ae.metric).argmin(1).cpu().numpy()
        else:
            cl = assignment(h[P], tok).cpu().numpy()
        n_pos = len(P)
        # up to cap_per_doc random rows per (doc, cluster in C)
        cand = np.nonzero(isC[cl])[0]
        keep_c = np.empty(0, dtype=np.int64)
        if len(cand):
            order = rng.permutation(cand)
            o2 = order[np.argsort(cl[order], kind="stable")]
            cs = cl[o2]
            first = np.r_[0, np.nonzero(np.diff(cs))[0] + 1]
            sizes = np.diff(np.r_[first, len(o2)])
            rank = np.arange(len(o2)) - np.repeat(first, sizes)
            keep_c = o2[rank < args.cap_per_doc]
        ctrl = rng.choice(n_pos, size=min(args.ctrl_per_doc, n_pos), replace=False)
        keep = np.union1d(keep_c, ctrl)                              # sorted unique local indices
        is_ctrl = np.isin(keep, ctrl)
        kp = P[torch.as_tensor(keep, device=dev)]

        if di == 0:                                                  # sanity: fp32 head from captured h vs model logits
            lg = out.logits[0].float()
            lp_model = torch.log_softmax(lg[kp], -1).gather(1, idsr[kp + 1][:, None])
            lp_mine = torch.log_softmax(head32(norm32(h[kp])), -1).gather(1, idsr[kp + 1][:, None])
            d = (lp_model - lp_mine).abs()
            sanity = {"max_abs_logprob_diff": float(d.max()), "mean_abs_logprob_diff": float(d.mean()),
                      "n_positions": int(len(kp))}
            log(f"sanity (doc 0, {len(kp)} kept positions): max |dlogp| = {sanity['max_abs_logprob_diff']:.4f}, "
                f"mean = {sanity['mean_abs_logprob_diff']:.4f}  (expect < 0.05; abort > 0.5)")
            if sanity["max_abs_logprob_diff"] > 0.5:
                raise SystemExit("sanity check failed: captured h does not reproduce the model's logits")
            del lg, out

        H.append(h[kp].half().cpu().numpy())
        TOK.append(idsr[kp].cpu().numpy())
        NXT.append(idsr[kp + 1].cpu().numpy())
        CL.append(cl[keep])
        DOC.append(np.full(len(keep), di, dtype=np.int64))
        CTRL.append(is_ctrl)
        counts += np.bincount(cl[keep], minlength=K)
        n_docs = di + 1
        if n_docs % 100 == 0 or n_docs == 1:
            cc = counts[C]
            log(f"docs {n_docs}: rows/cluster in C  min {cc.min()}  median {int(np.median(cc))}  "
                f"({time.time() - t0:.0f}s)")
        if counts[C].min() >= args.target_per_cluster:
            log(f"every cluster in C has >= {args.target_per_cluster} rows after {n_docs} docs")
            break
    handle.remove()
    cc = counts[C]
    log(f"collection done: {n_docs} docs, {sum(len(c) for c in CL)} rows, rows/cluster in C "
        f"min {cc.min()} median {int(np.median(cc))}")
    rows = dict(h=np.concatenate(H), tok=np.concatenate(TOK).astype(np.int64),
                nxt=np.concatenate(NXT).astype(np.int64), cluster=np.concatenate(CL).astype(np.int64),
                doc=np.concatenate(DOC), is_ctrl=np.concatenate(CTRL), C=np.asarray(C, dtype=np.int64),
                n_docs=np.asarray(n_docs), sanity=np.asarray(json.dumps(sanity)))
    return rows


# ---------------------------------------------------------------------------------------------
# 7. metrics: clean vs edited next-token distributions in float32
# ---------------------------------------------------------------------------------------------
class Scorer:
    def __init__(self, norm32, head32):
        self.norm, self.head = norm32, head32

    def logp(self, h):
        return torch.log_softmax(self.head(self.norm(h)), -1)

    def row_metrics(self, h, h2, edited, nxt):
        n = len(h)
        dnll = torch.zeros(n, device=dev)
        kl = torch.zeros(n, device=dev)
        idx = edited.nonzero(as_tuple=True)[0]
        for s in range(0, len(idx), EVAL_CHUNK):
            ch = idx[s:s + EVAL_CHUNK]
            lc, le = self.logp(h[ch]), self.logp(h2[ch])
            t = nxt[ch][:, None]
            dnll[ch] = (lc.gather(1, t) - le.gather(1, t)).squeeze(1)       # NLL_edited - NLL_clean
            kl[ch] = (lc.exp() * (lc - le)).sum(1)
            del lc, le
        return dnll, kl


# ---------------------------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------------------------
def boot(vals, stat, nboot=2000, seed=0):
    v = np.asarray(vals, dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (nboot, len(v)))
    s = stat(v[idx], axis=1)
    return float(stat(v)), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def curve(case, arm):
    """(target, sibling, random) dNLL over the arm's strengths plus the origin, sorted by target."""
    pts = [(0.0, 0.0, 0.0)]
    for s in GRID[arm]:
        r = case["res"][arm][skey(s)]
        pts.append((r["target"]["dnll"], r["sibling"]["dnll"], r["random"]["dnll"]))
    pts = np.asarray(pts)
    return pts[np.argsort(pts[:, 0], kind="stable")]


def at_level(pts, L):
    return float(np.interp(L, pts[:, 0], pts[:, 1])), float(np.interp(L, pts[:, 0], pts[:, 2]))


def summarise(cases):
    from scipy import stats as sps
    S = {}
    # ---- a. standard setting
    std = {}
    for arm in ARMS:
        rows = [c["res"][arm][skey(STD_STRENGTH[arm])] for c in cases]
        d = {"strength": STD_STRENGTH[arm], "n": len(rows)}
        for g in ("target", "sibling", "random"):
            d[f"{g}_dnll_median"] = float(np.median([r[g]["dnll"] for r in rows]))
            d[f"{g}_kl_median"] = float(np.median([r[g]["kl"] for r in rows]))
            d[f"{g}_frac_edited_median"] = float(np.median([r[g]["frac_edited"] for r in rows]))
        d["target_minus_sibling_median"] = float(np.median([r["target"]["dnll"] - r["sibling"]["dnll"] for r in rows]))
        for pos in ("inner", "outer"):
            sub = [c["res"][arm][skey(STD_STRENGTH[arm])] for c in cases if c["T_position"] == pos]
            if sub:
                d[pos] = {"n": len(sub)}
                for g in ("target", "sibling", "random"):
                    d[pos][f"{g}_dnll_median"] = float(np.median([r[g]["dnll"] for r in sub]))
                d[pos]["target_minus_sibling_median"] = float(
                    np.median([r["target"]["dnll"] - r["sibling"]["dnll"] for r in sub]))
        std[arm] = d
    S["standard"] = std

    # ---- b. matched target damage
    curves = [{a: curve(c, a) for a in ARMS} for c in cases]
    keep, Ls = [], []
    for i, cv in enumerate(curves):
        L = 0.5 * min(cv[a][:, 0].max() for a in ARMS)
        if L > 0:
            keep.append(i)
            Ls.append(L)
    matched = {"n_cases": len(keep), "n_skipped": len(cases) - len(keep),
               "L_median": float(np.median(Ls)) if Ls else float("nan"),
               "L_min": float(np.min(Ls)) if Ls else float("nan"),
               "L_max": float(np.max(Ls)) if Ls else float("nan"), "arms": {}}
    sib = {a: np.array([at_level(curves[i][a], L)[0] for i, L in zip(keep, Ls)]) for a in ARMS}
    rnd = {a: np.array([at_level(curves[i][a], L)[1] for i, L in zip(keep, Ls)]) for a in ARMS}
    ref = "dir_range_z"
    for a in ARMS:
        d = {"n": len(keep)}
        for nm, v in (("sibling", sib[a]), ("random", rnd[a])):
            m, lo, hi = boot(v, np.mean)
            md, mlo, mhi = boot(v, np.median)
            d[nm] = {"mean": m, "mean_ci": [lo, hi], "median": md, "median_ci": [mlo, mhi]}
            if a != ref:
                diff = v - (sib if nm == "sibling" else rnd)[ref]
                dm, dlo, dhi = boot(diff, np.mean)
                try:
                    p = 1.0 if np.all(diff == 0) else float(sps.wilcoxon(diff).pvalue)
                except ValueError:
                    p = float("nan")
                d[nm]["vs_dir_range_z"] = {"mean_diff": dm, "ci": [dlo, dhi], "wilcoxon_p": p}
        matched["arms"][a] = d
    S["matched"] = matched

    # ---- c. absolute target levels
    absolute = {}
    for L in ABS_LEVELS:
        reach = {a: int(sum(cv[a][:, 0].max() >= L for cv in curves)) for a in ARMS}
        allr = [i for i, cv in enumerate(curves) if all(cv[a][:, 0].max() >= L for a in ARMS)]
        ent = {"n_cases": len(cases), "n_all_reach": len(allr), "reach": reach, "arms": {}}
        for a in ARMS:
            if allr:
                v = np.array([at_level(curves[i][a], L) for i in allr])
                ent["arms"][a] = {"sibling_median": float(np.median(v[:, 0])), "random_median": float(np.median(v[:, 1]))}
        absolute[skey(L)] = ent
    S["absolute"] = absolute
    return S


def format_summary(S, cases):
    L = []
    w = max(len(a) for a in ARMS)
    L.append(f"=== (a) standard setting: nl_* p=0.3, dir_* alpha=1.0; median over {len(cases)} cases (dNLL, nats) ===")
    L.append(f"{'arm':<{w}} | {'target':>8} {'sibling':>8} {'random':>8} | {'tgt-sib':>8} | {'tgt KL':>7} {'sib KL':>7} {'rnd KL':>7} | "
             f"{'sib frac':>8} {'rnd frac':>8}")
    for a in ARMS:
        d = S["standard"][a]
        L.append(f"{a:<{w}} | {d['target_dnll_median']:8.3f} {d['sibling_dnll_median']:8.3f} {d['random_dnll_median']:8.3f} | "
                 f"{d['target_minus_sibling_median']:8.3f} | {d['target_kl_median']:7.3f} {d['sibling_kl_median']:7.3f} "
                 f"{d['random_kl_median']:7.3f} | {d['sibling_frac_edited_median']:8.2f} {d['random_frac_edited_median']:8.2f}")
    L.append("")
    L.append("    split by position of T on the shared direction (inner = closer to the control mean)")
    L.append(f"    {'arm':<{w}} {'T pos':>6} {'n':>3} | {'target':>8} {'sibling':>8} {'random':>8} | {'tgt-sib':>8}")
    for a in ("dir_range_z", "dir_all_z"):
        for pos in ("inner", "outer"):
            d = S["standard"][a].get(pos)
            if d:
                L.append(f"    {a:<{w}} {pos:>6} {d['n']:>3} | {d['target_dnll_median']:8.3f} {d['sibling_dnll_median']:8.3f} "
                         f"{d['random_dnll_median']:8.3f} | {d['target_minus_sibling_median']:8.3f}")
    L.append("")
    m = S["matched"]
    L.append(f"=== (b) matched target damage: L = 0.5 x min over arms of max target dNLL "
             f"(n={m['n_cases']} cases, {m['n_skipped']} skipped; L median {m['L_median']:.3f}, range "
             f"[{m['L_min']:.3f}, {m['L_max']:.3f}] nats) ===")
    for nm in ("sibling", "random"):
        L.append(f"  {nm} dNLL at L: mean [95% CI] | median [95% CI] | paired diff vs dir_range_z: mean diff [95% CI], Wilcoxon p")
        for a in ARMS:
            d = m["arms"][a][nm]
            s = (f"  {a:<{w}} | {d['mean']:7.3f} [{d['mean_ci'][0]:7.3f},{d['mean_ci'][1]:7.3f}] | "
                 f"{d['median']:7.3f} [{d['median_ci'][0]:7.3f},{d['median_ci'][1]:7.3f}]")
            if "vs_dir_range_z" in d:
                v = d["vs_dir_range_z"]
                s += f" | {v['mean_diff']:+7.3f} [{v['ci'][0]:+7.3f},{v['ci'][1]:+7.3f}] p={v['wilcoxon_p']:.2g}"
            else:
                s += " | (reference)"
            L.append(s)
        L.append("")
    L.append("=== (c) absolute target levels ===")
    for k, e in S["absolute"].items():
        L.append(f"  L = {k} nats: all arms reach it in {e['n_all_reach']}/{e['n_cases']} cases")
        L.append(f"  {'arm':<{w}} | {'cases reaching':>14} | {'sib med@L':>9} {'rnd med@L':>9}  (cases where all arms reach)")
        for a in ARMS:
            ar = e["arms"].get(a)
            s = (f"  {a:<{w}} | {e['reach'][a]:>14d} | "
                 + (f"{ar['sibling_median']:9.3f} {ar['random_median']:9.3f}" if ar else f"{'n/a':>9} {'n/a':>9}"))
            L.append(s)
        L.append("")
    show = [c for c in cases if c["showcase"]]
    L.append(f"=== (d) showcase {SHOWCASE} at the standard setting (dNLL / KL / frac edited) ===")
    for c in show:
        L.append(f"  case T={c['T']} S={c['S']} ({c['T_position']} member; fit rows T/S/ctrl {c['n_fit']['T']}/{c['n_fit']['S']}/"
                 f"{c['n_fit']['ctrl']}, eval rows {c['n_eval']['T']}/{c['n_eval']['S']}/{c['n_eval']['ctrl']})")
        L.append(f"  {'arm':<{w}} | {'tgt dNLL':>8} {'sib dNLL':>8} {'rnd dNLL':>8} | {'tgt KL':>7} {'sib KL':>7} {'rnd KL':>7} | "
                 f"{'tgt fr':>6} {'sib fr':>6} {'rnd fr':>6}")
        for a in ARMS:
            r = c["res"][a][skey(STD_STRENGTH[a])]
            L.append(f"  {a:<{w}} | {r['target']['dnll']:8.3f} {r['sibling']['dnll']:8.3f} {r['random']['dnll']:8.3f} | "
                     f"{r['target']['kl']:7.3f} {r['sibling']['kl']:7.3f} {r['random']['kl']:7.3f} | "
                     f"{r['target']['frac_edited']:6.2f} {r['sibling']['frac_edited']:6.2f} {r['random']['frac_edited']:6.2f}")
        L.append("  full strength curves (target / sibling / random dNLL):")
        for a in ARMS:
            parts = []
            for s in GRID[a]:
                r = c["res"][a][skey(s)]
                parts.append(f"{s:g}: {r['target']['dnll']:.2f}/{r['sibling']['dnll']:.2f}/{r['random']['dnll']:.2f}")
            L.append(f"    {a:<{w}} " + "  ".join(parts))
        L.append("  direction diagnostics (z, fit rows): " + json.dumps({k: round(v, 3) for k, v in c["dir_info"].items()}))
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def removal_main():
    args = _ARGS
    t_start = time.time()

    pairs, n_rp, n_uniq, n_qual = select_pairs(args)
    C = sorted({int(c) for p in pairs for c in (p["a"], p["b"])})
    log(f"{len(pairs)} pairs, |C| = {len(C)} clusters")
    if not pairs:
        Path(args.out).write_text(json.dumps({"status": "complete", "meta": {"args": vars(args),
            "n_qualifying": 0, "reason": "No range pairs satisfy the prespecified criteria"},
            "cases": [], "summary": {"matched": {"n_cases": 0}}}, indent=1))
        return

    from geoae.checkpoint import load_tokenizer
    ae, mean, std, _ck = load_ae_checkpoint(CKPT, dev, allow_token_bias=True)
    if not ae.has_token_bias:
        raise ValueError("This matched token-bypass protocol needs a bypass checkpoint")
    validate_checkpoint(_ck, args)
    tb = TokenBiasLookup(TBFILE, dev)
    assignment = None
    selection_K = int(ae.centroids.shape[0])
    codebook_metadata = {"path": str(Path(CKPT).resolve()), "n_clusters": selection_K,
                         "dimension": int(ae.centroids.shape[1]), "metric": ae.metric,
                         "layer": LAYER, "model_name": LM_NAME,
                         "checkpoint_sha256": file_sha256(CKPT),
                         "collection_seed": args.seed, "corpus_dir": CORPUS_DIR}
    if args.selection_space != "geoae":
        codebook_path = KM if args.selection_space == "residual" else KMT
        base_centroids, base_mean, base_std, selection_K, base_layer, base_model = load_baseline_kmeans(
            Path(codebook_path), dev, allow_token_bias=args.selection_space == "token_centred")
        assert base_layer == LAYER and base_model == LM_NAME, "selection codebook LM/layer mismatch"
        selection_tb = None
        codebook_metadata = {"path": str(Path(codebook_path).resolve()), "n_clusters": int(selection_K),
                             "dimension": int(base_centroids.shape[1]), "metric": "euclidean",
                             "layer": base_layer, "model_name": base_model}
        if args.selection_space == "token_centred":
            selection_tbfile = TBFILE
            selection_tb = TokenBiasLookup(selection_tbfile, dev)
            codebook_metadata["token_bias"] = str(Path(selection_tbfile).resolve())

        def assignment(h, tok):
            x = (h - base_mean) / base_std
            if selection_tb is not None:
                x = x - selection_tb(tok)
            return pairwise_sq_dist(x, base_centroids, metric="euclidean").argmin(1)

    lm = load_lm(LM_NAME, device=dev)
    tokenizer = load_tokenizer(LM_NAME)
    norm32, head32 = lm_head_fp32(lm)
    norm32, head32 = norm32.to(dev), head32.to(dev)
    log(f"loaded AE (K={ae.centroids.shape[0]}, d={ae.centroids.shape[1]}), LM, fp32 head")

    cache = Path(args.cache)
    if args.reuse_cache and cache.exists():
        log(f"reusing cache {cache}")
        d = np.load(cache, allow_pickle=False)
        rows = {k: d[k] for k in d.files}
        cached_space = str(rows.get("selection_space", np.asarray("geoae")))
        assert cached_space == args.selection_space, (
            f"cache label namespace {cached_space!r} differs from selection space {args.selection_space!r}")
        if "codebook_metadata" in rows:
            assert json.loads(str(rows["codebook_metadata"])) == codebook_metadata, (
                "cache labels came from a different selection codebook")
        else:
            assert args.selection_space == "geoae" and "selection_space" not in rows, (
                "only legacy GeoAE caches can omit selection/codebook metadata")
        assert set(C) <= set(rows["C"].tolist()), "cache was collected for a different cluster set"
    else:
        rows = collect(args, C, lm, tokenizer, ae, mean, std, norm32, head32,
                       assignment=assignment, n_clusters=selection_K)
        rows["selection_space"] = np.asarray(args.selection_space)
        rows["codebook_metadata"] = np.asarray(json.dumps(codebook_metadata, sort_keys=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **rows)
        log(f"saved cache {cache}")
    sanity = json.loads(str(rows["sanity"]))
    n_docs = int(rows["n_docs"])

    # ---- per-row tensors and the three spaces ------------------------------------------------
    H = torch.from_numpy(rows["h"]).to(dev).float()
    tok = torch.from_numpy(rows["tok"]).to(dev)
    nxt = torch.from_numpy(rows["nxt"]).to(dev)
    cl = rows["cluster"]
    doc = rows["doc"]
    is_ctrl = rows["is_ctrl"].astype(bool)
    fit = (doc % 2 == 0)
    N = len(H)
    X = (H - mean) / std
    Z = torch.cat([ae.encode(X[s:s + 4096], tok[s:s + 4096]) for s in range(0, N, 4096)])
    if assignment is None:
        cl2 = torch.cat([pairwise_sq_dist(Z[s:s + 4096], ae.centroids, metric=ae.metric).argmin(1)
                         for s in range(0, N, 4096)]).cpu().numpy()
    else:
        cl2 = torch.cat([assignment(H[s:s + 4096], tok[s:s + 4096])
                         for s in range(0, N, 4096)]).cpu().numpy()
    log(f"{N} rows ({int(fit.sum())} fit / {int((~fit).sum())} eval, {int(is_ctrl.sum())} controls); "
        f"cluster labels recomputed from cached h agree for {np.mean(cl2 == cl):.4%} of rows")
    AHB = X - tb(tok)
    del X
    SP = {"h": H, "hb": AHB, "z": Z}
    if args.selection_space != "geoae":
        SP["x"] = (H - base_mean) / base_std
    scorer = Scorer(norm32, head32)

    # ---- per-cluster row counts, fixed control subsample -------------------------------------
    fit_idx = {c: np.nonzero((cl == c) & fit)[0] for c in C}
    eval_idx = {c: np.nonzero((cl == c) & ~fit)[0] for c in C}
    counts = {int(c): {"fit": int(len(fit_idx[c])), "eval": int(len(eval_idx[c]))} for c in C}
    ev_med = int(np.median([v["eval"] for v in counts.values()]))
    log(f"eval rows per cluster in C: min {min(v['eval'] for v in counts.values())} median {ev_med}; "
        f"fit rows: min {min(v['fit'] for v in counts.values())} median "
        f"{int(np.median([v['fit'] for v in counts.values()]))}")
    ctrl_eval_all = np.nonzero(is_ctrl & ~fit)[0]
    rng_c = np.random.default_rng(args.seed + 2000)
    ctrl_eval_sub = np.sort(rng_c.choice(ctrl_eval_all, size=min(args.n_ctrl_eval, len(ctrl_eval_all)), replace=False))
    ctrl_fit_all = np.nonzero(is_ctrl & fit)[0]
    ctrl_fit_cl = cl[ctrl_fit_all]
    ctrl_fit_np = {s: SP[s][torch.as_tensor(ctrl_fit_all, device=dev)].cpu().numpy() for s in SP}
    log(f"controls: {len(ctrl_fit_all)} fit, {len(ctrl_eval_all)} eval ({len(ctrl_eval_sub)} subsampled)")

    # ---- edit application --------------------------------------------------------------------
    stdv = std

    def apply_edit(space, edit, rows_t):
        a = SP[space][rows_t]
        a2 = edit(a)
        m = (a2 != a).any(1)                                  # unedited rows keep h exactly: dNLL = KL = 0
        h = H[rows_t]
        h2 = h.clone()
        idx = m.nonzero(as_tuple=True)[0]
        if len(idx):
            if space == "h":
                h2[idx] = a2[idx]
            elif space == "hb":
                h2[idx] = h[idx] + (a2[idx] - a[idx]) * stdv         # = (a' + b[tok]) * std + mean
            elif space == "x":
                h2[idx] = h[idx] + (a2[idx] - a[idx]) * base_std
            else:
                h2, m = apply_latent_delta(h, a, a2, tok[rows_t], ae, stdv)
        return h, h2, m

    tok0 = torch.full((1,), 1000, dtype=torch.long, device=dev)    # any fixed token: the bias cancels

    def run_config(space, edit, rows_t, bounds):
        h, h2, m = apply_edit(space, edit, rows_t)
        dn, kl = scorer.row_metrics(h, h2, m, nxt[rows_t])
        out = {}
        for g, (s, e) in bounds.items():
            out[g] = {"dnll": float(dn[s:e].mean()), "kl": float(kl[s:e].mean()),
                      "frac_edited": float(m[s:e].float().mean()), "n": int(e - s)}
        return out

    def dir_edit(c, u, lo, hi, rep, alpha, gated):
        def edit(a):
            t = (a - c) @ u
            shift = float(alpha) * (rep - t)
            if gated:
                shift = shift * ((t >= lo) & (t <= hi)).to(shift.dtype)
            return a + shift[:, None] * u
        return edit

    def tt(ix):
        return torch.as_tensor(ix, dtype=torch.long, device=dev)

    # ---- cases -------------------------------------------------------------------------------
    cases, skipped = [], []
    todo = []
    for p in pairs:
        todo.append((p["a"], p["b"], p))
        todo.append((p["b"], p["a"], p))
    for ci, (T, S, p) in enumerate(todo):
        if min(len(fit_idx[T]), len(fit_idx[S]), len(eval_idx[T]), len(eval_idx[S])) < args.min_rows:
            skipped.append({"T": T, "S": S, "fit": [len(fit_idx[T]), len(fit_idx[S])],
                            "eval": [len(eval_idx[T]), len(eval_idx[S])]})
            log(f"case {ci + 1}/{len(todo)} T={T} S={S}: SKIPPED (fit {len(fit_idx[T])}/{len(fit_idx[S])}, "
                f"eval {len(eval_idx[T])}/{len(eval_idx[S])} < {args.min_rows})")
            continue
        t0 = time.time()
        keep_cf = ~((ctrl_fit_cl == T) | (ctrl_fit_cl == S))
        if not keep_cf.any():
            skipped.append({"T": T, "S": S, "reason": "No unrelated fit control rows"})
            continue
        iC_fit = tt(ctrl_fit_all[keep_cf])
        iT_fit, iS_fit = tt(fit_idx[T]), tt(fit_idx[S])
        ce = ctrl_eval_sub[~((cl[ctrl_eval_sub] == T) | (cl[ctrl_eval_sub] == S))]
        if not len(ce):
            skipped.append({"T": T, "S": S, "reason": "No unrelated evaluation control rows"})
            continue
        rows_all = tt(np.concatenate([eval_idx[T], eval_idx[S], ce]))
        nT, nS = len(eval_idx[T]), len(eval_idx[S])
        bounds = {"target": (0, nT), "sibling": (nT, nT + nS), "random": (nT + nS, nT + nS + len(ce))}
        res = {a: {} for a in ARMS}

        # NeuronLens in h, hb, z
        for sp, arm in (("h", "nl_h"), ("hb", "nl_hb"), ("z", "nl_z")):
            aT = SP[sp][iT_fit].cpu().numpy()
            aC = ctrl_fit_np[sp][keep_cf]
            acts = np.concatenate([aT, aC])
            is_c = np.r_[np.ones(len(aT), bool), np.zeros(len(aC), bool)]
            sal = dprime_saliency(acts, is_c)
            rep = aC.mean(0)
            for pp in NL_P:
                salient = select_top(sal, p=pp)
                lo, hi, _, _ = fit_ranges(aT, salient, tao=2.0)
                res[arm][skey(pp)] = run_config(sp, gate_edit(lo, hi, rep, dev), rows_all, bounds)
            del acts, aC

        # shared direction in z
        Zc, ZT, ZS = Z[iC_fit], Z[iT_fit], Z[iS_fit]
        c = Zc.mean(0)
        muT, muS = ZT.mean(0) - c, ZS.mean(0) - c
        u = muT / muT.norm() + muS / muS.norm()
        u = u / u.norm()
        tT, tS, tC = (ZT - c) @ u, (ZS - c) @ u, (Zc - c) @ u
        rep_t = float(tC.mean())
        lo_t, hi_t = float(tT.mean() - 2 * tT.std(correction=0)), float(tT.mean() + 2 * tT.std(correction=0))
        inner = abs(float(tT.mean()) - rep_t) < abs(float(tS.mean()) - rep_t)
        dir_info = {"cos_muT_muS": float((muT / muT.norm()) @ (muS / muS.norm())),
                    "mean_t_T": float(tT.mean()), "sd_t_T": float(tT.std(correction=0)),
                    "mean_t_S": float(tS.mean()), "sd_t_S": float(tS.std(correction=0)),
                    "rep_t": rep_t, "range_lo": lo_t, "range_hi": hi_t,
                    "dprime_T_S": float((tT.mean() - tS.mean()).abs() /
                                        torch.sqrt(0.5 * (tT.var(correction=0) + tS.var(correction=0)) + 1e-8))}
        for al in DIR_ALPHA:
            res["dir_range_z"][skey(al)] = run_config("z", dir_edit(c, u, lo_t, hi_t, rep_t, al, True), rows_all, bounds)
            res["dir_all_z"][skey(al)] = run_config("z", dir_edit(c, u, lo_t, hi_t, rep_t, al, False), rows_all, bounds)

        # the same direction in the residual
        v = (ae.decode((c + u)[None], tok0) - ae.decode(c[None], tok0))[0] * stdv
        v = v / v.norm()
        mh = H[iC_fit].mean(0)
        thT = (H[iT_fit] - mh) @ v
        rep_h = float(((H[iC_fit] - mh) @ v).mean())
        lo_h, hi_h = float(thT.mean() - 2 * thT.std(correction=0)), float(thT.mean() + 2 * thT.std(correction=0))
        for al in DIR_ALPHA:
            res["dir_range_h"][skey(al)] = run_config("h", dir_edit(mh, v, lo_h, hi_h, rep_h, al, True), rows_all, bounds)

        positions = {"z": "inner" if inner else "outer"}
        native_dir_info = {}
        if args.selection_space != "geoae":
            for sp, range_arm, all_arm in (("x", "dir_range_x", "dir_all_x"),
                                           ("h", "dir_range_h_native", "dir_all_h_native")):
                acts = SP[sp]
                center = acts[iC_fit].mean(0)
                mu_t, mu_s = acts[iT_fit].mean(0) - center, acts[iS_fit].mean(0) - center
                direction = mu_t / mu_t.norm() + mu_s / mu_s.norm()
                direction /= direction.norm()
                t_target = (acts[iT_fit] - center) @ direction
                t_sibling = (acts[iS_fit] - center) @ direction
                replacement = float(((acts[iC_fit] - center) @ direction).mean())
                lower = float(t_target.mean() - 2 * t_target.std(correction=0))
                upper = float(t_target.mean() + 2 * t_target.std(correction=0))
                positions[sp] = "inner" if abs(float(t_target.mean()) - replacement) < abs(float(t_sibling.mean()) - replacement) else "outer"
                native_dir_info[sp] = {"cos_muT_muS": float(torch.nn.functional.cosine_similarity(mu_t, mu_s, dim=0)),
                                      "dprime_T_S": float(abs(t_target.mean() - t_sibling.mean()) /
                                          torch.sqrt(.5 * (t_target.var(correction=0) + t_sibling.var(correction=0)) + 1e-8)),
                                      "mean_t_T": float(t_target.mean()), "mean_t_S": float(t_sibling.mean()),
                                      "sd_t_T": float(t_target.std(correction=0)), "sd_t_S": float(t_sibling.std(correction=0)),
                                      "range_lo": lower, "range_hi": upper, "rep_t": replacement}
                for al in DIR_ALPHA:
                    res[range_arm][skey(al)] = run_config(sp, dir_edit(center, direction, lower, upper, replacement, al, True), rows_all, bounds)
                    res[all_arm][skey(al)] = run_config(sp, dir_edit(center, direction, lower, upper, replacement, al, False), rows_all, bounds)

        rec = {"T": int(T), "S": int(S), "showcase": bool(p["showcase"]), "T_position": positions.get("x", positions["z"]),
               "pair": {k: p[k] for k in ("a", "b", "cos", "dprime", "n_a", "n_b")},
               "n_fit": {"T": len(fit_idx[T]), "S": len(fit_idx[S]), "ctrl": int(keep_cf.sum())},
               "n_eval": {"T": nT, "S": nS, "ctrl": len(ce)}, "dir_info": dir_info, "res": res}
        if native_dir_info:
            rec["native_dir_info"] = native_dir_info
            rec["T_position_by_space"] = positions
        cases.append(rec)
        std_r = {a: res[a][skey(STD_STRENGTH[a])] for a in ARMS}
        log(f"case {ci + 1}/{len(todo)} T={T} S={S} ({rec['T_position']}) {time.time() - t0:.0f}s | std dNLL tgt/sib/rnd: "
            + "  ".join(f"{a}={std_r[a]['target']['dnll']:.2f}/{std_r[a]['sibling']['dnll']:.2f}/{std_r[a]['random']['dnll']:.2f}"
                        for a in ARMS))

    log(f"evaluated {len(cases)} cases, skipped {len(skipped)}")
    S_ = summarise(cases) if cases else {"matched": {"n_cases": 0}}
    text = format_summary(S_, cases) if cases else "No cases met the minimum held-out fit/evaluation row counts."
    print("\n" + text, flush=True)

    out = {"status": "complete", "meta": {"args": vars(args), "selection_space": args.selection_space,
                    "codebook_metadata": codebook_metadata,
                    "n_range_pair_rows": n_rp, "n_unique_pairs": n_uniq, "n_qualifying": n_qual,
                    "pairs": pairs, "clusters": C, "cluster_row_counts": counts, "docs_streamed": n_docs,
                    "n_rows": int(N), "n_controls": int(is_ctrl.sum()), "n_ctrl_eval_used": int(len(ctrl_eval_sub)),
                    "sanity": sanity, "grid": {a: GRID[a] for a in ARMS}, "std_strength": STD_STRENGTH,
                    "skipped_cases": skipped, "seconds": time.time() - t_start,
                    "cluster_label_agreement": float(np.mean(cl2 == cl)),
                    "intervention": "h + std * (decode(edited_z,tok) - decode(z,tok)); current-state finite difference",
                    "inference_dtype": "bfloat16 LM capture; float32 final norm and head scoring",
                    "split": "Unseen source documents after dump consumption; even documents fit, odd documents eval"},
           "cases": cases, "summary": S_, "summary_text": text}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    Path(args.out).with_suffix(".txt").write_text(text + "\n")
    log(f"wrote {args.out} ({time.time() - t_start:.0f}s total)")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("counts", "removal", "all"), default="all")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--activations_dir", required=True)
    ap.add_argument("--km", required=True, help="Balanced residual k-means control")
    ap.add_argument("--kmt", required=True, help="Balanced token-removed k-means control")
    ap.add_argument("--token_bias_table", required=True)
    ap.add_argument("--layer", type=int, default=35)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--corpus_dir", default=None,
                    help="Held-out corpus root (manifest.json plus source jsonl.gz files); overrides dump metadata")
    ap.add_argument("--ranges", default=None, help="Counts JSON; defaults to out_dir/ranges_along_directions.json")
    ap.add_argument("--out", default=None, help="Removal JSON; defaults to out_dir/same_direction_removal.json")
    ap.add_argument("--cache", default=None, help="Collected rows NPZ; defaults to out_dir/same_direction_rows.npz")
    ap.add_argument("--reuse_cache", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--count_min_n", type=int, default=50)
    ap.add_argument("--count_dprime", type=float, default=2.0)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--selection_space", choices=("geoae", "residual", "token_centred"), default="geoae")
    ap.add_argument("--n_pairs", type=int, default=30)
    ap.add_argument("--min_n", type=int, default=100, help="Discovery rows per member for causal pair selection")
    ap.add_argument("--min_cos", type=float, default=0.9)
    ap.add_argument("--min_dprime", type=float, default=3.0)
    ap.add_argument("--max_docs", type=int, default=3000)
    ap.add_argument("--target_per_cluster", type=int, default=160)
    ap.add_argument("--cap_per_doc", type=int, default=4)
    ap.add_argument("--ctrl_per_doc", type=int, default=16)
    ap.add_argument("--n_ctrl_eval", type=int, default=2000)
    ap.add_argument("--min_rows", type=int, default=30, help="Minimum fit AND eval rows for each pair member")
    ap.add_argument("--eval_chunk", type=int, default=512, help="Rows per float32 vocabulary scoring batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    if not 0 < args.val_frac < 1:
        ap.error("--val_frac must be between zero and one")
    for name in ("count_min_n", "chunk", "n_pairs", "min_n", "max_docs", "target_per_cluster",
                 "cap_per_doc", "ctrl_per_doc", "n_ctrl_eval", "min_rows", "eval_chunk"):
        if getattr(args, name) <= 0:
            ap.error(f"--{name} must be positive")
    args.no_showcase = True     # a Llama-specific handpicked cluster pair is never added to Qwen
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.ranges = args.ranges or str(out_dir / "ranges_along_directions.json")
    args.out = args.out or str(out_dir / "same_direction_removal.json")
    args.cache = args.cache or str(out_dir / "same_direction_rows.npz")
    return args


def main(argv=None):
    global _ARGS, CKPT, TBFILE, RANGES, KM, KMT, DUMP, LM_NAME, LAYER, CORPUS_DIR, EVAL_CHUNK, dev
    args = parse_args(argv)
    _ARGS = args
    CKPT, TBFILE, RANGES = args.checkpoint, args.token_bias_table, args.ranges
    KM, KMT, DUMP = args.km, args.kmt, args.activations_dir
    LM_NAME, LAYER, CORPUS_DIR, EVAL_CHUNK = args.model_name, args.layer, args.corpus_dir, args.eval_chunk
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    from geoae.seeding import seed_everything
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.selection_space != "geoae":
        for arm in ("dir_range_x", "dir_all_x", "dir_range_h_native", "dir_all_h_native"):
            if arm not in ARMS:
                ARMS.append(arm)
                GRID[arm] = DIR_ALPHA
                STD_STRENGTH[arm] = 1.0
    with torch.no_grad():
        if args.task in ("counts", "all"):
            counts_main(args)
        if args.task in ("removal", "all"):
            if not Path(args.ranges).is_file():
                raise FileNotFoundError(f"Run --task counts first: missing {args.ranges}")
            removal_main()


if __name__ == "__main__":
    main()
