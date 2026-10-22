"""NeuronLens-style concept removal on pairs of concepts that share ONE direction of the GeoAE latent.

A "range pair" (results/delta/d6144/evals/ranges_along_directions.json) is two GeoAE clusters whose
centred means point the same way (cosine >= 0.9) yet separate along that single direction: the same
direction at two intensities. We remove one concept (target T) and measure the damage to T, to its
same-direction sibling S, and to random text, for six removal arms in the AE latent vs the base residual:

    nl_h / nl_hb / nl_z   NeuronLens (d' saliency, top-p dims, +-2 sigma range gate, complement-mean
                          replacement) in the raw residual / token-centred residual / AE latent
    dir_range_z           range gate on the shared direction u in z, push the in-range coordinate to the
                          control mean (strength alpha)
    dir_all_z             the same push applied to EVERY row (whole direction = one concept)
    dir_range_h           u mapped into the residual (decoded finite difference), same range gate

With --selection_space residual, membership comes entirely from native base
k-means. Additional dir_range_x/dir_all_x and dir_range_h_native/dir_all_h_native
arms fit their own shared directions in normalized and raw residual spaces.

Damage = next-token NLL increase (and KL(clean || edited)) through final_norm + lm_head in float32,
rows whose edit leaves h untouched cost exactly zero. Rows come from held-out documents (docs never used
for AE training) using EVERY position; fit rows = even doc index, eval rows = odd doc index.

    GEOAE_REMOTE=1 HF_HUB_OFFLINE=1 .venv/bin/python experiments/paper/analysis/paper_same_direction_removal.py
"""
from __future__ import annotations

from geoae.paths import resolve_input_path

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from geoae.checkpoint import load_ae_checkpoint, load_lm  # noqa: E402
from geoae.analysis.closest_tokens import load_baseline_kmeans
from geoae.analysis.closest_tokens import sampled_sources
from geoae.analysis.closest_tokens import stream_sampled_docs  # noqa: E402
from geoae.evaluation.interventions import dprime_saliency
from geoae.evaluation.interventions import fit_ranges
from geoae.evaluation.interventions import gate_edit
from geoae.evaluation.interventions import select_top  # noqa: E402
from geoae.lm_arch import decoder_layers, locate_lm_parts  # noqa: E402
from geoae.losses import pairwise_sq_dist  # noqa: E402
from geoae.token_bias import TokenBiasLookup  # noqa: E402

CKPT = os.environ.get("SDR_CKPT", "checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt")
TBFILE = "e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz"
RANGES = os.environ.get("SDR_RANGES", "results/delta/d6144/evals/ranges_along_directions.json")
KM = os.environ.get("SDR_KM", "e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled.npz")
KMT = os.environ.get("SDR_KMT", "e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled_tokmean.npz")
DUMP = "activations_sampled_10M"
LM_NAME = "meta-llama/Llama-3.2-3B"
LAYER = 27
SHOWCASE = (227, 1662)

ARMS = ["nl_h", "nl_hb", "nl_z", "dir_range_z", "dir_all_z", "dir_range_h"]
NL_P = [0.01, 0.03, 0.1, 0.3]
DIR_ALPHA = [0.25, 0.5, 1.0, 2.0]
GRID = {a: (NL_P if a.startswith("nl_") else DIR_ALPHA) for a in ARMS}
STD_STRENGTH = {a: (0.3 if a.startswith("nl_") else 1.0) for a in ARMS}
ABS_LEVELS = [0.25, 0.5, 1.0]
EVAL_CHUNK = 2048

dev = torch.device("cuda")
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False       # clean and edited logits share one exact fp32 path
torch.backends.cudnn.allow_tf32 = False


def skey(s: float) -> str:
    return f"{s:g}"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------------------------------------------------------------------------------------
# 1. pair selection
# ---------------------------------------------------------------------------------------------
def select_pairs(args):
    R = json.load(open(resolve_input_path(RANGES)))
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
    assert len(layers) == 28 and LAYER == 27, f"expected 28 decoder layers, got {len(layers)}"
    captured: list[torch.Tensor] = []

    def hook(_m, _i, out):
        captured.append(out[0] if isinstance(out, tuple) else out)

    handle = layers[LAYER].register_forward_hook(hook)
    meta, sources = sampled_sources(Path(DUMP))
    skip, ctx = meta["skip_leading"], meta["context_len"]
    stream = stream_sampled_docs(sources, meta, 64 * args.max_docs, tokenizer, 32, 7)
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
        h = captured[0][0].float()                                  # (T, 3072) decoder-layer output
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
                    p = float(sps.wilcoxon(diff).pvalue)
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection_space", choices=("geoae", "residual", "token_centred"), default="geoae",
                    help="native cluster membership and discovery pairs used for every treatment")
    ap.add_argument("--no_showcase", action="store_true", help="do not append the GeoAE showcase pair")
    ap.add_argument("--n_pairs", type=int, default=30)
    ap.add_argument("--min_n", type=int, default=100)
    ap.add_argument("--min_cos", type=float, default=0.9)
    ap.add_argument("--min_dprime", type=float, default=3.0)
    ap.add_argument("--max_docs", type=int, default=3000)
    ap.add_argument("--target_per_cluster", type=int, default=160)
    ap.add_argument("--cap_per_doc", type=int, default=4)
    ap.add_argument("--ctrl_per_doc", type=int, default=16)
    ap.add_argument("--n_ctrl_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_rows", type=int, default=30, help="a case needs >= this many fit and eval rows for T and S")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--reuse_cache", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.selection_space != "geoae":
        for arm in ("dir_range_x", "dir_all_x", "dir_range_h_native", "dir_all_h_native"):
            ARMS.append(arm)
            GRID[arm] = DIR_ALPHA
            STD_STRENGTH[arm] = 1.0
    suffix = "" if args.selection_space == "geoae" else f"_{args.selection_space}_selected"
    if args.cache is None:
        args.cache = f"results/delta/d6144/evals/same_direction_rows{suffix}.npz"
    if args.out is None:
        args.out = f"results/delta/d6144/evals/same_direction_removal{suffix}.json"
    t_start = time.time()

    pairs, n_rp, n_uniq, n_qual = select_pairs(args)
    C = sorted({int(c) for p in pairs for c in (p["a"], p["b"])})
    log(f"{len(pairs)} pairs, |C| = {len(C)} clusters")
    if not pairs:
        raise SystemExit(f"no {args.selection_space} pairs satisfy the selection criteria")

    from geoae.checkpoint import load_tokenizer
    ae, mean, std, _ck = load_ae_checkpoint(CKPT, dev, allow_token_bias=True)
    assert ae.has_token_bias
    tb = TokenBiasLookup(TBFILE, dev)
    assignment = None
    selection_K = int(ae.centroids.shape[0])
    codebook_metadata = {"path": str(Path(CKPT).resolve()), "n_clusters": selection_K,
                         "dimension": int(ae.centroids.shape[1]), "metric": ae.metric,
                         "layer": LAYER, "model_name": LM_NAME}
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
            with np.load(resolve_input_path(codebook_path), allow_pickle=True) as kmfile:
                selection_tbfile = str(kmfile["token_bias"])
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
    if args.reuse_cache and resolve_input_path(cache).exists():
        log(f"reusing cache {cache}")
        d = np.load(resolve_input_path(cache), allow_pickle=False)
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
                for s in range(0, len(idx), 2048):
                    ch = idx[s:s + 2048]
                    tk = tok[rows_t[ch]]
                    h2[ch] = h[ch] + (ae.decode(a2[ch], tk) - ae.decode(a[ch], tk)) * stdv
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
        iC_fit = tt(ctrl_fit_all[keep_cf])
        iT_fit, iS_fit = tt(fit_idx[T]), tt(fit_idx[S])
        ce = ctrl_eval_sub[~((cl[ctrl_eval_sub] == T) | (cl[ctrl_eval_sub] == S))]
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
    S_ = summarise(cases)
    text = format_summary(S_, cases)
    print("\n" + text, flush=True)

    out = {"meta": {"args": vars(args), "selection_space": args.selection_space,
                    "codebook_metadata": codebook_metadata,
                    "n_range_pair_rows": n_rp, "n_unique_pairs": n_uniq, "n_qualifying": n_qual,
                    "pairs": pairs, "clusters": C, "cluster_row_counts": counts, "docs_streamed": n_docs,
                    "n_rows": int(N), "n_controls": int(is_ctrl.sum()), "n_ctrl_eval_used": int(len(ctrl_eval_sub)),
                    "sanity": sanity, "grid": {a: GRID[a] for a in ARMS}, "std_strength": STD_STRENGTH,
                    "skipped_cases": skipped, "seconds": time.time() - t_start,
                    "cluster_label_agreement": float(np.mean(cl2 == cl))},
           "cases": cases, "summary": S_, "summary_text": text}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"wrote {args.out} ({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
