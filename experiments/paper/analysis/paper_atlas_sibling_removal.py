"""Phase 2 of the Atlas sibling test: does removing concept A by its RANGE on the shared direction spare sibling B?

experiments/paper/analysis/paper_same_direction_removal.py on frozen Atlas sibling pairs instead of GeoAE range pairs. NO geometry-based
selection: every sibling pair that passes the data rule (>= min_excl exclusive chunks per member, pairs.json) is run,
in both orders (target T = A, sibling S = B and the reverse); the pairs that phase 1 calls range pairs (GeoAE z) are
reported as a subgroup. The edit is applied at layer 27, which is the LAST decoder layer, so it only changes the same
position's logits: damage = next-token NLL increase (and KL(clean || edited)) through final_norm + lm_head in float32.

Rows are the cached token positions of A-ONLY chunks (target), B-ONLY chunks (sibling) and random Atlas chunks
(controls, chunks carrying A or B excluded). Everything is FIT on the fit half (documents with hash bit 1) and
EVALUATED on the eval half; the removal arms are exactly the original's:

    nl_h / nl_hb / nl_z   NeuronLens (d' saliency, top-p dims, +-2 sigma range gate, complement-mean replacement) in
                          the raw residual / token-centred residual / AE latent (base units vs latent units)
    dir_range_z           range gate on the shared direction u in z, push the in-range coordinate to the control mean
    dir_all_z             the same push applied to EVERY row (whole direction = one concept)
    dir_range_h           u mapped into the residual (decoded finite difference), same range gate

Arms are compared at EQUAL target damage (the original's summarise(): L = half the smallest maximum target damage).
The summary of the original (standard setting, matched damage, absolute levels) is reported for all cases and for
subgroups: phase-1 range pairs (GeoAE z) vs the rest, and by pair rule (E / C).

    scripts/py experiments/paper/analysis/paper_atlas_sibling_removal.py --cache /scratch/dogar/geoae/atlas_l27 \
        --pairs results/delta/d6144/evals/atlas_siblings/pairs.json --geometry results/delta/d6144/evals/atlas_siblings/geometry.json
"""
from __future__ import annotations

from geoae.paths import resolve_input_path

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import geoae.evaluation.atlas_geometry as G  # noqa: E402
import geoae.analysis.atlas_siblings as AS  # noqa: E402
from geoae.evaluation.interventions import dprime_saliency
from geoae.evaluation.interventions import fit_ranges
from geoae.evaluation.interventions import gate_edit
from geoae.evaluation.interventions import select_top  # noqa: E402


def _load_original():
    """The summary machinery of paper_same_direction_removal.py (Scorer, summarise, curves, ...): imported by path so the
    matched-damage definitions cannot drift from the original."""
    p = ROOT / "experiments/paper/analysis/paper_same_direction_removal.py"
    spec = importlib.util.spec_from_file_location("paper_same_direction_removal", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paper_same_direction_removal"] = mod
    spec.loader.exec_module(mod)
    return mod


O = _load_original()
ARMS, GRID, STD_STRENGTH, NL_P, DIR_ALPHA, skey = O.ARMS, O.GRID, O.STD_STRENGTH, O.NL_P, O.DIR_ALPHA, O.skey
LM_NAME, boot = O.LM_NAME, O.boot
dev = torch.device("cuda")
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------------------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------------------------
class Rows:
    """A set of cached rows in the three spaces the removal arms edit: raw h, token-centred normalised hb, latent z."""

    def __init__(self, cache, spaces, chunks):
        _, h16, tok, nxt = cache.gather(chunks)
        per = cache.row_start[chunks + 1] - cache.row_start[chunks]
        self.chunk = np.repeat(chunks, per)
        self.h = torch.as_tensor(h16).to(dev).float()
        self.tok = torch.as_tensor(tok).to(dev)
        self.nxt = torch.as_tensor(nxt).to(dev)
        x = (self.h - spaces.m_a) / spaces.s_a
        self.hb = x - spaces.tb(self.tok)
        self.z = torch.cat([spaces.ae.encode(x[s:s + 4096], self.tok[s:s + 4096]) for s in range(0, len(x), 4096)]) if len(x) else x[:, :0]
        del x

    def __len__(self):
        return len(self.h)

    def take(self, mask):
        out = object.__new__(Rows)
        mt = torch.as_tensor(mask, device=dev)
        out.chunk = self.chunk[mask]
        for k in ("h", "tok", "nxt", "hb", "z"):
            setattr(out, k, getattr(self, k)[mt])
        return out

    @staticmethod
    def cat(parts):
        out = object.__new__(Rows)
        out.chunk = np.concatenate([p.chunk for p in parts])
        for k in ("h", "tok", "nxt", "hb", "z"):
            setattr(out, k, torch.cat([getattr(p, k) for p in parts]))
        return out

    def space(self, name):
        return getattr(self, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rows_dir", default=None)
    ap.add_argument("--pairs", default="results/delta/d6144/evals/atlas_siblings/pairs.json")
    ap.add_argument("--geometry", default=None, help="geometry.json of phase 1: supplies the range-pair subgroup flags")
    ap.add_argument("--out", default="results/delta/d6144/evals/atlas_siblings/removal.json")
    ap.add_argument("--min_excl", type=int, default=100)
    ap.add_argument("--max_pairs", type=int, default=0, help="0 = every pair that passes the data rule; else a seeded random subset")
    ap.add_argument("--min_rows", type=int, default=30, help="a case needs >= this many fit and eval rows for T and S")
    ap.add_argument("--cap_chunks", type=int, default=300, help="chunks per member and half (seeded subsample)")
    ap.add_argument("--n_ctrl_fit", type=int, default=24000)
    ap.add_argument("--n_ctrl_eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()

    doc = json.load(open(resolve_input_path(args.pairs)))
    assert AS.pair_list_sha256(doc["pairs"]) == doc["pair_list_sha256"], "pairs.json does not match its recorded pair-list hash"
    cache = AS.AtlasCache(args.cache, args.rows_dir)
    smoke = cache.n_chunks != doc["meta"]["stream"]["n_chunks"]
    if not smoke:
        assert cache.fingerprint == doc["meta"]["stream"]["fingerprint_sha256"], "the cache is not the stream pairs.json was frozen on"
    fit = cache.fit_mask(0)
    sibs = doc["pairs"]
    rep, uid, M = AS.pair_exclusive_report(cache, sibs, args.min_excl)
    passing = [i for i, r in enumerate(rep) if r["passes"]]
    log(f"{len(sibs)} sibling pairs, {len(passing)} pass >= {args.min_excl} exclusive chunks per member at {cache.n_chunks:,} chunks")
    if args.max_pairs and len(passing) > args.max_pairs:
        rng = np.random.default_rng(args.seed)
        passing = sorted(int(i) for i in rng.choice(passing, args.max_pairs, replace=False))
        log(f"running a seeded random subset of {len(passing)} pairs")
    if not passing:
        raise SystemExit("no pair passes the data rule at this N; nothing to run")

    range_flag = {}
    if args.geometry and resolve_input_path(Path(args.geometry)).exists():
        geo = json.load(open(resolve_input_path(args.geometry)))
        for r in geo["pairs"]["sib"]:
            if "spaces" in r:
                range_flag[r["i"]] = bool(r["spaces"]["geoae"]["range_pair"])
        log(f"range-pair flags from {args.geometry}: {sum(range_flag.values())}/{len(range_flag)} sibling pairs are GeoAE range pairs")

    spaces = G.Spaces(dev)
    ae, std = spaces.ae, spaces.s_a
    from geoae.checkpoint import load_lm
    lm = load_lm(LM_NAME, device=dev)
    norm32, head32 = G.lm_head_fp32(lm)
    norm32, head32 = norm32.to(dev), head32.to(dev)
    del lm
    torch.cuda.empty_cache()
    scorer = O.Scorer(norm32, head32)
    log("loaded AE, fp32 head")

    # ---- controls: random Atlas chunks, fixed once ---------------------------------------------------------------------
    rng_c = np.random.default_rng(args.seed + 2000)
    per_chunk = max(1, int(round(cache.n_rows / cache.n_chunks)))
    fit_ch, ev_ch = np.flatnonzero(fit), np.flatnonzero(~fit)
    cf = np.sort(rng_c.choice(fit_ch, min(len(fit_ch), -(-args.n_ctrl_fit // per_chunk)), replace=False))
    ce = np.sort(rng_c.choice(ev_ch, min(len(ev_ch), -(-args.n_ctrl_eval // per_chunk)), replace=False))
    CF, CE = Rows(cache, spaces, cf), Rows(cache, spaces, ce)
    log(f"controls: {len(CF):,} fit rows ({len(cf)} chunks), {len(CE):,} eval rows ({len(ce)} chunks)")
    CF_np = {sp: CF.space(sp).cpu().numpy() for sp in ("h", "hb", "z")}

    tok0 = torch.full((1,), 1000, dtype=torch.long, device=dev)         # any fixed token: the bias cancels in a difference

    def apply_edit(space, edit, R):
        a = R.space(space)
        a2 = edit(a)
        m = (a2 != a).any(1)                                             # unedited rows keep h exactly: dNLL = KL = 0
        h2 = R.h.clone()
        idx = m.nonzero(as_tuple=True)[0]
        if len(idx):
            if space == "h":
                h2[idx] = a2[idx]
            elif space == "hb":
                h2[idx] = R.h[idx] + (a2[idx] - a[idx]) * std
            else:
                for s in range(0, len(idx), 2048):
                    ch = idx[s:s + 2048]
                    h2[ch] = R.h[ch] + (ae.decode(a2[ch], R.tok[ch]) - ae.decode(a[ch], R.tok[ch])) * std
        return h2, m

    def run_config(space, edit, R, bounds):
        h2, m = apply_edit(space, edit, R)
        dn, kl = scorer.row_metrics(R.h, h2, m, R.nxt)
        return {g: {"dnll": float(dn[s:e].mean()), "kl": float(kl[s:e].mean()),
                    "frac_edited": float(m[s:e].float().mean()), "n": int(e - s)} for g, (s, e) in bounds.items()}

    def dir_edit(c, u, lo, hi, rep, alpha, gated):
        def edit(a):
            t = (a - c) @ u
            shift = float(alpha) * (rep - t)
            if gated:
                shift = shift * ((t >= lo) & (t <= hi)).to(shift.dtype)
            return a + shift[:, None] * u
        return edit

    def case(T, S, TF, SF, TE, SE, meta):
        """T, S: concept ids; TF/SF fit rows, TE/SE eval rows of the target / sibling's exclusive chunks."""
        t0 = time.time()
        bad = np.concatenate([meta["chunks_T"], meta["chunks_S"]])
        keep_f = ~np.isin(CF.chunk, bad)
        keep_e = ~np.isin(CE.chunk, bad)
        CFx, CEx = CF.take(keep_f), CE.take(keep_e)
        EV = Rows.cat([TE, SE, CEx])
        nT, nS = len(TE), len(SE)
        bounds = {"target": (0, nT), "sibling": (nT, nT + nS), "random": (nT + nS, len(EV))}
        res = {a: {} for a in ARMS}
        for sp, arm in (("h", "nl_h"), ("hb", "nl_hb"), ("z", "nl_z")):
            aT = TF.space(sp).cpu().numpy()
            aC = CF_np[sp][keep_f]
            acts = np.concatenate([aT, aC])
            is_c = np.r_[np.ones(len(aT), bool), np.zeros(len(aC), bool)]
            sal = dprime_saliency(acts, is_c)
            rep = aC.mean(0)
            for pp in NL_P:
                salient = select_top(sal, p=pp)
                lo, hi, _, _ = fit_ranges(aT, salient, tao=2.0)
                res[arm][skey(pp)] = run_config(sp, gate_edit(lo, hi, rep, dev), EV, bounds)
            del acts, aC
        # shared direction in z
        Zc, ZT, ZS = CFx.z, TF.z, SF.z
        c = Zc.mean(0)
        muT, muS = ZT.mean(0) - c, SF.z.mean(0) - c
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
        # cross-fitted d' of the same direction on the eval half (used only if geometry.json is not supplied)
        eT, eS = (TE.z - c) @ u, (SE.z - c) @ u
        dir_info["dprime_T_S_eval"] = float((eT.mean() - eS.mean()).abs() / torch.sqrt(0.5 * (eT.var(correction=0) + eS.var(correction=0)) + 1e-8))
        for al in DIR_ALPHA:
            res["dir_range_z"][skey(al)] = run_config("z", dir_edit(c, u, lo_t, hi_t, rep_t, al, True), EV, bounds)
            res["dir_all_z"][skey(al)] = run_config("z", dir_edit(c, u, lo_t, hi_t, rep_t, al, False), EV, bounds)
        # the same direction in the residual
        v = (ae.decode((c + u)[None], tok0) - ae.decode(c[None], tok0))[0] * std
        v = v / v.norm()
        mh = CFx.h.mean(0)
        thT = (TF.h - mh) @ v
        rep_h = float(((CFx.h - mh) @ v).mean())
        lo_h, hi_h = float(thT.mean() - 2 * thT.std(correction=0)), float(thT.mean() + 2 * thT.std(correction=0))
        for al in DIR_ALPHA:
            res["dir_range_h"][skey(al)] = run_config("h", dir_edit(mh, v, lo_h, hi_h, rep_h, al, True), EV, bounds)
        rec = {"T": int(T), "S": int(S), "showcase": False, "T_position": "inner" if inner else "outer",
               "pair": meta["pair"], "n_fit": {"T": len(TF), "S": len(SF), "ctrl": len(CFx)},
               "n_eval": {"T": nT, "S": nS, "ctrl": len(CEx)}, "dir_info": dir_info, "res": res,
               "tag": meta["tag"], "pair_index": meta["i"], "range_pair_z": meta["range_pair_z"]}
        std_r = {a: res[a][skey(STD_STRENGTH[a])] for a in ARMS}
        log(f"  case T={meta['names'][0]!r} S={meta['names'][1]!r} ({rec['T_position']}) {time.time() - t0:.0f}s | std dNLL tgt/sib/rnd: "
            + "  ".join(f"{a}={std_r[a]['target']['dnll']:.2f}/{std_r[a]['sibling']['dnll']:.2f}/{std_r[a]['random']['dnll']:.2f}" for a in ARMS))
        return rec

    cases, skipped = [], []
    for k, i in enumerate(passing):
        r = sibs[i]
        a, b = uid[tuple(r["a_ids"])], uid[tuple(r["b_ids"])]
        sets = AS.pair_chunk_sets(M, a, b, fit)
        rng = np.random.default_rng([args.seed, i])
        capped = {n: AS.cap_chunks(sets[n], args.cap_chunks, rng) for n in ("a_fit", "a_eval", "b_fit", "b_eval")}
        rows_n = {n: sum(int(cache.row_start[c + 1] - cache.row_start[c]) for c in capped[n]) for n in capped}
        if min(rows_n.values()) < args.min_rows:
            skipped.append({"i": i, "a": r["a"], "b": r["b"], "rows": rows_n})
            log(f"pair {k + 1}/{len(passing)} {r['a_name']!r}~{r['b_name']!r}: SKIPPED (rows {rows_n} < {args.min_rows})")
            continue
        R = {n: Rows(cache, spaces, capped[n]) for n in capped}
        # a case's controls and its own-exclusive chunks must not overlap the pair: both members' chunks (any label) go out
        chunks_a = np.concatenate([sets["a_fit"], sets["a_eval"], sets["both"]])
        chunks_b = np.concatenate([sets["b_fit"], sets["b_eval"], sets["both"]])
        flag = range_flag.get(i)
        log(f"pair {k + 1}/{len(passing)} [{r['tag']}] {r['a_name']!r} ~ {r['b_name']!r}  rows fit/eval A {len(R['a_fit'])}/{len(R['a_eval'])} B {len(R['b_fit'])}/{len(R['b_eval'])}  GeoAE range pair: {flag}")
        base = {"i": i, "tag": r["tag"], "range_pair_z": flag}
        pair_meta = {"a": r["a"], "b": r["b"], "cos": r["emb_cos"], "dprime": None, "n_a": rows_n["a_fit"] + rows_n["a_eval"], "n_b": rows_n["b_fit"] + rows_n["b_eval"]}
        for (T, S, TF, SF, TE, SE, cT, cS, nm) in (
                (r["a"], r["b"], R["a_fit"], R["b_fit"], R["a_eval"], R["b_eval"], chunks_a, chunks_b, (r["a_name"], r["b_name"])),
                (r["b"], r["a"], R["b_fit"], R["a_fit"], R["b_eval"], R["a_eval"], chunks_b, chunks_a, (r["b_name"], r["a_name"]))):
            meta = dict(base, pair=pair_meta, chunks_T=cT, chunks_S=cS, names=nm)
            cases.append(case(T, S, TF, SF, TE, SE, meta))
        del R

    log(f"evaluated {len(cases)} cases from {len(passing) - len(skipped)} pairs, skipped {len(skipped)} pairs")
    # range-pair flag fallback when geometry.json is absent: cross-fitted d' >= 2 and fit-cosine >= 0.9 in z on the case's own rows
    if not range_flag:
        for c in cases:
            c["range_pair_z"] = bool(c["dir_info"]["cos_muT_muS"] >= G.COS and c["dir_info"]["dprime_T_S_eval"] >= G.DPRIME)

    groups = {"all": cases,
              "range_pairs_z": [c for c in cases if c["range_pair_z"]],
              "not_range_pairs_z": [c for c in cases if not c["range_pair_z"]],
              "rule_E": [c for c in cases if c["tag"] in ("E", "both")],
              "rule_C": [c for c in cases if c["tag"] in ("C", "both")]}
    summaries, text = {}, []
    for gname, cs in groups.items():
        if len(cs) < 2:
            summaries[gname] = {"n_cases": len(cs), "note": "too few cases"}
            continue
        summaries[gname] = O.summarise(cs)
        summaries[gname]["n_cases"] = len(cs)
        summaries[gname]["sibling_to_target_ratio_at_matched_damage"] = sparing_ratio(cs)
        if gname == "all":
            full = O.format_summary(summaries[gname], cs)
            text.append(full[:full.index("=== (d) showcase")].rstrip() if "=== (d) showcase" in full else full)
    text.append(format_groups(summaries))
    txt = ("*** SMOKE (tiny n) ***\n" if smoke else "") + "\n".join(text)
    print("\n" + txt, flush=True)
    out = {"meta": {"args": vars(args), "smoke": smoke, "pair_list_sha256": doc["pair_list_sha256"],
                    "cache": {"dir": args.cache, "n_chunks": cache.n_chunks, "n_rows": int(cache.n_rows), "fingerprint": cache.fingerprint},
                    "n_pairs_total": len(sibs), "n_pairs_pass_data_rule": int(sum(r["passes"] for r in rep)),
                    "n_pairs_run": len(passing) - len(skipped), "n_cases": len(cases), "skipped_pairs": skipped,
                    "range_flags_from": args.geometry if range_flag else "inline cross-fit in z (cos>=0.9, d'_eval>=2)",
                    "grid": {a: GRID[a] for a in ARMS}, "std_strength": STD_STRENGTH, "norm_check": spaces.norm_check,
                    "group_sizes": {g: len(cs) for g, cs in groups.items()}, "seconds": time.time() - t_start},
           "cases": cases, "summary": summaries, "summary_text": txt}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
    log(f"wrote {args.out} ({time.time() - t_start:.0f}s total)")


def sparing_ratio(cases):
    """Per arm: median over cases of sibling dNLL / target dNLL at the matched target damage L (0 = sibling untouched)."""
    curves = [{a: O.curve(c, a) for a in ARMS} for c in cases]
    out = {}
    for a in ARMS:
        r = []
        for cv in curves:
            L = 0.5 * min(cv[x][:, 0].max() for x in ARMS)
            if L > 0:
                r.append(O.at_level(cv[a], L)[0] / L)
        if r:
            m, lo, hi = boot(r, np.median)
            out[a] = {"median": m, "ci": [lo, hi], "n": len(r)}
    return out


def format_groups(summaries):
    L = ["", "=== subgroups: sibling dNLL at matched target damage (mean [95% CI]) and sibling/target ratio (median) ==="]
    w = max(len(a) for a in ARMS)
    for g, s in summaries.items():
        if "matched" not in s:
            L.append(f"-- {g}: {s.get('n_cases', 0)} cases ({s.get('note', '')})")
            continue
        m = s["matched"]
        L.append(f"-- {g}: {s['n_cases']} cases, matched on {m['n_cases']}, L median {m['L_median']:.3f} nats")
        L.append(f"   {'arm':<{w}} | {'sibling dNLL@L':>24} | {'random dNLL@L':>14} | {'sib/tgt median':>14}")
        for a in ARMS:
            d = m["arms"][a]
            ratio = s["sibling_to_target_ratio_at_matched_damage"].get(a, {})
            L.append(f"   {a:<{w}} | {d['sibling']['mean']:>7.3f} [{d['sibling']['mean_ci'][0]:.3f},{d['sibling']['mean_ci'][1]:.3f}] | "
                     f"{d['random']['mean']:>14.3f} | {ratio.get('median', float('nan')):>14.3f}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
