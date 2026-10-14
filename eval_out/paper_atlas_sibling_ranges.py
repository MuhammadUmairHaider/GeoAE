"""Phase 1 of the Atlas sibling test: are sibling concepts "range pairs" (one shared direction, two magnitudes)?

For every frozen sibling pair and its two null pairs (pairs.json: names chosen without any activation) and for each
of three spaces on the SAME rows -- base h normalised exactly as the AE input, token-centred h (h - b[tok]), and the
d6144 token-bypass GeoAE latent z = ae.encode(x, tok) -- this applies eval_out/paper_ranges_along_directions.py's
definitions (see geoae/interp/atlas_geometry.py) to the rows of A-ONLY and B-ONLY chunks:

  * rows of a chunk = the (up to 16) token positions the extractor kept; chunks split into a FIT and an EVAL half by
    DOCUMENT (a hash of doc_int_id), so no document is on both sides;
  * centred means mu_a, mu_b (relative to the space's global mean over ALL cached rows) and the shared direction u come
    from the fit half; cos = cosine(mu_a, mu_b); d' is measured on the eval half's projections on u;
  * range pair = cos >= 0.9 and d' >= 2 (the original thresholds);
  * a document-level bootstrap (resampling documents inside each of the four row sets, whole statistic recomputed)
    gives every pair a CI on cos and d' and P(range pair); a pair bootstrap over pairs gives the group CIs;
  * the unit of analysis is the token row (as in the original). A SECONDARY analysis pools each chunk's rows into one
    chunk-mean vector (`unit = chunk_mean`: token-level noise averages out; d' is then between chunk means), reported
    next to the row-level one and clearly labelled; the registered primary result is the row-level one;
  * a pair is analysed only if BOTH members have >= min_excl exclusive chunks (the data rule fixed in pairs.json) and at
    least 2 documents in each half; chunks per member and half are capped at --cap_chunks (seeded subsample).

    scripts/delta/py eval_out/paper_atlas_sibling_ranges.py --cache /scratch/dogar/geoae/atlas_l27 \
        --pairs results/delta/d6144/evals/atlas_siblings/pairs.json --out results/delta/d6144/evals/atlas_siblings/geometry.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoae.interp import atlas_geometry as G  # noqa: E402
from geoae.interp import atlas_siblings as AS  # noqa: E402

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def global_means(cache, spaces, dev, bs=32768):
    sums = {k: torch.zeros(spaces.dim if k == "geoae" else 3072, dtype=torch.float64, device=dev) for k in G.SPACES}
    n = 0
    for s in range(0, cache.n_rows, bs):
        e = min(s + bs, cache.n_rows)
        sp = spaces(np.array(cache.rows[s:e]), np.array(cache.tok[s:e], dtype=np.int64))
        for k in G.SPACES:
            sums[k] += sp[k].double().sum(0)
        n += e - s
    return {k: (v / n).float() for k, v in sums.items()}, n


def _one_space(units, mean, B, gen, dev):
    """units: {a_fit, b_fit, a_eval, b_eval} -> (X, doc idx). Returns stats (+ bootstrap summary)."""
    st = G.pair_stats(units["a_fit"][0], units["b_fit"][0], units["a_eval"][0], units["b_eval"][0], mean)
    st["range_pair"] = bool(st["cos"] >= G.COS and st["dprime"] >= G.DPRIME)
    if B > 0:
        bt = G.bootstrap_pair(units["a_fit"], units["b_fit"], units["a_eval"], units["b_eval"], mean, B, gen)
        isr = (bt["cos"] >= G.COS) & (bt["dprime"] >= G.DPRIME)
        q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]      # noqa: E731
        st["boot"] = {"B": B, "cos_ci": q(bt["cos"]), "dprime_ci": q(bt["dprime"]), "radial_share_ci": q(bt["radial_share"]),
                      "p_range": float(isr.mean())}
    return st


def analyse_pair(cache, spaces, means, sets, B, gen, dev):
    """sets: dict a_fit/a_eval/b_fit/b_eval -> chunk arrays. Returns {"row": {space: stats}, "chunk_mean": {space: stats}}."""
    names = ("a_fit", "b_fit", "a_eval", "b_eval")
    rows = {}
    for nm in names:
        _, h16, tok, _ = cache.gather(sets[nm])
        rows[nm] = spaces(h16, tok), G.row_docs(cache, sets[nm]).to(dev)
    out = {"row": {}, "chunk_mean": {}}
    for sp in G.SPACES:
        out["row"][sp] = _one_space({nm: (rows[nm][0][sp], rows[nm][1]) for nm in names}, means[sp], B, gen, dev)
        out["chunk_mean"][sp] = _one_space({nm: G.chunk_means(cache, sets[nm], rows[nm][0][sp]) for nm in names}, means[sp], B, gen, dev)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rows_dir", default=None, help="where the big row arrays live (a /dev/shm copy); default the cache dir")
    ap.add_argument("--pairs", default="results/delta/d6144/evals/atlas_siblings/pairs.json")
    ap.add_argument("--out", default="results/delta/d6144/evals/atlas_siblings/geometry.json")
    ap.add_argument("--min_excl", type=int, default=100)
    ap.add_argument("--min_half", type=int, default=2, help="documents' chunks needed per member in EACH half")
    ap.add_argument("--cap_chunks", type=int, default=500, help="chunks per member and half (seeded subsample)")
    ap.add_argument("--n_boot", type=int, default=200, help="document-bootstrap replicates per pair (0 = off)")
    ap.add_argument("--n_pair_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()
    dev = torch.device("cuda")

    doc = json.load(open(args.pairs))
    assert AS.pair_list_sha256(doc["pairs"]) == doc["pair_list_sha256"], "pairs.json does not match its recorded pair-list hash"
    cache = AS.AtlasCache(args.cache, args.rows_dir)
    log(f"cache {args.cache}: {cache.n_chunks:,} chunks, {cache.n_rows:,} rows; pairs.json hash {doc['pair_list_sha256'][:16]}")
    ref = doc["meta"]["stream"]
    if cache.n_chunks == ref["n_chunks"]:
        assert cache.fingerprint == ref["fingerprint_sha256"], "the cache is not the stream pairs.json was frozen on"
    fit = cache.fit_mask(0)

    sibs = doc["pairs"]
    nulls = {"null_same": doc["null_same"], "null_cross": doc["null_cross"]}
    allrecs = list(sibs) + [x for v in nulls.values() for x in v if x]
    rep, uid, M = AS.pair_exclusive_report(cache, allrecs, args.min_excl)

    spaces = G.Spaces(dev)
    log(f"norm-constant agreement (max abs diff): {spaces.norm_check}")
    t0 = time.time()
    means, n_rows = global_means(cache, spaces, dev)
    log(f"global means over {n_rows:,} rows in {time.time() - t0:.0f}s")

    def run(rec, kind, idx):
        a, b = uid[tuple(rec["a_ids"])], uid[tuple(rec["b_ids"])]
        sets = AS.pair_chunk_sets(M, a, b, fit)
        n_ex = [len(sets["a_fit"]) + len(sets["a_eval"]), len(sets["b_fit"]) + len(sets["b_eval"])]
        if min(n_ex) < args.min_excl:
            return {"skipped": f"exclusive chunks {n_ex} < {args.min_excl}", "n_exclusive": n_ex}
        if min(len(sets[k]) for k in ("a_fit", "a_eval", "b_fit", "b_eval")) < args.min_half:
            return {"skipped": "fewer than min_half chunks in a half", "n_exclusive": n_ex}
        rng = np.random.default_rng([args.seed, idx, {"sib": 0, "null_same": 1, "null_cross": 2}[kind]])
        capped = {k: AS.cap_chunks(sets[k], args.cap_chunks, rng) for k in ("a_fit", "a_eval", "b_fit", "b_eval")}
        gen = torch.Generator(device=dev).manual_seed(args.seed * 1_000_003 + idx * 3 + {"sib": 0, "null_same": 1, "null_cross": 2}[kind])
        res = analyse_pair(cache, spaces, means, capped, args.n_boot, gen, dev)
        return {"n_exclusive": n_ex, "n_chunks_used": {k: int(len(v)) for k, v in capped.items()},
                "n_shared_chunks": int(len(sets["both"])), "spaces": res["row"], "spaces_chunk_mean": res["chunk_mean"]}

    records = {"sib": [], "null_same": [], "null_cross": []}
    t0 = time.time()
    total = len(sibs) * 3
    done = 0
    for i, r in enumerate(sibs):
        out = {"i": i, "a": r["a"], "b": r["b"], "a_name": r["a_name"], "b_name": r["b_name"], "tag": r["tag"], "lcc": r["lcc"]}
        out.update(run(r, "sib", i))
        records["sib"].append(out)
        for kind in ("null_same", "null_cross"):
            x = nulls[kind][i]
            if x is None:
                records[kind].append({"i": i, "skipped": "no null found"})
                continue
            out = {"i": i, "a": x["a"], "b": x["b"], "a_name": x["a_name"], "b_name": x["b_name"], "lcc_a": x["lcc_a"], "lcc_b": x["lcc_b"]}
            out.update(run(x, kind, i))
            records[kind].append(out)
        done += 3
        if (i + 1) % 25 == 0 or i + 1 == len(sibs):
            log(f"pairs {i + 1}/{len(sibs)} ({time.time() - t0:.0f}s, {(time.time() - t0) / done:.2f}s per pair-kind)")

    # ---- aggregate ---------------------------------------------------------------------------------------------------
    def ok(rec):
        return "spaces" in rec

    tags = [r["tag"] for r in sibs]
    sel = {"union": list(range(len(sibs))),
           "E": [i for i, t in enumerate(tags) if t in ("E", "both")],
           "C": [i for i, t in enumerate(tags) if t in ("C", "both")],
           "both": [i for i, t in enumerate(tags) if t == "both"]}

    def aggregate(key):
        summary, paired = {}, {}
        for sp in G.SPACES:
            summary[sp], paired[sp] = {}, {}
            for g, idx in sel.items():
                summary[sp][g] = {}
                for kind in ("sib", "null_same", "null_cross"):
                    # a null is reported only next to its own sibling: matched sets use the indices where BOTH exist
                    if kind != "sib":
                        recs = [records[kind][i] for i in idx if ok(records[kind][i]) and ok(records["sib"][i])]
                    else:
                        recs = [records["sib"][i] for i in idx if ok(records["sib"][i])]
                    S = [r[key][sp] for r in recs]
                    s = G.summarise_group([x["cos"] for x in S], [x["dprime"] for x in S], [x["radial_share"] for x in S], args.n_pair_boot, args.seed)
                    if S and "boot" in S[0]:
                        pr = np.array([x["boot"]["p_range"] for x in S])
                        s["mean_p_range"] = float(pr.mean())
                        s["share_robust_range_p_ge_0.95"] = float((pr >= 0.95).mean())
                        s["median_dprime_ci_halfwidth"] = float(np.median([(x["boot"]["dprime_ci"][1] - x["boot"]["dprime_ci"][0]) / 2 for x in S]))
                        s["median_cos_ci_halfwidth"] = float(np.median([(x["boot"]["cos_ci"][1] - x["boot"]["cos_ci"][0]) / 2 for x in S]))
                    s["median_insample_dprime"] = float(np.median([x["insample_dprime"] for x in S])) if S else None
                    s["median_insample_cos"] = float(np.median([x["insample_cos"] for x in S])) if S else None
                    s["median_cos_eval"] = float(np.median([x["cos_eval"] for x in S])) if S else None
                    summary[sp][g][kind] = s
                for kind in ("null_same", "null_cross"):
                    both = [i for i in idx if ok(records["sib"][i]) and ok(records[kind][i])]
                    fs = [records["sib"][i][key][sp]["range_pair"] for i in both]
                    fn = [records[kind][i][key][sp]["range_pair"] for i in both]
                    paired[sp].setdefault(g, {})[kind] = G.paired_diff(fs, fn, args.n_pair_boot, args.seed)
        return summary, paired

    summary, paired = aggregate("spaces")
    summary_cm, paired_cm = aggregate("spaces_chunk_mean")
    n_ok = {k: sum(ok(r) for r in v) for k, v in records.items()}
    skipped = {k: sum(not ok(r) for r in v) for k, v in records.items()}
    out = {"meta": {"args": vars(args), "pair_list_sha256": doc["pair_list_sha256"], "pairs_file": args.pairs,
                    "cache": {"dir": args.cache, "n_chunks": cache.n_chunks, "n_rows": int(cache.n_rows), "fingerprint": cache.fingerprint},
                    "range_criteria": {"cos": G.COS, "dprime": G.DPRIME}, "norm_check": spaces.norm_check,
                    "n_analysed": n_ok, "n_skipped": skipped, "seconds": time.time() - t_start,
                    "definition": "fit-half means and shared direction, eval-half d'; global mean over all cached rows; document-level bootstrap",
                    "units": "summary / pairs[*].spaces = token rows (registered primary); summary_chunk_mean / spaces_chunk_mean = chunk-mean vectors (secondary)",
                    "smoke": cache.n_chunks != ref["n_chunks"]},
           "summary": summary, "paired_sibling_minus_null_range_share": paired,
           "summary_chunk_mean": summary_cm, "paired_chunk_mean": paired_cm, "pairs": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"wrote {args.out} ({time.time() - t_start:.0f}s); analysed {n_ok}, skipped {skipped}")
    print(format_table(out, "summary", "paired_sibling_minus_null_range_share", "token rows (registered primary)"))
    print(format_table(out, "summary_chunk_mean", "paired_chunk_mean", "chunk-mean vectors (secondary)"))


def format_table(out, skey_="summary", pkey="paired_sibling_minus_null_range_share", label="token rows") -> str:
    L = []
    sm = "  *** SMOKE (tiny n) ***" if out["meta"]["smoke"] else ""
    L.append(f"=== phase 1 [{label}]: share of range pairs (cos >= {G.COS}, d' >= {G.DPRIME}); cross-fitted; 95% CI over pairs{sm} ===")
    for sp in G.SPACES:
        L.append(f"-- {sp}")
        L.append(f"   {'group':<7}{'kind':<11}{'n':>4} | {'range share [CI]':>24} | {'cos med':>7} {'d med':>6} {'radial med':>10} | {'cos>=.9':>7} {'d>=2':>6}")
        for g, d in out[skey_][sp].items():
            for kind in ("sib", "null_same", "null_cross"):
                s = d[kind]
                if not s["n"]:
                    L.append(f"   {g:<7}{kind:<11}{0:>4} |")
                    continue
                rs = s["range_share"]
                L.append(f"   {g:<7}{kind:<11}{s['n']:>4} | {rs['value']:>8.3f} [{rs['ci'][0]:.3f},{rs['ci'][1]:.3f}] | "
                         f"{s['cos_median']['value']:>7.3f} {s['dprime_median']['value']:>6.2f} {s['radial_share_median']['value']:>10.3f} | "
                         f"{s['parallel_share']['value']:>7.3f} {s['dprime_ge_2_share']['value']:>6.3f}")
        pd_ = out[pkey][sp]["union"]
        L.append("   paired (sibling - null) range-share difference, union: "
                 + "; ".join(f"{k} {v['mean_diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] (n={v['n']})" for k, v in pd_.items()))
    return "\n".join(L)


if __name__ == "__main__":
    main()
