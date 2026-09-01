"""
Frequency-ranked view of a closest_tokens.json.

closest_tokens writes clusters keyed by cluster id, which is the order the AE
happens to number them in — not an order anyone wants to read. This re-emits the
same records sorted by corpus usage (descending `n_assigned`), so the head of the
file is the clusters that actually carry the corpus and the tail is the ones the
model barely uses.

Reads an existing JSON, so it costs seconds rather than re-running the ~1 h
collection pass. Two outputs:

  <stem>_by_freq.json  identical schema and formatting to the source file --
                       same {meta, clusters: {cluster_id: record}} shape, same
                       indent=2 / ensure_ascii=False -- with the cluster keys in
                       descending-usage order and two fields added per record:
                       `rank` and `cum_pct` (cumulative share of tokens)
  <stem>_freq.tsv      one line per cluster for eyeballing / sorting in a sheet:
                       rank, id, count, usage %, cumulative %, monosemanticity,
                       distance min/median, dominant domain, top token strings

It also prints the usage distribution summary: the fitted Zipf exponent, usage
perplexity (how many clusters the corpus effectively uses), and the counts above
/ below uniform — the numbers that say whether a balancing target was reached.

Usage:
    python -m geoae.interp.rank_clusters \
      e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_balance_phased/closest_tokens.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def top_tokens(cluster: dict, n: int) -> str:
    td = cluster.get("token_distribution", {})
    pairs = list(td.items()) if isinstance(td, dict) else list(td)
    return " ".join(repr(t)[1:-1] for t, _ in pairs[:n])


def dominant_domain(cluster: dict) -> str:
    dom = cluster.get("domains", {})
    if not dom:
        return "-"
    total = sum(dom.values()) or 1
    best = max(dom, key=dom.get)
    return f"{best}:{100 * dom[best] / total:.0f}%"


def usage_summary(counts: np.ndarray, K: int) -> dict:
    """Zipf exponent, effective cluster count, and spread against uniform."""
    ranked = np.sort(counts)[::-1]
    p = ranked / ranked.sum()
    nz = ranked[ranked > 0]
    slope, intercept = np.polyfit(np.log(np.arange(1, len(nz) + 1)), np.log(nz), 1)
    pred = np.exp(intercept + slope * np.log(np.arange(1, len(nz) + 1)))
    resid = ((np.log(nz) - np.log(pred)) ** 2).sum()
    total = ((np.log(nz) - np.log(nz).mean()) ** 2).sum()
    nzp = p[p > 0]
    uniform = 1.0 / K
    return {
        "zipf_alpha": float(-slope),
        "zipf_r2": float(1 - resid / total),
        "usage_perplexity": float(np.exp(-(nzp * np.log(nzp)).sum())),
        "top1_pct": float(p[0] * 100),
        "top10_pct": float(p[:10].sum() * 100),
        "top100_pct": float(p[:100].sum() * 100),
        "n_above_5x_uniform": int((p > 5 * uniform).sum()),
        "n_below_half_uniform": int((p < 0.5 * uniform).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", help="closest_tokens.json written by geoae.interp.closest_tokens")
    ap.add_argument("--out_json", default=None, help="default: <stem>_by_freq.json")
    ap.add_argument("--out_tsv", default=None, help="default: <stem>_freq.tsv")
    ap.add_argument("--n_tokens_col", type=int, default=6, help="Token strings per TSV row")
    ap.add_argument("--print_n", type=int, default=20, help="Rows to print to stdout")
    args = ap.parse_args()

    src = Path(args.json_path)
    data = json.load(open(src))
    meta, clusters = data["meta"], data["clusters"]
    K = meta["n_clusters"]

    ranked = sorted(clusters.items(), key=lambda kv: -kv[1]["n_assigned"])
    counts = np.array([c["n_assigned"] for _, c in ranked], dtype=float)
    summary = usage_summary(counts, K)
    cum = np.cumsum(counts) / counts.sum() * 100

    # Same dict-of-clusters shape as the source; JSON preserves insertion order,
    # so re-inserting in ranked order is what "sorted by frequency" means here.
    records = {
        cid: {"rank": i, "cum_pct": round(float(cp), 3), **c}
        for i, ((cid, c), cp) in enumerate(zip(ranked, cum), start=1)
    }

    out_json = Path(args.out_json) if args.out_json else src.with_name(src.stem + "_by_freq.json")
    payload = {"meta": {**meta, "sorted_by": "n_assigned", "usage_summary": summary},
               "clusters": records}
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    out_tsv = Path(args.out_tsv) if args.out_tsv else src.with_name(src.stem + "_freq.tsv")
    with open(out_tsv, "w") as f:
        f.write("rank\tcluster_id\tn_assigned\tusage_pct\tcum_pct\tmonosemanticity\t"
                "dist_min\tdist_p50\tdomain\ttop_tokens\n")
        for cid, r in records.items():
            f.write(f"{r['rank']}\t{cid}\t{r['n_assigned']}\t{r['usage_pct']}\t"
                    f"{r['cum_pct']}\t{r['monosemanticity']}\t{r['dist_min']}\t{r['dist_p50']}\t"
                    f"{dominant_domain(r)}\t{top_tokens(r, args.n_tokens_col)}\n")

    print(f"[rank] {src}")
    print(f"[rank] {len(records)} live clusters / K={K}, {meta['n_tokens']} tokens")
    print(f"[rank] Zipf alpha {summary['zipf_alpha']:.3f} (R^2 {summary['zipf_r2']:.2f})  "
          f"usage perplexity {summary['usage_perplexity']:.0f}/{K}")
    print(f"[rank] top1 {summary['top1_pct']:.2f}%  top10 {summary['top10_pct']:.2f}%  "
          f"top100 {summary['top100_pct']:.1f}%  "
          f">5x uniform {summary['n_above_5x_uniform']}  <0.5x uniform {summary['n_below_half_uniform']}")
    print(f"\n{'rank':>5} {'cid':>5} {'count':>7} {'use%':>6} {'cum%':>6} {'mono':>5}  {'domain':<10} tokens")
    for cid, r in list(records.items())[: args.print_n]:
        print(f"{r['rank']:>5} {cid:>5} {r['n_assigned']:>7} {r['usage_pct']:>6.3f} "
              f"{r['cum_pct']:>6.2f} {r['monosemanticity']:>5.2f}  {dominant_domain(r):<10} "
              f"{top_tokens(r, args.n_tokens_col)}")
    print(f"\n[rank] wrote {out_json}")
    print(f"[rank] wrote {out_tsv}")


if __name__ == "__main__":
    main()
