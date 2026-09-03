"""
STRUCTURAL generality of a cluster — computed from a closest_tokens JSON, no LLM.

Motivation: semantic coherence RISES toward the usage tail (4.10 -> 4.44 on the
b32k AE) because narrow clusters are TRIVIALLY easy to be coherent about. A
cluster that only ever fires on "#include", or on "the" when it precedes a band
name, scores 5 and tells you nothing. Coherence and breadth are separate axes and
ranking on coherence alone surfaces feature-split debris.

Three components, each normalised to 0..1, all deliberately independent of how
COHERENT the cluster is:

  token_div    normalised entropy over the closest-token STRINGS. A cluster that
               is one fixed string scores 0. NOTE this is 1 - the `monosemanticity`
               field closest_tokens already writes.
  domain_div   normalised entropy over the corpus domains (web/wiki/code/math/pile).
               A code-only cluster is narrower than one firing everywhere.
  context_div  normalised type-token ratio over the words SURROUNDING the marked
               token, with the marked token itself removed. Formulaic contexts
               ("day of month in birth/death dates") score low; a concept that
               appears in varied prose scores high.

generality = the geometric mean of the three, so a cluster must be broad on ALL
of them — a single fixed string in varied prose is still narrow.

    python -m geoae.interp.cluster_generality results/a.json results/b.json
    python -m geoae.interp.cluster_generality results/a.json --join results/llm_judge_generality.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[A-Za-z']+")
MARK = re.compile(r"«[^»]*»")


def norm_entropy(counts) -> float:
    tot = sum(counts.values())
    if tot <= 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / tot) * math.log(c / tot) for c in counts.values() if c)
    return h / math.log(len(counts))


def generality(cluster: dict) -> dict:
    top = cluster.get("top", [])
    tok_div = norm_entropy(Counter(e.get("token", "") for e in top))
    dom_div = norm_entropy(Counter(cluster.get("domains", {})))
    words = []
    for e in top:
        ctx = MARK.sub(" ", e.get("context", "") or "")   # drop the marked token
        words += [w.lower() for w in WORD.findall(ctx)]
    ctx_div = (len(set(words)) / len(words)) if words else 0.0
    g = (max(tok_div, 1e-9) * max(dom_div, 1e-9) * max(ctx_div, 1e-9)) ** (1 / 3)
    return {"token_div": tok_div, "domain_div": dom_div, "context_div": ctx_div,
            "generality_struct": g}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_paths", nargs="+")
    ap.add_argument("--join", default=None, help="llm_judge output, to correlate against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    judged = {}
    if args.join:
        for r in json.loads(Path(args.join).read_text()):
            judged[Path(r["source"]).name] = {c["cluster"]: c for c in r["clusters"]}

    out = []
    for p in args.json_paths:
        name = Path(p).name
        clusters = json.loads(Path(p).read_text())["clusters"]
        rows = []
        for cid, c in clusters.items():
            g = generality(c)
            g.update(cluster=int(cid), n_assigned=c["n_assigned"])
            rows.append(g)
        rows.sort(key=lambda r: -r["n_assigned"])
        print(f"\n=== {name}  ({len(rows)} clusters) ===")
        print(f"  token_div {st.mean(r['token_div'] for r in rows):.3f}   "
              f"domain_div {st.mean(r['domain_div'] for r in rows):.3f}   "
              f"context_div {st.mean(r['context_div'] for r in rows):.3f}   "
              f"GENERALITY {st.mean(r['generality_struct'] for r in rows):.3f}")
        print(f"  {'decile':<8}{'median n':>10}{'tok_div':>9}{'dom_div':>9}{'ctx_div':>9}{'GEN':>8}")
        for i in range(10):
            seg = rows[int(i / 10 * len(rows)):int((i + 1) / 10 * len(rows))]
            if not seg:
                continue
            print(f"  d{i+1:<7}{st.median(r['n_assigned'] for r in seg):>10.0f}"
                  f"{st.mean(r['token_div'] for r in seg):>9.3f}"
                  f"{st.mean(r['domain_div'] for r in seg):>9.3f}"
                  f"{st.mean(r['context_div'] for r in seg):>9.3f}"
                  f"{st.mean(r['generality_struct'] for r in seg):>8.3f}")
        if name in judged:
            pairs = [(r["generality_struct"], judged[name][r["cluster"]].get("generality"))
                     for r in rows if r["cluster"] in judged[name]
                     and isinstance(judged[name][r["cluster"]].get("generality"), (int, float))]
            if len(pairs) > 10:
                xs, ys = zip(*pairs)
                mx, my = st.mean(xs), st.mean(ys)
                num = sum((a - mx) * (b - my) for a, b in pairs)
                den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
                print(f"\n  structural vs LLM generality: r = {num/den:+.3f}  (n={len(pairs)})")
        out.append({"source": p, "rows": rows})

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
