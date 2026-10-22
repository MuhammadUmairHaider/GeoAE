"""Summaries for the bounded judge run: Gemma steering, Haiku head-to-head (Gemma 2x, Llama 2x), Haiku-vs-Opus agreement."""

from geoae.paths import resolve_input_path
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("results/delta")


def basic_ci(est, boots):
    b = np.asarray(boots)
    d = b - b.mean()
    return est + np.quantile(d, .025), est + np.quantile(d, .975)


print("=== named-concept steering, net usable per 100 at alpha 0.5 (k-means dirs / GeoAE dirs / supervised)")
for name, d in (("Llama 2x", "d6144"), ("Gemma 1x", "g1b_l25_d1152"), ("Gemma 2x", "g1b_l25_d2304"), ("Gemma 4x", "g1b_l25_d4608")):
    p = R / d / "concept/concept_steer_judge_llmfluency.json"
    if not resolve_input_path(p).exists():
        print(f"  {name}: missing"); continue
    j = json.loads(resolve_input_path(p).read_text())
    pa = j["scopes"]["pooled"]["per_alpha"]
    best = j.get("best_alpha", {})
    row = []
    for h in ("base_dir", "bypass_dir", "label"):
        u = pa[h]["0.5"]["usable"]; f = pa[h]["0.5"]["fluent_share"]
        row.append(f"{u[0] * 100:5.1f} [{u[1] * 100:.1f},{u[2] * 100:.1f}] fl {f[0]:.0%}")
    pw = j["scopes"]["pooled"].get("pairwise", {}).get("bypass_dir_minus_base_dir", {})
    diff = f"paired GeoAE-kmeans {pw.get('mean_diff', float('nan')) * 100:+.1f} [{pw['ci'][0] * 100:+.1f},{pw['ci'][1] * 100:+.1f}] p={pw.get('wilcoxon_p', float('nan')):.3f}" if pw else ""
    print(f"  {name}: " + " | ".join(row) + f"  {diff}  best alpha {best}")
    for scope in ("db14", "biasbios"):
        s = j["scopes"].get(scope, {}).get("pairwise", {}).get("bypass_dir_minus_base_dir")
        if s:
            print(f"      {scope}: GeoAE-kmeans {s['mean_diff'] * 100:+.1f} [{s['ci'][0] * 100:+.1f},{s['ci'][1] * 100:+.1f}]")


def summarise_pairs(path):
    recs = [r for r in json.loads(resolve_input_path(Path(path)).read_text())["records"]]
    out = {}
    pairs = defaultdict(lambda: defaultdict(list))
    lvl = defaultdict(lambda: defaultdict(list))
    for r in recs:
        a = r.get("answer")
        if not a:
            continue
        key = (r["kind"], r["first"], r["second"], r["pair"])
        for q in ("clearer_concept", "more_semantic"):
            v = a.get(q)
            if v in ("tie", "neither"):
                pairs[key][q].append(.5)
            elif v in ("A", "B"):
                pairs[key][q].append(float((v == "A") == r["second_is_a"]))
        if r["kind"] == "main":
            sec, fst = ("A", "B") if r["second_is_a"] else ("B", "A")
            lvl[(r["first"], r["second"])]["second"].append((a.get(sec) or {}).get("level"))
            lvl[(r["first"], r["second"])]["first"].append((a.get(fst) or {}).get("level"))
            for role, side in (("second_abs", sec), ("first_abs", fst)):
                x = (a.get(side) or {}).get("abstraction")
                if x is not None:
                    lvl[(r["first"], r["second"])][role].append(x)
    rng = np.random.default_rng(0)
    for (kind, x, y, _), qs in pairs.items():
        for q, v in qs.items():
            out.setdefault((kind, x, y, q), []).append(np.mean(v))
    res = {}
    for k, v in out.items():
        v = np.asarray(v)
        boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        lo, hi = basic_ci(v.mean(), boots)
        res[k] = (v.mean(), lo, hi, len(v))
    return res, lvl


def print_pairs(name, path):
    if not resolve_input_path(Path(path)).exists():
        print(f"  {name}: missing"); return None
    res, lvl = summarise_pairs(path)
    print(f"--- {name}")
    for (kind, x, y, q), (m, lo, hi, n) in sorted(res.items()):
        if kind == "control" and q != "clearer_concept":
            continue
        print(f"  {kind:7s} {x:>18s} vs {y:13s} {q:16s} {m:.3f} [{lo:.3f},{hi:.3f}] n={n}")
    for (x, y), d in lvl.items():
        tf = lambda L: np.mean([l == "topic" for l in L if l]) if L else float("nan")
        print(f"  levels {x} vs {y}: topic share {tf(d['first']):.0%} -> {tf(d['second']):.0%}; "
              f"abstraction {np.mean(d['first_abs']):.2f} -> {np.mean(d['second_abs']):.2f}")
    return res


print("\n=== head-to-head pairs (win rate of the second space; 0.5 = equal)")
print_pairs("Gemma 2x, Haiku 4.5", R / "g1b_l25_d2304/interp/pairwise_claude-haiku-4.5.json")
print_pairs("Llama 2x, Haiku 4.5", R / "d6144/interp/pairwise_claude-haiku-4.5.json")
print_pairs("Llama 2x, Opus agents (reference)", R / "d6144/interp/pairwise_claude-opus-agents.json")

print("\n=== Haiku vs Opus agents on identical Llama calls (decisive answers)")
hp, op = R / "d6144/interp/pairwise_claude-haiku-4.5.json", R / "d6144/interp/pairwise_claude-opus-agents.json"
if resolve_input_path(hp).exists() and resolve_input_path(op).exists():
    key = lambda r: (r["kind"], r["first"], r["second"], r["pair"], r["order"])
    H = {key(r): r.get("answer") for r in json.loads(resolve_input_path(hp).read_text())["records"]}
    O = {key(r): r.get("answer") for r in json.loads(resolve_input_path(op).read_text())["records"]}
    for q in ("clearer_concept", "more_semantic"):
        both = [(H[k][q], O[k][q]) for k in H if k in O and H[k] and O[k] and H[k].get(q) in ("A", "B") and O[k].get(q) in ("A", "B")]
        agree = np.mean([a == b for a, b in both]) if both else float("nan")
        pa = np.mean([a == "A" for a, _ in both]); pb = np.mean([b == "A" for _, b in both])
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (agree - pe) / (1 - pe) if both else float("nan")
        print(f"  {q:16s} agreement {agree:.1%} on {len(both)} calls both decisive, Cohen kappa {kappa:.2f}")
