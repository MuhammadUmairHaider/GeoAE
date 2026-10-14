"""Paired per-cluster ratings and conditional win rates from pairwise_judge outputs; seeded random example pairs."""
import json
import random
from collections import defaultdict

import numpy as np

I = "results/delta/d6144/interp/"
FILES = {"opus agents": I + "pairwise_claude-opus-agents.json", "opus api": I + "pairwise_claude-opus-5.5.json",
         "flash-lite": I + "pairwise_gemini-2.5-flash-lite.json", "gpt-4o-mini": I + "pairwise_gpt-4o-mini.json"}
LIT = {"surface", "morphology"}


def ci(vals, n_boot=2000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    est = v.mean()
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)])
    d = b - b.mean()
    return est, est + np.quantile(d, .025), est + np.quantile(d, .975), len(v)


def collect(path):
    pairs = defaultdict(lambda: {"first": defaultdict(list), "second": defaultdict(list), "clear": [], "sem": []})
    for r in json.load(open(path))["records"]:
        a = r.get("answer")
        if r["kind"] != "main" or not a:
            continue
        p = pairs[(r["first"], r["second"], r["pair"])]
        sides = {"second": "A" if r["second_is_a"] else "B", "first": "B" if r["second_is_a"] else "A"}
        for role, side in sides.items():
            s = a.get(side) or {}
            for m in ("abstraction", "coherence"):
                if s.get(m) is not None:
                    p[role][m].append(s[m])
            p[role]["level"].append(s.get("level"))
            p[role]["label"].append(s.get("label"))
        for q, k in (("clearer_concept", "clear"), ("more_semantic", "sem")):
            v = a.get(q)
            if v in ("tie", "neither"):
                p[k].append(0.5)
            elif v in ("A", "B"):
                p[k].append(float((v == "A") == r["second_is_a"]))
    return pairs


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}] (n {t[3]})"


for judge, path in FILES.items():
    try:
        pairs = collect(path)
    except FileNotFoundError:
        print(judge, "missing")
        continue
    print(f"\n######## {judge}")
    for x, y in sorted({k[:2] for k in pairs}):
        P = [pairs[k] for k in pairs if k[:2] == (x, y)]
        print(f"  {x} -> {y}  ({y} minus {x}, paired over pairs)")
        for m in ("abstraction", "coherence"):
            v = [np.mean(p["second"][m]) - np.mean(p["first"][m]) for p in P if p["second"][m] and p["first"][m]]
            print(f"    {m:18s} {fmt(ci(v))}")
        for L in ("topic", "surface", "semantics", "syntax", "discourse"):
            v = [np.mean([lv == L for lv in p["second"]["level"]]) - np.mean([lv == L for lv in p["first"]["level"]])
                 for p in P if p["second"]["level"] and p["first"]["level"]]
            print(f"    share {L:12s} {fmt(ci(v))}")
        for name, cond in (("all", lambda p: True),
                           ("x not literal", lambda p: not (set(p["first"]["level"]) & LIT)),
                           ("neither literal", lambda p: not (set(p["first"]["level"] + p["second"]["level"]) & LIT))):
            v = [np.mean(p["clear"]) for p in P if p["clear"] and cond(p)]
            w = [np.mean(p["sem"]) for p in P if p["sem"] and cond(p)]
            if v and w:
                print(f"    win | {name:15s} clearer {fmt(ci(v))}   more semantic {fmt(ci(w))}")

pairs = collect(FILES["opus agents"])
keys = sorted(k for k in pairs if k[:2] == ("base", "geoae"))
random.Random(7).shuffle(keys)
print("\n######## seeded-random base-vs-GeoAE pairs (opus agents); one order's labels")
for k in keys[:10]:
    p = pairs[k]

    def side(role):
        q = p[role]
        return (f"{q['label'][0]!r} (abs {q['abstraction'][0] if q['abstraction'] else None}, "
                f"{q['level'][0]})")
    c = np.mean(p["clear"]) if p["clear"] else None
    s = np.mean(p["sem"]) if p["sem"] else None
    print(f"  pair {k[2]:3d}: BASE {side('first')}\n            GEOAE {side('second')}\n"
          f"            geoae clearer {c}, more semantic {s}")
