"""Head-to-head LLM judging of two codebooks: which cluster is the clearer concept?

Absolute ratings sit near the ceiling (semantic coherence >= 4 for 92-98% of the text in every
space) and neighbour-intruder tests punish good geometry, because the intruder comes from each
codebook's own nearest centroids. This tool asks a RELATIVE question instead. Each call shows the
judge two clusters, A and B, drawn at random (by usage) from two spaces, 8 passages each, and asks:

  clearer_concept   which cluster's marked tokens share a clearer, more specific single concept
  more_semantic     which groups tokens by meaning rather than by the literal word or its grammar
  per cluster       a label (<= 10 words), coherence 1-10, abstraction 1-10, level (llm_judge's 8)

Every pair is asked in BOTH orders (A/B and B/A) to cancel position bias; each order gives the
second space 1 (win), 0 (loss) or 0.5 (tie / neither), and a pair scores the mean of its usable
orders. Win rate = mean pair score, with a basic bootstrap interval over pairs.

Control: each space's real clusters against FAKE clusters of the same size whose passages come from
8 different random clusters of that space. A judge that does not prefer the real cluster almost
always is not trustworthy here, and the run says so.

    python -m geoae.analysis.pairwise_judge \\
        --space base=results/delta/base_balanced_kmeans_k2000_dpc_sampled/interp/closest_tokens.json \\
        --space token_removed=results/delta/base_balanced_kmeans_k2000_dpc_sampled_tokmean/interp/closest_tokens.json \\
        --space geoae=results/delta/d6144/interp/closest_tokens.json \\
        --provider openrouter --model google/gemini-2.5-flash-lite --pairs 500 \\
        --out results/delta/d6144/interp/pairwise_flash_lite.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from geoae.analysis.llm_judge import CONCEPT_LEVELS
from geoae.analysis.llm_judge import LLM
from geoae.analysis.llm_judge import parse_json

CHOICES = ("A", "B", "tie", "neither")


# ------------------------------------------------------------------- clusters
def load_space(path: str, min_assigned: int, n_examples: int) -> dict:
    """{cluster id: (n_assigned, [passages])} for clusters with >= n_examples distinct passages."""
    clusters = json.loads(Path(path).read_text())["clusters"]
    out = {}
    for k, c in clusters.items():
        if c.get("n_assigned", 0) < min_assigned:
            continue
        ex, seen = [], set()
        for e in c.get("top", []) + c.get("spectrum", []):
            key = (e.get("token"), e.get("context"))
            if key in seen or not e.get("context"):
                continue
            seen.add(key)
            ex.append(e["context"].replace("\\n", " ").replace("\n", " ").strip())
            if len(ex) == n_examples:
                break
        if len(ex) == n_examples:
            out[int(k)] = (int(c["n_assigned"]), ex)
    return out


def sample_clusters(space: dict, n: int, rng: random.Random, weight: str) -> list[int]:
    ids = sorted(space)
    w = [space[i][0] for i in ids] if weight == "usage" else None
    return rng.choices(ids, weights=w, k=n)


def fake_cluster(space: dict, n_examples: int, rng: random.Random) -> list[str]:
    """Passages from n_examples DIFFERENT random clusters: no shared concept by construction."""
    picks = rng.sample(sorted(space), n_examples)
    return [rng.choice(space[k][1]) for k in picks]


# -------------------------------------------------------------------- prompt
def _block(ex: list[str]) -> str:
    return "\n".join(f"{i + 1:>3}. {e}" for i, e in enumerate(ex))


RUBRIC = f"""For EACH cluster give:
  "label": what its marked tokens share, at most 10 words
  "coherence": 1-10. 1-2 no common thread; 3-4 a vague or very broad commonality; 5-6 a clear but
      broad shared property; 7-8 one clear concept with a few exceptions; 9-10 one precise concept
      shared by every marked token
  "abstraction": 1-10, the kind of thing they share. 1-2 the literal string, spelling or
      punctuation; 3-4 word form or grammatical role (affixes, part of speech, function words);
      5-6 a word meaning or an entity type; 7-8 a topic or domain; 9-10 what the text is doing
      (discourse, stance, reasoning, rhetorical move)
  "level": one of {", ".join(CONCEPT_LEVELS)}

Then compare them:
  "clearer_concept": which cluster's marked tokens share a clearer, more specific single concept?
  "more_semantic": which cluster groups tokens by MEANING rather than by the literal word or its grammar?
Answer each with "A", "B", "tie" (equally) or "neither" (neither has one)."""


def pair_prompt(ex_a: list[str], ex_b: list[str]) -> str:
    return f"""Below are two clusters, A and B. Each was formed by grouping a language model's internal
representations. In every snippet ONE token is marked with «guillemets»; judge the MARKED token in its
context, not the surrounding text on its own.

Cluster A:
{_block(ex_a)}

Cluster B:
{_block(ex_b)}

{RUBRIC}

Answer as JSON only, no prose outside it:
{{"A": {{"label": "...", "coherence": <1-10>, "abstraction": <1-10>, "level": "..."}},
 "B": {{"label": "...", "coherence": <1-10>, "abstraction": <1-10>, "level": "..."}},
 "clearer_concept": "A|B|tie|neither", "more_semantic": "A|B|tie|neither"}}"""


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if 1 <= v <= 10 else None


def parse_answer(text: str) -> dict | None:
    d = parse_json(text)
    if not isinstance(d, dict):
        return None
    out = {}
    for q in ("clearer_concept", "more_semantic"):
        v = d.get(q)
        out[q] = v if v in CHOICES else None
    for side in ("A", "B"):
        s = d.get(side) if isinstance(d.get(side), dict) else {}
        lvl = str(s.get("level", "")).strip().lower()
        out[side] = {"label": s.get("label"), "coherence": _num(s.get("coherence")),
                     "abstraction": _num(s.get("abstraction")),
                     "level": lvl if lvl in CONCEPT_LEVELS else None}
    return out


def score_for_second(answer: str | None, second_is_a: bool) -> float | None:
    """The second space's score from one call: 1 win, 0 loss, 0.5 tie/neither, None unusable."""
    if answer is None:
        return None
    if answer in ("tie", "neither"):
        return 0.5
    return float((answer == "A") == second_is_a)


# ----------------------------------------------------------------- summaries
def basic_ci(est: float, boots) -> tuple[float, float]:
    b = np.asarray([x for x in boots if np.isfinite(x)])
    if len(b) == 0:
        return float("nan"), float("nan")
    d = b - b.mean()
    return float(est + np.quantile(d, 0.025)), float(est + np.quantile(d, 0.975))


def win_rate(pair_scores: list[float], n_boot: int, seed: int) -> dict:
    s = np.asarray([x for x in pair_scores if x is not None], float)
    if len(s) == 0:
        return {"n": 0, "win": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    boots = [s[rng.integers(0, len(s), len(s))].mean() for _ in range(n_boot)]
    lo, hi = basic_ci(float(s.mean()), boots)
    return {"n": int(len(s)), "win": float(s.mean()), "lo": lo, "hi": hi}


def summarise(records: list[dict], spaces: list[str], n_boot: int, seed: int) -> dict:
    out = {"pairings": {}, "controls": {}, "per_space": {}, "position": {}}
    # pair scores, keyed by (kind, first, second, pair id)
    by_pair: dict = {}
    first_slot, consistent, both = [], 0, 0
    for r in records:
        key = (r["kind"], r["first"], r["second"], r["pair"])
        by_pair.setdefault(key, {})[r["order"]] = r
    for (kind, x, y, _), orders in by_pair.items():
        for q in ("clearer_concept", "more_semantic"):
            vals = [score_for_second((o["answer"] or {}).get(q), o["second_is_a"]) for o in orders.values()]
            vals = [v for v in vals if v is not None]
            tgt = out["pairings"] if kind == "main" else out["controls"]
            tgt.setdefault(f"{x} vs {y}", {}).setdefault(q, []).append(float(np.mean(vals)) if vals else None)
        if kind == "main" and len(orders) == 2:
            a = [(o["answer"] or {}).get("clearer_concept") for o in orders.values()]
            if all(v in ("A", "B") for v in a):
                both += 1
                consistent += a[0] != a[1]          # the same CLUSTER won in both orders
        for o in orders.values():
            v = (o["answer"] or {}).get("clearer_concept")
            if kind == "main" and v in ("A", "B"):
                first_slot.append(v == "A")
    for tgt in (out["pairings"], out["controls"]):
        for name, qs in tgt.items():
            for q, vals in qs.items():
                qs[q] = win_rate(vals, n_boot, seed)
    out["position"] = {"first_slot_win": float(np.mean(first_slot)) if first_slot else None,
                       "order_consistent": consistent / both if both else None, "n_decisive_pairs": both}
    for s in spaces:
        coh, ab, lv = [], [], {}
        for r in records:
            if r["kind"] != "main" or not r["answer"]:
                continue
            for side, sp in (("A", r["space_a"]), ("B", r["space_b"])):
                if sp != s:
                    continue
                a = r["answer"][side]
                if a["coherence"] is not None:
                    coh.append(a["coherence"])
                if a["abstraction"] is not None:
                    ab.append(a["abstraction"])
                lv[a["level"] or "unparsed"] = lv.get(a["level"] or "unparsed", 0) + 1
        n = sum(lv.values()) or 1
        out["per_space"][s] = {"coherence_mean": float(np.mean(coh)) if coh else None,
                               "abstraction_mean": float(np.mean(ab)) if ab else None,
                               "abstraction_hist": {str(k): int(sum(1 for v in ab if round(v) == k)) for k in range(1, 11)},
                               "level_share": {k: v / n for k, v in sorted(lv.items(), key=lambda t: -t[1])}}
    return out


def print_summary(summ: dict) -> None:
    def f(w):
        return "  n/a" if w["win"] is None else f"{w['win']:.3f} [{w['lo']:.3f}, {w['hi']:.3f}]  (n {w['n']})"
    print("\n[pair] CONTROL: real cluster (2nd) vs fake cluster of random passages (1st) — should be ~1.0")
    for name, qs in summ["controls"].items():
        print(f"  {name:34s} clearer {f(qs['clearer_concept'])}")
    p = summ["position"]
    print(f"\n[pair] position: first slot wins {p['first_slot_win']:.3f} of decisive answers; the same cluster "
          f"wins in both orders for {p['order_consistent']:.3f} of {p['n_decisive_pairs']} decisive pairs"
          if p["first_slot_win"] is not None else "\n[pair] position: no decisive answers")
    print("\n[pair] win rate of the SECOND space (0.5 = equal)")
    for name, qs in summ["pairings"].items():
        print(f"  {name:34s} clearer concept {f(qs['clearer_concept'])}")
        print(f"  {'':34s} more semantic   {f(qs['more_semantic'])}")
    print("\n[pair] per space (every appearance in a main pair)")
    for s, d in summ["per_space"].items():
        top = ", ".join(f"{k} {v:.2f}" for k, v in list(d["level_share"].items())[:4])
        coh = "n/a" if d["coherence_mean"] is None else f"{d['coherence_mean']:.2f}"
        ab = "n/a" if d["abstraction_mean"] is None else f"{d['abstraction_mean']:.2f}"
        print(f"  {s:16s} coherence {coh}/10  abstraction {ab}/10  levels: {top}")


# ---------------------------------------------------------------------- main
def build_jobs(spaces: dict, args) -> list[dict]:
    rng = random.Random(args.seed)
    names = list(spaces)
    jobs = []
    for x, y in itertools.combinations(names, 2):
        cx = sample_clusters(spaces[x], args.pairs, rng, args.weight)
        cy = sample_clusters(spaces[y], args.pairs, rng, args.weight)
        for i, (a, b) in enumerate(zip(cx, cy)):
            jobs.append({"kind": "main", "first": x, "second": y, "pair": i,
                         "cluster_first": a, "cluster_second": b,
                         "ex_first": spaces[x][a][1], "ex_second": spaces[y][b][1]})
    for s in names:
        cs = sample_clusters(spaces[s], args.control_pairs, rng, args.weight)
        for i, c in enumerate(cs):
            jobs.append({"kind": "control", "first": f"fake {s}", "second": s, "pair": i,
                         "cluster_first": None, "cluster_second": c,
                         "ex_first": fake_cluster(spaces[s], args.n_examples, rng), "ex_second": spaces[s][c][1]})
    calls = []
    for j in jobs:
        for order in (0, 1):                        # 0: first space is A; 1: second space is A
            second_is_a = order == 1
            ex_a, ex_b = (j["ex_second"], j["ex_first"]) if second_is_a else (j["ex_first"], j["ex_second"])
            calls.append({**j, "order": order, "second_is_a": second_is_a,
                          "space_a": j["second"] if second_is_a else j["first"],
                          "space_b": j["first"] if second_is_a else j["second"],
                          "ex_a": ex_a, "ex_b": ex_b, "prompt": pair_prompt(ex_a, ex_b)})
    return calls


# ------------------------------------------------- offline judging (agents)
AGENT_HEADER = f"""Each ITEM below shows two clusters, A and B. Each cluster was formed by grouping a
language model's internal representations. In every snippet ONE token is marked with «guillemets»;
judge the MARKED token in its context, not the surrounding text on its own. Judge every item on its
own: the items are unrelated, in random order, and A and B carry no meaning across items.

{RUBRIC}

Write ONE JSON object per item, one per line (JSON Lines), nothing else:
{{"id": "<item id>", "A": {{"label": "...", "coherence": <1-10>, "abstraction": <1-10>, "level": "..."}}, "B": {{"label": "...", "coherence": <1-10>, "abstraction": <1-10>, "level": "..."}}, "clearer_concept": "A|B|tie|neither", "more_semantic": "A|B|tie|neither"}}
"""

KEY_DROP = ("prompt", "ex_first", "ex_second", "ex_a", "ex_b")


def export_batches(calls: list[dict], out_dir: Path, key_path: Path, batch_size: int, seed: int) -> None:
    """Blind batch files for agent judges + the answer key (kept apart: it names the spaces).
    Order-0 and order-1 calls go to different batches, so no judge sees both orders of a pair."""
    rng = random.Random(seed)
    for i, c in enumerate(calls):
        c["id"] = f"q{i:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = []
    for order in (0, 1):
        part = [c for c in calls if c["order"] == order]
        rng.shuffle(part)
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    for b, part in enumerate(batches):
        items = "\n".join(f"=== ITEM {c['id']} ===\nCluster A:\n{_block(c['ex_a'])}\n\nCluster B:\n"
                          f"{_block(c['ex_b'])}\n" for c in part)
        (out_dir / f"batch_{b:02d}.md").write_text(f"{AGENT_HEADER}\n{items}")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps([{k: v for k, v in c.items() if k not in KEY_DROP} for c in calls]))
    print(f"[pair] exported {len(calls)} items in {len(batches)} batches -> {out_dir}; key -> {key_path}")


def import_answers(key_path: Path, answers_dir: Path) -> list[dict]:
    """Records (as from the API run) from the key and the agents' answers_*.jsonl files."""
    got = {}
    for f in sorted(answers_dir.glob("answers_*.jsonl")):
        for line in f.read_text().splitlines():
            a = parse_answer(line)
            d = parse_json(line)
            if a is not None and isinstance(d, dict) and d.get("id"):
                got[d["id"]] = a
    key = json.loads(key_path.read_text())
    print(f"[pair] answers for {sum(c['id'] in got for c in key)}/{len(key)} items from {answers_dir}")
    return [c | {"answer": got.get(c["id"])} for c in key]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", action="append", default=[],
                    help="name=closest_tokens.json (repeat, 2+). Every pairing of the spaces is judged; "
                         "in 'x vs y' the win rate is y's, so list the base first")
    ap.add_argument("--pairs", type=int, default=500, help="pairs per pairing of spaces")
    ap.add_argument("--control_pairs", type=int, default=50, help="real-vs-fake pairs per space")
    ap.add_argument("--n_examples", type=int, default=8)
    ap.add_argument("--min_assigned", type=int, default=50)
    ap.add_argument("--weight", default="usage", choices=["usage", "uniform"])
    ap.add_argument("--provider", default="openrouter", choices=["openai", "anthropic", "openrouter"])
    ap.add_argument("--model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--cache_dir", default="cache/llm_judge")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry_run", action="store_true", help="print one prompt and the call count, no key needed")
    ap.add_argument("--export", default=None, help="write blind batch files for agent judges here, no API calls")
    ap.add_argument("--key", default=None, help="--export / --import_dir: the answer key (keep it away from judges)")
    ap.add_argument("--batch_size", type=int, default=75)
    ap.add_argument("--import_dir", default=None, help="score agents' answers_*.jsonl against --key")
    args = ap.parse_args(argv)

    if args.import_dir:
        records = import_answers(Path(args.key), Path(args.import_dir))
        names = list(dict.fromkeys(r["second"] for r in records if r["kind"] == "main")
                     | dict.fromkeys(r["first"] for r in records if r["kind"] == "main"))
        summ = summarise(records, names, args.n_boot, args.seed)
        print_summary(summ)
        if args.out:
            Path(args.out).write_text(json.dumps({"meta": {"judge": "agents", "key": args.key},
                                                  "summary": summ, "records": records}, indent=1, ensure_ascii=False))
            print(f"[pair] wrote {args.out}")
        return

    spaces = {}
    for s in args.space:
        name, _, path = s.partition("=")
        if not path:
            ap.error(f"--space {s!r}: expected name=path")
        spaces[name] = load_space(path, args.min_assigned, args.n_examples)
        print(f"[pair] {name:16s} {len(spaces[name]):5d} clusters with >= {args.min_assigned} tokens <- {path}")
    if len(spaces) < 2:
        ap.error("need at least two --space")

    calls = build_jobs(spaces, args)
    print(f"[pair] {len(calls)} calls ({args.pairs} pairs x {len(spaces) * (len(spaces) - 1) // 2} pairings "
          f"+ {args.control_pairs} controls x {len(spaces)} spaces, both orders) - judge {args.model}")
    if args.dry_run:
        print(calls[0]["prompt"])
        return
    if args.export:
        export_batches(calls, Path(args.export), Path(args.key), args.batch_size, args.seed)
        return

    llm = LLM(args.provider, args.model, Path(args.cache_dir))
    done = [0]
    t0 = time.time()

    def run(c):
        ans = parse_answer(llm.ask(c["prompt"], max_tokens=1500))
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"[pair] {done[0]}/{len(calls)} calls ({time.time() - t0:.0f}s, api {llm.n_calls}, "
                  f"cached {llm.n_cached})", flush=True)
        return ans

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        answers = list(ex.map(run, calls))
    records = [{k: v for k, v in c.items() if k not in KEY_DROP} | {"answer": a}
               for c, a in zip(calls, answers)]
    n_bad = sum(a is None for a in answers)
    print(f"[pair] api calls {llm.n_calls}, cache hits {llm.n_cached}, unparsed answers {n_bad}")

    summ = summarise(records, list(spaces), args.n_boot, args.seed)
    print_summary(summ)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"meta": {"spaces": dict(s.partition("=")[::2] for s in args.space),
                                            "model": args.model, "provider": args.provider,
                                            **{k: getattr(args, k) for k in ("pairs", "control_pairs", "n_examples",
                                                                             "min_assigned", "weight", "seed")}},
                                   "summary": summ, "records": records}, indent=1, ensure_ascii=False))
        print(f"[pair] wrote {out}")


if __name__ == "__main__":
    main()
