"""
LLM-as-judge auto-interpretability scoring for GeoAE clusters.

Replaces the `monosemanticity` field written by closest_tokens.py, which is
`1 - normalised token entropy` over the closest-token strings and therefore
measures TOKEN-STRING REPETITION, not semantic coherence. Cluster 247 of the
b32k run (" Lov  Tor  Cart  Ber  Ken  Mag" — a clean name-fragment cluster)
scores 0.011 under that metric.

PROTOCOL (Bills et al. 2023 explain-then-simulate, with the detection-scoring
refinement from EleutherAI's sae-auto-interp / Delphi). Assignments here are
HARD (nearest centroid), not graded activations, so detection is the right
scorer — there is no activation magnitude to simulate.

  1. EXPLAIN   judge sees N contexts from the cluster core plus its distance
               spectrum, and writes an explanation + structured labels.
  2. DETECT    judge sees held-out positives shuffled with HARD NEGATIVES drawn
               from the nearest OTHER centroids, and labels each. Score is
               balanced accuracy.
  3. NULL      the same detection run against a DELIBERATELY MISMATCHED
               explanation from another cluster. This is the empirical floor.

THREE CONTROLS, none of them optional:

  * HARD NEGATIVES, not random ones. Random tokens make detection trivial and
    every space scores ~1.0. Negatives come from the nearest competing
    centroids, so the task is "is this boundary real", which is exactly what
    separates a feature from a grab-bag.
  * HELD-OUT POSITIVES. The explainer sees examples [0:n_explain]; the scorer
    only ever sees examples after that. Otherwise the judge is graded on
    memorisation.
  * A NULL FLOOR. Report (score - null) as well as score. An auto-interp number
    with no floor is uninterpretable, and this project has twice been misled by
    metrics quoted without one (see the logged-silhouette and `dying` cases).

AND THE COMPARISON MATTERS MORE THAN THE ABSOLUTE NUMBER. Run the identical
protocol over the encoder-free baselines and read the DIFFERENCE:

    python -m geoae.interp.closest_tokens --baseline_kmeans <balanced.npz> --out results/ct_balanced.json
    python -m geoae.interp.llm_judge results/closest_tokens_b32k....json results/ct_balanced.json

The standing result in this project is that Sinkhorn balancing, not the encoder,
carries the win; a judge score for the AE alone cannot test that.

USAGE

    # validate prompts and cost with no API key at all
    python -m geoae.interp.llm_judge results/closest_tokens_b32k_lam1_d6144.json \
        --checkpoint checkpoints/.../step_0014200.pt --n_clusters 40 --dry_run

    # then, with OPENAI_API_KEY or ANTHROPIC_API_KEY exported
    python -m geoae.interp.llm_judge results/closest_tokens_b32k_lam1_d6144.json \
        --checkpoint checkpoints/.../step_0014200.pt --n_clusters 200 \
        --provider openai --model gpt-4o-mini --concurrency 8

Responses are cached on disk by hash(provider, model, prompt), so re-runs and
added clusters never re-pay for work already done.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CONCEPT_LEVELS = [
    "surface",      # literal token string / casing / whitespace / punctuation
    "morphology",   # affixes, inflection, subword fragments
    "syntax",       # part of speech, grammatical role
    "entity",       # named entities and their fragments
    "semantics",    # word meaning / lexical field
    "topic",        # document subject matter
    "discourse",    # register, stance, rhetorical function
    "mixed",        # no single concept covers the examples
]


# ----------------------------------------------------------------- providers
class LLM:
    """Thin adapter over OpenAI / Anthropic chat completions."""

    def __init__(self, provider: str, model: str, cache_dir: Path, dry_run: bool = False):
        self.provider, self.model, self.dry_run = provider, model, dry_run
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_calls = self.n_cached = 0
        self._client = None
        if not dry_run:
            self._client = self._make_client()

    def _make_client(self):
        if self.provider in ("openai", "openrouter"):
            # OpenRouter speaks the OpenAI wire protocol; it only needs a base_url
            # and its model ids carry a vendor prefix ("openai/gpt-4o-mini").
            env = "OPENROUTER_API_KEY" if self.provider == "openrouter" else "OPENAI_API_KEY"
            key = os.environ.get(env)
            if not key:
                sys.exit(f"{env} is not set. Export it, or pass --dry_run.")
            try:
                from openai import OpenAI
            except ImportError:
                sys.exit("pip install openai  (or: uv pip install openai)")
            if self.provider == "openrouter":
                return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
            return OpenAI(api_key=key)
        if self.provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                sys.exit("ANTHROPIC_API_KEY is not set. Export it, or pass --dry_run.")
            try:
                import anthropic
            except ImportError:
                sys.exit("pip install anthropic  (or: uv pip install anthropic)")
            return anthropic.Anthropic(api_key=key)
        sys.exit(f"unknown provider {self.provider!r}")

    def _cache_path(self, prompt: str) -> Path:
        h = hashlib.sha256(f"{self.provider}\0{self.model}\0{prompt}".encode()).hexdigest()[:32]
        return self.cache_dir / f"{h}.json"

    def ask(self, prompt: str, max_tokens: int = 900) -> str:
        cp = self._cache_path(prompt)
        if cp.exists():
            self.n_cached += 1
            return json.loads(cp.read_text())["response"]
        if self.dry_run:
            return ""
        last = None
        for attempt in range(5):
            try:
                if self.provider in ("openai", "openrouter"):
                    r = self._client.chat.completions.create(
                        model=self.model, max_completion_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    out = r.choices[0].message.content or ""
                else:
                    r = self._client.messages.create(
                        model=self.model, max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    out = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                cp.write_text(json.dumps({"prompt": prompt, "response": out}))
                self.n_calls += 1
                return out
            except Exception as e:                       # noqa: BLE001 - surface after retries
                last = e
                time.sleep(min(2 ** attempt, 20))
        print(f"  [judge] giving up after retries: {last}", file=sys.stderr)
        return ""


# ------------------------------------------------------------------- prompts
def _fmt(examples) -> str:
    return "\n".join(
        f"{i+1:>3}. [{e.get('domain','?')}] {e['context'].strip()}"
        for i, e in enumerate(examples)
    )


def explain_prompt(examples) -> str:
    return f"""Below are text snippets. In each, ONE token is marked with «guillemets».
All of these marked tokens were grouped together by a clustering of a language
model's internal representations.

{_fmt(examples)}

Identify what the MARKED TOKENS have in common. Judge the marked token in its
context — not the surrounding text on its own.

Answer as JSON only, no prose outside it:
{{
  "explanation": "<one sentence, max 25 words, describing what the marked tokens share>",
  "level": "<one of: {', '.join(CONCEPT_LEVELS)}>",
  "monosemanticity": <integer 1-5>,
  "surface_coherence": <integer 1-5>,
  "semantic_coherence": <integer 1-5>,
  "generality": <integer 1-5>,
  "coverage": <fraction 0.0-1.0 of the {len(examples)} examples your explanation covers>,
  "outliers": [<1-based indices that do not fit>],
  "confidence": <integer 1-5>
}}

"monosemanticity" — overall, does ONE concept explain the examples?
  5 = one crisp concept explains every example      4 = one concept, 1-2 outliers
  3 = two related concepts                          2 = several loosely related
  1 = no discernible common concept (a grab-bag)

"surface_coherence" — are the MARKED STRINGS THEMSELVES the same or near-identical?
  5 = nearly all the same string (e.g. every one is " the")
  1 = all different strings

"semantic_coherence" — RATE THIS INDEPENDENTLY OF surface_coherence. Ignoring the
spelling of the tokens, do they play the same role, carry related meaning, or occur
in the same kind of context? THIS IS THE KEY FIELD, so read the rule carefully:

  * If the tokens are mostly the SAME string, ask whether they share anything
    BEYOND being that string. A cluster of the identical token used in unrelated
    contexts is surface_coherence 5 and semantic_coherence 1-2.
  * If the tokens are ALL DIFFERENT strings but share a role or meaning, that is
    HIGH semantic coherence (4-5) even though surface_coherence is 1.
  * Repetition of one string is NEVER by itself a reason to score semantic
    coherence above 2.

"generality" — how BROAD is the concept? Coherence and generality are different
axes and a cluster can be high on one and low on the other. A narrow cluster is
TRIVIALLY easy to be coherent about, so do not let a high semantic_coherence pull
this up.
  5 = a concept that recurs across many contexts, domains and phrasings
      (e.g. "past-tense verbs", "names of countries", "units of measurement")
  3 = a concept tied to one domain or register but productive within it
      (e.g. "LaTeX math delimiters", "Python attribute names")
  1 = one word in one construction, or a single fixed string
      (e.g. "'the' when it precedes the name of a band", "the last digit of a
      year in a date", "the token #include")

Use "level" = "mixed" ONLY when monosemanticity is 1 or 2."""


def detect_prompt(explanation: str, items) -> str:
    return f"""A cluster of language-model tokens has been described as:

    "{explanation}"

Below are numbered snippets, each with ONE token marked with «guillemets». Some
marked tokens belong to that cluster; the others are from DIFFERENT but nearby
clusters, so the distinction may be subtle.

{_fmt(items)}

Decide for each whether the marked token belongs to the described cluster.
Answer as JSON only:
{{"belongs": [<1-based indices of snippets that belong>]}}"""


def intruder_prompt(members, intruder, pos: int) -> str:
    items = list(members)
    items.insert(pos, intruder)
    return f"""All but ONE of the snippets below contain a marked token drawn from the
same cluster of language-model representations. Exactly one is an INTRUDER taken
from a different cluster.

{_fmt(items)}

Which is the intruder? Judge the marked «token» in its context, not the
surrounding text. Answer as JSON only:
{{"intruder": <the 1-based index of the intruder>}}"""


# --------------------------------------------------------------------- utils
def parse_json(text: str):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def load_centroids(src: str):
    """Centroids from either an AE checkpoint (.pt) or a k-means baseline (.npz)."""
    import torch
    if str(src).endswith(".npz"):
        import numpy as np
        return torch.from_numpy(np.load(str(src))["centroids"]).float()
    from geoae.checkpoint import load_ae_checkpoint
    ae, _, _, _ = load_ae_checkpoint(str(src), "cpu")
    return ae.centroids.detach().float()


def neighbours_from_checkpoint(ckpt_path: str, n_neighbours: int):
    """(K, n) array of nearest OTHER centroid ids, for hard negatives."""
    import torch
    C = load_centroids(ckpt_path)
    out = []
    for i in range(0, len(C), 256):
        d = torch.cdist(C[i:i + 256], C)
        for r, row in enumerate(d):
            row[i + r] = float("inf")
        out.append(row_topk := d.topk(n_neighbours, largest=False).indices)
    return torch.cat(out).tolist()


# ---------------------------------------------------------------------- main
def judge_run(path: Path, args, llm: LLM, rng: random.Random):
    data = json.loads(path.read_text())
    clusters = data["clusters"]
    live = sorted(clusters.keys(), key=lambda k: -clusters[k]["n_assigned"])
    live = [k for k in live if len(clusters[k]["top"]) >= args.n_explain + args.n_pos]
    if args.min_assigned:
        live = [k for k in live if clusters[k]["n_assigned"] >= args.min_assigned]

    if args.sample == "top":
        chosen = live[: args.n_clusters]
    elif args.sample == "random":
        chosen = rng.sample(live, min(args.n_clusters, len(live)))
    else:                                   # stratified across the usage range
        step = max(1, len(live) // max(args.n_clusters, 1))
        chosen = live[::step][: args.n_clusters]

    # Each run needs ITS OWN centroids for hard negatives — comparing an AE scored
    # against AE-neighbour negatives with a baseline scored against random ones
    # would be meaningless. Resolve from the file's own meta; --checkpoint overrides.
    meta = data.get("meta", {})
    src = args.checkpoint or meta.get("checkpoint") or meta.get("baseline_kmeans")
    nbrs = None
    if src and Path(src).exists():
        print(f"[judge] hard negatives from centroids of {Path(src).name}")
        nbrs = neighbours_from_checkpoint(src, args.n_neighbours)
    else:
        print(f"[judge] WARNING: no centroid source for {path.name} (looked for {src!r}). "
              "Negatives will be RANDOM clusters, inflating detection scores and making "
              "this run NOT comparable to a hard-negative run.")

    def one(cid: str):
        # Per-cluster RNG seeded from (seed, cluster id). The shared `rng` is
        # consumed by several worker threads at once, so item order and null
        # pairing would otherwise depend on thread interleaving — which makes
        # runs unreproducible and a model-vs-model comparison unfair, since the
        # two models would see different shuffles of the same items.
        crng = random.Random(f"{args.seed}:{cid}")
        c = clusters[cid]
        pool = c["top"]
        # spectrum[0] IS top[0] by construction, and `top` itself carries repeats,
        # so dedupe on (token, context) before showing anything to the explainer.
        expl_ex, seen = [], set()
        for e in pool[: args.n_explain] + c.get("spectrum", []):
            k = (e.get("token"), e.get("context"))
            if k not in seen:
                seen.add(k)
                expl_ex.append(e)
            if len(expl_ex) >= args.n_explain + 2:
                break
        pos = pool[args.n_explain: args.n_explain + args.n_pos]

        # hard negatives from the nearest competing centroids
        if nbrs is not None:
            cand = [str(j) for j in nbrs[int(cid)] if str(j) in clusters]
        else:
            cand = [k for k in clusters if k != cid]
            crng.shuffle(cand)
        negs = []
        for nb in cand:
            for e in clusters[nb]["top"][:3]:
                negs.append(e)
                if len(negs) >= args.n_neg:
                    break
            if len(negs) >= args.n_neg:
                break

        e_raw = llm.ask(explain_prompt(expl_ex))
        e = parse_json(e_raw) or {}
        explanation = e.get("explanation", "")

        rec = {
            "cluster": int(cid), "n_assigned": c["n_assigned"], "usage_pct": c["usage_pct"],
            "mono_entropy_old": c.get("monosemanticity"),
            "explanation": explanation, "level": e.get("level"),
            "mono_llm": e.get("monosemanticity"),
            "surface_coh": e.get("surface_coherence"),
            "semantic_coh": e.get("semantic_coherence"),
            "generality": e.get("generality"),
            "coverage": e.get("coverage"),
            "confidence": e.get("confidence"), "outliers": e.get("outliers"),
        }
        if not explanation or not pos or not negs:
            rec["detect_acc"] = rec["null_acc"] = None
            return rec

        items = pos + negs
        order = list(range(len(items)))
        crng.shuffle(order)
        shown = [items[i] for i in order]
        # index in `shown` -> True if it came from the positive half
        is_pos = [order[i] < len(pos) for i in range(len(shown))]

        def score(expl):
            d = parse_json(llm.ask(detect_prompt(expl, shown))) or {}
            picked = {i - 1 for i in d.get("belongs", []) if isinstance(i, int)}
            tpr = sum(1 for i, p in enumerate(is_pos) if p and i in picked) / max(sum(is_pos), 1)
            tnr = sum(1 for i, p in enumerate(is_pos) if not p and i not in picked) / max(len(is_pos) - sum(is_pos), 1)
            # n_pred detects DEGENERATE answering: a judge that says "all belong"
            # or "none belong" scores exactly 0.5 balanced accuracy and looks like
            # an honest coin-flip. Without this you cannot tell the two apart.
            return 0.5 * (tpr + tnr), len(picked), tpr, tnr

        # WORD INTRUSION — behavioural coherence, independent of the explanation
        # and with a clean 1/(n+1) chance floor. This is the primary semantic
        # signal: it asks "do these belong together", never "is your sentence good".
        hits = 0
        for trial in range(args.n_intruder_trials):
            mem = pos[trial % max(len(pos) - args.intruder_k + 1, 1):][: args.intruder_k]
            if len(mem) < args.intruder_k or not negs:
                break
            intr_ex = negs[trial % len(negs)]
            slot = crng.randrange(args.intruder_k + 1)
            d = parse_json(llm.ask(intruder_prompt(mem, intr_ex, slot))) or {}
            got = d.get("intruder")
            if isinstance(got, int) and got - 1 == slot:
                hits += 1
        rec["intruder_trials"] = args.n_intruder_trials
        rec["intruder_acc"] = hits / args.n_intruder_trials if args.n_intruder_trials else None
        rec["intruder_chance"] = 1.0 / (args.intruder_k + 1)

        rec["detect_acc"], rec["n_pred"], rec["tpr"], rec["tnr"] = score(explanation)
        rec["n_items"] = len(shown)
        rec["n_pos_items"] = sum(is_pos)
        # NULL FLOOR: another cluster's explanation on these same items
        other = crng.choice([k for k in chosen if k != cid]) if len(chosen) > 1 else None
        rec["null_acc"] = None
        if other and args.null_control:
            o_raw = llm.ask(explain_prompt(clusters[other]["top"][: args.n_explain]))
            o = parse_json(o_raw) or {}
            if o.get("explanation"):
                rec["null_acc"], rec["null_n_pred"], _, _ = score(o["explanation"])
        return rec

    if args.dry_run:
        cid = chosen[0]
        c = clusters[cid]
        print("\n" + "=" * 78)
        print(f"DRY RUN — prompts for cluster {cid} (n={c['n_assigned']})")
        print("=" * 78)
        _ex, _seen = [], set()
        for e in c["top"][: args.n_explain] + c.get("spectrum", []):
            k = (e.get("token"), e.get("context"))
            if k not in _seen:
                _seen.add(k); _ex.append(e)
            if len(_ex) >= args.n_explain + 2:
                break
        print(explain_prompt(_ex))
        print("\n" + "-" * 78 + "\nDETECTION PROMPT (with a placeholder explanation)\n" + "-" * 78)
        demo_neg = clusters[chosen[1]]["top"][: args.n_neg] if len(chosen) > 1 else []
        print(detect_prompt("<explanation from stage 1>",
                            c["top"][args.n_explain: args.n_explain + args.n_pos] + demo_neg))
        n_calls = len(chosen) * (2 + (1 if args.null_control else 0)) + (len(chosen) if args.null_control else 0)
        print("\n" + "=" * 78)
        print(f"would judge {len(chosen)} clusters  ->  ~{n_calls} API calls "
              f"({'with' if args.null_control else 'without'} null control)")
        print("=" * 78)
        return None

    print(f"[judge] {path.name}: {len(chosen)} clusters, provider={args.provider} model={args.model}")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        recs = list(ex.map(one, chosen))
    return {"source": str(path), "provider": args.provider, "model": args.model,
            "n_clusters": len(recs), "clusters": recs}


def _wmean(recs, field):
    """Token-weighted mean of `field`, weighting each cluster by n_assigned."""
    v = [(r[field], r.get("n_assigned", 0)) for r in recs
         if isinstance(r.get(field), (int, float)) and r.get("n_assigned")]
    w = sum(n for _, n in v)
    return (sum(x * n for x, n in v) / w) if w else None


def summarise(res):
    recs = [r for r in res["clusters"] if r.get("mono_llm")]
    if not recs:
        print("  (no parsable results)")
        return
    import statistics as st
    mono = [r["mono_llm"] for r in recs]
    sem = [r["semantic_coh"] for r in recs if isinstance(r.get("semantic_coh"), (int, float))]
    gen = [r["generality"] for r in recs if isinstance(r.get("generality"), (int, float))]
    both = [(r["semantic_coh"], r["generality"]) for r in recs
            if isinstance(r.get("semantic_coh"), (int, float)) and isinstance(r.get("generality"), (int, float))]
    sur = [r["surface_coh"] for r in recs if isinstance(r.get("surface_coh"), (int, float))]
    intr = [r["intruder_acc"] for r in recs if isinstance(r.get("intruder_acc"), (int, float))]
    det = [r["detect_acc"] for r in recs if r.get("detect_acc") is not None]
    null = [r["null_acc"] for r in recs if r.get("null_acc") is not None]
    print(f"\n  clusters scored      : {len(recs)}")
    print(f"  monosemanticity 1-5  : mean {st.mean(mono):.2f}  median {st.median(mono):.1f}")
    if sem:
        se = st.stdev(sem) / len(sem) ** 0.5 if len(sem) > 1 else 0.0
        print(f"  SEMANTIC coherence   : mean {st.mean(sem):.2f} ± {se:.3f}   <- primary")
    if gen:
        se = st.stdev(gen) / len(gen) ** 0.5 if len(gen) > 1 else 0.0
        print(f"  GENERALITY 1-5       : mean {st.mean(gen):.2f} ± {se:.3f}   <- breadth of the concept")
    if both:
        prod = [c * g for c, g in both]
        print(f"  coherence x generality: mean {st.mean(prod):.2f}   "
              f"(useful clusters need BOTH; narrow clusters are trivially coherent)")
    if sur:
        print(f"  surface coherence    : mean {st.mean(sur):.2f}   (repetition of the same string)")
    if intr:
        ch = recs[0].get("intruder_chance", 1 / 6)
        se = st.stdev(intr) / len(intr) ** 0.5 if len(intr) > 1 else 0.0
        print(f"  WORD INTRUSION acc   : {st.mean(intr):.3f} ± {se:.3f}  (chance {ch:.3f}, "
              f"lift {st.mean(intr) - ch:+.3f})   <- behavioural")
    if det:
        print(f"  detection bal-acc    : mean {st.mean(det):.3f}")
        npr = [r["n_pred"] for r in recs if r.get("n_pred") is not None]
        ni  = [r["n_items"] for r in recs if r.get("n_items")]
        tpr = [r["tpr"] for r in recs if r.get("tpr") is not None]
        tnr = [r["tnr"] for r in recs if r.get("tnr") is not None]
        if npr:
            deg = sum(1 for r in recs if r.get("n_pred") in (0, r.get("n_items")))
            print(f"  judge picked         : {st.mean(npr):.1f} of {st.mean(ni):.0f} items "
                  f"(TPR {st.mean(tpr):.2f} / TNR {st.mean(tnr):.2f})")
            print(f"  DEGENERATE answers   : {deg}/{len(recs)} (picked all or none)")
    if null:
        print(f"  NULL floor           : mean {st.mean(null):.3f}")
        print(f"  LIFT over null       : {st.mean(det) - st.mean(null):+.3f}   <- the number that matters")
    # ---- TOKEN-WEIGHTED view -------------------------------------------
    # Cluster-averaged means let ~1000 tiny tail clusters outvote the ~200 that
    # carry most of the data. Weighting by n_assigned reports what the model
    # actually does to the corpus. It also separates a real improvement from
    # feature-split debris: adding narrow tail clusters raises the cluster
    # average and leaves the token-weighted number flat.
    ws, wi = _wmean(recs, "semantic_coh"), _wmean(recs, "intruder_acc")
    if ws is not None or wi is not None:
        print("  --- token-weighted (by n_assigned) ---")
        if ws is not None:
            print(f"  SEMANTIC coherence   : {ws:.2f}   (cluster-avg {st.mean(sem):.2f})")
        if wi is not None:
            print(f"  WORD INTRUSION acc   : {wi:.3f}   (cluster-avg {st.mean(intr):.3f})")
        byn = sorted([r for r in recs if isinstance(r.get("intruder_acc"), (int, float))],
                     key=lambda r: -r.get("n_assigned", 0))
        tot = sum(r.get("n_assigned", 0) for r in byn)
        if tot:
            print(f"  {'band':<16}{'%tokens':>9}{'intrusion':>11}{'semantic':>10}")
            for lbl, a, b in (("top 10% clusters", 0, .10), ("next 40%", .10, .50),
                              ("bottom 50%", .50, 1.0)):
                seg = byn[int(a * len(byn)):int(b * len(byn))]
                if not seg:
                    continue
                share = 100 * sum(r.get("n_assigned", 0) for r in seg) / tot
                si = st.mean([r["intruder_acc"] for r in seg])
                ss = st.mean([r["semantic_coh"] for r in seg
                              if isinstance(r.get("semantic_coh"), (int, float))] or [float("nan")])
                print(f"  {lbl:<16}{share:>8.1f}%{si:>11.3f}{ss:>10.2f}")

    lv = {}
    for r in recs:
        lv[r.get("level") or "?"] = lv.get(r.get("level") or "?", 0) + 1
    print("  concept level mix    : " + "  ".join(
        f"{k} {100*v/len(recs):.0f}%" for k, v in sorted(lv.items(), key=lambda kv: -kv[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_paths", nargs="*", help="closest_tokens.json files; 2+ are compared")
    ap.add_argument("--checkpoint", default=None,
                    help="Override the centroid source for HARD NEGATIVES (.pt or .npz). "
                         "By default each JSON's own meta is used, which is what you want "
                         "when comparing several runs.")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic", "openrouter"])
    ap.add_argument("--model", default=None, help="default: gpt-4o-mini / claude-sonnet-5")
    ap.add_argument("--n_clusters", type=int, default=100)
    ap.add_argument("--sample", default="stratified", choices=["top", "random", "stratified"],
                    help="'top' biases to the head of the usage distribution; "
                         "'stratified' spans it (default)")
    ap.add_argument("--min_assigned", type=int, default=50)
    ap.add_argument("--n_explain", type=int, default=8, help="examples shown to the explainer")
    ap.add_argument("--n_pos", type=int, default=6, help="HELD-OUT positives for detection")
    ap.add_argument("--n_neg", type=int, default=6)
    ap.add_argument("--n_neighbours", type=int, default=4, help="competing centroids for negatives")
    ap.add_argument("--n_intruder_trials", type=int, default=3,
                    help="word-intrusion trials per cluster (chance = 1/(intruder_k+1))")
    ap.add_argument("--intruder_k", type=int, default=5, help="genuine members per intrusion trial")
    ap.add_argument("--null_control", action="store_true", default=True)
    ap.add_argument("--no_null_control", dest="null_control", action="store_false")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--cache_dir", default="cache/llm_judge")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry_run", action="store_true", help="print prompts and cost, no key needed")
    ap.add_argument("--report_from", default=None,
                    help="re-render summaries from a saved llm_judge JSON; no API calls, no key")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.model is None:
        args.model = {"openai": "gpt-4o-mini",
                      "anthropic": "claude-sonnet-5",
                      "openrouter": "openai/gpt-4o-mini"}[args.provider]

    if args.report_from:
        for r in json.loads(Path(args.report_from).read_text()):
            print(f"\n=== {Path(r['source']).name} ===\n")
            summarise(r)
        return

    llm = LLM(args.provider, args.model, Path(args.cache_dir), dry_run=args.dry_run)
    rng = random.Random(args.seed)

    all_res = []
    for p in args.json_paths:
        res = judge_run(Path(p), args, llm, rng)
        if res is None:
            return
        all_res.append(res)
        print(f"\n=== {Path(p).name} ===")
        summarise(res)

    if not args.dry_run:
        print(f"\n[judge] api calls {llm.n_calls}, cache hits {llm.n_cached}")
        out = Path(args.out or "results/llm_judge.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_res, indent=2, ensure_ascii=False))
        print(f"[judge] wrote {out}")


if __name__ == "__main__":
    main()
