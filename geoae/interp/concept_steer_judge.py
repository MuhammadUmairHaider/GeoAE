"""
LLM judge for concept_steer_generate.py: did steering toward a NAMED concept with each
available handle (label, label_z, base, bypass, base_dir, bypass_dir -- whichever the gen
file actually has: see discover_handles) move the continuation on-topic? Forced choice among
the target concept and
3 distractor concepts of the SAME dataset (chance .25), shown as short topic descriptions
(concept_steer_generate.concept_description). The candidate set for a given (dataset,
concept, prompt) is drawn deterministically (crc32) from the dataset's FULL class list (not
just the concepts that were generated), so it is IDENTICAL across every handle and alpha --
the only thing that differs between judged cells is the continuation text, which is what
makes the per-concept comparisons in (b) paired.

alpha 0 judges the UNSTEERED continuation against each concept's candidate set -- the
per-cell prior every other alpha is scored net of. Since the alpha-0 prompt (unsteered text +
options) is IDENTICAL across every handle, and options are identical across alphas for a
given (dataset, concept, prompt), the full set of (prompt, continuation, options) strings has
many duplicates; these are deduped and asked ONCE before mapping answers back to every job
that needs them (a duplicate sent Nx concurrently would not be caught by the disk cache, and a
nondeterministic LLM answer could then make hit0 differ between handles that must share it).

Reports at THREE scopes: pooled (all concepts), and one per dataset present in the gen file
(db14, biasbios). Each handle's best alpha is chosen ONCE on the POOLED mean usable and reused
for every scope's paired comparison, so the per-dataset tables say "which handle wins at the
alpha that's best overall", not "the alpha that happens to be best for that one dataset".

    set -a; source .env; set +a
    python -u -m geoae.interp.concept_steer_judge --gen eval_out/concept_steer_generate.json \
        --provider openrouter --model google/gemini-2.5-flash-lite --out eval_out/concept_steer_judge.json
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from geoae.interp.llm_judge import LLM

LETTERS = "ABCD"
FLUENT_THRESHOLD = 0.6
DISPLAY_ORDER = ("label", "label_z", "base", "bypass", "base_dir", "bypass_dir")
PAIR_CANDIDATES = (
    ("label_z", "label"), ("label_z", "bypass"), ("bypass", "base"), ("bypass", "label"), ("base", "label"),
    ("bypass_dir", "base_dir"), ("bypass_dir", "label"), ("base_dir", "label"),
    ("bypass_dir", "bypass"), ("base_dir", "base"),
)


def discover_handles(steered: dict) -> list[str]:
    """Which handle names are present in a gen JSON's generations.steered dict, in the fixed
    DISPLAY_ORDER, with anything unrecognised appended alphabetically at the end."""
    present = set()
    for by_concept in steered.values():
        for by_handle in by_concept.values():
            present.update(by_handle.keys())
    ordered = [h for h in DISPLAY_ORDER if h in present]
    extra = sorted(present - set(DISPLAY_ORDER))
    return ordered + extra


def present_pairs(handles: list[str]) -> tuple[tuple[str, str], ...]:
    """PAIR_CANDIDATES restricted to pairs where BOTH names are in `handles`."""
    present = set(handles)
    return tuple((a, b) for a, b in PAIR_CANDIDATES if a in present and b in present)


def build_prompt(prompt: str, cont: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[k]}) {o}" for k, o in enumerate(options))
    return (
        "A language model was given a prompt and, from that point, wrote a continuation.\n\n"
        f"Prompt: {prompt!r}\nContinuation: {cont!r}\n\n"
        "Which of these topics does the CONTINUATION lean toward most? If none clearly, pick "
        f"the closest.\n\n{opts}\n\n"
        "Answer with a single letter (A, B, C or D) and nothing else."
    )


def pick_candidates(dataset: str, concept: str, p: int, all_concepts: list[str]):
    """Target + 3 distractors from the SAME dataset's full class list, and the shown order --
    both deterministic per (dataset, concept, prompt index), via zlib.crc32 (NOT hash(), which
    is per-process salted). Returns (candidate concept names, cand[0] is the target; display
    order: a permutation of range(len(cand)))."""
    others = [c for c in all_concepts if c != concept]
    rng = np.random.default_rng(zlib.crc32(f"{dataset}|{concept}|{p}".encode()))
    k = min(3, len(others))
    distractors = list(rng.choice(np.array(others, dtype=object), size=k, replace=False))
    cand = [concept] + distractors
    order_rng = np.random.default_rng(zlib.crc32(f"order|{dataset}|{concept}|{p}".encode()))
    order = list(order_rng.permutation(len(cand)))
    return cand, order


# ---------------------------------------------------------------------------
# Pure, unit-testable scoring (numpy in, dict/tuple out)
# ---------------------------------------------------------------------------

def centred_ci(estimate: float, boots: np.ndarray) -> tuple[float, float]:
    """95% CENTRED (basic) bootstrap CI: [2*m - q97.5, 2*m - q2.5]. `estimate` is the actual
    sample statistic -- NEVER the bootstrap mean, which centred/basic intervals deliberately
    do not use (that would be the ordinary percentile interval)."""
    boots = np.asarray(boots)
    lo = 2 * estimate - float(np.percentile(boots, 97.5))
    hi = 2 * estimate - float(np.percentile(boots, 2.5))
    return lo, hi


def boot_mean_ci(vals: np.ndarray, rng: np.random.Generator, n_boot: int = 2000):
    """Mean and centred 95% CI, bootstrapping over the (non-NaN) rows of `vals`."""
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(v.mean())
    if len(v) == 1:
        return m, m, m
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = v[idx].mean(1)
    lo, hi = centred_ci(m, boots)
    return m, lo, hi


def net_fluent_scores(hit: np.ndarray, hit0: np.ndarray, distinct2: np.ndarray,
                       threshold: float = FLUENT_THRESHOLD) -> dict:
    """hit, hit0, distinct2: (n_concepts, n_prompts). hit0 is the SAME cell's alpha=0 hit
    array (so a hit already present unsteered nets to 0, never counted as a win). fluent =
    distinct2 >= threshold; non-fluent continuations are excluded from net_hit. Returns
    per-concept fluent_share, net_hit (NaN if a concept has no fluent prompt) and usable
    (fluent_share * net_hit, 0 -- not NaN -- when fluent_share is 0)."""
    hit, hit0, distinct2 = np.asarray(hit, float), np.asarray(hit0, float), np.asarray(distinct2, float)
    fluent = distinct2 >= threshold
    diff = np.where(fluent, hit - hit0, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        net_hit = np.nanmean(diff, axis=1)
    fluent_share = fluent.mean(axis=1)
    usable = np.where(np.isnan(net_hit), 0.0, fluent_share * net_hit)
    return {"fluent_share": fluent_share, "net_hit": net_hit, "usable": usable}


def matched_at_fluent_share(fluent_share_by_alpha: np.ndarray, values_by_alpha: np.ndarray,
                             target_share: float) -> float:
    """fluent_share_by_alpha / values_by_alpha: aligned arrays over ascending alpha. fluent
    share is forced MONOTONE DECREASING (np.minimum.accumulate) before interpolating, and the
    pair is sorted ascending by fluent share (np.interp needs ascending x). NaN if
    target_share falls outside the (monotonised) range."""
    fs = np.minimum.accumulate(np.asarray(fluent_share_by_alpha, dtype=float))
    order = np.argsort(fs)
    fs_o, val_o = fs[order], np.asarray(values_by_alpha, dtype=float)[order]
    if target_share < fs_o.min() or target_share > fs_o.max():
        return float("nan")
    return float(np.interp(target_share, fs_o, val_o))


def paired_diff(a: np.ndarray, b: np.ndarray):
    """Per-concept a - b, restricted to concepts where BOTH are non-NaN. Returns (diff, mask)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    return a[mask] - b[mask], mask


def filter_scope(sc: dict, predicate=None) -> dict:
    """Restrict a net_fluent_scores-style result -- a dict with a 'concepts' list of
    (dataset, concept) keys aligned with its fluent_share/net_hit/usable arrays -- to the
    entries for which predicate((dataset, concept)) is True. predicate=None (the POOLED scope)
    returns `sc` unchanged. Used to slice one pooled computation into per-dataset scopes
    without re-running net_fluent_scores."""
    if predicate is None:
        return sc
    idx = [i for i, dc in enumerate(sc["concepts"]) if predicate(dc)]
    return {"concepts": [sc["concepts"][i] for i in idx],
            "fluent_share": np.asarray(sc["fluent_share"])[idx],
            "net_hit": np.asarray(sc["net_hit"])[idx],
            "usable": np.asarray(sc["usable"])[idx]}


def dataset_predicate(dataset: str):
    return lambda dc: dc[0] == dataset


def format_per_concept_row(row: dict, handles: list[str]) -> str:
    """Render one per_concept_table row (built in main()'s section (d)) the way it's printed:
    coverage precision to 2dp, and usable AS A PERCENTAGE -- per_concept_table['usable_<h>']
    is stored as a raw fraction (e.g. 0.275, consistent with every other table's JSON), so
    it prints as 'label=27.5', not 'label=0.3' (the un-multiplied fraction to 1dp)."""
    def pct(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) and x == x else "n/a"
    pb_s, py_s = pct(row.get("precision_base")), pct(row.get("precision_bypass"))
    us = "  ".join(f"{h}={100 * row[f'usable_{h}']:.1f}" if row.get(f"usable_{h}") is not None else f"{h}=n/a"
                   for h in handles)
    return f"{row['dataset']:9s} {row['concept']:24s} prec base={pb_s} bypass={py_s}   {us}"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="output of concept_steer_generate.py")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--cache_dir", default="cache/llm_judge")
    ap.add_argument("--max_alpha", type=float, default=3.0, help="judge alphas up to this")
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true", help="build prompts, call nothing")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    G = json.load(open(args.gen))
    prompts = G["meta"]["prompts"][: args.n_prompts]
    n_prompts = len(prompts)
    unsteered_text = G["generations"]["unsteered"]["text"]
    concept_desc = G["meta"]["concept_descriptions"]                  # {ds: {name: desc}} -- ALL classes
    steered = G["generations"]["steered"]                             # {ds: {concept: {handle: {alpha: {...}}}}}
    dataset_names = sorted(steered.keys())
    scope_names = ["pooled"] + dataset_names
    llm = LLM(args.provider, args.model, Path(args.cache_dir), dry_run=args.dry_run)

    HANDLES = discover_handles(steered)
    PAIRS = present_pairs(HANDLES)
    print(f"[judge] handles present: {', '.join(HANDLES)}; pairs: "
          + ", ".join(f"{a}-{b}" for a, b in PAIRS))

    # concepts available per handle (pooled across datasets, ds-qualified so names don't collide)
    concepts_by_handle = {h: [] for h in HANDLES}
    for ds, by_concept in steered.items():
        for concept, by_handle in by_concept.items():
            for h in HANDLES:
                if h in by_handle:
                    concepts_by_handle[h].append((ds, concept))

    option_cache: dict[tuple[str, str, int], tuple[list[str], list[int]]] = {}

    def options_for(ds, concept, p):
        key = (ds, concept, p)
        if key not in option_cache:
            option_cache[key] = pick_candidates(ds, concept, p, list(concept_desc[ds].keys()))
        return option_cache[key]

    jobs = []   # (ds, concept, handle, alpha_str, p, cand, order, prompt_text)
    for ds, by_concept in steered.items():
        for concept, by_handle in by_concept.items():
            for h, by_alpha in by_handle.items():
                for a_str in sorted(by_alpha, key=float):
                    a = float(a_str)
                    if a > 0 and a > args.max_alpha:
                        continue
                    cell = by_alpha[a_str]
                    texts = unsteered_text if a == 0.0 else cell["text"]
                    for p in range(min(n_prompts, len(texts))):
                        cand, order = options_for(ds, concept, p)
                        shown = [concept_desc[ds][cand[k]] for k in order]
                        jobs.append((ds, concept, h, a_str, p, cand, order,
                                    build_prompt(prompts[p], texts[p], shown)))

    # DEDUPE identical prompt strings before calling the LLM. alpha=0 prompts are identical
    # across every handle (same unsteered text, same options), and options are identical
    # across alphas for a given (dataset, concept, prompt) -- so many jobs share a prompt
    # verbatim. Asking each unique string once (rather than once per job, concurrently, which
    # the disk cache cannot catch) also guarantees every job sharing a prompt gets the SAME
    # answer, which the paired handle comparisons below rely on.
    unique_prompts = list(dict.fromkeys(j[7] for j in jobs))     # preserves first-seen order
    prompt_idx = {t: i for i, t in enumerate(unique_prompts)}
    print(f"[judge] {len(jobs):,} judgements, {len(unique_prompts):,} unique prompts "
          f"({args.model}); cached ones are free")

    def ask(text):
        return "A" if args.dry_run else llm.ask(text, max_tokens=5)

    with ThreadPoolExecutor(args.workers) as ex:
        unique_answers = list(ex.map(ask, unique_prompts))

    matches = [re.search(r"[ABCD]", (ans or "").upper()) for ans in unique_answers]
    bad_unique = sum(1 for m in matches if not m)

    hits: dict = {}          # ds -> concept -> handle -> alpha_str -> {p: hit}
    dis_check = set()
    job_bad = 0
    for (ds, concept, h, a_str, p, cand, order, text) in jobs:
        m = matches[prompt_idx[text]]
        if not m:
            job_bad += 1
        hit = float(bool(m) and order[LETTERS.index(m.group(0))] == 0)
        hits.setdefault(ds, {}).setdefault(concept, {}).setdefault(h, {}).setdefault(a_str, {})[p] = hit
        dis_check.add((ds, concept, p, tuple(cand), tuple(order)))
    print(f"[judge] unparseable answers: {bad_unique} of {len(unique_prompts)} unique prompts "
          f"(affecting {job_bad} of {len(jobs)} judgements; counted as misses)")
    # sanity: exactly one (cand, order) per (ds, concept, p) -- identical across handle/alpha
    n_cells = len({(d, c, p) for d, c, p, *_ in dis_check})
    if len(dis_check) != n_cells:
        print("[judge] WARNING: option set varied across handle/alpha for some (dataset, concept, prompt)")

    def get_arrays(ds, concept, h, a_str):
        d = hits.get(ds, {}).get(concept, {}).get(h, {}).get(a_str, {})
        hit = np.array([d.get(p, np.nan) for p in range(n_prompts)])
        distinct2 = np.asarray(steered[ds][concept][h][a_str]["distinct2"][:n_prompts]) if a_str != "0.0" \
            else np.asarray(G["generations"]["unsteered"]["distinct2"][:n_prompts])
        return hit, distinct2

    alphas = sorted({a for by_concept in steered.values() for by_handle in by_concept.values()
                     for h in HANDLES if h in by_handle for a in by_handle[h] if float(a) > 0},
                    key=float)
    alphas = [a for a in alphas if float(a) <= args.max_alpha]

    # ---- (a) per (handle, alpha): fluent share / net hit / usable, POOLED over ALL concepts --
    # (computed once, pooled; every scope below is a slice of this via filter_scope)
    per_alpha = {h: {} for h in HANDLES}       # per_alpha[h][a_str] = {"concepts":[...], "fluent_share":arr,...}
    for h in HANDLES:
        concepts = concepts_by_handle[h]
        if not concepts:
            continue
        for a_str in alphas:
            avail = [(ds, c) for ds, c in concepts if a_str in steered[ds][c][h]]
            if not avail:
                continue
            hit = np.stack([get_arrays(ds, c, h, a_str)[0] for ds, c in avail])
            d2 = np.stack([get_arrays(ds, c, h, a_str)[1] for ds, c in avail])
            h0 = np.stack([get_arrays(ds, c, h, "0.0")[0] for ds, c in avail])
            sc = net_fluent_scores(hit, h0, d2)
            sc["concepts"] = avail
            per_alpha[h][a_str] = sc

    print("\n[judge] FLUENT continuations only, net of the same cell's unsteered hit (per 100) "
          "[95% CI over concepts] -- by scope")
    scopes_per_alpha = {}
    for scope in scope_names:
        pred = None if scope == "pooled" else dataset_predicate(scope)
        print(f"  scope={scope}:")
        summary_alpha = {h: {} for h in HANDLES}
        for h in HANDLES:
            if not per_alpha[h]:
                continue
            printed_header = False
            for a_str in alphas:
                if a_str not in per_alpha[h]:
                    continue
                sc = filter_scope(per_alpha[h][a_str], pred)
                if len(sc["concepts"]) == 0:
                    continue
                fm, flo, fhi = boot_mean_ci(sc["fluent_share"], rng, args.n_boot)
                nm, nlo, nhi = boot_mean_ci(sc["net_hit"], rng, args.n_boot)
                um, ulo, uhi = boot_mean_ci(sc["usable"], rng, args.n_boot)
                summary_alpha[h][a_str] = {"fluent_share": (fm, flo, fhi), "net_hit": (nm, nlo, nhi),
                                           "usable": (um, ulo, uhi), "n_concepts": len(sc["concepts"])}
                if not printed_header:
                    print(f"    {h}:")
                    printed_header = True
                print(f"      a{a_str:<5} fluent {100*fm:3.0f}%  net {100*nm:+5.1f} [{100*nlo:+.0f},{100*nhi:+.0f}]  "
                      f"usable {100*um:5.1f} [{100*ulo:.0f},{100*uhi:.0f}]  (n={len(sc['concepts'])})")
        scopes_per_alpha[scope] = summary_alpha

    # ---- (b) best alpha per handle, chosen ONCE on the POOLED mean usable ---------------------
    best_alpha = {}
    pooled_summary = scopes_per_alpha["pooled"]
    for h in HANDLES:
        if not pooled_summary[h]:
            continue
        best_alpha[h] = max(pooled_summary[h], key=lambda a: pooled_summary[h][a]["usable"][0])
    print(f"\n[judge] best alpha per handle (max POOLED mean usable), reused for every scope below: "
          + ", ".join(f"{h}={best_alpha[h]}" for h in best_alpha))

    def usable_vector(h, a_str, universe):
        """(len(universe),) usable per (ds,concept) in `universe`, NaN where handle h has no
        data at a_str for that concept."""
        sc = per_alpha[h].get(a_str)
        if sc is None:
            return np.full(len(universe), np.nan)
        lookup = {dc: u for dc, u in zip(sc["concepts"], sc["usable"])}
        return np.array([lookup.get(dc, np.nan) for dc in universe])

    all_concepts_union = sorted({dc for h in HANDLES for dc in concepts_by_handle[h]})

    def scope_universe(scope):
        return all_concepts_union if scope == "pooled" else [dc for dc in all_concepts_union if dc[0] == scope]

    print("\n[judge] paired per-concept usable, at each handle's own best alpha (chosen on POOLED "
          "usable) -- by scope")
    scopes_pairwise = {}
    for scope in scope_names:
        universe = scope_universe(scope)
        print(f"  scope={scope}:")
        pairwise = {}
        for A, B in PAIRS:
            if A not in best_alpha or B not in best_alpha:
                continue
            uA = usable_vector(A, best_alpha[A], universe)
            uB = usable_vector(B, best_alpha[B], universe)
            diff, mask = paired_diff(uA, uB)
            n = len(diff)
            m, lo, hi = boot_mean_ci(diff, rng, args.n_boot)
            eps = 1e-9
            wins, ties, loses = int((diff > eps).sum()), int((np.abs(diff) <= eps).sum()), int((diff < -eps).sum())
            p_val = float("nan")
            if n >= 1 and np.any(np.abs(diff) > eps):
                from scipy.stats import wilcoxon
                try:
                    p_val = float(wilcoxon(diff).pvalue)
                except ValueError:
                    p_val = float("nan")
            pairwise[f"{A}_minus_{B}"] = {"mean_diff": m, "ci": [lo, hi], "wilcoxon_p": p_val,
                                          "n": n, "wins": wins, "ties": ties, "loses": loses,
                                          "alpha_A": best_alpha[A], "alpha_B": best_alpha[B]}
            print(f"    {A} - {B} (a{best_alpha[A]} vs a{best_alpha[B]}): mean {100*m:+5.1f} "
                  f"[{100*lo:+.0f},{100*hi:+.0f}]  p={p_val:.3f}  {A} wins/ties/loses {wins}/{ties}/{loses}  (n={n})")
        scopes_pairwise[scope] = pairwise

    # ---- (c) matched fluent share: net hit / usable at 90/80/70/50% fluent, by scope ----------
    print("\n[judge] at MATCHED fluent share: net hit / usable -- by scope")
    scopes_matched = {}
    for scope in scope_names:
        summary_alpha = scopes_per_alpha[scope]
        print(f"  scope={scope}:")
        matched = {h: {} for h in HANDLES}
        for h in HANDLES:
            al = [a for a in alphas if a in summary_alpha[h]]
            if not al:
                continue
            fs = np.array([summary_alpha[h][a]["fluent_share"][0] for a in al])
            nh = np.array([summary_alpha[h][a]["net_hit"][0] for a in al])
            us = np.array([summary_alpha[h][a]["usable"][0] for a in al])
            for target in (0.9, 0.8, 0.7, 0.5):
                matched[h][str(target)] = {"net_hit": matched_at_fluent_share(fs, nh, target),
                                           "usable": matched_at_fluent_share(fs, us, target)}
            print(f"    {h}: " + "  ".join(
                f"{int(100*t)}%: net {100*matched[h][str(t)]['net_hit']:+.1f} / usable "
                f"{100*matched[h][str(t)]['usable']:.1f}"
                if matched[h][str(t)]["net_hit"] == matched[h][str(t)]["net_hit"] else f"{int(100*t)}%: (out of range)"
                for t in (0.9, 0.8, 0.7, 0.5)))
        scopes_matched[scope] = matched

    # ---- (d) per-concept table (already per-dataset via the "dataset" column) -----------------
    print("\n[judge] per concept: coverage precision (base/bypass) and usable at each handle's best alpha")
    per_concept_table = []
    for ds, concept in all_concepts_union:
        cov = G["coverage"].get(ds, {}).get(concept, {})
        row = {"dataset": ds, "concept": concept,
               "precision_base": (cov.get("base") or {}).get("precision"),
               "precision_bypass": (cov.get("bypass") or {}).get("precision")}
        for h in HANDLES:
            if h in best_alpha:
                v = usable_vector(h, best_alpha[h], [(ds, concept)])[0]
                row[f"usable_{h}"] = None if np.isnan(v) else float(v)
            else:
                row[f"usable_{h}"] = None
        per_concept_table.append(row)
        print(f"  {format_per_concept_row(row, HANDLES)}")

    res = {
        "meta": {**vars(args), "gen": args.gen, "n_jobs": len(jobs), "n_unique_prompts": len(unique_prompts)},
        "best_alpha": best_alpha,
        "scopes": {
            scope: {
                "per_alpha": {h: {a: {k: v for k, v in s.items() if k != "concepts"} for a, s in by_a.items()}
                             for h, by_a in scopes_per_alpha[scope].items()},
                "pairwise": scopes_pairwise[scope],
                "matched_fluent_share": scopes_matched[scope],
            }
            for scope in scope_names
        },
        "per_concept": per_concept_table,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n[judge] wrote {args.out}  (calls {llm.n_calls}, cached {llm.n_cached})")


if __name__ == "__main__":
    main()
