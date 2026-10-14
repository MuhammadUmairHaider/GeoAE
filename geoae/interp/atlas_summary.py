"""
Atlas summary: how much of a codebook can an LLM NAME, and what KIND of concept is it?

Input is one llm_judge run list ([{source, provider, model, n_clusters, clusters}, ...]; one run per
space: the balanced k-means "base" and the token-bypass AE "bypass", each judged from its
closest_tokens --doc_view JSON) and those two closest_tokens JSONs. The judge records carry the
explanation, level and scores; the closest_tokens files add what the judge never saw as numbers —
usage, token mix, document spread, hub flag, example contexts — and the full list of LIVE clusters,
which is the denominator for "how much of the codebook did the judge cover".

Writes, into --out_dir:
  atlas_summary.json   base vs bypass numbers (below) + meta (inputs, thresholds, drops)
  atlas_base.json      every judged cluster of that space, biggest first: explanation, scores, top
  atlas_bypass.json    token + share, doc spread, hub flag, 3 example contexts -> a browsable atlas

WHAT IS COUNTED. Everything is over the JUDGED clusters (llm_judge's stratified sample of the live
ones). Each is weighted by its usage (n_assigned, renormalised over the judged set), because the
question is "how much of the TEXT sits in a cluster the judge could name", not "how many clusters".
The unweighted cluster-count version is reported beside it. `judged usage mass` (judged / all live
usage) says how much of the codebook the sample stands for.

  token_detector  the commonest token string covers >= 80% of the cluster's closest tokens
                  (closest_tokens' token_distribution is over the closest min(200, n) tokens)
  nameable        intruder_acc >= 2/3 AND semantic_coh >= 4. With 3 intrusion trials at chance 1/6 a
                  cluster clears the first test by luck with p = 0.074, so absolute shares carry that
                  floor in both spaces
  conceptual      nameable AND level in {semantics, topic, discourse, entity} AND NOT token_detector
                  -> HEADLINE: the share of text in clusters nameable as a concept, not as a string
  level_mass      share of mass per judge `level` (8 categories + "unparsed")
  hub             doc_view hub flag; plus usage-weighted means of semantic_coh, surface_coh,
                  generality, intruder_acc and (detect_acc - null_acc)

A cluster whose judge answer failed to parse stays in the judged set, with level "unparsed", and counts
as NOT nameable; it is left out of the means of the fields it lacks (each reported with its n). Records
that cannot be placed at all (no cluster id, id absent from the closest_tokens file, no usage) are
dropped and counted.

CIs: bypass - base, with the judged clusters of each space resampled with replacement (independently:
they are different clusters). Interval = est + quantile(boots - mean(boots), [.025, .975]); `est` is
the full-sample value, never mean(boots). They cover which clusters were judged — not judge noise, the
held-out text sample or the training seed.

READ DIFFERENCES, NOT LEVELS. The `level` mix is the judge's opinion (judge_agreement.py: "syntax"
for 20% / 48% / 70% of the same clusters under three judges). Only bypass - base under one judge and
one protocol is interpretable.

    python -m geoae.interp.atlas_summary \
        --judge results/delta/d6144/interp/judge_flash_lite.json \
        --ct_base results/delta/base_balanced_kmeans_k2000_dpc_sampled/interp/closest_tokens.json \
        --ct_bypass results/delta/d6144/interp/closest_tokens.json \
        --out_dir results/delta/d6144/interp
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from geoae.interp.llm_judge import CONCEPT_LEVELS

LEVELS = list(CONCEPT_LEVELS) + ["unparsed"]            # the judge's 8 + answers that did not parse
CONCEPT_LEVEL_SET = ("semantics", "topic", "discourse", "entity")
TOKEN_DETECTOR_SHARE = 0.8
INTRUDER_MIN = 2 / 3
SEMANTIC_MIN = 4
# closest_tokens builds token_distribution from `core = sorted_idx[:min(200, M)]` and keeps only the
# 15 commonest strings, so the share's denominator is the core size, NOT the sum of the listed counts
# (that would over-state the share of a cluster with a long singleton tail).
CORE_TOKENS = 200
MIN_SUFFIX = 3          # path components that must agree to match a judge `source` by suffix
WEIGHTINGS = ("mass", "count")
EPS = 1e-9

# key -> (label, kind, decimals); "share" prints as %, "mean" as a plain number
METRICS = {
    "nameable": ("nameable", "share", 1),
    "conceptual": ("conceptual", "share", 1),
    "token_detector": ("token_detector", "share", 1),
    "hub": ("hub clusters", "share", 1),
    "semantic_coh": ("semantic_coh (1-5)", "mean", 2),
    "surface_coh": ("surface_coh (1-5)", "mean", 2),
    "generality": ("generality (1-5)", "mean", 2),
    "intruder_acc": ("intruder_acc", "mean", 3),
    "detect_minus_null": ("detect_acc - null_acc", "mean", 3),
}
VALUE_METRICS = ("hub", "semantic_coh", "surface_coh", "generality", "intruder_acc", "detect_minus_null")
LEVEL_PREFIX = "level:"


# ------------------------------------------------------------------ loading / matching
def load_json(path):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[atlas] {p} does not exist")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[atlas] {p} is not valid JSON ({e}) - a truncated write? "
                         f"redo the step that wrote it (FORCE=1)")


def _canon(p) -> str:
    return os.path.realpath(os.path.abspath(str(p)))


def _suffix_len(a, b) -> int:
    n = 0
    for x, y in zip(Path(str(a)).parts[::-1], Path(str(b)).parts[::-1]):
        if x != y:
            break
        n += 1
    return n


def match_runs(runs: list, ct_base, ct_bypass):
    """-> (base run, bypass run). llm_judge records `source` = the closest_tokens path exactly as it
    was typed on its command line, so match that to --ct_base / --ct_bypass: first as resolved paths;
    failing that (judged from another cwd or machine) by the longest common path tail, only if it is
    unambiguous. Never by position in the file."""
    sources = [r.get("source") if isinstance(r, dict) else None for r in runs]
    picked = {}
    for name, target in (("base", ct_base), ("bypass", ct_bypass)):
        hits = [i for i, s in enumerate(sources) if isinstance(s, str) and _canon(s) == _canon(target)]
        if not hits:
            ranked = sorted(((_suffix_len(s, target) if isinstance(s, str) else 0, i)
                             for i, s in enumerate(sources)), reverse=True)
            if ranked and ranked[0][0] >= MIN_SUFFIX and (len(ranked) == 1 or ranked[1][0] < ranked[0][0]):
                hits = [ranked[0][1]]
        if len(hits) != 1:
            raise SystemExit(f"[atlas] cannot tell which judge run is the {name} space: --ct_{name} "
                             f"{target} matches {len(hits)} of the run sources {sources}")
        picked[name] = hits[0]
    if picked["base"] == picked["bypass"]:
        raise SystemExit(f"[atlas] --ct_base and --ct_bypass both match the judge run "
                         f"{sources[picked['base']]!r}")
    return runs[picked["base"]], runs[picked["bypass"]]


def check_modes(ct_base: dict, ct_bypass: dict, allow: bool) -> dict:
    """A swapped --ct_base / --ct_bypass would silently flip every sign, so refuse it."""
    got = {"base": (ct_base.get("meta") or {}).get("mode"), "bypass": (ct_bypass.get("meta") or {}).get("mode")}
    want = {"base": "raw_kmeans_baseline", "bypass": "learned_ae"}
    bad = [f"--ct_{k} is {got[k]!r}, expected {want[k]!r}" for k in got if got[k] is not None and got[k] != want[k]]
    if bad and not allow:
        raise SystemExit("[atlas] " + "; ".join(bad) + " (swapped arguments? --allow_mode_mismatch overrides)")
    if bad:
        print("[atlas] WARNING: " + "; ".join(bad))
    return {"found": got, "expected": want, "ok": not bad}


def same_text_diff(meta_b: dict, meta_y: dict) -> list:
    """Fields saying the two closest_tokens runs did NOT read the same held-out text."""
    flat = {}
    for tag, m in (("base", meta_b), ("bypass", meta_y)):
        d = m.get("data") or {}
        flat[tag] = {"n_tokens": m.get("n_tokens"), "n_docs": m.get("n_docs"), "model_name": m.get("model_name"),
                     "target_layer": m.get("target_layer"), "data.seed": d.get("seed"),
                     "data.doc_offsets": d.get("doc_offsets"), "data.context_len": d.get("context_len")}
    return [k for k in flat["base"] if flat["base"][k] != flat["bypass"][k]]


# ------------------------------------------------------------------ per-cluster records
def _num(x, lo=None, hi=None):
    """x as a finite float, or None: missing, non-numeric (a bool is not a score), NaN or outside
    the range the judge prompt documents."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    if not math.isfinite(x) or (lo is not None and x < lo) or (hi is not None and x > hi):
        return None
    return x


def token_stats(c: dict, n_assigned=None):
    """(top token, its share of the closest tokens) from a closest_tokens cluster; (None, None) if
    the cluster has no usable token_distribution."""
    best, total = None, 0.0
    for ent in c.get("token_distribution") or []:
        try:
            tok, cnt = ent[0], _num(ent[1], lo=0)
        except (TypeError, IndexError, KeyError):
            continue
        if cnt is None:
            continue
        total += cnt
        if best is None or cnt > best[1]:
            best = (tok, cnt)
    if best is None or total <= 0:
        return None, None
    core = max(total, min(CORE_TOKENS, n_assigned)) if n_assigned else total
    return best[0], best[1] / core


def norm_level(x) -> str:
    if isinstance(x, str) and x.strip().lower() in CONCEPT_LEVELS:
        return x.strip().lower()
    return "unparsed"


def _cluster_id(x):
    if isinstance(x, str) and x.strip().lstrip("-").isdigit():
        x = int(x)
    return x if isinstance(x, int) and not isinstance(x, bool) else None


def build_space(run: dict, ct: dict):
    """Join one judge run with its closest_tokens file -> (cluster records, info).

    Records the judge could not score are KEPT (level "unparsed", None scores): dropping them would
    reward a space whose answers fail to parse. Only records that cannot be placed are dropped."""
    clusters = ct.get("clusters") or {}
    live_tokens = sum(_num(c.get("n_assigned"), lo=0) or 0 for c in clusters.values() if isinstance(c, dict))
    live_pct = sum(_num(c.get("usage_pct"), lo=0) or 0 for c in clusters.values() if isinstance(c, dict))
    drops, raw_unparsed, seen, recs = Counter(), Counter(), set(), []
    mismatch = 0
    judge_recs = run.get("clusters") or []
    for jr in judge_recs:
        if not isinstance(jr, dict):
            drops["not a record"] += 1
            continue
        cid = _cluster_id(jr.get("cluster"))
        if cid is None:
            drops["no cluster id"] += 1
            continue
        if cid in seen:
            drops["duplicate cluster id"] += 1
            continue
        c = clusters.get(str(cid))
        if not isinstance(c, dict):
            drops["cluster not in the closest_tokens file"] += 1
            continue
        n = _num(c.get("n_assigned"), lo=1) or _num(jr.get("n_assigned"), lo=1)
        if n is None:
            drops["no usage (n_assigned)"] += 1
            continue
        seen.add(cid)
        jn = _num(jr.get("n_assigned"), lo=1)
        if jn is not None and abs(jn - n) > max(5.0, 0.02 * n):
            mismatch += 1                       # judge and closest_tokens files are out of sync
        lvl = norm_level(jr.get("level"))
        if lvl == "unparsed":
            raw_unparsed[repr(jr.get("level"))] += 1
        tok, share = token_stats(c, n)
        is_det = share is not None and share >= TOKEN_DETECTOR_SHARE - EPS
        sem, intr = _num(jr.get("semantic_coh"), 1, 5), _num(jr.get("intruder_acc"), 0, 1)
        nameable = intr is not None and sem is not None and intr >= INTRUDER_MIN - EPS and sem >= SEMANTIC_MIN
        hub = c.get("hub")
        expl = jr.get("explanation")
        usage_pct = _num(c.get("usage_pct"), lo=0)
        recs.append({
            "cluster": cid,
            "usage_pct": usage_pct if usage_pct is not None else _num(jr.get("usage_pct"), lo=0),
            "level": lvl,
            "explanation": expl if isinstance(expl, str) else None,
            "semantic_coh": sem,
            "surface_coh": _num(jr.get("surface_coh"), 1, 5),
            "generality": _num(jr.get("generality"), 1, 5),
            "intruder_acc": intr,
            "detect_acc": _num(jr.get("detect_acc"), 0, 1),
            "null_acc": _num(jr.get("null_acc"), 0, 1),
            "top_token": tok,
            "top_token_share": None if share is None else round(share, 4),
            "token_detector": bool(is_det),
            "n_docs": c.get("n_docs"), "eff_docs": c.get("eff_docs"),
            "hub": hub if isinstance(hub, bool) else None,
            "examples": [e["context"] for e in (c.get("top") or [])[:3]
                         if isinstance(e, dict) and isinstance(e.get("context"), str)],
            "nameable": bool(nameable),
            "conceptual": bool(nameable and lvl in CONCEPT_LEVEL_SET and not is_det),
            "n_assigned": int(n),                # weight; dropped from the atlas files
        })
    judged_tokens = sum(r["n_assigned"] for r in recs)
    judged_pct = sum(r["usage_pct"] or 0 for r in recs)
    recs.sort(key=lambda r: (-r["n_assigned"], r["cluster"]))
    info = {
        "n_judge_records": len(judge_recs), "n_judged": len(recs),
        "n_dropped": int(sum(drops.values())), "dropped": dict(drops),
        "n_live": len(clusters), "n_clusters": (ct.get("meta") or {}).get("n_clusters"),
        "judged_usage_mass": judged_tokens / live_tokens if live_tokens else None,
        "judged_usage_pct": round(judged_pct, 3), "live_usage_pct": round(live_pct, 3),
        "n_assigned_mismatch": mismatch,
        "unparsed_level_values": dict(raw_unparsed.most_common(10)),
    }
    return recs, info


# ------------------------------------------------------------------ metrics
def metric_columns(recs: list) -> dict:
    """key -> (value X, present P), both (n,) float arrays. X is 0 where the value is missing; a metric
    is sum(w X P) / sum(w P), so a missing value leaves that metric's mean and denominator, while the
    indicators (nameable, conceptual, token_detector, level shares) are always present (a cluster that
    could not be scored is simply not nameable)."""
    n = len(recs)

    def val(k):
        return np.array([np.nan if r[k] is None else r[k] for r in recs], float)

    def flag(k):
        return np.array([float(r[k]) for r in recs])

    def present(x):
        return np.isfinite(x).astype(float), np.where(np.isfinite(x), x, 0.0)

    cols, ones = {}, np.ones(n)
    for k in ("nameable", "conceptual", "token_detector"):
        cols[k] = (flag(k), ones)
    hub = np.array([np.nan if r["hub"] is None else float(r["hub"]) for r in recs], float)
    for k, x in (("hub", hub), ("semantic_coh", val("semantic_coh")), ("surface_coh", val("surface_coh")),
                 ("generality", val("generality")), ("intruder_acc", val("intruder_acc")),
                 ("detect_minus_null", val("detect_acc") - val("null_acc"))):
        p, x0 = present(x)
        cols[k] = (x0, p)
    levels = np.array([r["level"] for r in recs])
    for lv in LEVELS:
        cols[LEVEL_PREFIX + lv] = ((levels == lv).astype(float), ones)
    return cols


def space_matrices(recs: list):
    """Stacked per-cluster numerators / denominators, (n, 2 * n_metrics): column j is
    (weighting, key) = index[j]; weights are usage ("mass") or 1 ("count")."""
    cols = metric_columns(recs)
    w = {"mass": np.array([r["n_assigned"] for r in recs], float), "count": np.ones(len(recs))}
    num, den, index = [], [], []
    for wk in WEIGHTINGS:
        for k, (x, p) in cols.items():
            num.append(w[wk] * x * p)
            den.append(w[wk] * p)
            index.append((wk, k))
    return np.column_stack(num), np.column_stack(den), index


def ratios(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return num.sum(0) / den.sum(0)


def bootstrap(num: np.ndarray, den: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """(n_boot, m) metric values over clusters resampled with replacement. A resample is a vector of
    multinomial counts, so every metric of every draw is one matrix product."""
    n = num.shape[0]
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (counts @ num) / (counts @ den)


def centred_ci(est: float, boots) -> tuple:
    """est + quantile(boots - mean(boots), [.025, .975]). `est` is the full-sample statistic, never
    mean(boots): a resampled share is biased, and this interval is built to be centred on `est`."""
    b = np.asarray(boots, float)
    b = b[np.isfinite(b)]
    if len(b) < 2 or not math.isfinite(est):
        return None, None
    lo, hi = np.quantile(b - b.mean(), [0.025, 0.975])
    return float(est + lo), float(est + hi)


def _f(x):
    return None if x is None or not math.isfinite(x) else round(float(x), 6)


def intruder_chance_pass_rate(*runs) -> dict:
    """P(a cluster clears intruder_acc >= INTRUDER_MIN by guessing): binomial tail at the judge's own
    trial count and chance level (3 trials, 1/6 -> 0.074)."""
    trials, chance = Counter(), Counter()
    for run in runs:
        for r in run.get("clusters") or []:
            if isinstance(r, dict) and _num(r.get("intruder_trials"), lo=1):
                trials[int(r["intruder_trials"])] += 1
                if _num(r.get("intruder_chance"), 0, 1) is not None:
                    chance[round(r["intruder_chance"], 6)] += 1
    if not trials:
        return {}
    n, p = trials.most_common(1)[0][0], (chance.most_common(1)[0][0] if chance else 1 / 6)
    need = math.ceil(INTRUDER_MIN * n - EPS)
    return {"n_trials": n, "chance": round(p, 4), "pass_rate": round(sum(
        math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(need, n + 1)), 4)}


def atlas_rows(recs: list) -> list:
    """The browsable list: the records minus the internal weight (usage_pct is the readable form)."""
    return [{k: v for k, v in r.items() if k != "n_assigned"} for r in recs]


def summarize(run_b: dict, run_y: dict, ct_b: dict, ct_y: dict, n_boot: int = 1000, seed: int = 0):
    """-> (summary dict, base atlas list, bypass atlas list)."""
    recs_b, info_b = build_space(run_b, ct_b)
    recs_y, info_y = build_space(run_y, ct_y)
    for tag, recs, info in (("base", recs_b, info_b), ("bypass", recs_y, info_y)):
        if not recs:
            raise SystemExit(f"[atlas] no usable judged clusters in the {tag} space "
                             f"({info['n_judge_records']} judge records, dropped: {info['dropped']})")
    num_b, den_b, index = space_matrices(recs_b)
    num_y, den_y, _ = space_matrices(recs_y)
    est_b, est_y = ratios(num_b, den_b), ratios(num_y, den_y)
    boot_b = boot_y = None
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        boot_b, boot_y = bootstrap(num_b, den_b, n_boot, rng), bootstrap(num_y, den_y, n_boot, rng)

    def space(recs, info, est):
        cols = metric_columns(recs)
        u = np.array([r["n_assigned"] for r in recs], float)
        out = {"coverage": {k: info[k] for k in ("n_judge_records", "n_judged", "n_live", "n_clusters",
                                                  "judged_usage_mass", "judged_usage_pct", "live_usage_pct")},
               "dropped": {"n_dropped": info["n_dropped"], "reasons": info["dropped"]},
               "n_assigned_mismatch": info["n_assigned_mismatch"],
               "unparsed_level_values": info["unparsed_level_values"],
               "mass": {}, "count": {}, "level_mass": {}, "level_count": {}, "n_present": {}, "mass_present": {}}
        for j, (wk, k) in enumerate(index):
            if k.startswith(LEVEL_PREFIX):
                out["level_" + wk][k[len(LEVEL_PREFIX):]] = _f(est[j])
            else:
                out[wk][k] = _f(est[j])
        for k in VALUE_METRICS:                  # how many clusters (and how much mass) had a usable value
            p = cols[k][1]
            out["n_present"][k] = int(p.sum())
            out["mass_present"][k] = _f((u * p).sum() / u.sum())
        return out

    diff = {wk: {} for wk in ("mass", "count", "level_mass", "level_count")}
    for j, (wk, k) in enumerate(index):
        d = est_y[j] - est_b[j]
        lo = hi = None
        if boot_b is not None and math.isfinite(d):
            lo, hi = centred_ci(d, boot_y[:, j] - boot_b[:, j])
        sect, name = (("level_" + wk, k[len(LEVEL_PREFIX):]) if k.startswith(LEVEL_PREFIX) else (wk, k))
        diff[sect][name] = {"base": _f(est_b[j]), "bypass": _f(est_y[j]), "diff": _f(d),
                            "lo": _f(lo), "hi": _f(hi)}

    summary = {"base": space(recs_b, info_b, est_b), "bypass": space(recs_y, info_y, est_y), "diff": diff}
    return summary, atlas_rows(recs_b), atlas_rows(recs_y)


# ------------------------------------------------------------------ printing
def _val(v, kind, nd):
    if v is None:
        return "n/a"
    return f"{100 * v:.1f}%" if kind == "share" else f"{v:.{nd}f}"


def _signed(x, nd):
    """+x.xx, with a rounded-to-zero value shown as +0 rather than -0."""
    s = f"{x:+.{nd}f}"
    return s.replace("-", "+") if float(s) == 0 else s


def _dif(e, kind, nd):
    if e["diff"] is None:
        return "n/a"
    k, nd = (100, 1) if kind == "share" else (1, nd)
    s = f"{_signed(k * e['diff'], nd)}" + (" pp" if kind == "share" else "")
    return s + (f" [{_signed(k * e['lo'], nd)}, {_signed(k * e['hi'], nd)}]" if e["lo"] is not None else "")


def print_report(summary: dict, meta: dict) -> None:
    b, y, d = summary["base"], summary["bypass"], summary["diff"]
    print(f"\n[atlas] judge: {meta['judge'].get('provider')} / {meta['judge'].get('model')}   "
          f"({meta['bootstrap']['n_boot']} bootstrap draws over judged clusters, seed {meta['bootstrap']['seed']})")
    print(f"\n{'coverage':<28}{'base':>12}{'bypass':>12}")
    for label, key, fmt in (("live clusters", "n_live", "{:d}"), ("judged clusters", "n_judged", "{:d}"),
                            ("judged usage mass", "judged_usage_mass", "pct")):
        cells = []
        for s in (b, y):
            v = s["coverage"][key]
            cells.append("n/a" if v is None else (f"{100 * v:.1f}%" if fmt == "pct" else fmt.format(v)))
        print(f"  {label:<26}{cells[0]:>12}{cells[1]:>12}")
    print(f"  {'judge records dropped':<26}{b['dropped']['n_dropped']:>12}{y['dropped']['n_dropped']:>12}")

    def table(title, section, keys):
        print(f"\n{title}")
        print(f"  {'metric':<24}{'base':>9}{'bypass':>9}   bypass - base [95% CI]")
        for k in keys:
            label, kind, nd = METRICS[k]
            e = d[section][k]
            print(f"  {label:<24}{_val(e['base'], kind, nd):>9}{_val(e['bypass'], kind, nd):>9}   {_dif(e, kind, nd)}")

    table("usage-weighted (share of the text's tokens; means weighted by usage)", "mass", list(METRICS))
    table("cluster-count (unweighted)", "count", ["nameable", "conceptual", "token_detector",
                                                  "semantic_coh", "intruder_acc"])
    print("\nlevel_mass (judge `level`, share of usage mass)")
    print(f"  {'level':<24}{'base':>9}{'bypass':>9}   bypass - base [95% CI]")
    for lv in LEVELS:
        e = d["level_mass"][lv]
        print(f"  {lv:<24}{_val(e['base'], 'share', 1):>9}{_val(e['bypass'], 'share', 1):>9}   {_dif(e, 'share', 1)}")

    notes = []
    for k in VALUE_METRICS:
        nb, ny = b["n_present"][k], y["n_present"][k]
        if nb < b["coverage"]["n_judged"] or ny < y["coverage"]["n_judged"]:
            notes.append(f"  {k:<18} base {nb}/{b['coverage']['n_judged']} "
                         f"({100 * (1 - (b['mass_present'][k] or 0)):.1f}% of mass lacks it)   "
                         f"bypass {ny}/{y['coverage']['n_judged']} ({100 * (1 - (y['mass_present'][k] or 0)):.1f}% of mass lacks it)")
    if notes:
        print("\nclusters with a usable value (a cluster lacking one leaves that mean; it is still judged, "
              "and counts as NOT nameable):")
        print("\n".join(notes))
    for tag, s in (("base", b), ("bypass", y)):
        if s["dropped"]["n_dropped"]:
            print(f"[atlas] {tag}: dropped {s['dropped']['n_dropped']} of {s['coverage']['n_judge_records']} "
                  f"judge records: {s['dropped']['reasons']}")
        if s["n_assigned_mismatch"]:
            print(f"[atlas] WARNING {tag}: {s['n_assigned_mismatch']} clusters' n_assigned differs between the "
                  f"judge file and the closest_tokens file - were they made from different runs?")
        if s["unparsed_level_values"]:
            print(f"[atlas] {tag}: `level` values counted as unparsed: {s['unparsed_level_values']}")
    if meta["text_differs"]:
        print(f"[atlas] WARNING: base and bypass closest_tokens runs differ in {meta['text_differs']} - "
              f"they did not read identical held-out text")


# ------------------------------------------------------------------ main
def _ct_meta(ct: dict) -> dict:
    m = ct.get("meta") or {}
    return {k: m.get(k) for k in ("mode", "checkpoint", "model_name", "target_layer", "n_clusters",
                                  "n_tokens", "n_docs", "n_live")} | {"doc_view": "doc_view" in m}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True, nargs="+",
                    help="llm_judge output(s): JSON lists of runs, one run per space; several files are "
                         "concatenated (e.g. the base/bypass run plus a separately judged token-mean control)")
    ap.add_argument("--ct_base", required=True, help="closest_tokens JSON of the base (k-means) space")
    ap.add_argument("--ct_bypass", required=True, help="closest_tokens JSON of the bypass AE space")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_mode_mismatch", action="store_true",
                    help="do not abort when --ct_base is not a k-means file / --ct_bypass not an AE file")
    args = ap.parse_args(argv)

    runs = []
    for jf in args.judge:
        got = load_json(jf)
        if isinstance(got, dict) and "clusters" in got:
            got = [got]
        if not isinstance(got, list):
            raise SystemExit(f"[atlas] {jf}: expected the llm_judge JSON list of runs")
        runs += got
    ct_b, ct_y = load_json(args.ct_base), load_json(args.ct_bypass)
    run_b, run_y = match_runs(runs, args.ct_base, args.ct_bypass)
    modes = check_modes(ct_b, ct_y, args.allow_mode_mismatch)
    print(f"[atlas] base   <- {args.ct_base}  ({len(run_b.get('clusters') or [])} judged records)")
    print(f"[atlas] bypass <- {args.ct_bypass}  ({len(run_y.get('clusters') or [])} judged records)")

    summary, atlas_b, atlas_y = summarize(run_b, run_y, ct_b, ct_y, args.n_boot, args.seed)
    if run_b.get("model") != run_y.get("model") or run_b.get("provider") != run_y.get("provider"):
        print(f"[atlas] WARNING: the two runs were judged by different models "
              f"({run_b.get('model')} vs {run_y.get('model')}) - the comparison is not like for like")
    meta = {
        "judge": {"file": " ".join(map(str, args.judge)), "provider": run_b.get("provider"), "model": run_b.get("model"),
                  "model_bypass_run": run_y.get("model")},
        "inputs": {"ct_base": str(args.ct_base), "ct_bypass": str(args.ct_bypass),
                   "judge_source_base": run_b.get("source"), "judge_source_bypass": run_y.get("source")},
        "ct_meta": {"base": _ct_meta(ct_b), "bypass": _ct_meta(ct_y)},
        "mode_check": modes,
        "text_differs": same_text_diff(ct_b.get("meta") or {}, ct_y.get("meta") or {}),
        "thresholds": {"token_detector_top_share": TOKEN_DETECTOR_SHARE, "intruder_acc_min": INTRUDER_MIN,
                       "semantic_coh_min": SEMANTIC_MIN, "conceptual_levels": list(CONCEPT_LEVEL_SET),
                       "token_share_denominator": f"min({CORE_TOKENS}, n_assigned), the closest_tokens core"},
        "intruder_chance_pass_rate": intruder_chance_pass_rate(run_b, run_y),
        "weights": "mass = n_assigned renormalised over the judged clusters; count = 1 per cluster",
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed,
                      "resampling": "judged clusters with replacement, independently per space",
                      "interval": "est + quantile(boots - mean(boots), [0.025, 0.975]); est = full-sample value",
                      "covers": "which clusters were judged; not judge noise, text sample or training seed"},
        "missing_policy": "a cluster lacking a field stays judged: it leaves the means of the fields it lacks "
                          "(n_present), is level 'unparsed' if its level did not parse, and is not nameable",
    }
    print_report(summary, meta)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in (("atlas_summary.json", {"meta": meta, **summary}),
                          ("atlas_base.json", atlas_b), ("atlas_bypass.json", atlas_y)):
        with open(out / name, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[atlas] wrote {out / name}")


if __name__ == "__main__":
    main()
