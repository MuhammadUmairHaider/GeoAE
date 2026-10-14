"""Why does GeoAE look better at control edits at α 0.5? Competing explanations tested on per-document confidences.

E1 substrate:   GeoAE edits run on the AE-spliced model, which starts less confident in the gold label.
E2 edit size:   the same nominal α (class-gap units) is a bigger functional edit in the latent.
E3 coordinates: the latent's ranges/directions let the same-size edit hit the concept more precisely.
E4 scale:       the relative scale divides by each space's own (lower, for GeoAE) starting confidence.

T1 matched start (all datasets, α 0.5 + range removal): only documents whose gold-label confidence the splice barely
   changed (|conf_z - conf_h| <= --tol); paired per-document ABSOLUTE drops (target docs) and losses (other-class docs).
   Removes E1 and E4: same documents, same starting confidence, no division.
T2 matched damage on the dense grid α 0/.1/.2/.3/.4/.5/1/2 (DBpedia-14, Bias in Bios, AG News; every model with a
   *_conf_small.json): per concept, interpolate along the α curve: removal at equal collateral, collateral at equal
   removal, removal at equal perplexity rise (each space vs its own unedited perplexity). Separates E2 from E3.
T3 per concept: where base is good and GeoAE is bad (and the reverse) at α 0.5 range steering, with candidate
   explanations; Spearman (dataset-demeaned) of the per-concept gap with each candidate across concepts.

Units: points of gold-label probability (absolute) unless marked rel (% of that space's own unedited confidence).
Aggregation: 10%-trimmed mean over concepts of the per-concept paired GeoAE - base difference; 95% CI = bootstrap over
concepts, stratified by dataset (10k, seed 0). Jigsaw is binary and left out of pooled rows.
Usage: python eval_out/paper_control_mechanism.py [--tol 0.05] [--trim 0.10] > results/delta/control_mechanism.md
"""
import argparse
import functools
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tol", type=float, default=0.05, help="max |conf_z - conf_h| for a matched-start document")
ap.add_argument("--trim", type=float, default=0.10)
ap.add_argument("--min_docs", type=int, default=5, help="min matched docs per concept and side")
args = ap.parse_args()
rng = np.random.default_rng(0)
B = 10000
ROOT = Path("results/delta")
ARMS = [("d6144", "Llama-3.2-3B L27"), ("g4b_l33_d5120", "Gemma 3 4B L33"), ("qwen25_l35_d4096", "Qwen2.5-3B L35")]
DSN = {"db14": "DBpedia-14", "biasbios": "Bias in Bios", "ag_news": "AG News", "emotion6": "Emotion",
       "news20": "20 Newsgroups", "trec6": "TREC", "atlas": "FineWeb-Atlas", "jigsaw2": "Jigsaw"}
CURATED = ["db14", "biasbios", "ag_news", "emotion6", "news20", "trec6"]
DENSE = ["db14", "biasbios", "ag_news"]
f1 = lambda v: "n/a" if not np.isfinite(v) else format(v, ".1f").replace("-", "−")
fs = lambda v: "n/a" if not np.isfinite(v) else format(v, "+.1f").replace("-", "−")


def tmean(x):
    x = np.sort(np.asarray(x, float), axis=-1); n = x.shape[-1]; k = int(args.trim * n)
    return x[..., k:n - k].mean(axis=-1)


def boot(strata):
    """trimmed mean, 95% CI (stratified bootstrap), n, wins, losses for {dataset: per-concept values}."""
    strata = {g: np.asarray([v for v in xs if np.isfinite(v)]) for g, xs in strata.items()}
    strata = {g: v for g, v in strata.items() if len(v)}
    if not strata:
        return np.nan, np.nan, np.nan, 0, 0, 0
    allx = np.concatenate(list(strata.values()))
    bs = tmean(np.concatenate([v[rng.integers(0, len(v), (B, len(v)))] for v in strata.values()], axis=1))
    lo, hi = np.quantile(bs, [.025, .975])
    return tmean(allx), lo, hi, len(allx), int((allx > 1e-9).sum()), int((allx < -1e-9).sum())


def cell(t, bold_min=14):
    est, lo, hi, n, w, l = t
    if not n:
        return "n/a", "", ""
    sig = "**" if (lo > 0 or hi < 0) and n >= bold_min else ""
    return f"{sig}{fs(est)} [{fs(lo)}, {fs(hi)}]{sig}", str(n), f"{w} / {l}"


@functools.lru_cache(maxsize=None)
def _ok(p):
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "summary" not in d or not d.get("concepts") or "ref" not in next(iter(d["concepts"].values())):
        return None
    return d


def main_files(arm, ds):
    I = ROOT / arm / "interventions"
    if ds == "atlas":
        out = []
        for s in (0, 1, 2):
            for t in ("_conf", ""):
                d = _ok(I / "atlas_classes" / f"range_intervention_atlas_s{s}_dprime{t}.json")
                if d is not None:
                    out.append((f"s{s}", d)); break
        return out
    for t in ("_conf", ""):
        d = _ok(I / f"range_intervention_{ds}_dprime{t}.json")
        if d is not None:
            return [("", d)]
    return []


def concepts(arm, ds):
    """[(concept name, record, meta)] from the main (α .5/1/2 + removal) files."""
    out = []
    for tag, d in main_files(arm, ds):
        names = d["meta"].get("classes")
        for c, rec in d["concepts"].items():
            nm = names[int(c)] if names and int(c) < len(names) else c
            if ds == "jigsaw2" and nm != "toxic":
                continue
            out.append((f"{tag}:{nm}" if tag else nm, rec, d["meta"]))
    return out


def arrays(rec, s, key):
    r = np.asarray(rec["ref"][s]["doc_conf"], float); e = np.asarray(rec[f"{s}_{key}"]["doc_conf"], float)
    return r, e, len(rec["eval_idx"]["tgt"])


def point(rec, s, key, meta):
    """(removed rel, removed abs, collateral rel, collateral abs, ppl rise %) for one edit, vs the space's own unedited run."""
    r, e, nt = arrays(rec, s, key)
    bt, at = r[:nt].mean(), e[:nt].mean()
    loss = np.maximum(0, r[nt:] - e[nt:])
    ppl0 = meta["base_ppl"] if s == "h" else meta["zrecon_ppl"]
    return dict(rem_rel=100 * (bt - at) / bt, rem_abs=100 * (bt - at), coll_rel=100 * loss.mean() / r[nt:].mean(),
                coll_abs=100 * loss.mean(), ppl=100 * rec[f"{s}_{key}"]["ppl_rise"] / ppl0)


# --------------------------------------------------------------------------------------------------------------------
print(f"# Control mechanism checks\n\nmatched-start tolerance |conf_z − conf_h| ≤ {args.tol}; {args.trim:.0%}-trimmed means "
      f"over concepts; bold = 95% CI excludes 0 with ≥ 14 concepts. Positive Δ = GeoAE larger.\n")

OPS = [("st_range_a0.5", "Range steering α 0.5"), ("st_global_a0.5", "Plain steering α 0.5"), ("rm_range_comp", "Range removal")]

# ---------------- T1 matched start ----------------
print("## T1. Same documents, same starting confidence (absolute points of gold-label probability)\n")
print("Only documents whose gold-label confidence the splice moved by at most the tolerance. Target = drop on the "
      "concept's own documents; collateral = loss on other-class documents; selectivity = target − collateral. "
      "'All docs' repeats the absolute comparison without the matching; 'rel (headline)' is the α-0.5 table metric.\n")
print("| Model | Pool | Edit | Target docs matched | Start conf: base / GeoAE | Target drop: base → GeoAE | Collateral: base → GeoAE "
      "| Δ selectivity, matched [95% CI] | n | better / worse | Δ selectivity, all docs (abs) | Δ selectivity, rel (headline) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
t3rows = {}
for arm, mname in ARMS:
    data = {ds: concepts(arm, ds) for ds in DSN}
    for pool, group in (("Curated", CURATED), ("Atlas", ["atlas"])):
        group = [g for g in group if data[g]]
        if not group:
            continue
        for key, ename in OPS:
            dm, dall, drel = {g: [] for g in group}, {g: [] for g in group}, {g: [] for g in group}
            tot_t = tot_m = 0; st_h, st_z, th, tz, ch, cz = [], [], [], [], [], []
            for g in group:
                for nm, rec, meta in data[g]:
                    if f"z_{key}" not in rec:
                        continue
                    rh, eh, nt = arrays(rec, "h", key); rz, ez, _ = arrays(rec, "z", key)
                    m = np.abs(rz - rh) <= args.tol
                    tot_t += nt; tot_m += int(m[:nt].sum())
                    # all-doc absolute and headline relative selectivity differences
                    ph, pz = point(rec, "h", key, meta), point(rec, "z", key, meta)
                    dall[g].append((pz["rem_abs"] - pz["coll_abs"]) - (ph["rem_abs"] - ph["coll_abs"]))
                    drel[g].append((pz["rem_rel"] - pz["coll_rel"]) - (ph["rem_rel"] - ph["coll_rel"]))
                    t3rows.setdefault((arm, key), []).append((g, nm, rec, meta, ph, pz))
                    tm, cm = m[:nt], m[nt:]
                    if tm.sum() < args.min_docs or cm.sum() < args.min_docs:
                        continue
                    tdh, tdz = 100 * (rh - eh)[:nt][tm], 100 * (rz - ez)[:nt][tm]
                    clh, clz = 100 * np.maximum(0, rh - eh)[nt:][cm], 100 * np.maximum(0, rz - ez)[nt:][cm]
                    dm[g].append((tdz.mean() - clz.mean()) - (tdh.mean() - clh.mean()))
                    st_h.append(100 * rh[:nt][tm].mean()); st_z.append(100 * rz[:nt][tm].mean())
                    th.append(tdh.mean()); tz.append(tdz.mean()); ch.append(clh.mean()); cz.append(clz.mean())
            c1, n1, wl = cell(boot(dm))
            print(f"| {mname} | {pool} | {ename} | {tot_m}/{tot_t} ({100 * tot_m / max(tot_t, 1):.0f}%) | {f1(tmean(st_h))} / {f1(tmean(st_z))} "
                  f"| {f1(tmean(th))} → {f1(tmean(tz))} | {f1(tmean(ch))} → {f1(tmean(cz))} | {c1} | {n1} | {wl} "
                  f"| {cell(boot(dall))[0]} | {cell(boot(drel))[0]} |")

# ---------------- T2 matched damage on the dense grid ----------------
print("\n## T2. Equal damage on the dense grid (α 0, 0.1–0.5, 1, 2; DBpedia-14 + Bias in Bios + AG News)\n")
print("Per concept, each space's (collateral, removal, perplexity) curve over α, read at fixed levels by linear "
      "interpolation (first crossing; concepts that never reach a level are dropped). rel = % of the space's own starting "
      "confidence (favours the less confident GeoAE splice); abs = probability points (penalises it: less to remove). "
      "Perplexity = % rise over the space's own unedited perplexity on Wikipedia text.\n")


def curve(rec_small, rec_main, s, mode, meta_s, meta_m):
    pts = [dict(rem_rel=0., rem_abs=0., coll_rel=0., coll_abs=0., ppl=0.)]
    for a in ("0.1", "0.2", "0.3", "0.4", "0.5"):
        if f"{s}_st_{mode}_a{a}" in rec_small:
            pts.append(point(rec_small, s, f"st_{mode}_a{a}", meta_s))
    for a in ("1.0", "2.0"):
        if rec_main is not None and f"{s}_st_{mode}_a{a}" in rec_main:
            pts.append(point(rec_main, s, f"st_{mode}_a{a}", meta_m))
    return pts


def y_at(pts, xk, yk, x0):
    for p, q in zip(pts, pts[1:]):
        xa, xb, ya, yb = p[xk], q[xk], p[yk], q[yk]
        if xa <= x0 <= xb and xb > xa:
            return ya + (yb - ya) * (x0 - xa) / (xb - xa)
    return np.nan


LEVELS = [("removed (rel) at collateral (rel) = {}%", "coll_rel", "rem_rel", [1, 2, 5], +1),
          ("removed (abs) at collateral (abs) = {} pts", "coll_abs", "rem_abs", [0.5, 1, 3], +1),
          ("collateral (rel) at removed (rel) = {}%", "rem_rel", "coll_rel", [10, 20, 30], -1),
          ("removed (rel) at perplexity +{}%", "ppl", "rem_rel", [0.5, 1, 2], +1),
          ("removed (abs) at perplexity +{}%", "ppl", "rem_abs", [0.5, 1, 2], +1)]
print("| Model | Edit | Compared at | Base | GeoAE | Δ GeoAE − base [95% CI] | concepts | better / worse | verdict |")
print("|---|---|---|---|---|---|---|---|---|")
for arm, mname in ARMS:
    for mode, ename in (("range", "Range steering"), ("global", "Plain steering")):
        for lab, xk, yk, levels, sign in LEVELS:
            for lv in levels:
                diffs, H, Z = {}, [], []
                for ds in DENSE:
                    ps = ROOT / arm / "interventions" / f"range_intervention_{ds}_dprime_conf_small.json"
                    dsm = _ok(ps)
                    dmain = main_files(arm, ds)
                    if dsm is None:
                        continue
                    dmain = dmain[0][1] if dmain else None
                    for c, rec in dsm["concepts"].items():
                        rm = dmain["concepts"].get(c) if dmain else None
                        yh = y_at(curve(rec, rm, "h", mode, dsm["meta"], dmain["meta"] if dmain else None), xk, yk, lv)
                        yz = y_at(curve(rec, rm, "z", mode, dsm["meta"], dmain["meta"] if dmain else None), xk, yk, lv)
                        if np.isfinite(yh) and np.isfinite(yz):
                            diffs.setdefault(ds, []).append(yz - yh); H.append(yh); Z.append(yz)
                if not diffs:
                    continue
                t = boot(diffs)
                c1, n1, wl = cell(t)
                good = (t[1] > 0) if sign > 0 else (t[2] < 0)
                bad = (t[2] < 0) if sign > 0 else (t[1] > 0)
                verdict = "GeoAE better" if good else ("base better" if bad else "tie")
                print(f"| {mname} | {ename} | {lab.format(lv)} | {f1(tmean(H))} | {f1(tmean(Z))} | {c1} | {n1} | {wl} | {verdict} |")

# ---------------- T3 per concept ----------------
print("\n## T3. Per concept, range steering α 0.5 (relative selectivity, the headline metric)\n")


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 8:
        return np.nan, np.nan, len(x)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    r = np.corrcoef(rx, ry)[0, 1]
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(4000)])
    return r, (np.abs(perm) >= abs(r)).mean(), len(x)


def collateral_sink(rec, s, key, meta):
    """Class that absorbs most flipped other-class predictions under the edit, with its count."""
    names = meta.get("classes") or []
    nt = len(rec["eval_idx"]["tgt"])
    before = rec["ref"][s]["doc_pred"][nt:]; after = rec[f"{s}_{key}"]["doc_pred"][nt:]
    flips = [a for b, a in zip(before, after) if a != b]
    if not flips:
        return "—"
    vals, cnt = np.unique(flips, return_counts=True)
    i = int(cnt.argmax())
    return f"{vals[i]} ({cnt[i]}/{len(before)})"


for arm, mname in ARMS:
    rows = t3rows.get((arm, "st_range_a0.5"), [])
    rows = [r for r in rows if r[0] != "jigsaw2"]
    if not rows:
        continue
    recs = []
    for g, nm, rec, meta, ph, pz in rows:
        hs, zs = ph["rem_rel"] - ph["coll_rel"], pz["rem_rel"] - pz["coll_rel"]
        nt = len(rec["eval_idx"]["tgt"])
        sh = 100 * np.mean(rec["ref"]["h"]["doc_conf"][:nt]); sz = 100 * np.mean(rec["ref"]["z"]["doc_conf"][:nt])
        fh, fz = rec["fire"]["h"], rec["fire"]["z"]
        recs.append(dict(ds=g, nm=nm, hs=hs, zs=zs, d=zs - hs, gap=sh - sz, sh=sh, sz=sz,
                         fire=(fz["tgt"] - fz["comp"]) - (fh["tgt"] - fh["comp"]), dppl=pz["ppl"] - ph["ppl"],
                         drem=pz["rem_rel"] - ph["rem_rel"], dcoll=pz["coll_rel"] - ph["coll_rel"],
                         sink_h=collateral_sink(rec, "h", "st_range_a0.5", meta), sink_z=collateral_sink(rec, "z", "st_range_a0.5", meta)))
    # dataset-demeaned Spearman of the per-concept gap with each candidate
    def dm(k):
        out = np.array([r[k] for r in recs], float)
        for g in set(r["ds"] for r in recs):
            idx = [i for i, r in enumerate(recs) if r["ds"] == g]
            out[idx] -= out[idx].mean()
        return out
    D = dm("d")
    print(f"### {mname}: {len(recs)} concepts\n")
    print("| Candidate explanation (per concept, dataset-demeaned) | Spearman with Δ selectivity | permutation p | n |")
    print("|---|---|---|---|")
    for k, lab in (("gap", "starting-confidence loss from the splice (base − GeoAE, pts)  [E1]"),
                   ("dppl", "extra perplexity rise of the GeoAE edit (pts)  [E2]"),
                   ("drem", "extra target removal (rel)  [E2: bigger edit]"),
                   ("dcoll", "extra collateral (rel)"),
                   ("fire", "range-gate selectivity, GeoAE − base (fire on target − fire on others)  [E3]"),
                   ("hs", "base selectivity (is GeoAE worse where base is already good?)")):
        r, p, n = spearman(dm(k), D)
        print(f"| {lab} | {format(r, '+.2f').replace('-', '−')} | {p:.3f} | {n} |")
    srt = sorted(recs, key=lambda r: r["d"])
    good_base = [r for r in srt if r["hs"] >= np.median([q["hs"] for q in recs])]
    print("\nBase good, GeoAE worse (base selectivity at or above the median, largest GeoAE deficits):\n")
    print("| Dataset | Concept | Selectivity: base → GeoAE | Δ | Start conf: base / GeoAE | Removed Δ (rel) | Collateral Δ (rel) | Collateral goes to: base / GeoAE |")
    print("|---|---|---|---|---|---|---|---|")
    for r in [q for q in good_base if q["d"] < 0][:8]:
        print(f"| {DSN[r['ds']]} | {r['nm']} | {f1(r['hs'])} → {f1(r['zs'])} | {fs(r['d'])} | {f1(r['sh'])} / {f1(r['sz'])} "
              f"| {fs(r['drem'])} | {fs(r['dcoll'])} | {r['sink_h']} / {r['sink_z']} |")
    print("\nLargest GeoAE gains:\n")
    print("| Dataset | Concept | Selectivity: base → GeoAE | Δ | Start conf: base / GeoAE | Removed Δ (rel) | Collateral Δ (rel) | Collateral goes to: base / GeoAE |")
    print("|---|---|---|---|---|---|---|---|")
    for r in srt[::-1][:8]:
        print(f"| {DSN[r['ds']]} | {r['nm']} | {f1(r['hs'])} → {f1(r['zs'])} | {fs(r['d'])} | {f1(r['sh'])} / {f1(r['sz'])} "
              f"| {fs(r['drem'])} | {fs(r['dcoll'])} | {r['sink_h']} / {r['sink_z']} |")
    print()

# ---------------- T2b equal damage on matched-start documents (subtle regime, α 0–0.5) ----------------
print("\n## T2b. Equal damage, matched-start documents only (subtle regime α 0, 0.1–0.5; absolute points)\n")
print("T2 restricted to the documents whose starting confidence the splice moved by at most the tolerance (per concept "
      "≥ min_docs target and other-class documents), so neither space starts from a weaker model and no division is "
      "needed. Curves stop at α 0.5 (one run, one document set).\n")
print("| Model | Edit | Compared at | Base | GeoAE | Δ GeoAE − base [95% CI] | concepts | better / worse | verdict |")
print("|---|---|---|---|---|---|---|---|---|")


def mcurve(rec, s, mode, meta, m, nt):
    pts = [dict(rem=0., coll=0., ppl=0.)]
    r = np.asarray(rec["ref"][s]["doc_conf"], float)
    ppl0 = meta["base_ppl"] if s == "h" else meta["zrecon_ppl"]
    for a in ("0.1", "0.2", "0.3", "0.4", "0.5"):
        k = f"{s}_st_{mode}_a{a}"
        if k not in rec:
            continue
        e = np.asarray(rec[k]["doc_conf"], float)
        pts.append(dict(rem=100 * (r - e)[:nt][m[:nt]].mean(), coll=100 * np.maximum(0, r - e)[nt:][m[nt:]].mean(),
                        ppl=100 * rec[k]["ppl_rise"] / ppl0))
    return pts


LEVELS_M = [("removed at collateral = {} pts", "coll", "rem", [0.5, 1, 2], +1),
            ("collateral at removed = {} pts", "rem", "coll", [5, 10, 15], -1),
            ("removed at perplexity +{}%", "ppl", "rem", [0.25, 0.5, 1], +1)]
for arm, mname in ARMS:
    for mode, ename in (("range", "Range steering"), ("global", "Plain steering")):
        for lab, xk, yk, levels, sign in LEVELS_M:
            for lv in levels:
                diffs, H, Z = {}, [], []
                for ds in DENSE:
                    dsm = _ok(ROOT / arm / "interventions" / f"range_intervention_{ds}_dprime_conf_small.json")
                    if dsm is None:
                        continue
                    for c, rec in dsm["concepts"].items():
                        nt = len(rec["eval_idx"]["tgt"])
                        rh = np.asarray(rec["ref"]["h"]["doc_conf"], float); rz = np.asarray(rec["ref"]["z"]["doc_conf"], float)
                        m = np.abs(rz - rh) <= args.tol
                        if m[:nt].sum() < args.min_docs or m[nt:].sum() < args.min_docs:
                            continue
                        yh = y_at(mcurve(rec, "h", mode, dsm["meta"], m, nt), xk, yk, lv)
                        yz = y_at(mcurve(rec, "z", mode, dsm["meta"], m, nt), xk, yk, lv)
                        if np.isfinite(yh) and np.isfinite(yz):
                            diffs.setdefault(ds, []).append(yz - yh); H.append(yh); Z.append(yz)
                if not diffs:
                    continue
                t = boot(diffs)
                c1, n1, wl = cell(t)
                good = (t[1] > 0) if sign > 0 else (t[2] < 0)
                bad = (t[2] < 0) if sign > 0 else (t[1] > 0)
                print(f"| {mname} | {ename} | {lab.format(lv)} | {f1(tmean(H))} | {f1(tmean(Z))} | {c1} | {n1} | {wl} "
                      f"| {'GeoAE better' if good else ('base better' if bad else 'tie')} |")
