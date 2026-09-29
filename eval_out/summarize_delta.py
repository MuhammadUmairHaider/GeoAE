#!/usr/bin/env python3
"""Base vs bypass decision tables for the delta round-2 arms.

Reads the browsable tree results/delta/<arm>/{evals,interventions,steering,
concept}/*.json (default arms: d768, d3072, d6144, d12288, d6144k4000, d12288k4000; --root
overrides the repo root). For any file not yet migrated to that layout, falls
back to its pre-migration location (eval_out/delta/<arm>/*.json or
results/delta/<arm>/*.json at top level — see scripts/delta/organize_delta_results.sh),
so this keeps working whether or not the migration has run yet on a given tree.
Prints one base-vs-bypass table per headline per arm, in plain units, and writes
results/delta/summary.json (eval_out/delta/summary.json is kept as a copy for
anything still reading the old path). Every read is defensive: a missing or
partial file prints "-" for that cell rather than raising, so this can run
mid-sweep.

Field names / headline choices are ported from the schemas actually written by
the committed tools, cross-checked against the old-box outputs already in this
tree and the summarizer scripts that read them:
  eval_out/summarize_tokbias.py              token_structure / probe / mmlu / cq
  eval_out/summarize_tokbias_interventions.py steer / range_db14 / range_bios / number head-to-head
  eval_out/summarize_tokbias_hb.py            hb (matched-base) framing
  eval_out/summarize_tokbias_d12288.py        3-arm layout this generalises to 5
  eval_out/summarize_tokbias_controls.py      base_tokmean handling
  eval_out/summarize_width_series.py          matched_collateral() (erasure-matched
                                               range-intervention comparison), reused
                                               here rather than re-derived
None of those five are imported (summarize_tokbias_interventions.py and
summarize_width_series.py hardcode ROOT/ARMS at *module* scope from fixed
eval_out/*.json paths, which is exactly the trap this script exists to avoid),
so the read logic is re-implemented against the new eval_out/delta/<arm>/ /
results/delta/<arm>/ layout.

ARM-NAME ALIASING. Every reader accepts SEVERAL names for the same role so this
script is meaningful against both the new per-arm delta layout (canonical names
base / base_tokmean / bypass) and ad-hoc test trees built by copying old-box
files (which used km_new / km_tokmean / d6144_new / tokbias / bypass6k, etc.).
See ALIASES below.

    scripts/delta/py eval_out/summarize_delta.py
    scripts/delta/py eval_out/summarize_delta.py --root /some/other/tree --arms d6144
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARMS_DEFAULT = ["d768", "d3072", "d6144", "d12288", "d6144k4000", "d12288k4000"]

ALIASES = {
    "base": ["base", "km_new", "km"],
    "base_tokmean": ["base_tokmean", "km_tokmean", "km_tok64"],
    "bypass": ["bypass", "tokbias", "bypass6k", "bypass12k"],
}
SEQ_RUNGS = ["sentiment", "sentiment_long", "subjectivity", "formality",
             "language", "domain", "topic4", "topic14", "topic20"]
POS_RUNGS = ["pos_coarse", "pos_fine"]


# --------------------------------------------------------------------------- #
# generic helpers — every one of these tolerates missing/malformed input
# --------------------------------------------------------------------------- #
def load(p: Path):
    if not p or not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def resolve(root: Path, arm: str, group: str, name: str, old_group: str) -> Path:
    """New results/delta/<arm>/<group>/<name>, falling back to the pre-migration
    path when the new file doesn't exist yet (the migration may not have run, or
    a still-running old-version job hasn't produced it there). old_group is
    'eval_out' (was under eval_out/delta/<arm>/<name>) or 'results' (was under
    results/delta/<arm>/<name> directly) — see scripts/delta/organize_delta_results.sh
    for the exact old->new mapping this mirrors."""
    new = root / "results" / "delta" / arm / group / name
    if new.exists():
        return new
    old = root / old_group / "delta" / arm / name if old_group == "eval_out" \
        else root / "results" / "delta" / arm / name
    return old if old.exists() else new


def pick(d, role):
    """d.get(<one of ALIASES[role]>) — first alias present, else None."""
    if not isinstance(d, dict):
        return None
    for name in ALIASES[role]:
        if name in d:
            return d[name]
    return None


def find_prefixed(d, role, suffixes):
    """For 'name:variant'-keyed dicts (cluster_steering.json): the value for the
    first key 'alias:suffix' matching this role, trying each suffix in order."""
    if not isinstance(d, dict):
        return None, None
    for alias in ALIASES[role]:
        for suf in suffixes:
            k = f"{alias}:{suf}"
            if k in d:
                return k, d[k]
    return None, None


def dash(v, fmt="{:.3f}") -> str:
    if v is None:
        return "–"
    try:
        f = float(v)
        if f != f:  # NaN
            return "–"
        return fmt.format(f)
    except (TypeError, ValueError):
        return "–"


def header(t: str) -> None:
    print("\n" + "=" * 92 + f"\n{t}\n" + "=" * 92)


def row(label: str, cells, fmt="{:.3f}", w=24) -> None:
    print(f"  {label:<{w}}" + "".join(f"{dash(c, fmt):>14}" for c in cells))


def cols(names, w=24) -> None:
    print(f"  {'':<{w}}" + "".join(f"{n:>14}" for n in names))


# --------------------------------------------------------------------------- #
# matched-erasure collateral (ported from summarize_width_series.matched_collateral)
# --------------------------------------------------------------------------- #
def matched_collateral(concepts: dict, mode: str, alphas: list[float]):
    """(matched_erasure_pct, comp_drop_h_pct, comp_drop_z_pct, n_concepts) or None.

    Per concept, interpolates each space's (tgt_drop, comp_drop) curve over the
    alpha grid to the largest tgt_drop BOTH spaces reach (E), then reports the
    mean collateral (comp_drop) each space pays at that matched target-removal
    level. Concepts with < 2 usable alphas in either space, or E <= 0, are
    dropped.
    """
    es, chs, czs = [], [], []
    for c in concepts.values():
        curves = {}
        for space in ("h", "z"):
            pts = [(c[k]["tgt_drop"], c[k]["comp_drop"])
                   for al in alphas if (k := f"{space}_st_{mode}_a{al}") in c]
            if len(pts) < 2:
                curves = None
                break
            pts.sort()
            curves[space] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
        if not curves:
            continue
        e = min(curves["h"][0].max(), curves["z"][0].max())
        if e <= 0:
            continue
        es.append(e)
        chs.append(float(np.interp(e, *curves["h"])))
        czs.append(float(np.interp(e, *curves["z"])))
    if not es:
        return None
    return (100 * float(np.mean(es)), 100 * float(np.mean(chs)), 100 * float(np.mean(czs)), len(es))


# --------------------------------------------------------------------------- #
# per-headline sections
# --------------------------------------------------------------------------- #
def section_token_structure(root: Path, arms: list[str], out: dict) -> None:
    header("RECONSTRUCTION FVE + TOKEN STRUCTURE (held-out dump rows)")
    print("  FVE = 1 - fvu (fraction of variance explained by the reconstruction, higher better).")
    print("  trivial_cur_80 = share of clusters where >=80% of members share the SAME current token.")
    cols(arms)
    fve, triv = [], []
    for a in arms:
        d = load(resolve(root, a, "evals", "token_structure.json", "eval_out"))
        b, byp = pick(d, "base"), pick(d, "bypass")
        # base (k-means, no encoder) legitimately has no fvu — only the AE reconstructs.
        fve_b = None if not b or b.get("fvu") is None else 1 - b["fvu"]
        fve_y = None if not byp or byp.get("fvu") is None else 1 - byp["fvu"]
        fve.append((fve_b, fve_y))
        triv_b = None if not b else b.get("trivial_cur_80")
        triv_y = None if not byp else byp.get("trivial_cur_80")
        triv.append((triv_b, triv_y))
        out.setdefault(a, {})["token_structure"] = {
            "fve_base": fve_b, "fve_bypass": fve_y,
            "trivial_cur_80_base": triv_b, "trivial_cur_80_bypass": triv_y,
        }
    row("FVE base",         [v[0] for v in fve])
    row("FVE bypass",       [v[1] for v in fve])
    row("trivial_cur_80 base",   [v[0] for v in triv])
    row("trivial_cur_80 bypass", [v[1] for v in triv])


def section_probe(root: Path, arms: list[str], out: dict) -> None:
    header("CONCEPT PROBE — chance-corrected NMI (cache_ids/); base vs base_tokmean vs bypass")
    for a in arms:
        d = load(resolve(root, a, "evals", "probe.json", "eval_out"))
        if not d:
            print(f"  {a}: – (missing)")
            out.setdefault(a, {})["probe"] = None
            continue
        print(f"  {a}:")
        seq_vals = {"base": [], "base_tokmean": [], "bypass": []}
        topic14 = {}
        pos_vals = {"base": [], "base_tokmean": [], "bypass": []}
        for rung, v in d.items():
            arms_d = v.get("arms", {})
            b, bt, byp = pick(arms_d, "base"), pick(arms_d, "base_tokmean"), pick(arms_d, "bypass")
            if rung in SEQ_RUNGS:
                for role, cell in (("base", b), ("base_tokmean", bt), ("bypass", byp)):
                    if cell is not None:
                        seq_vals[role].append(cell.get("cnmi"))
                if rung == "topic14":
                    topic14 = {"base": b.get("cnmi") if b else None,
                                "base_tokmean": bt.get("cnmi") if bt else None,
                                "bypass": byp.get("cnmi") if byp else None}
            if rung in POS_RUNGS:
                for role, cell in (("base", b), ("base_tokmean", bt), ("bypass", byp)):
                    if cell is not None:
                        pos_vals[role].append(cell.get("cnmi"))
        seq_mean = {k: (float(np.mean(v)) if v else None) for k, v in seq_vals.items()}
        pos_mean = {k: (float(np.mean(v)) if v else None) for k, v in pos_vals.items()}
        cols_ = ["base", "base_tokmean", "bypass"]
        row("  topic14 cNMI", [topic14.get(k) for k in cols_])
        row("  mean sequence-rung cNMI", [seq_mean.get(k) for k in cols_])
        row("  mean POS cNMI", [pos_mean.get(k) for k in cols_])
        out.setdefault(a, {})["probe"] = {"topic14": topic14, "seq_mean": seq_mean, "pos_mean": pos_mean}


def section_mmlu(root: Path, arms: list[str], out: dict) -> None:
    header("MMLU UNDER THE AE SPLICE (n=2000)")
    cols(arms)
    base_acc, recon_acc, delta = [], [], []
    for a in arms:
        d = load(resolve(root, a, "evals", "mmlu.json", "eval_out"))
        m = (d or {}).get("meta", {})
        base_acc.append(m.get("base_acc"))
        recon_acc.append(m.get("recon_acc"))
        delta.append(m.get("delta"))
        out.setdefault(a, {})["mmlu"] = m or None
    row("base acc", base_acc)
    row("recon acc (bypass spliced in)", recon_acc)
    row("delta (recon - base)", delta)


def section_cq(root: Path, arms: list[str], out: dict) -> None:
    header("CLUSTERING QUALITY (1M rows, seed 0) — silhouette-family metrics; dunn NOT reported")
    print("  NOTE: clustering_quality's dunn_index subsamples with an UNSEEDED RNG and is not")
    print("  comparable across runs (see docs/notes/cq-dunn-and-steer-sign-traps.md); omitted here.")
    keys = [("silhouette", "silhouette (up)"), ("cluster_balance", "balance H (up)"),
            ("effective_k", "effective K (up)"), ("effective_rank", "effective rank (up)")]
    for a in arms:
        d = load(resolve(root, a, "evals", "cq.json", "eval_out"))
        b, byp = pick(d, "base"), pick(d, "bypass")
        print(f"  {a}:")
        cols(["base", "bypass"])
        cell = {}
        for k, label in keys:
            vb = (b or {}).get(k)
            vy = (byp or {}).get(k)
            row(f"  {label}", [vb, vy])
            cell[k] = {"base": vb, "bypass": vy}
        out.setdefault(a, {})["cq"] = cell


def section_range(root: Path, arms: list[str], key: str, title: str, out: dict) -> None:
    header(title)
    print("  matched erasure: h and z interpolated to the largest tgt_drop BOTH reach; collateral")
    print("  (comp_drop) reported there. NEGATIVE collateral delta (z - h) = AE does less damage.")
    for a in arms:
        d = load(resolve(root, a, "interventions", f"{key}.json", "results"))
        if not d or "summary" not in d or "concepts" not in d:
            print(f"  {a}: – (missing)")
            out.setdefault(a, {})[key] = None
            continue
        meta = d.get("meta", {})
        steer_modes = meta.get("steer_modes", ["global", "salient", "range", "transport"])
        alphas = meta.get("alphas", [0.5, 1.0, 2.0])
        print(f"  {a}:")
        arm_out = {}
        for sm in steer_modes:
            r = matched_collateral(d["concepts"], sm, alphas)
            if r is None:
                print(f"    st_{sm:<10} –")
                arm_out[sm] = None
                continue
            e, ch, cz, n = r
            print(f"    st_{sm:<10} target removed {e:5.1f}/100  collateral h {ch:+6.2f}/100  "
                  f"z {cz:+6.2f}/100  Δ(z-h) {cz - ch:+6.2f}/100  (n={n})")
            arm_out[sm] = {"target_removed_pct": e, "collateral_h_pct": ch, "collateral_z_pct": cz, "n": n}
        # The headline form used in docs/token_bypass_story.md: each operator at its own
        # setting, target removed / collateral per 100, base residual (h) -> bypass latent (z).
        for op in ("rm_range_comp", "st_range_a1.0"):
            s = d["summary"].get(op)
            if not s:
                continue
            print(f"    {op:<14} own setting: removed/collateral h {100 * s['h_tgt_drop']:4.0f}/{100 * s['h_comp_drop']:<3.0f}"
                  f" -> z {100 * s['z_tgt_drop']:4.0f}/{100 * s['z_comp_drop']:<3.0f}  selectivity z-h {100 * s['z_minus_h']:+5.1f}"
                  f"  (z wins {s['z_wins']}/{s['n']})")
            arm_out[op] = {k: s.get(k) for k in ("h_tgt_drop", "h_comp_drop", "z_tgt_drop", "z_comp_drop",
                                                 "z_minus_h", "z_minus_h_se", "z_wins", "n")}
        out.setdefault(a, {})[key] = arm_out


def section_number(root: Path, arms: list[str], out: dict) -> None:
    header("NUMBER CONTROL — strict target-flip rate, range mode, alpha=1 (z minus h)")
    cols(arms)
    vals = []
    for a in arms:
        rev = load(resolve(root, a, "interventions", "number_review.json", "eval_out"))
        comp = ((rev or {}).get("paired_alpha1_comparisons", {}) or {}).get("range:z_minus_h")
        v = (comp or {}).get("difference")
        ci = (comp or {}).get("noun_bootstrap_95")
        vals.append(v)
        out.setdefault(a, {})["number"] = {"z_minus_h_range_a1": v, "ci95": ci}
    row("Δ strict flip (z - h)", vals)


def section_cluster_steering(root: Path, arms: list[str], out: dict) -> None:
    header("NEXT-TOKEN CLUSTER TRANSFER AT MATCHED DISRUPTION")
    for a in arms:
        d = load(resolve(root, a, "steering", "cluster_steering.json", "eval_out"))
        if not d or "matched_disrupt" not in d:
            print(f"  {a}: – (missing)")
            out.setdefault(a, {})["cluster_steering"] = None
            continue
        md = d["matched_disrupt"]
        levels = sorted(next(iter(md.values())).keys(), key=float) if md else []
        print(f"  {a}:  (transfer = fraction of steered rows that land in the target cluster's signature)")
        cols([f"disrupt={lv}" for lv in levels], w=16)
        arm_out = {}
        for role in ("base", "bypass"):
            k_km, v_km = find_prefixed(md, role, ["km"])
            k_ae, v_ae = find_prefixed(md, role, ["ae"])
            key_found, vals_d = (k_ae, v_ae) if v_ae is not None else (k_km, v_km)
            def _transfer(lv):
                t = ((vals_d or {}).get(lv, {}) or {}).get("transfer")
                # transfer is {"est","lo","hi"} in the current tool; tolerate a bare
                # float too, in case an older/other file shape is read.
                return t.get("est") if isinstance(t, dict) else t
            cells = [_transfer(lv) for lv in levels]
            row(f"  {role} ({key_found or '–'})", cells, w=16)
            arm_out[role] = {"key": key_found, "transfer_by_disrupt": dict(zip(levels, cells))}
        out.setdefault(a, {})["cluster_steering"] = arm_out


def _best_alpha_usable(net_fluent: dict, arm_name: str):
    """cluster_steer_judge.py's res['net_fluent'][alpha][arm]['usable'] -> (best_alpha, usable_pct)."""
    if not net_fluent:
        return None, None
    best_a, best_u = None, None
    for a_str, by_arm in net_fluent.items():
        cell = by_arm.get(arm_name)
        if not cell:
            continue
        u = cell.get("usable")
        if u is None or u != u:
            continue
        if best_u is None or u > best_u:
            best_u, best_a = u, a_str
    return best_a, (None if best_u is None else 100 * best_u)


def section_cluster_steer_judge(root: Path, arms: list[str], out: dict) -> None:
    header("CLUSTER STEERING — usable (fluent, on-target, net of unsteered) generations per 100, best alpha")
    cols(arms)
    base_vals, byp_vals = [], []
    for a in arms:
        d = load(resolve(root, a, "steering", "cluster_steer_judge.json", "eval_out"))
        net = (d or {}).get("net_fluent")
        ba, bu = _best_alpha_usable(net, "base") if net else (None, None)
        ya, yu = _best_alpha_usable(net, "bypass") if net else (None, None)
        base_vals.append(bu)
        byp_vals.append(yu)
        out.setdefault(a, {})["cluster_steer_judge"] = {
            "base_best_alpha": ba, "base_usable_per_100": bu,
            "bypass_best_alpha": ya, "bypass_usable_per_100": yu,
        }
    row("base usable/100 (best alpha)",   base_vals, "{:.1f}")
    row("bypass usable/100 (best alpha)", byp_vals, "{:.1f}")


def section_concept_judge(root: Path, arms: list[str], out: dict) -> None:
    header("NAMED-CONCEPT STEERING — usable per 100 at best alpha, per handle (pooled db14+biasbios)")
    for a in arms:
        d = load(resolve(root, a, "concept", "concept_steer_judge.json", "eval_out"))
        if not d:
            print(f"  {a}: – (missing)")
            out.setdefault(a, {})["concept_judge"] = None
            continue
        best_alpha = d.get("best_alpha", {})
        pooled = (d.get("scopes", {}).get("pooled", {}) or {}).get("per_alpha", {})
        handles = sorted(set(best_alpha) | set(pooled))
        print(f"  {a}:")
        arm_out = {}
        for h in handles:
            a_str = best_alpha.get(h)
            cell = (pooled.get(h, {}) or {}).get(a_str, {}) if a_str else {}
            usable = cell.get("usable")
            u = None if not usable else 100 * usable[0]
            print(f"    {h:<14} best alpha {a_str or '–':<6} usable/100 {dash(u, '{:.1f}')}")
            arm_out[h] = {"best_alpha": a_str, "usable_per_100": u}
        out.setdefault(a, {})["concept_judge"] = arm_out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--arms", default=",".join(ARMS_DEFAULT), help="comma list")
    ap.add_argument("--out", default=None, help="default: <root>/results/delta/summary.json")
    args = ap.parse_args()
    root = Path(args.root)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_path = Path(args.out) if args.out else root / "results/delta/summary.json"

    print("=" * 92)
    print("  GeoAE delta round 2 — base vs bypass, plain units")
    print("=" * 92)

    out: dict = {}
    section_token_structure(root, arms, out)
    section_probe(root, arms, out)
    section_mmlu(root, arms, out)
    section_cq(root, arms, out)
    section_range(root, arms, "range_intervention_db14_dprime", "DB14 RANGE INTERVENTIONS", out)
    section_range(root, arms, "range_intervention_biasbios_dprime", "BIAS_IN_BIOS RANGE INTERVENTIONS", out)
    section_number(root, arms, out)
    section_cluster_steering(root, arms, out)
    section_cluster_steer_judge(root, arms, out)
    section_concept_judge(root, arms, out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=lambda o: None))
    print(f"\n[summary] wrote {out_path}")


if __name__ == "__main__":
    main()
