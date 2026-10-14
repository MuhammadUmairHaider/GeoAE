"""Side-by-side summary of the L27 K=2000 latent-width series: d3072 / d6144 / d12288.

Reads whatever eval JSONs exist and prints one table per family, so it can be run
mid-suite. Every number is recomputed here from the per-concept records, so the three
arms are always compared the same way — figures may differ slightly from earlier
ad-hoc analyses of the d3072/d6144 arms.

Sign conventions:
  MMLU delta        recon_acc - base_acc, higher (less negative) is better.
  z - h selectivity sign depends on the tool. range_intervention_compare uses
                    tgt_drop - comp_drop (HIGHER better); steering_concept_compare uses
                    tgt_acc_delta - comp_acc_delta, both negative (MORE NEGATIVE better).
                    Each table below states which convention it is in.
  collateral margin comp_drop(z) - comp_drop(h) at MATCHED erasure, so NEGATIVE means
                    the AE does less collateral damage for the same amount of erasure.
                    Matched erasure: per concept, E = min over spaces of the largest
                    tgt_drop that space reaches over the alpha grid; comp_drop for each
                    space is linearly interpolated to E over its own (tgt_drop, comp_drop)
                    curve. Concepts where either space has fewer than 2 usable alphas,
                    or where E <= 0, are dropped (n is reported).

Sections: provenance, reconstruction cost, clustering quality, probes, steering,
range interventions (both datasets), per-concept reliability, effect vs spread, and the
base-quality moderation test.

Usage: uv run python eval_out/summarize_width_series.py [--with_valmse]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CKPT = "checkpoints/llama3.2-3B/layer27"

# Each arm: where its results live. None = that eval was never run for that arm.
ARMS: dict[str, dict[str, str | list[str] | None]] = {
    "d3072": {
        "ckpt":        f"{CKPT}/k2000_bnh_b32k_lam1_d3072_dpc/step_0014200.pt",
        "mmlu":        "eval_out/mmlu_d3072_50.json",
        "cq":          "eval_out/cq_d3072_50.json",
        "probe":       "eval_out/probe_d3072_50.json",
        "steer_db14":  "results/steer_db14_d3072_50.json",
        "range_db14":  "results/range_intervention_db14_d3072_50_dprime.json",
        "range_bios":  "results/range_intervention_biasbios_d3072_50_dprime.json",
    },
    "d6144": {
        "ckpt":        f"{CKPT}/k2000_bnh_b32k_lam1_d6144_dpc/step_0014200.pt",
        "mmlu":        "eval_out/mmlu_gelu50.json",
        "cq":          "eval_out/cq_d12288_50.json",     # falls back to cq_d3072_50.json
        "probe":       "eval_out/probe_d12288_50.json",  # falls back to probe_d3072_50.json
        # The d6144 steer/range JSONs on disk were run on best_val.pt (epoch 47), not
        # step_0014200.pt (epoch 50) — the provenance table flags that. Each entry is a
        # candidate list: run eval_out/run_d6144_ep50_evals.sh and the epoch-50 files
        # below are picked up automatically, making the whole series epoch-matched.
        "steer_db14":  ["results/steer_db14_d6144_50.json", "results/steer_db14_dpc.json"],
        "range_db14":  ["results/range_intervention_db14_d6144_50_dprime.json",
                        "results/range_intervention_db14_b32k_dpc_dprime.json"],
        "range_bios":  ["results/range_intervention_biasbios_d6144_50_dprime.json",
                        "results/range_intervention_biasbios_b32k_dpc_dprime.json"],
    },
    "d12288": {
        "ckpt":        f"{CKPT}/k2000_bnh_b32k_lam1_d12288_dpc/step_0014200.pt",
        "mmlu":        "eval_out/mmlu_d12288_50.json",
        "cq":          "eval_out/cq_d12288_50.json",
        "probe":       "eval_out/probe_d12288_50.json",
        "steer_db14":  "results/steer_db14_d12288_50.json",
        "range_db14":  "results/range_intervention_db14_d12288_50_dprime.json",
        "range_bios":  "results/range_intervention_biasbios_d12288_50_dprime.json",
    },
}
# For the multi-arm files, always prefer the newest run that holds ALL arms: silhouette and
# dunn_index subsample with an UNSEEDED np.random.choice, so numbers from two different runs
# are not comparable even on identical data (measured: d3072 dunn 0.134 in the 2-arm run vs
# 0.071 in the 3-arm run). PREFERRED is tried before the per-arm entry in ARMS.
PREFERRED = {"cq": "eval_out/cq_d12288_50.json", "probe": "eval_out/probe_d12288_50.json"}
FALLBACK = {"cq": "eval_out/cq_d3072_50.json", "probe": "eval_out/probe_d3072_50.json"}
NAMES = list(ARMS)


def load(arm: str, key: str) -> dict | None:
    """Load arm[key], falling back to the older file that also carries this arm."""
    entry = ARMS[arm].get(key)
    cands = [PREFERRED.get(key), *(entry if isinstance(entry, list) else [entry]), FALLBACK.get(key)]
    for cand in cands:
        if not cand:
            continue
        p = ROOT / cand
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if key in ("cq", "probe"):
            # These files hold several arms; only useful if this arm is in there.
            holder = d if key == "cq" else next(iter(d.values()), {})
            if arm not in holder:
                continue
        d["_path"] = cand
        return d
    return None


def fmt(v, spec=".4f", width=10) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'—':>{width}}"
    if spec == "d":
        return f"{int(v):>{width}d}"
    return f"{v:>{width}{spec}}"


def header(title: str) -> None:
    print(f"\n{title}")
    print("-" * max(len(title), 74))


def row(label: str, vals, spec=".4f", w=26) -> None:
    print(f"  {label:<{w}}" + "".join(fmt(v, spec) for v in vals))


def cols(w: int = 10, left: bool = False) -> None:
    a = "<" if left else ">"
    print(f"  {'':<26}" + "".join(f"{n:{a}{w}}" for n in NAMES))


def cell(mean: float, p: float, wins: int, n: int) -> str:
    """Fixed-width cell so the columns line up whatever the sign and star count."""
    return f"{mean:+.3f}{stars(p):<4}{wins:>2}/{n:<3}"


def stars(p: float | None) -> str:
    if p is None or np.isnan(p):
        return "    "
    return "*** " if p < 0.001 else " ** " if p < 0.01 else "  * " if p < 0.05 else "    "


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def section_provenance(with_valmse: bool) -> None:
    header("PROVENANCE — which file each column comes from")
    for arm in NAMES:
        print(f"  {arm}:")
        for key in ("mmlu", "cq", "probe", "steer_db14", "range_db14", "range_bios"):
            d = load(arm, key)
            entry = ARMS[arm].get(key)
            first = entry[0] if isinstance(entry, list) else entry
            path = d["_path"] if d else (first or "—")
            ck = (d or {}).get("meta", {}).get("checkpoint", "")
            flag = ""
            if ck and "best_val" in ck:
                flag = "   <-- best_val, NOT the epoch-matched step_0014200"
            print(f"    {key:<12} {'ok ' if d else 'MISSING'} {path}{flag}")
        if with_valmse:
            try:
                import torch
                c = torch.load(ROOT / ARMS[arm]["ckpt"], map_location="cpu", weights_only=False)
                print(f"    {'checkpoint':<12} epoch {c['epoch']}, val_mse {c['val_mse']:.5f}")
            except Exception as e:                       # noqa: BLE001
                print(f"    {'checkpoint':<12} unreadable ({e})")


# --------------------------------------------------------------------------- #
# Reconstruction cost
# --------------------------------------------------------------------------- #
def section_mmlu() -> None:
    header("RECONSTRUCTION COST — MMLU under the AE round-trip (n=2000, higher delta better)")
    ms = [(load(a, "mmlu") or {}).get("meta", {}) for a in NAMES]
    cols()
    row("base acc", [m.get("base_acc") for m in ms])
    row("recon acc", [m.get("recon_acc") for m in ms])
    row("delta (recon - base)", [m.get("delta") for m in ms])


# --------------------------------------------------------------------------- #
# Clustering quality
# --------------------------------------------------------------------------- #
CQ_ROWS = [
    ("silhouette", "silhouette  (up)", ".4f"),
    ("dunn_index", "dunn  (up)", ".6f"),
    ("calinski_harabasz", "calinski-harabasz  (up)", ".1f"),
    ("davies_bouldin", "davies-bouldin  (down)", ".4f"),
    ("separability_ratio", "separability ratio  (up)", ".4f"),
    ("inter_centroid_dist_min", "min centroid dist  (up)", ".4f"),
    ("intra_cluster_var", "intra-cluster var  (down)", ".4f"),
    ("effective_rank", "effective rank  (up)", ".2f"),
    ("effective_k", "effective K  (up)", "d"),
    ("n_empty_clusters", "empty clusters  (down)", "d"),
    ("cluster_balance", "balance H  (up)", ".4f"),
]


def section_cq() -> None:
    header("CLUSTERING QUALITY — 1M diverse tokens, seed 0 (scale-free metrics first)")
    print("  NB dunn is approximated on an UNSEEDED 5k subsample and is a min/max statistic:")
    print("  it moved 0.134 -> 0.071 for the SAME d3072 checkpoint between two runs. Do not")
    print("  read the dunn row. Silhouette (10k, averaged) is stable to ~0.0006.")
    per = []
    for a in NAMES:
        d = load(a, "cq")
        per.append((d or {}).get(a, {}))
    cols()
    for key, label, spec in CQ_ROWS:
        row(label, [p.get(key) for p in per], spec)


# --------------------------------------------------------------------------- #
# Concept probes
# --------------------------------------------------------------------------- #
def section_probe() -> None:
    header("CONCEPT PROBE — cluster-label NMI per rung (up is better)")
    per = {a: (load(a, "probe") or {}) for a in NAMES}
    rungs = []
    for d in per.values():
        for r in d:
            if r != "_path" and r not in rungs:
                rungs.append(r)
    if not rungs:
        print("  (no probe results yet)")
        return
    cols()
    means = {a: [] for a in NAMES}
    for r in rungs:
        vals = []
        for a in NAMES:
            v = per[a].get(r, {}).get(a, {}).get("nmi")
            vals.append(v)
            if v is not None:
                means[a].append(v)
        row(r, vals)
    row("MEAN over rungs", [np.mean(means[a]) if means[a] else None for a in NAMES])


# --------------------------------------------------------------------------- #
# Steering (whole-concept direction, DB14)
# --------------------------------------------------------------------------- #
def section_steer() -> None:
    header("DB14 STEERING — paired z - h selectivity per alpha (NEGATIVE = AE better; * = p<.05)")
    print("  NB steer selectivity is tgt_acc_delta - comp_acc_delta, both negative, so a MORE")
    print("  NEGATIVE value is the cleaner steer. This is the opposite sign convention to the")
    print("  range-intervention tables below (tgt_drop - comp_drop, higher better).")
    data = {a: load(a, "steer_db14") for a in NAMES}
    alphas = []
    for d in data.values():
        for al in (d or {}).get("meta", {}).get("alphas", []):
            if al not in alphas:
                alphas.append(al)
    if not alphas:
        print("  (no steering results yet)")
        return
    cols(12, left=True)
    for al in sorted(alphas):
        cells, notes = [], []
        for a in NAMES:
            d = data[a]
            if not d:
                cells.append(None); notes.append("    "); continue
            hs, zs = [], []
            for c in d["concepts"].values():
                hk, zk = f"h_a{al}", f"z_a{al}"
                if hk in c and zk in c:
                    hs.append(c[hk]["selectivity"]); zs.append(c[zk]["selectivity"])
            if len(hs) < 2:
                cells.append(None); notes.append("    "); continue
            delta = np.array(zs) - np.array(hs)
            p = stats.ttest_rel(zs, hs).pvalue
            cells.append(float(delta.mean())); notes.append(stars(p))
        print(f"  {'alpha ' + str(al):<26}" + "".join(
            f"{('—' if v is None else f'{v:+.4f}' + n):<12}" for v, n in zip(cells, notes)))
    n_c = [len((data[a] or {}).get("concepts", {})) or None for a in NAMES]
    print(f"  {'n concepts':<26}" + "".join(f"{('—' if v is None else str(v)):<12}" for v in n_c))


# --------------------------------------------------------------------------- #
# Range interventions
# --------------------------------------------------------------------------- #
def paired(z: list[float], h: list[float]) -> tuple[float, float, int, int]:
    d = np.array(z) - np.array(h)
    p = stats.ttest_rel(z, h).pvalue if len(d) > 1 else float("nan")
    return float(d.mean()), float(p), int((d > 0).sum()), len(d)


def matched_collateral(concepts: dict, mode: str, alphas: list[float]) -> tuple | None:
    """comp_drop(z) - comp_drop(h) interpolated to a matched erasure level, per concept."""
    margins = []
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
        ch = float(np.interp(e, *curves["h"]))
        cz = float(np.interp(e, *curves["z"]))
        margins.append(cz - ch)
    if len(margins) < 2:
        return None
    m = np.array(margins)
    p = stats.ttest_1samp(m, 0.0).pvalue
    return float(m.mean()), float(p), int((m < 0).sum()), len(m)


def section_range(key: str, title: str) -> None:
    header(title)
    data = {a: load(a, key) for a in NAMES}
    if not any(d and "summary" in d for d in data.values()):
        print("  (no completed range results yet)")
        return

    modes = []
    for d in data.values():
        for m in (d or {}).get("summary", {}):
            if m.startswith(("rm_", "st_")) and m not in modes:
                modes.append(m)

    print("\n  z - h selectivity, paired over concepts (up = AE better; * p<.05 ** .01 *** .001)")
    cols(18, left=True)
    for m in modes:
        cells = []
        for a in NAMES:
            d = data[a]
            if not d or "summary" not in d:
                cells.append("—"); continue
            hs, zs = [], []
            for c in d["concepts"].values():
                hk, zk = f"h_{m}", f"z_{m}"
                if hk in c and zk in c:
                    hs.append(c[hk]["selectivity"]); zs.append(c[zk]["selectivity"])
            if len(hs) < 2:
                cells.append("—"); continue
            mean, p, wins, n = paired(zs, hs)
            cells.append(cell(mean, p, wins, n))
        print(f"  {m:<26}" + "".join(f"{c:<18}" for c in cells))

    print("\n  collateral at MATCHED erasure, comp_drop(z) - comp_drop(h)  (NEGATIVE = AE better)")
    cols(18, left=True)
    steer_modes, alphas = [], []
    for d in data.values():
        meta = (d or {}).get("meta", {})
        for sm in meta.get("steer_modes", []):
            if sm not in steer_modes:
                steer_modes.append(sm)
        for al in meta.get("alphas", []):
            if al not in alphas:
                alphas.append(al)
    for sm in steer_modes:
        cells = []
        for a in NAMES:
            d = data[a]
            r = matched_collateral(d["concepts"], sm, sorted(alphas)) if d and "summary" in d else None
            cells.append("—" if r is None else cell(*r))
        print(f"  {'st_' + sm:<26}" + "".join(f"{c:<18}" for c in cells))

    print("\n  target drop / collateral drop at the operator's own setting (h -> z)")
    cols(24, left=True)
    for m in modes:
        cells = []
        for a in NAMES:
            s = (data[a] or {}).get("summary", {}).get(m)
            cells.append("—" if not s else
                         f"{s['h_tgt_drop']:.2f}/{s['h_comp_drop']:.2f}→{s['z_tgt_drop']:.2f}/{s['z_comp_drop']:.2f}")
        print(f"  {m:<26}" + "".join(f"{c:<24}" for c in cells))


# --------------------------------------------------------------------------- #
# Per-concept reliability
# --------------------------------------------------------------------------- #
RANGE_SETS = (("range_db14", "DBpedia-14"), ("range_bios", "bias_in_bios"))
TRANSPORT = "st_transport_a1.0"


def _concepts(key: str) -> dict[str, dict]:
    """{arm: {concept: record}} for one dataset, for whichever arms have the file."""
    out = {}
    for a in NAMES:
        d = load(a, key)
        if d and "concepts" in d and "summary" in d:
            out[a] = d["concepts"]
    return out


def _zmh(rec: dict, mode: str) -> float:
    return rec[f"z_{mode}"]["selectivity"] - rec[f"h_{mode}"]["selectivity"]


def section_reliability() -> None:
    header("PER-CONCEPT RELIABILITY — do per-concept effects replicate across arms?")
    se = float(np.sqrt(2) * np.sqrt(.85 * .15 / 50 + .2 * .8 / 80))
    print(f"  Binomial SE of ONE concept's z-h at n_eval=50 / n_comp=80: ~{se:.3f} — the size of")
    print("  the effects themselves, so only dataset-level means are interpretable.")
    for key, lab in RANGE_SETS:
        C = _concepts(key)
        arms = [a for a in NAMES if a in C]
        if len(arms) < 2:
            continue
        keys = [k for k in C[arms[0]] if all(k in C[a] for a in arms)]
        pairs = [(x, y) for i, x in enumerate(arms) for y in arms[i + 1:]]
        print(f"\n  {lab}: n = {len(keys)} concepts — pairwise Spearman rho across runs")
        print(f"  {'quantity':<26}" + "".join(f"{x + '~' + y:>18}" for x, y in pairs))
        rows = [("h rm_range_comp", "h_rm_range_comp"), ("z rm_range_comp", "z_rm_range_comp"),
                ("h transport a1", f"h_{TRANSPORT}"), ("z transport a1", f"z_{TRANSPORT}"),
                ("z-h transport a1", None)]
        for qlab, fld in rows:
            v = {a: [(_zmh(C[a][k], TRANSPORT) if fld is None else C[a][k][fld]["selectivity"])
                     for k in keys] for a in arms}
            cells = "".join(f"{stats.spearmanr(v[x], v[y])[0]:>+18.2f}" for x, y in pairs)
            print(f"  {qlab:<26}{cells}")
        print("  ^ the base-side (h) rows replicate; the z-h differences are the ones that do not.")


def section_effect_noise() -> None:
    header("EFFECT SIZE vs SPREAD — mean |z-h| over all operators against its per-concept SD")
    print(f"  {'dataset':<14}{'arm':<10}{'mean |z-h|':>12}{'mean SD':>10}{'effect/spread':>15}")
    for key, lab in RANGE_SETS:
        C = _concepts(key)
        for a in NAMES:
            if a not in C:
                continue
            recs = list(C[a].values())
            modes = [k[2:] for k in recs[0] if k.startswith("h_")]
            eff, sd = [], []
            for m in modes:
                d = np.array([_zmh(r, m) for r in recs if f"z_{m}" in r])
                eff.append(abs(d.mean()))
                sd.append(d.std(ddof=1))
            print(f"  {lab:<14}{a:<10}{np.mean(eff):>12.3f}{np.mean(sd):>10.3f}"
                  f"{np.mean(eff) / np.mean(sd):>15.2f}")


# --------------------------------------------------------------------------- #
# Base-quality moderation
# --------------------------------------------------------------------------- #
def section_moderation(mode: str = TRANSPORT) -> None:
    header(f"BASE-QUALITY MODERATION — {mode} (is a narrow latent a regulariser?)")
    print("  x = that concept's BASE (h) rm_range_comp selectivity, averaged over the OTHER arms'")
    print("      runs, so the moderator shares no noise with the y it explains.")
    print("  y = z - h selectivity (positive = AE better). Negative rho = the AE helps where the")
    print("      base is weak and hurts where it is strong.")
    pool: dict[str, dict[str, dict]] = {a: {} for a in NAMES}
    for key, lab in RANGE_SETS:
        for a, cs in _concepts(key).items():
            for k, rec in cs.items():
                pool[a][f"{lab}:{k}"] = rec
    arms = [a for a in NAMES if pool[a]]
    if len(arms) < 2:
        print("  (need at least two arms)")
        return
    print(f"\n  {'arm':<10}{'n':>4}{'rho':>8}{'p':>9}   {'terciles weak->strong':<26}"
          f"{'within DB14':>16}{'within bios':>16}")
    for a in arms:
        x, y, ds = [], [], []
        for k, rec in pool[a].items():
            others = [pool[o][k]["h_rm_range_comp"]["selectivity"] for o in arms
                      if o != a and k in pool[o]]
            if not others or f"z_{mode}" not in rec:
                continue
            x.append(float(np.mean(others)))
            y.append(_zmh(rec, mode))
            ds.append(k.split(":")[0])
        if len(x) < 6:
            continue
        x, y, ds = np.array(x), np.array(y), np.array(ds)
        rho, p = stats.spearmanr(x, y)
        terc = [y[i].mean() for i in np.array_split(np.argsort(x), 3)]
        sub = []
        for lab in ("DBpedia-14", "bias_in_bios"):
            m = ds == lab
            r2, p2 = stats.spearmanr(x[m], y[m]) if m.sum() > 2 else (float("nan"),) * 2
            sub.append(f"{r2:+.2f} ({p2:.2f})")
        print(f"  {a:<10}{len(x):>4}{rho:>8.2f}{p:>9.4f}   "
              f"{' / '.join(f'{t:+.3f}' for t in terc):<26}{sub[0]:>16}{sub[1]:>16}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_valmse", action="store_true",
                    help="also load each checkpoint (1-2 GB each) to print its epoch and val_mse")
    args = ap.parse_args()

    print("=" * 74)
    print("  L27 K=2000 LATENT-WIDTH SERIES — d3072 / d6144 / d12288 (dpc init, ep50)")
    print("=" * 74)
    section_provenance(args.with_valmse)
    section_mmlu()
    section_cq()
    section_probe()
    section_steer()
    section_range("range_db14", "RANGE INTERVENTIONS — DB14 (14 concepts, d' saliency, tao 2.0)")
    section_range("range_bios", "RANGE INTERVENTIONS — bias_in_bios (27 professions, d' saliency, tao 2.0)")
    section_reliability()
    section_effect_noise()
    section_moderation()
    print()


if __name__ == "__main__":
    main()
