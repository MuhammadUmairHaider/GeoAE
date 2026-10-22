"""Summarize the focused prospective control experiment without selecting wins."""
from __future__ import annotations

from geoae.paths import resolve_input_path

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def first_crossing(points, level):
    """Interpolate only a first upward crossing in the recorded alpha order."""
    for left, right in zip(points, points[1:]):
        if left[0] <= level <= right[0] and right[0] > left[0]:
            weight = (level - left[0]) / (right[0] - left[0])
            return [(1 - weight) * a + weight * b for a, b in zip(left[1:], right[1:])]
    return None


def rows_by_key(result):
    return {(r["pair"], r["target_label"], r["arm"], r["requested_target_delta_nll"]): r
            for r in result["records"] if r.get("status") == "matched"}


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="results/ssl_concepts_20261009_v1")
    args = parser.parse_args()
    out = Path(args.experiment)
    out.mkdir(parents=True, exist_ok=True)
    source = out / "controls.json"
    result = json.loads(resolve_input_path(source).read_text())
    capture = json.loads((resolve_input_path(out / "capture/capture.json")).read_text())
    manifest = capture["manifest"]
    names = {p["pair_id"]: (p["a_name"], p["b_name"]) for p in manifest["pairs"]}
    pairs = [p for p in names if p != "native_newline"]
    # Paired document resampling holds the fitted directions fixed. This
    # quantifies test-document variation, not training/fit uncertainty.
    geometry_boot = {}
    for pair in pairs:
        for kind in ("shared", "difference"):
            group = {g["space"]: g for g in result["geometry"]
                     if g["target_label"] == 1 and g["pair"] == pair and g["direction"] == kind}
            if not group:
                continue
            ref = group["base"]["test"]
            rng = np.random.default_rng(20261009)
            ti = rng.integers(0, len(ref["target"]["projection"]["doc_ids"]),
                              (2000, len(ref["target"]["projection"]["doc_ids"])))
            si = rng.integers(0, len(ref["sibling"]["projection"]["doc_ids"]),
                              (2000, len(ref["sibling"]["projection"]["doc_ids"])))
            samples = {}
            for space, g in group.items():
                for label in ("target", "sibling"):
                    assert g["test"][label]["projection"]["doc_ids"] == ref[label]["projection"]["doc_ids"]
                a = np.asarray(g["test"]["target"]["projection"]["doc_means"])[ti]
                b = np.asarray(g["test"]["sibling"]["projection"]["doc_means"])[si]
                samples[space] = np.abs(a.mean(1) - b.mean(1)) / np.maximum(
                    ((a.var(1) + b.var(1)) / 2) ** .5, 1e-12)
            for space, draws in samples.items():
                geometry_boot[pair, kind, space] = {
                    "ci95": np.quantile(draws, [.025, .975]).tolist(),
                    "difference_vs_base_ci95": np.quantile(draws - samples["base"], [.025, .975]).tolist()}
    geometry = []
    for g in result["geometry"]:
        if g["target_label"] != 1:
            continue
        boot = geometry_boot[g["pair"], g["direction"], g["space"]]
        geometry.append({"pair": g["pair"], "direction": g["direction"], "space": g["space"],
                         "test_document_dprime": g["test_document_dprime"],
                         "test_position_dprime": g["test_position_dprime_equal_document_weight"],
                         "target_gate_fraction": g["test"]["target"]["gate_fraction"]["mean"],
                         "sibling_gate_fraction": g["test"]["sibling"]["gate_fraction"]["mean"],
                         "dprime_ci_lo": boot["ci95"][0], "dprime_ci_hi": boot["ci95"][1],
                         "difference_vs_base_ci_lo": boot["difference_vs_base_ci95"][0],
                         "difference_vs_base_ci_hi": boot["difference_vs_base_ci95"][1]})
    write_csv(out / "geometry.csv", geometry)
    lookup = rows_by_key(result)
    records = []
    for r in result["records"]:
        flat = {"pair": r["pair"], "target_label": r["target_label"], "arm": r["arm"],
                "requested_target_nll": r["requested_target_delta_nll"], "status": r["status"],
                "alpha": r.get("alpha"), "actual_target_nll": None, "sibling_nll": None,
                "random_nll": None, "total_sibling_nll_vs_raw": None,
                "sibling_positive_document_damage": None, "test_match_status": r.get("test_match_status")}
        if "evaluation" in r:
            ev = r["evaluation"]
            flat.update(actual_target_nll=ev["target"]["delta_nll"]["mean"],
                        sibling_nll=ev["sibling"]["delta_nll"]["mean"],
                        random_nll=ev["random"]["delta_nll"]["mean"],
                        total_sibling_nll_vs_raw=ev["sibling"]["total_delta_nll_vs_raw"]["mean"],
                        sibling_positive_document_damage=float(np.maximum(
                            0, ev["sibling"]["delta_nll"]["doc_means"]).mean()))
        records.append(flat)
    write_csv(out / "all_controls.csv", records)
    descriptive = []
    for pair in pairs:
        for target in (0, 1):
            for kind in ("difference", "shared"):
                for gate in ("range", "all"):
                    values = {}
                    for space in ("base", "hb", "ae_recon"):
                        selected = [r for key, r in lookup.items()
                                    if key[:3] == (pair, target, f"{space}/{kind}/{gate}")]
                        selected.sort(key=lambda r: r["alpha"])
                        points = [(0., 0., 0.)] + [
                            (r["evaluation"]["target"]["delta_nll"]["mean"],
                             r["evaluation"]["sibling"]["delta_nll"]["mean"],
                             r["evaluation"]["random"]["delta_nll"]["mean"]) for r in selected]
                        values[space] = first_crossing(points, .25)
                    flat = {"pair": pair, "target_label": target, "direction": kind, "gate": gate}
                    for space, v in values.items():
                        flat[f"{space}_sibling_at_025_test_target"] = None if v is None else v[0]
                        flat[f"{space}_random_at_025_test_target"] = None if v is None else v[1]
                    descriptive.append(flat)
    write_csv(out / "descriptive_test_matching.csv", descriptive)
    lines = [
        "# Fresh Atlas geometry and control results",
        "",
        "This is a prospective test of previously selected contrasts. Directions/ranges were fit on 50 documents "
        "per class, strengths calibrated on 30, and evaluated on 50 untouched documents per class. "
        "Random controls use 64 documents per split. Each document has equal weight.",
        "",
        f"The source scan excluded {capture['input_prep']['excluded_prefix_documents']:,} discovery documents "
        f"and captured {capture['n_rows']:,} positions. Text labels describe whole passages; the captured positions "
        "are in their final quarter. This outcome measures next-token predictive disruption, not demonstrated "
        "removal of a named meaning.",
        "",
        "The References/External-links case lacked fresh source coverage and was excluded. "
        "All three Atlas contrasts retain their original 50/30/50 documents per class.",
        "",
        "## Separation on untouched documents",
        "",
        "d′ below is between per-document mean projections on a direction fitted without test documents. "
        "The shared direction tests radial differentiation; the mean-difference direction tests a direct contrast.",
        "",
        "| Contrast | Direction | Base | Token-centered base | GeoAE |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in pairs:
        for kind in ("shared", "difference"):
            values = {g["space"]: g["test_document_dprime"] for g in geometry
                      if g["pair"] == pair and g["direction"] == kind}
            lines.append(f"| {names[pair][0]} / {names[pair][1]} | {kind} | "
                         f"{values.get('base', float('nan')):.3f} | "
                         f"{values.get('hb', float('nan')):.3f} | "
                         f"{values.get('latent', float('nan')):.3f} |")
    lines += ["", "| Contrast | Direction | GeoAE−base d′ [paired 95% document interval] |",
              "|---|---|---:|"]
    for pair in pairs:
        for kind in ("shared", "difference"):
            values = {g["space"]: g for g in geometry if g["pair"] == pair and g["direction"] == kind}
            ae, base = values["latent"], values["base"]
            lines.append(f"| {pair} | {kind} | {ae['test_document_dprime'] - base['test_document_dprime']:+.3f} "
                         f"[{ae['difference_vs_base_ci_lo']:+.3f}, {ae['difference_vs_base_ci_hi']:+.3f}] |")
    lines += ["", "These intervals resample the same test documents in both spaces, conditional on the frozen "
              "fitted directions; they omit training and fitting variation and multiplicity correction.",
              "", "![Fresh Atlas separation with document intervals](separation.png)",
              "", "[Publication-size PDF](separation.pdf)"]
    lines += [
        "",
        "## Calibration-selected operating points",
        "",
        "The table shows gated edits calibrated to +0.25 nats target damage. Each arrow is target/sibling damage "
        "on TEST documents in the base and AE spaces. It includes every contrast and direction. "
        "A target gap above 0.02 nats is marked unmatched; such a row cannot support a matched-selectivity win.",
        "",
        "| Contrast | Target | Direction | Base target / sibling | AE target / sibling | Actual target gap AE−base | Match |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for pair in pairs:
        for target in (0, 1):
            for kind in ("difference", "shared"):
                base = lookup.get((pair, target, f"base/{kind}/range", .25))
                ae = lookup.get((pair, target, f"ae_recon/{kind}/range", .25))
                if not base or not ae:
                    lines.append(f"| {pair} | {names[pair][target]} | {kind} | "
                                 f"{'unreachable/unresolved' if not base else 'calibrated'} | "
                                 f"{'unreachable/unresolved' if not ae else 'calibrated'} | — | unavailable |")
                    continue
                be, ze = base["evaluation"], ae["evaluation"]
                bt, zt = be["target"]["delta_nll"]["mean"], ze["target"]["delta_nll"]["mean"]
                bs, zs = be["sibling"]["delta_nll"]["mean"], ze["sibling"]["delta_nll"]["mean"]
                gap = zt - bt
                match = "matched within 0.02" if abs(gap) <= .02 else "unmatched"
                lines.append(f"| {pair} | {names[pair][target]} | {kind} | {bt:.3f} / {bs:.3f} | "
                             f"{zt:.3f} / {zs:.3f} | {gap:+.3f} | {match} |")
    lines += [
        "",
        "## Descriptive matching at the same observed test damage",
        "",
        "This secondary table interpolates the test outcomes of the three calibration-selected strengths. "
        "It preserves strength order, takes the first upward crossing, and never extrapolates. "
        "It describes the saved curve and does not validate an operating point selected on new documents. "
        "No significance interval is attached to this interpolated table.",
        "",
        "| Contrast | Target | Direction | Base sibling | Token-centered sibling | AE sibling |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in descriptive:
        if row["gate"] != "range":
            continue
        number = lambda x: "unreachable" if x is None else f"{x:.3f}"
        lines.append(f"| {row['pair']} | {names[row['pair']][row['target_label']]} | {row['direction']} | "
                     f"{number(row['base_sibling_at_025_test_target'])} | "
                     f"{number(row['hb_sibling_at_025_test_target'])} | "
                     f"{number(row['ae_recon_sibling_at_025_test_target'])} |")
    diagnostic_path = out / "controls_error_preserved.json"
    if resolve_input_path(diagnostic_path).exists():
        diagnostic = json.loads(resolve_input_path(diagnostic_path).read_text())
        delta_lookup = rows_by_key(diagnostic)
        delta_rows = []
        lines += [
            "", "## Diagnostic with the original residual error preserved", "",
            "This diagnostic was added after inspecting reconstruction costs. It reuses every frozen target and "
            "document and evaluates both directions; it is secondary evidence on the same test set. "
            "The edit is h + decode(z_edited) − decode(z_original), so its zero-strength baseline is the original "
            "model and no reconstruction loss is charged separately.", "",
            "The following values interpolate at +0.25 nats observed target damage using the same first-crossing "
            "rule. They are descriptive curve estimates, with no extrapolation or significance claim.", "",
            "| Contrast | Target | Direction | Base sibling | Error-preserved AE sibling |",
            "|---|---|---|---:|---:|",
        ]
        for pair in pairs:
            for target in (0, 1):
                for kind in ("difference", "shared"):
                    values = {}
                    for space in ("base", "ae_delta"):
                        selected = [r for key, r in delta_lookup.items()
                                    if key[:3] == (pair, target, f"{space}/{kind}/range")]
                        selected.sort(key=lambda r: r["alpha"])
                        points = [(0., 0., 0.)] + [
                            (r["evaluation"]["target"]["delta_nll"]["mean"],
                             r["evaluation"]["sibling"]["delta_nll"]["mean"],
                             r["evaluation"]["random"]["delta_nll"]["mean"]) for r in selected]
                        values[space] = first_crossing(points, .25)
                    flat = {"pair": pair, "target_label": target, "direction": kind,
                            "base_sibling_at_025": None if values["base"] is None else values["base"][0],
                            "ae_delta_sibling_at_025": None if values["ae_delta"] is None else values["ae_delta"][0],
                            "base_random_at_025": None if values["base"] is None else values["base"][1],
                            "ae_delta_random_at_025": None if values["ae_delta"] is None else values["ae_delta"][1]}
                    delta_rows.append(flat)
                    lines.append(f"| {pair} | {names[pair][target]} | {kind} | "
                                 f"{number(flat['base_sibling_at_025'])} | "
                                 f"{number(flat['ae_delta_sibling_at_025'])} |")
        write_csv(out / "error_preserved_descriptive.csv", delta_rows)
        (out / "error_preserved_summary.json").write_text(json.dumps({
            "source_sha256": hashlib.sha256(resolve_input_path(diagnostic_path).read_bytes()).hexdigest(),
            "n_controls": len(diagnostic["records"]),
            "statuses": dict(Counter(r["status"] for r in diagnostic["records"])),
            "paired_actual_target_matches": sum(c["test_target_match_status"] == "within_tolerance"
                                              for c in diagnostic["paired_comparisons"]),
            "n_comparisons": len(diagnostic["paired_comparisons"]),
            "limits": "Same test documents; secondary error-preservation diagnostic; curve estimates interpolate "
                      "calibration-selected strengths. No independent semantic removal or SSL causal attribution."
        }, indent=2) + "\n")
        lines += ["", "For the graphics-hardware shared-direction example, the fixed calibration-selected "
                  "strengths achieve 0.114 versus 0.091 nats actual test target damage at the 0.1 calibration level, "
                  "and 0.287 versus 0.243 at the 0.25 level. Those are not within the registered 0.02-nat "
                  "matching tolerance. The interpolated table estimates equal damage; it must not be described "
                  "as an independently measured exact-match operating point."]
    lines += [
        "",
        "AE incremental damage is measured relative to its reconstructed baseline. The full output also records "
        "raw/reconstructed baseline NLL and total raw-to-edited damage. These must accompany any interpretation "
        "of the incremental table; reconstruction costs are not removed by normalization.",
        "",
        "The full output contains document-level paired bootstrap intervals, calibration reachability, "
        "test target mismatches, gate firing, reconstruction costs, and every requested strength. There is no "
        "bootstrap over just three concepts and no selection of successful directions for a general claim.",
        "",
        "The experiment confirms or challenges specific geometric observations. It does not isolate pull, "
        "separation, or VICReg causally; that requires matched-duration objective ablations.",
        "",
        "Files: [all controls](controls.json), [geometry](geometry.csv), [complete control table](all_controls.csv), "
        "[descriptive matching](descriptive_test_matching.csv), [frozen targets/protocol](manifest.json), "
        "[capture checks](capture/capture.json).",
    ]
    if resolve_input_path(diagnostic_path).exists():
        lines += ["", "Secondary diagnostic: [full error-preserved controls](controls_error_preserved.json), "
                  "[interpolated estimates](error_preserved_descriptive.csv), "
                  "[diagnostic plan and limits](diagnostic_plan.md)."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    diagnostics = {"controls_sha256": hashlib.sha256(resolve_input_path(source).read_bytes()).hexdigest(),
                   "n_controls": len(result["records"]), "statuses": dict(Counter(
                       r["status"] for r in result["records"])), "excluded": result["excluded"],
                   "capture_sanity": capture["sanity"]}
    if resolve_input_path(diagnostic_path).exists():
        diagnostics["error_preserved_controls_sha256"] = hashlib.sha256(resolve_input_path(diagnostic_path).read_bytes()).hexdigest()
    (out / "summary.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = {"base": "#64748b", "hb": "#0f766e", "latent": "#9333ea"}
    labels = {"base": "Base", "hb": "Token-centered", "latent": "GeoAE"}
    x = np.arange(len(pairs))
    for axis, kind in zip(axes, ("shared", "difference")):
        for offset, space in enumerate(("base", "hb", "latent")):
            vals = [next(g["test_document_dprime"] for g in geometry
                         if g["pair"] == pair and g["direction"] == kind and g["space"] == space)
                    for pair in pairs]
            gs = [next(g for g in geometry
                       if g["pair"] == pair and g["direction"] == kind and g["space"] == space) for pair in pairs]
            errors = np.array([[max(0, v - g["dprime_ci_lo"]) for v, g in zip(vals, gs)],
                               [max(0, g["dprime_ci_hi"] - v) for v, g in zip(vals, gs)]])
            axis.bar(x + (offset - 1) * .24, vals, .23, color=colors[space], label=labels[space],
                     yerr=errors, capsize=2)
        axis.set_xticks(x, ["Cooking", "Graphics", "Construction"])
        axis.set_ylabel("Held-out document projection d′")
        axis.set_title("Shared direction" if kind == "shared" else "Mean-difference direction")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Fresh Atlas contrasts: same documents in each representation")
    fig.tight_layout()
    fig.savefig(out / "separation.png", dpi=180)
    fig.savefig(out / "separation.pdf")
    plt.close(fig)
    print(json.dumps(diagnostics), flush=True)


if __name__ == "__main__":
    main()
