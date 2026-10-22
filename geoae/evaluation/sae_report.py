"""Write a concise, claim-aware Markdown report of frozen Gemma Scope comparisons.

python -m geoae.evaluation.sae_report results/.../gemma_scope_compare --out report.md
Reads coding/reconstruction/removal JSON artifacts; no model or GPU is needed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from geoae.evaluation.sae_matched import analyze_observed_matching


def _number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}g}"


def _load(path):
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("analysis")


def _table(headers, rows):
    if not rows:
        return ["No usable observations.", ""]
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(str(v) for v in row) + " |" for row in rows], ""]


def _family_mean(rows, group, metric):
    values = defaultdict(list)
    for row in rows:
        value = row["evaluation"][group][metric]
        if value is not None:
            values[row["family"]].append(value)
    return mean(mean(v) for v in values.values()) if values else None


def _coding(coding):
    summary = coding["summary"]
    lines = [f"Held-out association support: {summary['n_heldout_supported']}/{summary['n_labels']} labels; "
             f"fit assignment: {summary['n_fit_assigned']}/{summary['n_labels']}.", ""]
    lines += _table(["Sibling coding interpretation", "Pairs"],
                    sorted(summary["interpretation_counts"].items()))
    rows = []
    for family in coding["families"]:
        rows.append([family["family"], family["n_labels"], family["n_heldout_supported"],
                     family["sae_union_atoms_by_handle_budget"].get("1", "—"),
                     family.get("geoae_direction_count") if family.get("geoae_direction_count") is not None else "unverified"])
    lines += _table(["Family", "Labels", "Supported labels", "Union of top-1 SAE handles", "GeoAE directions"], rows)
    lines += ["Handle counts describe fit-selected label associations. They do not establish exhaustive concept coding, "
              "and differ from per-token active atoms and complete dictionary size.", ""]
    return lines


def _reconstruction(reconstruction):
    dictionary = reconstruction["dictionary"]
    lines = [f"SAE dictionary: {dictionary['feature_atoms']:,} atoms; "
             f"{dictionary['decoder_scalar_parameters']:,} decoder scalar weights; "
             f"total pretrained scalar weights: {dictionary.get('total_scalar_parameters', '—')}.", ""]
    if reconstruction.get("geoae_parameter_count") is not None:
        lines += [f"GeoAE trainable scalar parameters: {reconstruction['geoae_parameter_count']:,}; "
                  f"token-bypass stored values: {reconstruction.get('geoae_token_bias_values', 0):,}.", ""]
    rows = [["Full SAE", _number(reconstruction["observed_l0"]["all"]),
             _number(reconstruction["full_sae"]["all"]["normalized_mse"])]]
    if reconstruction.get("supplied_geoae_full"):
        rows.append(["Full GeoAE", "dense latent", _number(reconstruction["supplied_geoae_full"]["all"]["normalized_mse"])])
    lines += _table(["Reconstruction", "Mean active coefficients", "Normalized raw MSE"], rows)
    matched = reconstruction.get("matched_active_budget_reconstructions", [])
    if matched:
        rows = [[r["requested_budget"], _number(r["sae"]["observed_retained_l0"]["all"]),
                 _number(r["geoae"]["observed_retained_l0"]["all"]),
                 _number(r["sae"]["metrics"]["all"]["normalized_mse"]),
                 _number(r["geoae"]["metrics"]["all"]["normalized_mse"])] for r in matched]
        lines += _table(["Active-contribution cap", "SAE retained L0", "GeoAE retained L0", "SAE normalized MSE", "GeoAE normalized MSE"], rows)
        lines += ["The active-contribution caps match. Complete dictionary parameters and retained decoder/token biases remain separately reported.", ""]
    else:
        lines += ["Matched-budget GeoAE reconstruction was not supplied; full GeoAE versus SAE top-k does not establish matched-budget reconstruction.", ""]
    return lines


def _removal(removal):
    lines = ["Strengths are calibrated on fitting data. Held-out target damage is reported independently; "
             "these are not claims of equal held-out target damage. ΔNLL and KL are in nats; negative ΔNLL means improved next-token prediction.", "",
             f"Matching statuses: {json.dumps(removal['status_counts'], sort_keys=True)}. "
             f"Excluded pairs/directions: {len(removal['excluded_pairs'])}.", ""]
    matched = [r for r in removal["records"] if r["status"] == "matched"]
    lookup = {(r["pair_id"], r["target"], r["requested_damage"], r["arm"], r["feature_budget"]): r
              for r in matched}
    groups = defaultdict(list)
    for row in matched:
        if row["arm"] in ("sae", "sae_range"):
            groups[(row["arm"], row["feature_budget"], row["requested_damage"])].append(row)
    rows = []
    for (arm, budget, level), values in sorted(groups.items()):
        pairs = [(r, lookup.get((r["pair_id"], r["target"], level, "geoae_range", None))) for r in values]
        pairs = [(r, g) for r, g in pairs if g is not None]
        if not pairs:
            rows.append([arm, budget, level, 0, "—", "—", "—", "—", "—", "—", "—", "—"])
            continue
        sae_rows, geo_rows = zip(*pairs)
        selected = [r["selected_feature_count"] for r in sae_rows]
        rows.append([arm, budget, level, f"{len(pairs)} / {len({r['family'] for r in sae_rows})}",
                     f"{min(selected)}–{max(selected)}",
                     _number(_family_mean(sae_rows, "target", "delta_nll")),
                     _number(_family_mean(geo_rows, "target", "delta_nll")),
                     _number(_family_mean(sae_rows, "sibling", "delta_nll")),
                     _number(_family_mean(geo_rows, "sibling", "delta_nll")),
                     _number(_family_mean(sae_rows, "random", "delta_nll")),
                     _number(_family_mean(geo_rows, "random", "delta_nll")),
                     _number(_family_mean(sae_rows, "random", "kl")) + " / " + _number(_family_mean(geo_rows, "random", "kl"))])
    lines += _table(["SAE operation", "SAE cap", "Fit damage requested", "Paired targets / families", "Actual selected atoms", "SAE target ΔNLL", "GeoAE target ΔNLL", "SAE sibling ΔNLL", "GeoAE sibling ΔNLL", "SAE random ΔNLL", "GeoAE random ΔNLL", "SAE / GeoAE random KL"], rows)
    lines += ["All table averages weight each decision family equally and use the same reachable target intersection for both arms.", ""]
    ci_rows = []
    for comparison in removal["paired_comparisons"]:
        stat = comparison["sibling_delta_nll"]
        ci = stat["ci95"]
        ci_rows.append([comparison["arm"].replace("_minus_geoae_range", ""), comparison["feature_budget"], comparison["requested_damage"],
                        comparison["n_paired_targets"], stat["n_families"], _number(stat["mean"]),
                        "—" if ci is None else f"[{_number(ci[0])}, {_number(ci[1])}]"])
    lines += _table(["SAE operation", "SAE cap", "Fit damage requested", "Paired targets", "Families", "SAE minus GeoAE sibling ΔNLL", "Family-bootstrap 95% CI"], ci_rows)
    return lines


def _observed_matching(removal, sae_arm="sae"):
    analysis = analyze_observed_matching(removal, sae_arm=sae_arm)
    lines = ["This secondary descriptive analysis interpolates collateral between measured held-out outcomes at "
             "fixed target ΔNLL levels 0.025, 0.05, and 0.1. These intervention points were selected by the primary fit-only "
             "calibration. Interpolated strengths were not rerun and are not independently held-out calibrated estimates.", "",
             "Curves require at least two measured positive strengths, remain in alpha order, must have monotone observed "
             "target damage, and are never extrapolated. Both arms use the same reachable target intersection.", "",
             f"Observed-curve statuses: {json.dumps(analysis['status_counts'], sort_keys=True)}.", ""]
    if sae_arm == "sae_range":
        diagnostic = analysis["operator_diagnostic"]
        lines += [f"Operator diagnostic: {diagnostic['n_can_activate_inactive_coefficient']}/{diagnostic['n_targets']} "
                  "fit intervals include zero and have a positive control coefficient mean. Moving toward that mean can "
                  "activate an originally inactive SAE feature on a gated row. Actual zero-coefficient row counts are not stored; "
                  "this centered-coordinate operator differs from moving a coefficient to zero.", ""]
    rows = []
    for comparison in analysis["comparisons"]:
        sae, geo = comparison["sae"], comparison["geoae_range"]
        diff = comparison["difference"]["sibling_delta_nll"]
        ci = diff["ci95"]
        random_diff = comparison["difference"]["random_delta_nll"]
        random_ci = random_diff["ci95"]
        rows.append([
            comparison["feature_budget"], comparison["level"],
            f"{comparison['n_paired_targets']} / {comparison['n_families']}",
            _number(sae["target_delta_nll"]["mean"]) + " / " + _number(geo["target_delta_nll"]["mean"]),
            _number(sae["sibling_delta_nll"]["mean"]) + " / " + _number(geo["sibling_delta_nll"]["mean"]),
            _number(sae["random_delta_nll"]["mean"]) + " / " + _number(geo["random_delta_nll"]["mean"]),
            _number(diff["mean"]),
            "—" if ci is None else f"[{_number(ci[0])}, {_number(ci[1])}]",
            _number(random_diff["mean"]) + (" [CI unavailable]" if random_ci is None else
                                            f" [{_number(random_ci[0])}, {_number(random_ci[1])}]"),
        ])
    lines += _table(["SAE cap", "Observed target level", "Paired targets / families", "SAE / GeoAE target ΔNLL",
                     "SAE / GeoAE sibling ΔNLL", "SAE / GeoAE random ΔNLL", "Sibling SAE minus GeoAE", "Family-bootstrap 95% CI",
                     "Random SAE minus GeoAE [95% CI]"], rows)
    lines += ["Averages weight decision families equally. Confidence intervals resample families and do not include "
              "within-document sampling or interpolation-model uncertainty. This table estimates conditional collateral "
              "at equal observed target damage and remains separate from the primary held-out experiment.", ""]
    return lines


def render_report(result_root: str | Path) -> str:
    """Render completed and partial result directories, without implying missing results."""
    root = Path(result_root)
    dirs = sorted({p.parent for name in ("coding.json", "reconstruction.json", "removal.json")
                   for p in root.rglob(name)})
    lines = ["# Gemma Scope 2 versus GeoAE", "",
             "Pretrained SAEs are compared at Gemma 3 4B's final residual block. "
             "Interpretations below follow held-out support and fit-only selection.", ""]
    if not dirs:
        lines += ["No comparison result JSON files found.", ""]
    for directory in dirs:
        lines += [f"## {directory.relative_to(root)}", ""]
        coding, reconstruction, removal = (_load(directory / name) for name in ("coding.json", "reconstruction.json", "removal.json"))
        if coding:
            lines += ["### Sibling coding", ""] + _coding(coding)
        if reconstruction:
            lines += ["### Reconstruction and space", ""] + _reconstruction(reconstruction)
        if removal:
            lines += ["### Fit-calibrated removal", ""] + _removal(removal)
            lines += ["### Secondary interpolation at equal observed target damage (SAE ablation)", ""] + _observed_matching(removal)
            if any(r["arm"] == "sae_range" for r in removal["records"]):
                lines += ["### Secondary interpolation at equal observed target damage (SAE coefficient range edit)", ""] + _observed_matching(removal, "sae_range")
        else:
            lines += ["Removal result is not present in this directory.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = render_report(args.result_root)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
