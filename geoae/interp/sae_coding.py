"""Held-out sibling coding and raw-space reconstruction for a frozen SAE.

This module does not train an SAE or select features on evaluation rows. A
``handle`` is a feature selected for association with a supplied label; counting
handles is distinct from counting active features or all dictionary parameters.
The runner is responsible for document-level split construction and label
provenance. Statistics here give each supplied row equal weight.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch


DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 20, 32)
_VARIANCE_FLOOR = 1e-12


def _mask(value: Any, n_rows: int, device: torch.device, name: str) -> torch.Tensor:
    result = torch.as_tensor(value, device=device)
    if result.ndim != 1 or len(result) != n_rows or result.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean vector with {n_rows} rows")
    return result


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating point matrix")
    if not all(value.shape):
        raise ValueError(f"{name} must have nonempty dimensions")
    # Avoid a full-size boolean allocation for 262k-wide dictionaries.
    for start in range(0, len(value), 128):
        if not bool(torch.isfinite(value[start:start + 128]).all()):
            raise ValueError(f"{name} contains nonfinite values")
    return value.detach()


def _budgets(values: Sequence[int], width: int) -> tuple[int, ...]:
    values = tuple(values)
    if not values or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError("budgets must contain positive integers")
    return tuple(sorted(set(values)))


def _moments(features: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Population moments without gathering an entire wide labeled matrix."""
    width = features.shape[1]
    means = torch.empty(width, dtype=torch.float64, device=features.device)
    variances = torch.empty_like(means)
    for start in range(0, width, 4096):
        stop = min(start + 4096, width)
        values = features[:, start:stop][rows].float()
        var, mean = torch.var_mean(values, dim=0, unbiased=False)
        means[start:stop] = mean
        variances[start:stop] = var
    return means, variances


def _decoder_norms(directions: torch.Tensor) -> torch.Tensor:
    norms = torch.empty(len(directions), dtype=torch.float32, device=directions.device)
    for start in range(0, len(directions), 4096):
        norms[start:start + 4096] = directions[start:start + 4096].float().norm(dim=1)
    return norms


def _auc(positive: torch.Tensor, negative: torch.Tensor) -> float | None:
    """Exact Mann--Whitney AUROC with half credit for ties, no sklearn dependency."""
    if not len(positive) or not len(negative):
        return None
    positive = positive.detach().double().cpu()
    negative = negative.detach().double().cpu()
    values = torch.cat((positive, negative))
    order = torch.argsort(values, stable=True)
    _, counts = torch.unique_consecutive(values[order], return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    midranks = torch.repeat_interleave((starts.double() + ends.double() + 1) / 2, counts)
    rank_sum = midranks[order < len(positive)].sum().item()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def _effect(positive: torch.Tensor, negative: torch.Tensor) -> dict[str, Any]:
    positive, negative = positive.detach().double(), negative.detach().double()
    result: dict[str, Any] = {"n_positive": len(positive), "n_negative": len(negative)}
    if not len(positive) or not len(negative):
        result.update(auc=None, mean_positive=None, mean_negative=None, mean_gap=None, dprime=None)
        return result
    mp, mn = positive.mean().item(), negative.mean().item()
    pooled = ((positive.var(unbiased=False) + negative.var(unbiased=False)) / 2).item()
    result.update(
        auc=_auc(positive, negative), mean_positive=mp, mean_negative=mn, mean_gap=mp - mn,
        dprime=float((mp - mn) / math.sqrt(max(pooled, _VARIANCE_FLOOR))),
        active_positive=float((positive != 0).double().mean().item()),
        active_negative=float((negative != 0).double().mean().item()),
    )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    # numpy scalar metadata is common in pair tables.
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return str(value)


@torch.no_grad()
def analyze_coding(
    features: torch.Tensor,
    memberships: Mapping[Any, Any],
    pairs: Sequence[Mapping[str, Any]],
    decoder_directions: torch.Tensor,
    fit_mask: Any,
    eval_mask: Any,
    *,
    controls: Any | None = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    parallel_cosine_threshold: float = 0.9,
    min_fit_examples: int = 5,
    min_eval_examples: int = 5,
    min_heldout_auc: float = 0.6,
) -> dict[str, Any]:
    """Associate siblings with SAE features using fit rows, then report held-out evidence.

    ``memberships`` maps labels to boolean row masks. Each pair contains ``a`` and
    ``b`` labels and optionally ``family``, ``geoae_direction`` or
    ``geoae_direction_count`` metadata. Explicit ``controls`` should mark random
    text; without them each label's nonmembers are its comparison group. Controls
    overlapping a label are excluded from that label's comparison group.

    Positive pooled-variance d-prime ranks features; zero decoder atoms and
    features with no positive target/control mean gap are excluded. Ranking and
    orientation of sibling scalar/contrast tests use fit rows only. The geometry
    category is descriptive; ``interpretation`` is unresolved when either label
    lacks held-out association. ``min_heldout_auc`` is a declared descriptive
    threshold, not a significance test or estimated confidence probability.
    """
    features = _matrix(features, "features")
    directions = _matrix(decoder_directions, "decoder_directions")
    n_rows, width = features.shape
    if directions.shape[0] != width:
        raise ValueError("decoder_directions must have shape [features, hidden]")
    if not 0 <= parallel_cosine_threshold <= 1 or not 0.5 <= min_heldout_auc <= 1:
        raise ValueError("cosine and held-out AUROC thresholds are out of range")
    if min_fit_examples < 1 or min_eval_examples < 1:
        raise ValueError("minimum example counts must be positive")
    budgets = _budgets(budgets, width)
    fit = _mask(fit_mask, n_rows, features.device, "fit_mask")
    evaluation = _mask(eval_mask, n_rows, features.device, "eval_mask")
    if bool((fit & evaluation).any()):
        raise ValueError("fit_mask and eval_mask must be disjoint")
    if not bool(fit.any()) or not bool(evaluation.any()):
        raise ValueError("fit_mask and eval_mask must both have rows")
    control_mask = None if controls is None else _mask(controls, n_rows, features.device, "controls")
    members = {label: _mask(mask, n_rows, features.device, f"memberships[{label!r}]")
               for label, mask in memberships.items()}
    if len({str(label) for label in members}) != len(members):
        raise ValueError("membership labels must have distinct string representations")
    for pair in pairs:
        if pair.get("a") not in members or pair.get("b") not in members:
            raise ValueError("each pair must contain a and b labels present in memberships")
        if pair["a"] == pair["b"]:
            raise ValueError("a sibling pair must contain two different labels")

    decoder_norms = _decoder_norms(directions).to(features.device)
    nonzero_decoder = decoder_norms > 0
    label_reports: dict[Any, dict[str, Any]] = {}
    for label, member_mask in members.items():
        comparison = ~member_mask if control_mask is None else control_mask & ~member_mask
        positives, negatives = fit & member_mask, fit & comparison
        n_positive, n_negative = int(positives.sum().item()), int(negatives.sum().item())
        base: dict[str, Any] = {
            "feature_id": None, "handles_by_budget": {str(k): [] for k in budgets},
            "fit_n_target": n_positive, "fit_n_control": n_negative,
            "fit_eligible_features": 0, "fit_selection_margin": None,
            "fit_selection_score": None, "fit": None, "heldout": None,
            "heldout_supported": False,
        }
        if min(n_positive, n_negative) < min_fit_examples:
            base["status"] = "insufficient_fit_rows"
            label_reports[label] = base
            continue
        target_mean, target_var = _moments(features, positives)
        control_mean, control_var = _moments(features, negatives)
        gap = target_mean - control_mean
        score = gap / ((target_var + control_var) / 2).clamp_min(_VARIANCE_FLOOR).sqrt()
        eligible = (gap > 0) & nonzero_decoder
        eligible_ids = torch.nonzero(eligible, as_tuple=True)[0]
        base["fit_eligible_features"] = len(eligible_ids)
        if not len(eligible_ids):
            base["status"] = "no_positive_fit_association"
            label_reports[label] = base
            continue
        # Stable sorting resolves exact ties by the lower feature id.
        ranking = eligible_ids[torch.argsort(score[eligible_ids], descending=True, stable=True)]
        selected = int(ranking[0].item())
        top = ranking[:min(max(budgets), len(ranking))].tolist()
        first_score = float(score[selected].item())
        second_score = float(score[ranking[1]].item()) if len(ranking) > 1 else None
        fit_stats = _effect(features[positives, selected], features[negatives, selected])
        eval_stats = _effect(features[evaluation & member_mask, selected],
                             features[evaluation & comparison, selected])
        supported = (min(eval_stats["n_positive"], eval_stats["n_negative"]) >= min_eval_examples
                     and eval_stats["auc"] is not None and eval_stats["auc"] >= min_heldout_auc
                     and eval_stats["mean_gap"] > 0)
        base.update(
            feature_id=selected, handles_by_budget={str(k): top[:k] for k in budgets},
            fit_selection_score=first_score,
            fit_selection_margin=None if second_score is None else (first_score - second_score) / first_score,
            fit=fit_stats, heldout=eval_stats, heldout_supported=bool(supported),
            status="heldout_supported" if supported else "heldout_unresolved",
        )
        label_reports[label] = base

    pair_reports = []
    family_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        a, b = pair["a"], pair["b"]
        ha, hb = label_reports[a]["feature_id"], label_reports[b]["feature_id"]
        family = str(pair.get("family") if pair.get("family") is not None else f"pair:{a}|{b}")
        exclusive_a, exclusive_b = members[a] & ~members[b], members[b] & ~members[a]
        geometry, cosine, discrimination = "unassigned", None, None
        interpretation = "unresolved"
        supported = label_reports[a]["heldout_supported"] and label_reports[b]["heldout_supported"]
        if ha is not None and hb is not None:
            da, db = directions[ha].double(), directions[hb].double()
            cosine = float((da @ db / (da.norm() * db.norm())).clamp(-1, 1).item())
            geometry = "same_feature" if ha == hb else (
                "distinct_parallel" if cosine >= parallel_cosine_threshold else "distinct_unrelated")
            scalar = features[:, ha] if ha == hb else (
                features[:, ha] * decoder_norms[ha] - features[:, hb] * decoder_norms[hb])
            fit_a, fit_b = scalar[fit & exclusive_a], scalar[fit & exclusive_b]
            if len(fit_a) and len(fit_b):
                gap = float(fit_a.double().mean().item() - fit_b.double().mean().item())
                orientation = 1 if gap >= 0 else -1
                fit_stats = _effect(orientation * fit_a, orientation * fit_b)
                eval_stats = _effect(orientation * scalar[evaluation & exclusive_a],
                                     orientation * scalar[evaluation & exclusive_b])
                separated = (abs(gap) > 0
                             and min(eval_stats["n_positive"], eval_stats["n_negative"]) >= min_eval_examples
                             and eval_stats["auc"] is not None and eval_stats["auc"] >= min_heldout_auc
                             and eval_stats["mean_gap"] > 0)
                discrimination = {
                    "score": "shared_feature_scalar" if ha == hb else "raw_contribution_norm_contrast",
                    "orientation_fit": orientation, "fit": fit_stats, "heldout": eval_stats,
                    "heldout_separated": bool(separated),
                    "exclusive_rows_only": True,
                }
                if supported:
                    interpretation = geometry if ha != hb else (
                        "same_feature_intensity_separation" if separated else
                        "shared_feature_without_heldout_separation")
        record = dict(pair)
        record.update(index=index, family=family, feature_a=ha, feature_b=hb,
                      decoder_cosine=cosine, feature_geometry=geometry,
                      heldout_associations_supported=bool(supported), interpretation=interpretation,
                      sibling_discrimination=discrimination,
                      overlapping_member_rows=int((members[a] & members[b]).sum().item()))
        pair_reports.append(record)
        family_pairs[family].append(record)

    families = []
    for family, records in family_pairs.items():
        labels = list(dict.fromkeys(label for p in records for label in (p["a"], p["b"])))
        atom_counts = {
            str(k): len(set(handle for label in labels for handle in label_reports[label]["handles_by_budget"][str(k)]))
            for k in budgets
        }
        directions_named = {str(p["geoae_direction"]) for p in records if p.get("geoae_direction") is not None}
        supplied_counts = {p["geoae_direction_count"] for p in records
                           if p.get("geoae_direction_count") is not None}
        geoae_count = (len(directions_named) if directions_named else
                       next(iter(supplied_counts)) if len(supplied_counts) == 1 else None)
        assigned = sum(label_reports[label]["feature_id"] is not None for label in labels)
        supported = sum(label_reports[label]["heldout_supported"] for label in labels)
        families.append({
            "family": family, "labels": labels, "n_labels": len(labels),
            "n_fit_assigned": assigned, "n_heldout_supported": supported,
            "fit_assignment_coverage": assigned / len(labels),
            "heldout_assignment_coverage": supported / len(labels),
            "sae_union_atoms_by_handle_budget": atom_counts,
            "geoae_direction_count": geoae_count,
            "count_unit": "union of fit-selected label handles; not per-token active atoms",
        })

    return _json_safe({
        "n_rows": n_rows, "n_features": width, "n_fit": int(fit.sum().item()),
        "n_eval": int(evaluation.sum().item()), "budgets": budgets,
        "control_source": "explicit_mask" if controls is not None else "label_nonmembers",
        "selection": {"score": "positive target-control pooled-population dprime",
                      "variance_floor": _VARIANCE_FLOOR, "rows": "fit_only",
                      "tie_break": "lower_feature_id", "min_fit_examples": min_fit_examples,
                      "min_eval_examples": min_eval_examples, "min_heldout_auc": min_heldout_auc,
                      "parallel_cosine_threshold": parallel_cosine_threshold},
        "labels": label_reports, "pairs": pair_reports, "families": families,
        "summary": {
            "n_labels": len(members),
            "n_fit_assigned": sum(x["feature_id"] is not None for x in label_reports.values()),
            "n_heldout_supported": sum(x["heldout_supported"] for x in label_reports.values()),
            "feature_geometry_counts": dict(Counter(p["feature_geometry"] for p in pair_reports)),
            "interpretation_counts": dict(Counter(p["interpretation"] for p in pair_reports)),
        },
        "assumptions": [
            "Labels and document-disjoint split masks are supplied by the caller; all metrics weight rows equally.",
            "Handles measure label association and do not establish causal or exhaustive concept coding.",
            "Decoder cosine is signed; antiparallel atoms are not counted as near-parallel atoms.",
            "Association support is a descriptive AUROC threshold, not a calibrated confidence or significance test.",
            "Sibling discrimination excludes rows carrying both labels and learns score orientation on fit rows.",
            "Distinct-feature contrast weights coefficients by raw decoder norm, preserving equivalent positive feature/decoder rescaling.",
            "Handle budget, per-token active budget and complete dictionary parameter budget are different quantities.",
            "GeoAE direction counts are reported only when supplied in family metadata.",
        ],
    })


def _interval_overlap(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any] | None:
    """Projection intervals centered at mean +/- two population standard deviations."""
    if not len(a) or not len(b):
        return None
    a, b = a.double(), b.double()
    ma, mb = a.mean().item(), b.mean().item()
    sa, sb = a.std(unbiased=False).item(), b.std(unbiased=False).item()
    alo, ahi, blo, bhi = ma - 2 * sa, ma + 2 * sa, mb - 2 * sb, mb + 2 * sb
    intersection = max(0.0, min(ahi, bhi) - max(alo, blo))
    union = (ahi - alo) + (bhi - blo) - intersection
    identical_points = alo == ahi == blo == bhi
    return {"a": [alo, ahi], "b": [blo, bhi], "n_population_sd": 2,
            "intersection_width": intersection, "union_width": union,
            "intersection_over_union": intersection / union if union > 0 else float(identical_points),
            "disjoint_gap": max(0.0, max(alo, blo) - min(ahi, bhi))}


@torch.no_grad()
def analyze_geoae_geometry(
    z: torch.Tensor,
    memberships: Mapping[Any, Any],
    pairs: Sequence[Mapping[str, Any]],
    fit_mask: Any,
    eval_mask: Any,
    controls: Any | None = None,
    *,
    parallel_cosine_threshold: float = 0.9,
    min_fit_examples: int = 5,
    min_eval_examples: int = 5,
) -> dict[str, Any]:
    """Fit-selected magnitude-versus-direction diagnostics in GeoAE latent space.

    Fit control mean centers both siblings. The shared scalar is projection onto
    the normalized sum of their fit mean unit directions, with its sign chosen
    from the fit sibling projection gap. Evaluation rows cannot change that
    direction, orientation, or the fit parallelism flag. Each pair excludes
    overlapping members. The result is a latent-coordinate diagnostic, not a
    claim about shared native cluster decoder directions or raw residual space.
    """
    z = _matrix(z, "z")
    n_rows = len(z)
    fit = _mask(fit_mask, n_rows, z.device, "fit_mask")
    evaluation = _mask(eval_mask, n_rows, z.device, "eval_mask")
    if bool((fit & evaluation).any()):
        raise ValueError("fit_mask and eval_mask must be disjoint")
    if not bool(fit.any()) or not bool(evaluation.any()):
        raise ValueError("fit_mask and eval_mask must both have rows")
    if not 0 <= parallel_cosine_threshold <= 1 or min_fit_examples < 1 or min_eval_examples < 1:
        raise ValueError("invalid geometry threshold or minimum example count")
    control = None if controls is None else _mask(controls, n_rows, z.device, "controls")
    members = {label: _mask(mask, n_rows, z.device, f"memberships[{label!r}]")
               for label, mask in memberships.items()}
    records = []
    for index, pair in enumerate(pairs):
        if pair.get("a") not in members or pair.get("b") not in members or pair["a"] == pair["b"]:
            raise ValueError("each geometry pair requires two different membership labels")
        ma, mb = members[pair["a"]], members[pair["b"]]
        a, b = ma & ~mb, mb & ~ma
        comparison = ~(ma | mb) if control is None else control & ~(ma | mb)
        fit_a, fit_b, fit_c = fit & a, fit & b, fit & comparison
        eval_a, eval_b = evaluation & a, evaluation & b
        counts = {"fit_a": int(fit_a.sum().item()), "fit_b": int(fit_b.sum().item()),
                  "fit_controls": int(fit_c.sum().item()), "eval_a": int(eval_a.sum().item()),
                  "eval_b": int(eval_b.sum().item()), "overlapping_members": int((ma & mb).sum().item())}
        record = dict(pair)
        record.update(index=index, rows=counts, fit_geo_shared_cos=None, fit_shared_direction=False,
                      fit_mean_norm_a=None, fit_mean_norm_b=None, fit_radial_gap_share=None,
                      orientation_fit=None, fit=None, heldout=None, eval_auc=None, eval_dprime=None,
                      fit_2sd_interval_overlap=None, eval_2sd_interval_overlap=None,
                      meaningful_shared_scalar=False, degenerate=False,
                      heldout_evidence_sufficient=min(counts["eval_a"], counts["eval_b"]) >= min_eval_examples)
        if min(counts["fit_a"], counts["fit_b"], counts["fit_controls"]) < min_fit_examples:
            record["status"] = "insufficient_fit_rows"
            records.append(record)
            continue
        center = z[fit_c].double().mean(dim=0)
        mean_a, mean_b = z[fit_a].double().mean(dim=0) - center, z[fit_b].double().mean(dim=0) - center
        norm_a, norm_b = mean_a.norm().item(), mean_b.norm().item()
        record.update(fit_mean_norm_a=norm_a, fit_mean_norm_b=norm_b)
        if min(norm_a, norm_b) <= 1e-12:
            record.update(status="degenerate_centered_mean", degenerate=True)
            records.append(record)
            continue
        unit_a, unit_b = mean_a / norm_a, mean_b / norm_b
        cosine = float((unit_a @ unit_b).clamp(-1, 1).item())
        shared = unit_a + unit_b
        record.update(fit_geo_shared_cos=cosine, fit_shared_direction=cosine >= parallel_cosine_threshold)
        if shared.norm().item() <= 1e-12:
            record.update(status="degenerate_shared_direction", degenerate=True)
            records.append(record)
            continue
        shared /= shared.norm()
        gap = mean_a - mean_b
        radial = float(((gap @ shared).square() / gap.square().sum()).item()) if gap.norm().item() > 1e-12 else None
        projection_gap = float((gap @ shared).item())
        orientation = 1 if projection_gap >= 0 else -1
        # Evaluate only two selected row subsets instead of materializing all
        # rows in float64 for a potentially wide GeoAE latent representation.
        fit_pa = ((z[fit_a].double() - center) @ shared) * orientation
        fit_pb = ((z[fit_b].double() - center) @ shared) * orientation
        eval_pa = ((z[eval_a].double() - center) @ shared) * orientation
        eval_pb = ((z[eval_b].double() - center) @ shared) * orientation
        fit_stats, heldout = _effect(fit_pa, fit_pb), _effect(eval_pa, eval_pb)
        record.update(status="measured", orientation_fit=orientation, fit_radial_gap_share=radial,
                      fit=fit_stats, heldout=heldout, eval_auc=heldout["auc"], eval_dprime=heldout["dprime"],
                      fit_2sd_interval_overlap=_interval_overlap(fit_pa, fit_pb),
                      eval_2sd_interval_overlap=_interval_overlap(eval_pa, eval_pb),
                      meaningful_shared_scalar=bool(record["fit_shared_direction"]
                                                    and record["heldout_evidence_sufficient"]))
        records.append(record)
    return _json_safe({
        "space": "geoae_latent_euclidean", "n_rows": n_rows, "latent_width": z.shape[1],
        "n_fit": int(fit.sum().item()), "n_eval": int(evaluation.sum().item()),
        "control_source": "explicit_mask" if controls is not None else "pair_nonmembers",
        "selection": {"centering": "fit_control_mean", "direction": "normalized_sum_of_fit_sibling_unit_means",
                      "orientation": "fit_sibling_projection_gap", "parallel_cosine_threshold": parallel_cosine_threshold,
                      "min_fit_examples": min_fit_examples, "min_eval_examples": min_eval_examples},
        "pairs": records,
        "summary": {"n_pairs": len(records), "status_counts": dict(Counter(p["status"] for p in records)),
                    "n_fit_parallel": sum(p["fit_shared_direction"] for p in records),
                    "n_meaningful_shared_scalar": sum(p["meaningful_shared_scalar"] for p in records)},
        "assumptions": [
            "Geometry is measured in GeoAE latent Euclidean coordinates, not raw residual space or native decoder directions.",
            "Membership provenance and document-disjoint masks are supplied by the caller; rows receive equal weight.",
            "Controls center, sibling direction selection and scalar orientation use fit rows only.",
            "Fit parallelism is fixed before evaluation; held-out scalar statistics do not discover or select pairs.",
            "Two-SD intervals are population dispersion summaries, not confidence intervals or guaranteed probability coverage.",
            "A scalar along nonparallel means is reported for diagnosis and is not treated as evidence for a shared range.",
        ],
    })


def _group_metrics(error_sums: dict[str, float], counts: dict[str, int], hidden: int,
                   normalizer: float) -> dict[str, Any]:
    result = {}
    for group, count in counts.items():
        mse = error_sums[group] / (count * hidden) if count else None
        result[group] = {"n_rows": count, "raw_mse": mse,
                         "normalized_mse": mse / normalizer if mse is not None and normalizer > 0 else None}
    return result


@torch.no_grad()
def reconstruction_report(
    h_raw: torch.Tensor,
    features: torch.Tensor,
    sae: Any,
    *,
    controls: Any | None = None,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    geoae_reconstruction: torch.Tensor | None = None,
    geoae_features: torch.Tensor | None = None,
    geoae_decoder_directions: torch.Tensor | None = None,
    geoae_bias: torch.Tensor | None = None,
    geoae_parameter_count: int | None = None,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Compare full and top-contribution SAE reconstructions in raw residual space.

    ``sae.decode(z)`` must be affine in coefficients and output raw residuals;
    ``sae.decoder_directions`` must contain its *unnormalized raw-space* decoder
    rows. Top-budget selection ranks ``abs(z_j) * norm(decoder_j)`` independently
    per row. The decode(0) bias is always retained, including at budget zero.

    One control-centered variance (per residual component) normalizes every MSE
    so groups and budgets share a denominator. If controls are omitted, all rows
    define this reference. Optional ``geoae_features`` and raw-space decoder
    rows provide an equivalent top-contribution-budget GeoAE curve. Its required
    ``geoae_bias`` can be [1, hidden] or [rows, hidden] to preserve token-dependent
    bypass contributions. Dictionary sizes and parameters remain separate from
    this matched active-contribution budget. No label association is used to
    choose active atoms. Optional ``geoae_reconstruction`` supplies its original
    full reconstruction independently of the budget curve.
    """
    h_raw, features = _matrix(h_raw, "h_raw"), _matrix(features, "features")
    if len(h_raw) != len(features):
        raise ValueError("h_raw and features must have the same number of rows")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n_rows, hidden = h_raw.shape
    width = features.shape[1]
    budgets = _budgets(budgets, width)
    directions = sae.decoder_directions
    if callable(directions):
        directions = directions()
    directions = _matrix(directions, "sae.decoder_directions")
    if tuple(directions.shape) != (width, hidden):
        raise ValueError("sae.decoder_directions must have shape [features, hidden]")
    if directions.device != features.device:
        raise ValueError("features and sae.decoder_directions must be on the same device")
    control = (torch.ones(n_rows, dtype=torch.bool, device=h_raw.device) if controls is None else
               _mask(controls, n_rows, h_raw.device, "controls"))
    if not bool(control.any()):
        raise ValueError("controls must contain at least one row")
    control_h = h_raw[control].float()
    reference_mean = control_h.mean(dim=0)
    normalizer = float((control_h - reference_mean).double().square().mean().item())
    masks = {"all": torch.ones_like(control), "controls": control, "noncontrols": ~control}
    counts = {name: int(mask.sum().item()) for name, mask in masks.items()}
    report_keys = ["bias_only", "full_sae"] + [f"budget_{k}" for k in budgets]
    geoae = None
    if geoae_reconstruction is not None:
        geoae = _matrix(geoae_reconstruction, "geoae_reconstruction")
        if geoae.shape != h_raw.shape:
            raise ValueError("geoae_reconstruction must match h_raw shape")
        report_keys.append("supplied_geoae_full")
    errors = {key: {name: 0.0 for name in masks} for key in report_keys}
    active_sums = {name: 0.0 for name in masks}
    budget_active_sums = {str(k): {name: 0.0 for name in masks} for k in budgets}
    geo_features, geo_directions, geo_bias, geo_norms = None, None, None, None
    supplied_geo = (geoae_features, geoae_decoder_directions, geoae_bias)
    if any(value is not None for value in supplied_geo):
        if any(value is None for value in supplied_geo):
            raise ValueError("GeoAE budget curves require features, decoder directions and bias together")
        geo_features = _matrix(geoae_features, "geoae_features")
        geo_directions = _matrix(geoae_decoder_directions, "geoae_decoder_directions")
        geo_bias = _matrix(geoae_bias, "geoae_bias")
        if len(geo_features) != n_rows or tuple(geo_directions.shape) != (geo_features.shape[1], hidden):
            raise ValueError("GeoAE features and decoder directions must match rows and hidden width")
        if geo_bias.shape[1] != hidden or geo_bias.shape[0] not in (1, n_rows):
            raise ValueError("geoae_bias must have shape [1, hidden] or [rows, hidden]")
        if geo_features.device != geo_directions.device:
            raise ValueError("GeoAE features and decoder directions must be on the same device")
        geo_norms = _decoder_norms(geo_directions)
        for key in ["geoae_bias_only", "geoae_affine_full"] + [f"geoae_budget_{k}" for k in budgets]:
            errors[key] = {name: 0.0 for name in masks}
    geo_active_sums = {name: 0.0 for name in masks}
    geo_budget_active_sums = {str(k): {name: 0.0 for name in masks} for k in budgets}
    norms = _decoder_norms(directions)
    zero = torch.zeros((1, width), dtype=features.dtype, device=features.device)
    bias = sae.decode(zero).detach().float()
    if tuple(bias.shape) != (1, hidden) or not bool(torch.isfinite(bias).all()):
        raise ValueError("sae.decode must produce finite raw residual rows")
    max_k = min(max(budgets), width)
    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        coefficients = features[start:stop]
        raw = h_raw[start:stop].to(features.device).float()
        full = sae.decode(coefficients).detach().float()
        if full.shape != raw.shape or not bool(torch.isfinite(full).all()):
            raise ValueError("sae.decode must produce finite raw residual rows")
        scores = coefficients.float().abs() * norms
        indices = scores.topk(max_k, dim=1, sorted=True).indices
        kept = coefficients.gather(1, indices).float()
        contributions = kept[:, :, None] * directions[indices].float()
        partial_sums = contributions.cumsum(dim=1)
        reconstructions = {"bias_only": bias.expand_as(raw), "full_sae": full}
        for k in budgets:
            reconstructions[f"budget_{k}"] = bias + partial_sums[:, min(k, width) - 1, :]
        if geoae is not None:
            reconstructions["supplied_geoae_full"] = geoae[start:stop].to(features.device).float()
        geo_active, geo_kept = None, None
        if geo_features is not None:
            geo_coefficients = geo_features[start:stop]
            geo_width = geo_features.shape[1]
            geo_indices = (geo_coefficients.float().abs() * geo_norms).topk(
                min(max(budgets), geo_width), dim=1, sorted=True).indices
            geo_kept = geo_coefficients.gather(1, geo_indices).float()
            geo_contributions = geo_kept[:, :, None] * geo_directions[geo_indices].float()
            geo_partial_sums = geo_contributions.cumsum(dim=1)
            raw_geo_bias = geo_bias if len(geo_bias) == 1 else geo_bias[start:stop]
            raw_geo_bias = raw_geo_bias.to(geo_features.device).float()
            reconstructions["geoae_bias_only"] = raw_geo_bias.to(features.device).expand_as(raw)
            reconstructions["geoae_affine_full"] = (
                raw_geo_bias + geo_coefficients.float() @ geo_directions.float()).to(features.device)
            for k in budgets:
                reconstructions[f"geoae_budget_{k}"] = (
                    raw_geo_bias + geo_partial_sums[:, min(k, geo_width) - 1, :]).to(features.device)
            geo_active = (geo_coefficients != 0).sum(dim=1)
        active = (coefficients != 0).sum(dim=1)
        for name, mask in masks.items():
            selected_rows = mask[start:stop].to(features.device)
            active_sums[name] += float(active[selected_rows].double().sum().item())
            for k in budgets:
                retained_active = (kept[:, :min(k, width)] != 0).sum(dim=1)
                budget_active_sums[str(k)][name] += float(retained_active[selected_rows].double().sum().item())
            if geo_active is not None:
                geo_selected = selected_rows.to(geo_features.device)
                geo_active_sums[name] += float(geo_active[geo_selected].double().sum().item())
                for k in budgets:
                    geo_retained = (geo_kept[:, :min(k, geo_features.shape[1])] != 0).sum(dim=1)
                    geo_budget_active_sums[str(k)][name] += float(geo_retained[geo_selected].double().sum().item())
            for key, reconstruction in reconstructions.items():
                row_errors = (raw[selected_rows] - reconstruction[selected_rows]).double().square().sum()
                errors[key][name] += float(row_errors.item())

    metrics = {key: _group_metrics(value, counts, hidden, normalizer) for key, value in errors.items()}
    parameter_count = getattr(sae, "parameter_count", None)
    if callable(parameter_count):
        parameter_count = parameter_count()
    if parameter_count is None and hasattr(sae, "parameters"):
        parameter_count = sum(p.numel() for p in sae.parameters())
    sae_curve = [{
        "requested_budget": k, "maximum_retained_atoms": min(k, width),
        "observed_retained_l0": {name: budget_active_sums[str(k)][name] / count if count else None
                                 for name, count in counts.items()},
        "metrics": metrics[f"budget_{k}"],
    } for k in budgets]
    geo_curve = None
    if geo_features is not None:
        geo_curve = [{
            "requested_budget": k, "maximum_retained_atoms": min(k, geo_features.shape[1]),
            "observed_retained_l0": {name: geo_budget_active_sums[str(k)][name] / count if count else None
                                     for name, count in counts.items()},
            "metrics": metrics[f"geoae_budget_{k}"],
        } for k in budgets]
    return _json_safe({
        "n_rows": n_rows, "hidden_width": hidden,
        "dictionary": {"feature_atoms": width, "decoder_scalar_parameters": width * hidden,
                       "total_scalar_parameters": parameter_count},
        "normalization": {"reference": "explicit_controls" if controls is not None else "all_rows",
                          "denominator": "mean squared raw residual deviation from reference mean",
                          "reference_n_rows": counts["controls"], "raw_variance_per_component": normalizer},
        "observed_l0": {name: active_sums[name] / count if count else None for name, count in counts.items()},
        "full_sae": metrics["full_sae"], "bias_only": metrics["bias_only"],
        "active_budget_reconstructions": sae_curve,
        "geoae_dictionary": None if geo_features is None else {
            "feature_atoms": geo_features.shape[1],
            "decoder_scalar_parameters": geo_features.shape[1] * hidden,
            "total_scalar_parameters": geoae_parameter_count,
        },
        "geoae_observed_l0": None if geo_features is None else {
            name: geo_active_sums[name] / count if count else None for name, count in counts.items()},
        "geoae_bias_only": metrics.get("geoae_bias_only"),
        "geoae_affine_full": metrics.get("geoae_affine_full"),
        "geoae_active_budget_reconstructions": geo_curve,
        "matched_active_budget_reconstructions": None if geo_curve is None else [
            {"requested_budget": k, "sae": sae_row, "geoae": geo_row}
            for k, sae_row, geo_row in zip(budgets, sae_curve, geo_curve)],
        "supplied_geoae_full": metrics.get("supplied_geoae_full"),
        "assumptions": [
            "SAE decoder rows include raw residual scaling and decode is affine in feature coefficients.",
            "Budget ranking is coefficient magnitude times decoder norm; decoder bias is retained.",
            "All rows receive equal weight; document-level confidence intervals require caller resampling.",
            "Active reconstruction budgets do not measure dictionary parameters or family handle counts.",
            "Supplied GeoAE full reconstruction does not imply matched dictionary size or active budget.",
            "Matched curves retain up to the same number of raw decoder contributions per row, with each model's bias retained.",
            "Token-dependent GeoAE bypass in the supplied bias is retained at every budget and reported in the bias-only baseline.",
            "Normalized MSE is null when the control-centered reference has zero variance.",
        ],
    })
