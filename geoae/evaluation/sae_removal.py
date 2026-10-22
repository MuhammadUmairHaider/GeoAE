"""Held-out, error-preserving SAE versus GeoAE concept removal.

This module deliberately knows nothing about model loading or dataset discovery.
The caller supplies raw final-block residuals and the *actual* final norm/logit
path, including any model-specific logit softcap. Feature selection, GeoAE ranges,
and strength matching use fit/calibration rows only. Evaluation rows never choose
an atom, interval, direction, or intervention strength.

SAE edits subtract selected feature contributions from the original residual;
GeoAE edits add a decoder difference to it. Neither replaces a residual with an
autoencoder reconstruction. Confidence intervals resample decision families,
retaining both sibling targets and all pairs of a family in the same cluster.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


def _mask(value, n: int, name: str) -> torch.Tensor:
    out = torch.as_tensor(value, dtype=torch.bool, device="cpu")
    if out.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {tuple(out.shape)}")
    return out


def _module_device(module, fallback: torch.device) -> torch.device:
    if isinstance(module, torch.nn.Module):
        for parameter in module.parameters():
            return parameter.device
        for buffer in module.buffers():
            return buffer.device
    return fallback


def _stats(matrix: torch.Tensor, rows: torch.Tensor, chunk: int = 64):
    """Streaming population moments, without a rows-by-full-SAE-width copy."""
    count = len(rows)
    if not count:
        raise ValueError("moments need at least one row")
    total = torch.zeros(matrix.shape[1], dtype=torch.float64, device=matrix.device)
    square = torch.zeros_like(total)
    for ids in rows.split(chunk):
        batch = matrix.index_select(0, ids.to(matrix.device)).double()
        total += batch.sum(0)
        square += batch.square().sum(0)
    mean = total / count
    var = (square / count - mean.square()).clamp_min(0)
    return mean.float(), var.float()


def select_features(features, target_mask, control_mask, fit_mask, budget: int):
    """Select target-enriched atoms by signed d-prime on fit rows only.

    A negative score identifies a feature absent from the target; ablating it is
    not a target removal. Such atoms and inactive target atoms are excluded.
    Returns ordered ids and their scores; the requested budget may be unmet.
    """
    features = torch.as_tensor(features)
    if features.ndim != 2 or budget < 1:
        raise ValueError("features must be a matrix and budget must be positive")
    n = len(features)
    fit = _mask(fit_mask, n, "fit_mask")
    target = _mask(target_mask, n, "target_mask")
    control = _mask(control_mask, n, "control_mask") & ~target
    mu_t, var_t = _stats(features, torch.where(fit & target)[0])
    mu_c, var_c = _stats(features, torch.where(fit & control)[0])
    dprime = (mu_t - mu_c) / ((var_t + var_c).mul(.5).sqrt().clamp_min(1e-8))
    valid = torch.isfinite(dprime) & (dprime > 0) & (mu_t > 0)
    scores = torch.where(valid, dprime, torch.full_like(dprime, -torch.inf))
    # Stable sorting makes ties reproducible across calls and feature budgets.
    order = torch.argsort(scores, descending=True, stable=True)
    ids = order[valid[order]][:budget]
    return ids.cpu().tolist(), dprime[ids].cpu().tolist()


def fit_range(z, target_mask, sibling_mask, control_mask, fit_mask, width: float = 2.):
    """Fit one shared direction and the target interval using only fit data.

    Each sibling's mean is centered on the same control mean. Their normalized
    directions receive equal weight, so a stronger sibling does not determine
    the axis. The target gate is its projected mean plus/minus ``width`` SD;
    removal replaces its centered coordinate with zero (the control mean).
    """
    z = torch.as_tensor(z)
    if z.ndim != 2 or width < 0:
        raise ValueError("z must be a matrix and range width must be nonnegative")
    n = len(z)
    fit = _mask(fit_mask, n, "fit_mask")
    target = _mask(target_mask, n, "target_mask")
    sibling = _mask(sibling_mask, n, "sibling_mask")
    control = _mask(control_mask, n, "control_mask") & ~target & ~sibling
    def mean(mask):
        return _stats(z, torch.where(fit & mask)[0])[0]
    center = mean(control)
    a, b = mean(target) - center, mean(sibling) - center
    na, nb = a.norm(), b.norm()
    if min(float(na), float(nb)) <= 1e-10:
        raise ValueError("degenerate sibling direction")
    ua, ub = a / na, b / nb
    u = ua + ub
    if float(u.norm()) <= 1e-8:
        # Antiparallel siblings still define an axis; report their alignment.
        u = ua
    u = u / u.norm()
    rows = torch.where(fit & target)[0].to(z.device)
    t = (z[rows].float() - center) @ u
    mu, sd = t.mean(), t.std(unbiased=False)
    lo, hi = float(mu - width * sd), float(mu + width * sd)
    sib_rows = torch.where(fit & sibling)[0].to(z.device)
    sib_t = (z[sib_rows].float() - center) @ u
    return {
        "center": center, "u": u, "lo": lo, "hi": hi,
        "mean": float(mu), "sd": float(sd),
        "sibling_alignment": float(ua @ ub),
        "fit_target_gate_fraction": float(((t >= lo) & (t <= hi)).float().mean()),
        "fit_sibling_gate_fraction": float(((sib_t >= lo) & (sib_t <= hi)).float().mean()),
    }


def calibrate_strength(strengths, fit_damage, level: float, monotonic_tol: float = 1e-6):
    """Interpolate alpha on a fit-only monotone curve, without extrapolation.

    Nonmonotone and unreachable curves are explicit exclusions, not sorted by
    damage or replaced by a running maximum. This preserves the intervention's
    strength ordering and avoids inventing a matching point.
    """
    x, y = np.asarray(strengths, dtype=float), np.asarray(fit_damage, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        raise ValueError("strength and damage curves must be matching nonempty vectors")
    if not np.isfinite(level) or level < 0 or np.any(np.diff(x) <= 0):
        raise ValueError("level must be nonnegative and strengths strictly increasing")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"status": "nonfinite", "alpha": None}
    if np.any(np.diff(y) < -monotonic_tol):
        return {"status": "nonmonotonic", "alpha": None}
    if level < y[0] or level > y[-1]:
        return {"status": "unreachable", "alpha": None,
                "fit_damage_min": float(y[0]), "fit_damage_max": float(y[-1])}
    # Earliest exact hit handles flat segments without dividing by zero.
    exact = np.flatnonzero(np.abs(y - level) <= 1e-12)
    if len(exact):
        return {"status": "matched", "alpha": float(x[exact[0]])}
    upper = int(np.flatnonzero(y >= level)[0])
    lower = upper - 1
    alpha = x[lower] + (x[upper] - x[lower]) * (level - y[lower]) / (y[upper] - y[lower])
    return {"status": "matched", "alpha": float(alpha),
            "bracket": [float(x[lower]), float(x[upper])]}


@torch.no_grad()
def score_edits(h, nxt, rows, edit: Callable, norm32, head32, strengths,
                logit_chunk: int = 64, device=None):
    """Per-row next-token delta NLL and KL(clean || edited), both in nats.

    ``edit(rows, clean_fp32, alpha)`` returns raw edited residuals. Only the
    supplied rows are scored. Unchanged rows are assigned exact zeros and never
    passed through a second logit computation, avoiding roundoff-only damage.
    Clean log probabilities are reused across strengths within each small chunk.
    """
    h, nxt = torch.as_tensor(h), torch.as_tensor(nxt)
    rows = torch.as_tensor(rows, dtype=torch.long, device="cpu")
    if logit_chunk < 1:
        raise ValueError("logit_chunk must be positive")
    device = torch.device(device) if device is not None else _module_device(head32, h.device)
    result = {float(a): {"delta_nll": np.zeros(len(rows), dtype=np.float64),
                        "kl": np.zeros(len(rows), dtype=np.float64)} for a in strengths}
    for start in range(0, len(rows), logit_chunk):
        ids = rows[start:start + logit_chunk]
        clean = h.index_select(0, ids.to(h.device)).to(device=device, dtype=torch.float32)
        labels = nxt.index_select(0, ids.to(nxt.device)).to(device=device, dtype=torch.long)
        clean_lp = torch.log_softmax(head32(norm32(clean)).float(), dim=-1)
        clean_nll = -clean_lp.gather(1, labels[:, None]).squeeze(1)
        for alpha, values in result.items():
            if alpha == 0:
                continue
            edited = edit(ids, clean, alpha).to(device=device, dtype=torch.float32)
            if edited.shape != clean.shape:
                raise ValueError("edit must preserve the residual shape")
            changed = (edited != clean).any(dim=1)
            if not bool(changed.any()):
                continue
            edited_lp = torch.log_softmax(head32(norm32(edited[changed])).float(), dim=-1)
            original_lp = clean_lp[changed]
            edited_nll = -edited_lp.gather(1, labels[changed, None]).squeeze(1)
            dnll = edited_nll - clean_nll[changed]
            kl = (original_lp.exp() * (original_lp - edited_lp)).sum(-1).clamp_min(0)
            positions = torch.where(changed)[0].cpu().numpy() + start
            values["delta_nll"][positions] = dnll.cpu().double().numpy()
            values["kl"][positions] = kl.cpu().double().numpy()
    return result


def _mean_metrics(scored, positions):
    if not len(positions):
        return {"n": 0, "delta_nll": None, "kl": None}
    return {"n": int(len(positions)),
            "delta_nll": float(np.mean(scored["delta_nll"][positions])),
            "kl": float(np.mean(scored["kl"][positions]))}


def _sample_rows(mask, cap, rng):
    rows = torch.where(mask)[0].numpy()
    if cap is not None and len(rows) > cap:
        rows = np.sort(rng.choice(rows, size=cap, replace=False))
    return torch.as_tensor(rows, dtype=torch.long)


def family_bootstrap(values: Mapping[str, Sequence[float]], samples: int = 500, seed: int = 0):
    """Equal-family average and percentile CI, with all sibling rows clustered."""
    means = np.asarray([np.mean(v) for v in values.values() if len(v)], dtype=np.float64)
    means = means[np.isfinite(means)]
    if not len(means):
        return {"n_families": 0, "mean": None, "ci95": None}
    out = {"n_families": int(len(means)), "mean": float(means.mean()), "ci95": None}
    if samples > 0 and len(means) > 1:
        rng = np.random.default_rng(seed)
        boot = means[rng.integers(0, len(means), size=(samples, len(means)))].mean(1)
        out["ci95"] = [float(x) for x in np.quantile(boot, [.025, .975])]
    return out


def summarize_removal(records, bootstrap_samples=500, seed=0):
    """Summaries and paired SAE-minus-GeoAE differences on identical targets."""
    matched = [r for r in records if r["status"] == "matched"]
    grouped = defaultdict(list)
    for row in matched:
        grouped[(row["arm"], row["feature_budget"], row["requested_damage"])].append(row)
    summaries = []
    for (arm, budget, level), rows in grouped.items():
        item = {"arm": arm, "feature_budget": budget, "requested_damage": level,
                "n_targets": len(rows)}
        for group in ("target", "sibling", "random"):
            for metric in ("delta_nll", "kl"):
                values = defaultdict(list)
                for row in rows:
                    values[row["family"]].append(row["evaluation"][group][metric])
                item[f"{group}_{metric}"] = family_bootstrap(values, bootstrap_samples, seed)
        summaries.append(item)
    lookup = {(r["pair_id"], r["target"], r["requested_damage"], r["arm"],
               r["feature_budget"]): r for r in matched}
    comparisons = []
    for (arm, budget, level), rows in grouped.items():
        if arm not in ("sae", "sae_range"):
            continue
        paired = []
        for row in rows:
            other = lookup.get((row["pair_id"], row["target"], level, "geoae_range", None))
            if other is not None:
                paired.append((row, other))
        item = {"arm": f"{arm}_minus_geoae_range", "feature_budget": budget,
                "requested_damage": level, "n_paired_targets": len(paired),
                "n_unpaired_sae_targets": len(rows) - len(paired)}
        for group in ("target", "sibling", "random"):
            for metric in ("delta_nll", "kl"):
                values = defaultdict(list)
                for row, other in paired:
                    values[row["family"]].append(
                        row["evaluation"][group][metric] - other["evaluation"][group][metric])
                item[f"{group}_{metric}"] = family_bootstrap(values, bootstrap_samples, seed)
        comparisons.append(item)
    return {"summaries": summaries, "paired_comparisons": comparisons}


@torch.no_grad()
def run_removal(
    h, z, tok, nxt, memberships, pairs, fit_mask, eval_mask, controls,
    ae, norm_mean, norm_std, sae, features, norm32, head32, *,
    strengths=(0., .25, .5, 1., 2.), feature_budgets=(1, 2, 4, 8),
    target_damage=(.05, .1, .2), logit_chunk=64, seed=0, min_rows=8,
    bootstrap_samples=500, calibration_mask=None, max_rows_per_group=128,
    range_width=2., monotonic_tol=1e-6, calibration_abs_tol=.01,
    calibration_rel_tol=.1, calibration_max_steps=12, include_sae_range=False, device=None,
) -> dict[str, Any]:
    """Run both removal directions of every sibling pair and return JSON data.

    ``controls`` is a common boolean mask, or a label->mask mapping. For each
    pair, its two control masks are intersected, then both siblings are excluded.
    ``sae.decoder_directions`` is the unnormalized raw-residual matrix [F,H].
    ``norm_std`` converts GeoAE decoder deltas back to raw residual scale.

    ``fit_mask`` selects feature/range fitting rows. ``calibration_mask`` defaults
    to the same fit rows, or may select a separate disjoint non-evaluation set.
    The default reuses fitting data for calibration and states that in metadata.
    ``max_rows_per_group`` caps logit-scoring rows only; fitting uses all fit rows.
    Every arm uses the same sampled calibration and evaluation rows for a target.
    ``include_sae_range`` adds an auxiliary top-1 SAE coefficient range edit:
    its fit mean plus/minus range_width SD gates the coefficient, which is moved
    toward the common fit control mean. This tests the conditional edit operator
    on the same selected SAE atom used by ordinary feature ablation.

    Main matching uses next-token delta NLL. Alphas start from calibration-grid
    interpolation and are refined by fit-only bracket bisection until attained
    damage is within min(absolute_tolerance, relative_tolerance*level). Failed
    refinement is excluded. Both attained damages are reported, since held-out
    damage need not equal requested fit damage. Evaluation curves never choose
    strengths and are never interpolated or extrapolated.
    """
    h, z = torch.as_tensor(h), torch.as_tensor(z)
    tok, nxt, features = torch.as_tensor(tok), torch.as_tensor(nxt), torch.as_tensor(features)
    n = len(h)
    if h.ndim != 2 or z.ndim != 2 or features.ndim != 2 or len(z) != n or len(features) != n:
        raise ValueError("h, z, and features must be row-aligned matrices")
    if tok.shape != (n,) or nxt.shape != (n,):
        raise ValueError("tok and nxt must have shape [N]")
    fit = _mask(fit_mask, n, "fit_mask")
    evaluate = _mask(eval_mask, n, "eval_mask")
    if bool((fit & evaluate).any()):
        raise ValueError("fit and evaluation masks overlap")
    calibration = fit if calibration_mask is None else _mask(calibration_mask, n, "calibration_mask")
    if bool((calibration & evaluate).any()):
        raise ValueError("calibration and evaluation masks overlap")
    if min_rows < 1 or (max_rows_per_group is not None and max_rows_per_group < min_rows):
        raise ValueError("row cap must be at least min_rows, and min_rows must be positive")
    if calibration_abs_tol <= 0 or calibration_rel_tol <= 0 or calibration_max_steps < 0:
        raise ValueError("calibration tolerances must be positive and max steps nonnegative")
    strengths = sorted(set(float(a) for a in strengths))
    budgets = sorted(set(int(b) for b in feature_budgets))
    levels = sorted(set(float(d) for d in target_damage))
    if not strengths or strengths[0] != 0 or not np.isfinite(strengths).all():
        raise ValueError("strengths must be finite, nonnegative, and include zero")
    if not budgets or budgets[0] < 1 or not levels or levels[0] < 0 or not np.isfinite(levels).all():
        raise ValueError("feature budgets must be positive and target damages nonnegative")
    masks = {k: _mask(v, n, f"membership {k}") for k, v in memberships.items()}
    std = torch.as_tensor(norm_std).float()
    mean = torch.as_tensor(norm_mean)
    if std.numel() not in (1, h.shape[1]) or mean.numel() not in (1, h.shape[1]):
        raise ValueError("normalization moments must be scalar or hidden-width vectors")
    if not torch.isfinite(std).all() or bool((std < 0).any()):
        raise ValueError("normalization std must be finite and nonnegative")
    std = std.reshape(-1)
    directions = sae.decoder_directions
    if callable(directions):
        directions = directions()
    if tuple(directions.shape) != (features.shape[1], h.shape[1]):
        raise ValueError("SAE decoder_directions must have shape [features, hidden]")
    scoring_device = torch.device(device) if device is not None else _module_device(head32, h.device)
    ae_device = _module_device(ae, z.device)
    rng = np.random.default_rng(seed)
    records, exclusions, curves = [], [], []
    for pair_index, pair in enumerate(pairs):
        a, b = pair["a"], pair["b"]
        family = str(pair.get("family", f"{a}|{b}"))
        pair_id = str(pair.get("id", f"{a}|{b}|{pair_index}"))
        base = {"pair_id": pair_id, "family": family, "a": str(a), "b": str(b)}
        if a not in masks or b not in masks:
            exclusions.append({**base, "reason": "missing_membership"})
            continue
        ma, mb = masks[a], masks[b]
        # Ambiguous multilabel rows cannot count as either held-out alternative.
        target_masks = {a: ma & ~mb, b: mb & ~ma}
        if isinstance(controls, Mapping):
            if a not in controls or b not in controls:
                exclusions.append({**base, "reason": "missing_control_mask"})
                continue
            control = _mask(controls[a], n, "control_a") & _mask(controls[b], n, "control_b")
        else:
            control = _mask(controls, n, "controls")
        control = control & ~ma & ~mb
        counts = {f"{split}_{group}": int((mask & split_mask).sum())
                  for split, split_mask in (("fit", fit), ("calibration", calibration), ("eval", evaluate))
                  for group, mask in (("a", target_masks[a]), ("b", target_masks[b]), ("random", control))}
        if min(counts.values()) < min_rows:
            exclusions.append({**base, "reason": "insufficient_rows", "counts": counts})
            continue
        for target, sibling in ((a, b), (b, a)):
            target_mask, sibling_mask = target_masks[target], target_masks[sibling]
            direction_base = {**base, "target": str(target), "sibling": str(sibling), "counts": counts}
            try:
                geometry = fit_range(z, target_mask, sibling_mask, control, fit, range_width)
            except ValueError as error:
                exclusions.append({**direction_base, "reason": "degenerate_direction", "detail": str(error)})
                continue
            geometry_json = {k: v for k, v in geometry.items() if k not in ("center", "u")}
            cal_rows = _sample_rows(calibration & target_mask, max_rows_per_group, rng)
            eval_groups = {name: _sample_rows(evaluate & mask, max_rows_per_group, rng)
                           for name, mask in (("target", target_mask), ("sibling", sibling_mask), ("random", control))}
            eval_rows = torch.cat(list(eval_groups.values()))
            positions, offset = {}, 0
            for name, ids in eval_groups.items():
                positions[name] = np.arange(offset, offset + len(ids))
                offset += len(ids)
            all_features, all_scores = select_features(features, target_mask, control, fit, max(budgets))
            sae_range_geometry = None

            def geo_edit(ids, clean, alpha, gated=True):
                latent = z.index_select(0, ids.to(z.device)).to(ae_device, dtype=torch.float32)
                tokens = tok.index_select(0, ids.to(tok.device)).to(ae_device)
                center = geometry["center"].to(ae_device)
                u = geometry["u"].to(ae_device)
                coordinate = (latent - center) @ u
                gate = ((coordinate >= geometry["lo"]) & (coordinate <= geometry["hi"])) if gated else torch.ones_like(coordinate, dtype=torch.bool)
                displacement = -alpha * coordinate * gate
                edited_latent = latent + displacement[:, None] * u
                delta = (ae.decode(edited_latent, tokens) - ae.decode(latent, tokens)).float()
                delta *= std.to(ae_device)
                return clean + delta.to(clean.device)

            arms = [("geoae_range", None, lambda ids, clean, alpha: geo_edit(ids, clean, alpha, True), [], []),
                    ("geoae_direction", None, lambda ids, clean, alpha: geo_edit(ids, clean, alpha, False), [], [])]
            for budget in budgets:
                selected = all_features[:budget]
                scores = all_scores[:budget]
                # Only the chosen columns are copied to the scoring device.
                ids = torch.tensor(selected, dtype=torch.long, device=directions.device)
                selected_directions = directions.index_select(0, ids).to(scoring_device, dtype=torch.float32)
                def sae_edit(rows, clean, alpha, feature_ids=selected, decoder=selected_directions):
                    ids0 = torch.tensor(feature_ids, dtype=torch.long, device=features.device)
                    values = features[rows.to(features.device)[:, None], ids0[None, :]].to(clean.device, dtype=torch.float32)
                    return clean - alpha * (values @ decoder)
                arms.append(("sae", budget, sae_edit, selected, scores))
            if include_sae_range and all_features:
                feature_id = all_features[0]
                coefficient = features[:, feature_id]
                fit_target = coefficient[(fit & target_mask).to(features.device)].float()
                fit_control = coefficient[(fit & control).to(features.device)].float()
                fit_sibling = coefficient[(fit & sibling_mask).to(features.device)].float()
                coefficient_mean = float(fit_target.mean())
                coefficient_sd = float(fit_target.std(unbiased=False))
                control_mean = float(fit_control.mean())
                lo = coefficient_mean - range_width * coefficient_sd
                hi = coefficient_mean + range_width * coefficient_sd
                sae_range_geometry = {
                    "feature_id": feature_id, "mean": coefficient_mean, "sd": coefficient_sd,
                    "control_mean": control_mean, "lo": lo, "hi": hi,
                    "fit_target_gate_fraction": float(((fit_target >= lo) & (fit_target <= hi)).float().mean()),
                    "fit_sibling_gate_fraction": float(((fit_sibling >= lo) & (fit_sibling <= hi)).float().mean()),
                }
                direction = directions[feature_id].to(scoring_device, dtype=torch.float32)
                def sae_range_edit(rows, clean, alpha):
                    value = coefficient.index_select(0, rows.to(coefficient.device)).to(clean.device, dtype=torch.float32)
                    gate = (value >= lo) & (value <= hi)
                    delta = alpha * (control_mean - value) * gate
                    return clean + delta[:, None] * direction
                arms.append(("sae_range", 1, sae_range_edit, [feature_id], all_scores[:1]))
            elif include_sae_range:
                exclusions.append({**direction_base, "arm": "sae_range", "reason": "no_positive_fit_association"})
            for arm, budget, edit, selected, scores in arms:
                arm_base = {**direction_base, "arm": arm, "feature_budget": budget,
                            "selected_features": selected, "feature_dprime": scores,
                            "selected_feature_count": len(selected) if arm in ("sae", "sae_range") else None,
                            "geometry": geometry_json}
                if arm == "sae_range":
                    arm_base["sae_feature_range"] = sae_range_geometry
                fitted = score_edits(h, nxt, cal_rows, edit, norm32, head32, strengths,
                                     logit_chunk, scoring_device)
                damage_curve = [float(fitted[alpha]["delta_nll"].mean()) for alpha in strengths]
                curves.append({**arm_base, "strengths": strengths, "calibration_target_delta_nll": damage_curve,
                               "calibration_target_kl": [float(fitted[alpha]["kl"].mean()) for alpha in strengths],
                               "calibration_n": len(cal_rows)})
                for level in levels:
                    match = calibrate_strength(strengths, damage_curve, level, monotonic_tol)
                    row = {**arm_base, "requested_damage": level, **match}
                    if match["status"] != "matched":
                        records.append(row)
                        continue
                    alpha = match["alpha"]
                    attained_cal = fitted.get(alpha)
                    if attained_cal is None:
                        attained_cal = score_edits(h, nxt, cal_rows, edit, norm32, head32, [alpha],
                                                  logit_chunk, scoring_device)[alpha]
                    tolerance = min(calibration_abs_tol, calibration_rel_tol * level) if level > 0 else calibration_abs_tol
                    iterations = 0
                    if "bracket" in match:
                        lower, upper = match["bracket"]
                        while (abs(float(attained_cal["delta_nll"].mean()) - level) > tolerance
                               and iterations < calibration_max_steps):
                            if float(attained_cal["delta_nll"].mean()) < level:
                                lower = alpha
                            else:
                                upper = alpha
                            alpha = (lower + upper) * .5
                            attained_cal = score_edits(h, nxt, cal_rows, edit, norm32, head32, [alpha],
                                                      logit_chunk, scoring_device)[alpha]
                            iterations += 1
                    row.update(alpha=alpha, calibration_tolerance=tolerance,
                               calibration_refinement_steps=iterations)
                    row["calibration_attained"] = _mean_metrics(attained_cal, np.arange(len(cal_rows)))
                    if abs(row["calibration_attained"]["delta_nll"] - level) > tolerance:
                        row["status"] = "calibration_unresolved"
                        records.append(row)
                        continue
                    evaluated = score_edits(h, nxt, eval_rows, edit, norm32, head32, [alpha],
                                            logit_chunk, scoring_device)[alpha]
                    row["evaluation"] = {name: _mean_metrics(evaluated, ids0) for name, ids0 in positions.items()}
                    records.append(row)
    status_counts = defaultdict(int)
    for row in records:
        status_counts[row["status"]] += 1
    return {
        "metadata": {
            "protocol": "fit-calibrated held-out error-preserving ablation",
            "calibration_metric": "next_token_delta_nll_nats", "kl_direction": "clean||edited",
            "calibration_reuses_feature_and_range_fit_rows": bool((calibration & fit).any()),
            "strengths": strengths, "feature_budgets": budgets, "target_damage": levels,
            "range_width_sd": range_width, "max_rows_per_group": max_rows_per_group,
            "include_sae_range_operator_control": include_sae_range,
            "min_rows": min_rows, "seed": seed, "bootstrap_unit": "decision_family",
            "bootstrap_samples": bootstrap_samples,
            "feature_selection": "positive target-versus-random-control fit dprime",
            "matching": "fit interpolation and bracket bisection; held-out strength rerun; no evaluation interpolation",
            "calibration_absolute_tolerance": calibration_abs_tol,
            "calibration_relative_tolerance": calibration_rel_tol,
            "calibration_tolerance_rule": "min(absolute, relative * requested_damage)",
            "calibration_max_refinement_steps": calibration_max_steps,
            "row_counts": {"total": n, "fit": int(fit.sum()), "calibration": int(calibration.sum()), "eval": int(evaluate.sum())},
        },
        "records": records, "curves": curves, "excluded_pairs": exclusions,
        "status_counts": dict(status_counts),
        **summarize_removal(records, bootstrap_samples, seed),
    }
