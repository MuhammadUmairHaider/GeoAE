"""Frozen-cohort, document-disjoint controls for named Atlas distinctions.

Input NPZ arrays: h[N,D], tok/nxt[N], doc[N], pair[N], label[N] (1 target,
0 sibling, -1 unrelated), split[N] (fit/calibration/test). Geometry is fit
only on FIT. Intervention strength is chosen only on CALIBRATION. TEST NLL
increments measure predictive disruption, not semantic concept erasure.

The AE has two deliberately separate readouts: ``ae_recon`` edits relative
to its decoded baseline; ``ae_delta`` adds only the decoded change to the
original residual. The latter preserves reconstruction error. Both use the
same fitted latent geometry, and neither mixes its baseline with the other.
Importing this module does not load checkpoints, a language model, or CUDA.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def validate_data(data):
    """Reject ambiguous memberships, invalid labels, and document split leakage."""
    required = {"h", "tok", "nxt", "doc", "pair", "label", "split"}
    if required - data.keys():
        raise ValueError(f"missing arrays: {sorted(required - data.keys())}")
    h = np.asarray(data["h"])
    if h.ndim != 2 or not len(h) or not np.isfinite(h).all():
        raise ValueError("h must be a nonempty finite N-by-D matrix")
    for key in required - {"h"}:
        if np.asarray(data[key]).shape != (len(h),):
            raise ValueError(f"{key} must have shape (N,)")
    label = np.asarray(data["label"])
    if not np.isin(label, [-1, 0, 1]).all():
        raise ValueError("label must be -1 (random), 0 (sibling), or 1 (target)")
    split = np.asarray(data["split"]).astype(str)
    if not np.isin(split, ["fit", "calibration", "test"]).all():
        raise ValueError("split must be fit, calibration, or test")
    doc_splits = {}
    for doc, part in zip(np.asarray(data["doc"]).astype(str), split):
        previous = doc_splits.setdefault(doc, part)
        if previous != part:
            raise ValueError(f"document split leakage: {doc!r} appears in {previous} and {part}")
    for key in ("tok", "nxt"):
        values = np.asarray(data[key])
        if not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
            raise ValueError(f"{key} must contain nonnegative integer token IDs")
    # One document cannot be both siblings of one pair. Random rows may occur
    # in a concept document if the producer explicitly samples other positions.
    memberships = {}
    for pair, doc, lab in zip(np.asarray(data["pair"]).astype(str),
                              np.asarray(data["doc"]).astype(str), label):
        if lab == -1:
            continue
        key = (pair, doc)
        if key in memberships and memberships[key] != int(lab):
            raise ValueError(f"ambiguous sibling membership for pair/document {key}")
        memberships[key] = int(lab)


def data_fingerprint(data):
    """Content fingerprint includes cohorts and cached residuals, not file paths."""
    digest = hashlib.sha256()
    for name in sorted(data):
        arr = np.asarray(data[name])
        digest.update(name.encode())
        digest.update(str(arr.shape).encode())
        if arr.dtype.kind in "OUS":
            digest.update(json.dumps(arr.astype(str).tolist(), ensure_ascii=False,
                                     separators=(",", ":")).encode())
        else:
            digest.update(arr.dtype.str.encode())
            digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def document_means(values, docs):
    """Equal document weights, regardless of the number of captured positions."""
    x = torch.as_tensor(values)
    keys, inverse = np.unique(np.asarray(docs).astype(str), return_inverse=True)
    if not len(keys):
        return x[:0], keys
    index = torch.as_tensor(inverse, dtype=torch.long, device=x.device)
    total = torch.zeros((len(keys),) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    total.index_add_(0, index, x)
    counts = torch.bincount(index, minlength=len(keys)).to(x.dtype)
    return total / counts.reshape((-1,) + (1,) * (x.ndim - 1)), keys


def _document_weights(docs, device):
    _, inverse, counts = np.unique(np.asarray(docs).astype(str), return_inverse=True,
                                    return_counts=True)
    return torch.as_tensor(1.0 / counts[inverse], dtype=torch.float64, device=device)


def _weighted_quantile(values, weights, q):
    order = values.argsort()
    cumulative = weights[order].cumsum(0)
    index = torch.searchsorted(cumulative, cumulative[-1] * q).clamp_max(len(values) - 1)
    return float(values[order[index]])


@torch.no_grad()
def fit_direction(x, docs, target, sibling, fit, *, kind="difference", quantiles=(.05, .95), background=None):
    """Fit a direction/range with equal document weights; never inspect other splits.

    Difference uses the contrast of class means. Shared uses the bisector of
    the class means after centering on the FIT background mean, matching the radial
    separation claim. Every substrate fits its own direction on the same docs.
    """
    x = torch.as_tensor(x).float()
    target = torch.as_tensor(target, dtype=torch.bool, device=x.device)
    sibling = torch.as_tensor(sibling, dtype=torch.bool, device=x.device)
    fit = torch.as_tensor(fit, dtype=torch.bool, device=x.device)
    docs = np.asarray(docs).astype(str)
    t, s = fit & target, fit & sibling
    if not t.any() or not s.any():
        raise ValueError("direction needs FIT rows from both siblings")
    tdoc, sdoc = docs[t.cpu().numpy()], docs[s.cpu().numpy()]
    mu_t = document_means(x[t], tdoc)[0].double().mean(0)
    mu_s = document_means(x[s], sdoc)[0].double().mean(0)
    if kind == "difference":
        direction = mu_t - mu_s
    elif kind == "shared":
        if background is None:
            raise ValueError("shared direction requires FIT background rows")
        background = torch.as_tensor(background, dtype=torch.bool, device=x.device) & fit
        if not background.any():
            raise ValueError("shared direction requires FIT background rows")
        center = document_means(x[background], docs[background.cpu().numpy()])[0].double().mean(0)
        a, b = mu_t - center, mu_s - center
        direction = a / a.norm().clamp_min(1e-12) + b / b.norm().clamp_min(1e-12)
    else:
        raise ValueError(f"unknown direction kind: {kind}")
    if direction.norm() < 1e-8:
        raise ValueError(f"degenerate {kind} direction")
    u = (direction / direction.norm()).float()
    pt, ps = x[t] @ u, x[s] @ u
    weights = _document_weights(tdoc, x.device)
    lo = _weighted_quantile(pt, weights, quantiles[0])
    hi = _weighted_quantile(pt, weights, quantiles[1])
    return {"u": u, "lo": lo, "hi": hi,
            "target_mean": float(mu_t @ u.double()), "sibling_mean": float(mu_s @ u.double()),
            "kind": kind, "quantiles": list(quantiles),
            "fit_target_docs": sorted(set(tdoc)), "fit_sibling_docs": sorted(set(sdoc)),
            "fit_target_gate_fraction": float(document_means(((pt >= lo) & (pt <= hi)).float(), tdoc)[0].mean()),
            "fit_sibling_gate_fraction": float(document_means(((ps >= lo) & (ps <= hi)).float(), sdoc)[0].mean())}


@torch.no_grad()
def direction_edit(x, fitted, alpha, *, gated=True):
    """Push the projected coordinate toward the sibling mean.

    A zero edit returns the exact input object. Ungated uses exactly the same
    push, so this is an ablation of the range gate alone.
    """
    if alpha == 0:
        return x, torch.zeros(len(x), dtype=torch.bool, device=x.device)
    projection = x @ fitted["u"]
    gate = ((projection >= fitted["lo"]) & (projection <= fitted["hi"])) if gated else torch.ones(
        len(x), dtype=torch.bool, device=x.device)
    delta = alpha * (fitted["sibling_mean"] - projection)
    delta = torch.where(gate, delta, torch.zeros_like(delta))
    changed = delta != 0
    edited = x.clone()
    edited[changed] += delta[changed, None] * fitted["u"]
    return edited, changed


def _summary(values, docs, *, bootstrap=1000, seed=0):
    means, keys = document_means(torch.as_tensor(values, dtype=torch.float64), docs)
    means = means.cpu().numpy()
    if not len(means):
        return {"mean": None, "ci95": None, "n_docs": 0, "n_rows": 0}
    ci = None
    # A single document provides no sampling uncertainty estimate.
    if bootstrap and len(means) >= 2:
        rng = np.random.default_rng(seed)
        estimates = np.empty(bootstrap)
        for start in range(0, bootstrap, 128):
            count = min(128, bootstrap - start)
            estimates[start:start + count] = means[rng.integers(0, len(means), (count, len(means)))].mean(1)
        ci = np.quantile(estimates, [.025, .975]).tolist()
    return {"mean": float(means.mean()), "ci95": ci, "n_docs": len(keys),
            "n_rows": len(values), "doc_ids": keys.tolist(), "doc_means": means.tolist()}


def _projection_dprime(projection, docs, target_idx, sibling_idx):
    """Position-level moments with equal document weight, plus doc-mean dprime."""
    moments = []
    doc_values = []
    for idx in (target_idx, sibling_idx):
        p = projection[idx].double()
        first, _ = document_means(p, docs[idx])
        second, _ = document_means(p.square(), docs[idx])
        moments.append((first.mean(), second.mean() - first.mean().square()))
        doc_values.append(first)
    ma, va = moments[0]
    mb, vb = moments[1]
    doc_a, doc_b = doc_values
    return {"test_position_dprime_equal_document_weight": float((ma - mb).abs() / (
        (va + vb) / 2).clamp_min(1e-24).sqrt()), "test_document_dprime": float((
        doc_a.mean() - doc_b.mean()).abs() / ((doc_a.var(unbiased=False) + doc_b.var(unbiased=False)
                                             ) / 2).clamp_min(1e-24).sqrt())}


class Readout:
    """The final norm/head in fp32, with architecture-specific logit softcap."""
    def __init__(self, norm, head, *, chunk_size=128, logit_softcap=None):
        self.norm, self.head = norm, head
        self.chunk_size = chunk_size
        self.logit_softcap = logit_softcap

    @torch.no_grad()
    def nll(self, h, nxt):
        result = torch.empty(len(h), dtype=torch.float32, device=h.device)
        for start in range(0, len(h), self.chunk_size):
            end = start + self.chunk_size
            logits = self.head(self.norm(h[start:end].float())).float()
            if self.logit_softcap:
                logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
            result[start:end] = -torch.log_softmax(logits, -1).gather(1, nxt[start:end, None]).squeeze(1)
        return result

    @torch.no_grad()
    def delta_nll(self, baseline, edited, nxt, changed):
        result = torch.zeros(len(baseline), dtype=torch.float32, device=baseline.device)
        if changed.any():
            result[changed] = self.nll(edited[changed], nxt[changed]) - self.nll(baseline[changed], nxt[changed])
        return result


def calibrate_strength(alpha_grid, damages, level):
    """First ascending bracket in alpha order; never sort or extrapolate damage."""
    alpha = np.asarray(alpha_grid, float)
    damage = np.asarray(damages, float)
    if len(alpha) != len(damage) or not len(alpha) or alpha[0] != 0 or np.any(np.diff(alpha) <= 0):
        raise ValueError("alpha grid must start at zero and strictly increase")
    if level < 0:
        raise ValueError("requested damage must be nonnegative")
    if not np.isfinite(damage).all():
        return {"status": "nonfinite", "alpha": None}
    if level == 0:
        return {"status": "matched", "alpha": 0., "bracket": [0., 0.]}
    for i in range(1, len(alpha)):
        if damage[i - 1] <= level <= damage[i] and damage[i] > damage[i - 1]:
            fraction = (level - damage[i - 1]) / (damage[i] - damage[i - 1])
            return {"status": "bracketed", "alpha": float(alpha[i - 1] + fraction * (alpha[i] - alpha[i - 1])),
                    "bracket": alpha[i - 1:i + 1].tolist(),
                    "calibration_curve_monotone": bool(np.all(np.diff(damage) >= -1e-6))}
    return {"status": "unreachable", "alpha": None,
            "calibration_curve_monotone": bool(np.all(np.diff(damage) >= -1e-6))}


def _serial_fit(fitted):
    return {key: (value.cpu().tolist() if isinstance(value, torch.Tensor) else value)
            for key, value in fitted.items()}


@torch.no_grad()
def run_experiment(data, ae, norm_mean, norm_std, readout, *, device="cpu",
                   alpha_grid=(0., .125, .25, .5, 1., 2., 4.), target_nll=(.1, .25, .5),
                   direction_kinds=("difference", "shared"), substrates=("base", "hb", "ae_recon"),
                   min_docs=3, bootstrap=1000, seed=42, calibration_tolerance=.005,
                   calibration_steps=12, encode_chunk=2048, metadata=None, test_match_tolerance=.02,
                   progress=None):
    """Evaluate frozen cohorts; returns JSON-safe records and paired doc CIs."""
    validate_data(data)
    # Validate grids before expensive model scoring.
    calibrate_strength(alpha_grid, np.zeros(len(alpha_grid)), 0)
    if set(substrates) - {"base", "raw", "hb", "ae_recon", "ae_delta"}:
        raise ValueError("unknown substrate")
    if min_docs < 1 or calibration_steps < 0 or calibration_tolerance <= 0:
        raise ValueError("invalid minimum docs or calibration settings")
    if any(level < 0 for level in target_nll):
        raise ValueError("requested damage must be nonnegative")
    h = torch.as_tensor(data["h"], dtype=torch.float32, device=device)
    tok = torch.as_tensor(data["tok"], dtype=torch.long, device=device)
    nxt = torch.as_tensor(data["nxt"], dtype=torch.long, device=device)
    docs = np.asarray(data["doc"]).astype(str)
    pair_ids = np.asarray(data["pair"]).astype(str)
    split = np.asarray(data["split"]).astype(str)
    labels = np.asarray(data["label"])
    mean = torch.as_tensor(norm_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(norm_std, dtype=torch.float32, device=device)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("normalization moments must be finite with positive standard deviation")
    x = (h - mean) / std
    zlist, rlist, blist = [], [], []
    for start in range(0, len(h), encode_chunk):
        end = start + encode_chunk
        z = ae.encode(x[start:end], tok[start:end]).float()
        zlist.append(z)
        rlist.append(ae.decode(z, tok[start:end]).float() * std + mean)
        if "hb" in substrates:
            if not callable(getattr(ae, "token_bias", None)):
                raise ValueError("hb requires the checkpoint's stored token-bias lookup")
            bias = ae.token_bias(tok[start:end])
            if bias is None:
                raise ValueError("hb requires a token-bias checkpoint")
            blist.append(bias.float())
    latent, recon = torch.cat(zlist), torch.cat(rlist)
    values_by_space = {"base": x, "raw": h, "latent": latent}
    if blist:
        values_by_space["hb"] = x - torch.cat(blist)
    raw_nll = readout.nll(h, nxt)
    recon_nll = readout.nll(recon, nxt)
    result = {"protocol": "ssl_named_concept_controls_v1", "data_sha256": data_fingerprint(data),
              "metadata": metadata or {}, "settings": {"alpha_grid": list(alpha_grid),
              "target_nll": list(target_nll), "direction_kinds": list(direction_kinds),
              "substrates": list(substrates), "min_docs": min_docs, "bootstrap": bootstrap,
              "seed": seed, "calibration_tolerance": calibration_tolerance,
              "calibration_steps": calibration_steps, "equal_document_weight": True,
              "test_match_tolerance": test_match_tolerance},
              "interpretation": "Next-token NLL increment is predictive damage, not semantic erasure.",
              "baseline_semantics": {"base": "normalized residual direction; exact original residual baseline",
              "raw": "raw residual direction; exact original residual baseline",
              "hb": "normalized residual minus checkpoint token bias; exact original residual baseline",
              "ae_recon": "decoded AE residual; edit damage relative to reconstruction",
              "ae_delta": "original residual plus decoded change; original error preserved"},
              "baselines": [], "geometry": [], "curves": [], "records": [], "excluded": [],
              "paired_comparisons": [], "document_manifest": {
              part: sorted(set(docs[split == part])) for part in ("fit", "calibration", "test")}}

    for pair, target_label in ((p, lab) for p in np.unique(pair_ids) for lab in (1, 0)):
        member = pair_ids == pair
        case = {"pair": pair, "target_label": target_label}
        target = member & (labels == target_label)
        sibling = member & (labels == 1 - target_label)
        fit = member & (split == "fit")
        cal = member & (split == "calibration")
        test = member & (split == "test")
        groups = {"target": target, "sibling": sibling, "random": member & (labels == -1)}
        counts = {part: {group: len(set(docs[mask & (split == part)])) for group, mask in groups.items()}
                  for part in ("fit", "calibration", "test")}
        if progress is not None:
            progress({**case, "phase": "begin", "document_counts": counts})
        if any(counts[part][group] < min_docs for part in counts for group in ("target", "sibling")):
            result["excluded"].append({**case, "reason": "insufficient_documents", "counts": counts})
            continue
        for part in ("calibration", "test"):
            for group, mask in groups.items():
                idx = np.flatnonzero(mask & (split == part))
                result["baselines"].append({**case, "split": part, "group": group,
                    "raw_nll": _summary(raw_nll[idx].cpu(), docs[idx], bootstrap=bootstrap, seed=seed),
                    "reconstructed_nll": _summary(recon_nll[idx].cpu(), docs[idx], bootstrap=bootstrap, seed=seed),
                    "reconstruction_delta_nll": _summary((recon_nll[idx] - raw_nll[idx]).cpu(), docs[idx],
                                                          bootstrap=bootstrap, seed=seed)})
        for kind in direction_kinds:
            fitted_spaces = {}
            for space, values in values_by_space.items():
                if space not in substrates and not (space == "latent" and any(a.startswith("ae_") for a in substrates)):
                    continue
                try:
                    fitted_spaces[space] = fit_direction(values, docs, target, sibling, fit, kind=kind,
                                                         background=groups["random"])
                except ValueError as exc:
                    result["excluded"].append({**case, "space": space, "direction": kind,
                                               "reason": str(exc)})
                    continue
                fitted = fitted_spaces[space]
                geometry = {**case, "space": space, "direction": kind, "fit": _serial_fit(fitted)}
                projection = values @ fitted["u"]
                geometry["test"] = {}
                for group, mask in groups.items():
                    idx = np.flatnonzero(mask & test)
                    geometry["test"][group] = {"projection": _summary(projection[idx].cpu(), docs[idx],
                        bootstrap=bootstrap, seed=seed), "gate_fraction": _summary(
                        ((projection[idx] >= fitted["lo"]) & (projection[idx] <= fitted["hi"])).float().cpu(),
                        docs[idx], bootstrap=bootstrap, seed=seed)}
                geometry.update(_projection_dprime(projection, docs, np.flatnonzero(target & test),
                                                    np.flatnonzero(sibling & test)))
                result["geometry"].append(geometry)
            for substrate in substrates:
                space = "latent" if substrate.startswith("ae_") else substrate
                if space not in fitted_spaces:
                    continue
                fitted = fitted_spaces[space]
                for gated in (True, False):
                    arm = f"{substrate}/{kind}/{'range' if gated else 'all'}"
                    if progress is not None:
                        progress({**case, "phase": "calibration_and_test", "arm": arm})

                    def evaluate(idx, alpha):
                        index = torch.as_tensor(idx, dtype=torch.long, device=h.device)
                        original = h[index]
                        baseline = recon[index] if substrate == "ae_recon" else original
                        values = values_by_space[space][index]
                        edited_values, changed = direction_edit(values, fitted, alpha, gated=gated)
                        if substrate == "raw":
                            edited = edited_values
                        elif substrate in ("base", "hb"):
                            edited = original.clone()
                            edited[changed] += (edited_values[changed] - values[changed]) * std
                        else:
                            # Reuse baseline exactly on untouched rows; do not
                            # leak decoder arithmetic drift into a zero edit.
                            edited = baseline.clone()
                            if changed.any():
                                decoded = ae.decode(edited_values[changed], tok[index][changed]).float() * std + mean
                                edited[changed] = decoded if substrate == "ae_recon" else original[changed] + (
                                    decoded - recon[index][changed])
                        damage = readout.delta_nll(baseline, edited, nxt[index], changed)
                        return damage, changed, (edited - baseline).norm(dim=1)

                    cal_target = np.flatnonzero(target & cal)
                    damages = []
                    for alpha in alpha_grid:
                        damage, _, _ = evaluate(cal_target, alpha)
                        damages.append(float(document_means(damage, docs[cal_target])[0].mean()))
                    result["curves"].append({**case, "arm": arm, "alpha": list(alpha_grid),
                                              "calibration_target_delta_nll": damages})
                    for level in target_nll:
                        selection = calibrate_strength(alpha_grid, damages, level)
                        record = {**case, "arm": arm, "requested_target_delta_nll": level,
                                  "calibration": selection, "counts": counts, "status": selection["status"]}
                        result["records"].append(record)
                        if selection["alpha"] is None:
                            continue
                        low, high = selection["bracket"]
                        alpha = selection["alpha"]
                        damage, changed, _ = evaluate(cal_target, alpha)
                        achieved = float(document_means(damage, docs[cal_target])[0].mean())
                        steps = 0
                        while abs(achieved - level) > calibration_tolerance and steps < calibration_steps:
                            if achieved < level:
                                low = alpha
                            else:
                                high = alpha
                            alpha = (low + high) / 2
                            damage, changed, _ = evaluate(cal_target, alpha)
                            achieved = float(document_means(damage, docs[cal_target])[0].mean())
                            steps += 1
                        record["alpha"] = alpha
                        record["calibration_attained"] = achieved
                        record["calibration_refinement_steps"] = steps
                        record["status"] = "matched" if abs(achieved - level) <= calibration_tolerance else "calibration_unresolved"
                        if record["status"] != "matched":
                            continue
                        record["evaluation"] = {}
                        for group, mask in groups.items():
                            idx = np.flatnonzero(mask & test)
                            damage, changed, edit_l2 = evaluate(idx, alpha)
                            total_damage = damage + (recon_nll[idx] - raw_nll[idx]) if substrate == "ae_recon" else damage
                            record["evaluation"][group] = {
                                "delta_nll": _summary(damage.cpu(), docs[idx], bootstrap=bootstrap, seed=seed),
                                "total_delta_nll_vs_raw": _summary(total_damage.cpu(), docs[idx], bootstrap=bootstrap, seed=seed),
                                "edit_l2": _summary(edit_l2.cpu(), docs[idx], bootstrap=0),
                                "changed_fraction": _summary(changed.float().cpu(), docs[idx], bootstrap=0)}
                        record["test_target_minus_requested"] = record["evaluation"]["target"]["delta_nll"]["mean"] - level
                        record["test_match_status"] = "within_tolerance" if abs(
                            record["test_target_minus_requested"]) <= test_match_tolerance else "unmatched_target_damage"

    # Pair exact document identities; no artificial CI for a single concept.
    lookup = {(r["pair"], r["target_label"], r["arm"], r["requested_target_delta_nll"]): r for r in result["records"]
              if r.get("status") == "matched"}
    for (pair, target_label, arm, level), ae_record in lookup.items():
        substrate, kind, gate = arm.split("/")
        if substrate == "base":
            continue
        base = lookup.get((pair, target_label, f"base/{kind}/{gate}", level))
        if base is None:
            continue
        comparison = {"pair": pair, "target_label": target_label, "arm": arm, "base_arm": base["arm"],
                      "requested_target_delta_nll": level, "group_differences_ae_minus_base": {}}
        for group in ("target", "sibling", "random"):
            left = ae_record["evaluation"][group]["delta_nll"]
            right = base["evaluation"][group]["delta_nll"]
            if left.get("doc_ids", []) != right.get("doc_ids", []):
                raise RuntimeError("paired comparison lost fixed document identities")
            difference = np.asarray(left.get("doc_means", [])) - np.asarray(right.get("doc_means", []))
            comparison["group_differences_ae_minus_base"][group] = _summary(
                difference, left.get("doc_ids", []), bootstrap=bootstrap, seed=seed)
        target_gap = comparison["group_differences_ae_minus_base"]["target"]["mean"]
        comparison["test_target_match_status"] = "within_tolerance" if abs(target_gap) <= test_match_tolerance else "unmatched_target_damage"
        result["paired_comparisons"].append(comparison)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha-grid", default="0,.125,.25,.5,1,2,4")
    parser.add_argument("--target-nll", default=".1,.25,.5")
    parser.add_argument("--direction-kinds", default="difference,shared")
    parser.add_argument("--substrates", default="base,hb,ae_recon")
    parser.add_argument("--min-docs", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--test-match-tolerance", type=float, default=.02)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Frozen controls output already exists: {output}")
    with np.load(args.data, allow_pickle=False) as arrays:
        data = {name: arrays[name] for name in arrays.files}
    metadata = json.loads(Path(args.metadata).read_text()) if args.metadata else {}
    validate_data(data)
    from geoae.checkpoint import load_ae_checkpoint, load_lm
    from geoae.lm_arch import locate_lm_parts
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ae, mean, std, checkpoint = load_ae_checkpoint(args.checkpoint, args.device, allow_token_bias=True)
    lm = load_lm(args.model, device=args.device)
    _, layers, norm, head = locate_lm_parts(lm)
    layer = metadata.get("layer")
    if layer is None or int(layer) != len(layers) - 1:
        raise ValueError("metadata.layer must explicitly identify the final decoder layer for this cached readout")
    recorded_model = metadata.get("model", metadata.get("model_name"))
    if recorded_model is not None and recorded_model != args.model:
        raise ValueError("cached activations model differs from requested model")
    text_config = getattr(lm.config, "text_config", lm.config)
    norm32, head32 = copy.deepcopy(norm).float().eval(), copy.deepcopy(head).float().eval()
    for parameter in list(norm32.parameters()) + list(head32.parameters()):
        parameter.requires_grad_(False)
    readout = Readout(norm32, head32, chunk_size=args.chunk_size,
                      logit_softcap=getattr(text_config, "final_logit_softcapping", None))
    del lm, layers, norm, head
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    metadata.update({"checkpoint": str(Path(args.checkpoint).resolve()),
                     "checkpoint_step": checkpoint.get("step"), "model": args.model,
                     "input_npz": str(Path(args.data).resolve())})
    checkpoint_digest = hashlib.sha256()
    with Path(args.checkpoint).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checkpoint_digest.update(block)
    metadata["checkpoint_sha256"] = checkpoint_digest.hexdigest()
    result = run_experiment(data, ae, mean, std, readout, device=args.device, metadata=metadata,
                            alpha_grid=tuple(map(float, args.alpha_grid.split(","))),
                            target_nll=tuple(map(float, args.target_nll.split(","))),
                            direction_kinds=tuple(args.direction_kinds.split(",")),
                            substrates=tuple(args.substrates.split(",")),
                            min_docs=args.min_docs, bootstrap=args.bootstrap,
                            test_match_tolerance=args.test_match_tolerance,
                            progress=lambda event: print(json.dumps(event), flush=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(result['records'])} controls and {len(result['paired_comparisons'])} comparisons to {output}",
          flush=True)


if __name__ == "__main__":
    main()
