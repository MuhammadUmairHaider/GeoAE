"""Paper-inspired dispersion on a document-disjoint sampled-dump tail.

Within = row-to-empirical-centroid cosine distance (equal cluster weights);
between = pairwise empirical-centroid cosine distance. Learned codebooks assign
groups only. Centred variants are supplementary. Fold ranges describe grouped
variability, not confidence intervals. Document proxies use each selected doc's
latest AVAILABLE sampled row, not its true last token/text representation.
Two streaming passes avoid a full hidden matrix and full cosine pair matrix.

    scripts/py experiments/paper/analysis/paper_dispersion.py --device cpu --max_rows 4096 --folds 2
"""
from __future__ import annotations

from geoae.paths import resolve_input_path

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from geoae.evaluation.dispersion import ClusterStatistics
from geoae.evaluation.dispersion import vendi_score

METRICS = ("dispersion", "within", "within_sample_weighted", "between", "between_over_within",
           "pairwise_within", "pairwise_between", "pairwise_between_over_within")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default="activations_sampled_10M")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--ckpt", default="checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt")
    parser.add_argument("--km", default="e2e/checkpoints/general/llama3.2-3B/layer27/delta/balanced_kmeans_k2000_dpc_sampled.npz")
    parser.add_argument("--km_tok", default=None, help="optional token-centred k-means control")
    parser.add_argument("--val_frac", type=float, default=0.05, help="must match the checkpoint's training split")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max_rows", type=int, default=0, help="0 = all eligible rows; otherwise uniform sampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vendi_rows", type=int, default=0, help="optional Vendi subset, 0 disables; maximum 2048")
    parser.add_argument("--out", default="results/delta/d6144/evals/dispersion_v2.json")
    args = parser.parse_args(argv)
    if not 0 < args.val_frac < 1:
        parser.error("--val_frac must be strictly between 0 and 1")
    if args.batch < 1 or args.folds < 1:
        parser.error("--batch and --folds must be positive")
    if args.max_rows < 0 or not 0 <= args.vendi_rows <= 2048:
        parser.error("--max_rows must be nonnegative; --vendi_rows must be in [0, 2048]")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


def select_rows(doc_ids, val_start, max_rows, seed):
    """Exclude train-overlapping documents before seeded uniform row sampling."""
    tail_docs = np.asarray(doc_ids[val_start:])
    overlap = np.isin(tail_docs, np.unique(doc_ids[:val_start]))
    rows = np.flatnonzero(~overlap).astype(np.int64) + val_start
    eligible = len(rows)
    if max_rows and max_rows < eligible:
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=max_rows, replace=False))
    return rows, {
        "held_out_tail_rows": len(tail_docs), "eligible_held_out_rows": eligible,
        "excluded_train_overlap_rows": int(overlap.sum()),
        "excluded_train_overlap_documents": int(len(np.unique(tail_docs[overlap]))),
        "selected_rows": len(rows), "seed": seed,
        "sampling": "seeded uniform without replacement" if len(rows) < eligible else "all eligible rows",
    }


def latest_sampled_rows(doc_ids, positions):
    if not len(doc_ids):
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((np.arange(len(doc_ids)), positions, doc_ids))
    ends = np.r_[np.flatnonzero(np.diff(doc_ids[order])), len(order) - 1]
    return np.sort(order[ends])


def summarized(stat, variant, centre=None):
    result = stat.metrics(variant, centre=centre)
    folds = [stat.metrics(variant, fold=i, centre=centre) for i in range(stat.n_folds)]
    result["fold_range"], result["fold_defined_counts"] = {}, {}
    for name in METRICS:
        values = [f[name] for f in folds if f[name] is not None]
        result["fold_range"][name] = [min(values), max(values)] if values else None
        result["fold_defined_counts"][name] = len(values)
    result["fold_rows"] = [f["rows"] for f in folds]
    result["fold_zero_centroid_clusters"] = [f["zero_centroid_clusters"] for f in folds]
    return result


def validate_norm(mean, std, dimension, label):
    if mean.shape != (dimension,) or std.shape != (dimension,):
        raise ValueError(f"{label}: normalization dimension differs from activation dump")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise ValueError(f"{label}: normalization needs finite mean and positive finite std")


def validate_checkpoint_data(checkpoint, layer, val_frac):
    data = checkpoint.get("config", {}).get("data", {})
    if data.get("target_layer") is not None and int(data["target_layer"]) != layer:
        raise ValueError("GeoAE checkpoint layer differs from requested layer")
    if data.get("val_frac") is not None and not math.isclose(float(data["val_frac"]), val_frac):
        raise ValueError("--val_frac must match the checkpoint's training split; otherwise rows may be trained on")


@torch.inference_mode()
def run(args):
    from geoae.checkpoint import load_ae_checkpoint
    from geoae.analysis.closest_tokens import load_baseline_kmeans
    from geoae.losses import pairwise_sq_dist
    from geoae.token_bias import TokenBiasLookup

    start = time.time()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    dump = Path(args.dump)
    activations = np.load(resolve_input_path(dump / f"layer_{args.layer}.npy"), mmap_mode="r")
    if activations.ndim != 2:
        raise ValueError("activation dump must have shape (N, D)")
    n_total, dimension = activations.shape
    tokens = np.load(resolve_input_path(dump / "rows_tok.npy"), mmap_mode="r")
    documents = np.load(resolve_input_path(dump / "rows_doc.npy"), mmap_mode="r")
    positions = np.load(resolve_input_path(dump / "rows_pos.npy"), mmap_mode="r")
    if any(a.shape != (n_total,) for a in (tokens, documents, positions)):
        raise ValueError("rows_tok, rows_doc, rows_pos must match activation row count")
    val_start = int(n_total * (1.0 - args.val_frac))
    rows, selection = select_rows(documents, val_start, args.max_rows, args.seed)
    if not len(rows):
        raise ValueError("no eligible held-out rows after excluding train-overlapping documents")
    tok, doc, pos = (np.asarray(a[rows], dtype=np.int64) for a in (tokens, documents, positions))
    fold = doc % args.folds
    proxy_mask = np.zeros(len(rows), dtype=bool)
    proxy_mask[latest_sampled_rows(doc, pos)] = True
    vendi_n = min(args.vendi_rows, len(rows))
    vendi_mask = np.zeros(len(rows), dtype=bool)
    if vendi_n:
        vendi_mask[np.random.default_rng(args.seed).choice(len(rows), size=vendi_n, replace=False)] = True
    print(f"[disp] {len(rows):,} selected rows, {proxy_mask.sum():,} document proxies, device={device}; "
          f"excluded {selection['excluded_train_overlap_rows']:,} train-overlap rows", flush=True)

    c_base, m_base, s_base, k_base, layer, _ = load_baseline_kmeans(Path(args.km), device)
    if layer != args.layer:
        raise ValueError("base k-means layer differs from requested layer")
    validate_norm(m_base, s_base, dimension, "base k-means")
    ae, m_ae, s_ae, checkpoint = load_ae_checkpoint(args.ckpt, device, allow_token_bias=True)
    validate_checkpoint_data(checkpoint, args.layer, args.val_frac)
    del checkpoint
    if ae.hidden_size != dimension:
        raise ValueError("--ckpt must be a GeoAE matching the activation dimension")
    if ae.has_token_bias and ae.token_bias_mode != "bypass":
        raise ValueError("--ckpt must be a plain or token-bypass GeoAE")
    validate_norm(m_ae, s_ae, dimension, "GeoAE")
    control = None
    if args.km_tok:
        c, m, s, k, layer, _ = load_baseline_kmeans(Path(args.km_tok), device, allow_token_bias=True)
        if layer != args.layer:
            raise ValueError("token-centred k-means layer differs from requested layer")
        validate_norm(m, s, dimension, "token-centred k-means")
        with np.load(resolve_input_path(args.km_tok), allow_pickle=True) as archive:
            if "token_bias" not in archive.files:
                raise ValueError("--km_tok needs a codebook referencing its token_bias table")
            bias = TokenBiasLookup(str(archive["token_bias"]), device)
        if not (np.allclose(bias.norm_mean, m.cpu().numpy()) and
                np.allclose(bias.norm_std, s.cpu().numpy())):
            raise ValueError("token-centred k-means and its token_bias table use different norm stats")
        control = c, m, s, k, bias

    def spaces(h, t):
        x = (h - m_base) / s_base
        labels = pairwise_sq_dist(x, c_base, metric="euclidean").argmin(1)
        z = ae.encode((h - m_ae) / s_ae, t if ae.has_token_bias else None)
        ae_labels = pairwise_sq_dist(z, ae.centroids, metric=ae.metric).argmin(1)
        result = {"residual_raw": (h, labels, k_base), "residual": (x, labels, k_base),
                  "geoae": (z, ae_labels, ae.n_clusters), "geoae_base_partition": (z, labels, k_base)}
        if control:
            c, m, s, k, bias = control
            xt = (h - m) / s - bias(t)
            result["token_centred"] = (xt, pairwise_sq_dist(xt, c, metric="euclidean").argmin(1), k)
        return result

    def batches():
        for i in range(0, len(rows), args.batch):
            end = min(i + args.batch, len(rows))
            h = torch.from_numpy(np.array(activations[rows[i:end]], dtype=np.float32)).to(device)
            t, f = (torch.from_numpy(a[i:end]).to(device) for a in (tok, fold))
            proxy, vendi = (torch.from_numpy(a[i:end]).to(device) for a in (proxy_mask, vendi_mask))
            yield i, h, t, f, proxy, vendi

    stats, doc_stats = {}, {}
    for i, h, t, f, proxy, _ in batches():
        for name, (v, labels, k) in spaces(h, t).items():
            if name not in stats:
                stats[name] = ClusterStatistics(k, v.shape[1], args.folds, device)
                doc_stats[name] = ClusterStatistics(1, v.shape[1], args.folds, device)
            try:
                stats[name].add_vectors(v, labels, f)
                doc_stats[name].add_vectors(v[proxy], torch.zeros_like(labels[proxy]), f[proxy])
            except ValueError as error:
                raise ValueError(f"{name}, batch selected rows {i}..{i + len(h)}: {error}") from error
    means = {name: s.mean() for name, s in stats.items()}
    doc_means = {name: s.mean() for name, s in doc_stats.items()}
    print(f"[disp] first pass complete in {time.time() - start:.1f}s", flush=True)
    vendi_parts = {}
    for i, h, t, f, proxy, vendi in batches():
        for name, (v, labels, _) in spaces(h, t).items():
            v = v.double()
            try:
                for variant, w in (("raw", v), ("centred", v - means[name])):
                    stats[name].add_units(w, labels, f, variant)
                for variant, w in (("raw", v[proxy]), ("centred", v[proxy] - doc_means[name])):
                    doc_stats[name].add_units(w, torch.zeros_like(labels[proxy]), f[proxy], variant)
            except ValueError as error:
                raise ValueError(f"{name}, batch selected rows {i}..{i + len(h)}: {error}") from error
            if vendi_n and name != "geoae_base_partition":
                vendi_parts.setdefault(name, []).append(v[vendi].cpu())
    print(f"[disp] second pass complete in {time.time() - start:.1f}s", flush=True)

    result = {"meta": {
        "paper": "On the Predictive Power of Representation Dispersion in Language Models",
        "paper_url": "https://arxiv.org/abs/2506.24106v2", "metric_version": 2,
        "within": "equal-occupied-cluster mean of row-to-empirical-centroid cosine distances",
        "within_sample_weighted": "row-weighted mean of row-to-empirical-centroid cosine distances",
        "between": "mean pairwise cosine distance among occupied empirical centroids",
        "centroid_convention": "arithmetic original-vector means, then normalize; paper's averaging convention unspecified",
        "dispersion": "mean pairwise cosine distance among distinct rows",
        "pairwise_metrics": "supplementary row-pair metrics, distinct from paper centroid metrics",
        "zero_vector_policy": "reject rows with undefined cosine distance",
        "zero_centroid_policy": "all centroid metrics null if any occupied centroid is zero",
        "folds": args.folds, "fold_assignment": "document id modulo folds",
        "fold_interpretation": "document-grouped fold range is variability, not confidence interval",
        "centring": "separate global selected-token and selected-document-proxy means, fixed across folds",
        "adaptations": ["learned clusters instead of repeated continuations sharing a context",
                        "arbitrary sampled-token residuals instead of true last-token text representations",
                        "default L27 dump is pre-final-RMSNorm residual, unlike author's final post-norm hidden state",
                        "centred variants are supplementary diagnostics",
                        "does not test dispersion-perplexity correlation or predictive power"],
        "partition_comparison": "geoae native groups; geoae_base_partition uses identical base groups",
        "document_proxy": "one latest AVAILABLE row within selected sample per doc; not true last-token/text representation",
        "dump": str(dump), "layer": args.layer, "val_start": val_start,
        "checkpoint": args.ckpt, "km": args.km, "km_tok": args.km_tok,
        "ae_mode": "bypass" if ae.has_token_bias else "plain",
        "token_bias": bool(ae.has_token_bias),
        "token_bias_mode": ae.token_bias_mode if ae.has_token_bias else None,
        "device": str(device), "batch": args.batch, "selection": selection,
        "space_clusters": {name: s.n_clusters for name, s in stats.items()},
        "vendi": {"rows": vendi_n, "seed": args.seed, "sampling": "uniform subset of selected token rows",
                  "kernel": "cosine Gram U @ U.T; eigvalsh / N; eigenvalues > 1e-10",
                  "centred": "supplementary, uses full selected-token global mean"},
    }, "token_level": {}, "doc_level": {}}
    for name, s in stats.items():
        result["token_level"][name], result["doc_level"][name] = {}, {}
        for variant in ("raw", "centred"):
            result["token_level"][name][variant] = summarized(s, variant, means[name] if variant == "centred" else None)
            d = summarized(doc_stats[name], variant, doc_means[name] if variant == "centred" else None)
            result["doc_level"][name][variant] = {"dispersion": d["dispersion"], "n_docs": d["rows"],
                "fold_range": d["fold_range"]["dispersion"], "fold_rows": d["fold_rows"],
                "fold_defined_count": d["fold_defined_counts"]["dispersion"]}
    for name, parts in vendi_parts.items():
        v = torch.cat(parts)
        for variant in ("raw", "centred"):
            w = v - means[name].cpu() if variant == "centred" else v
            result["token_level"][name][variant].update(vendi_score=vendi_score(w), vendi_rows=vendi_n)
    result["meta"]["elapsed_seconds"] = time.time() - start
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"[disp] wrote {output} ({time.time() - start:.1f}s)", flush=True)
    print(f"{'space':18s} {'variant':8s} {'disp(tok)':>10s} {'within':>10s} {'between':>10s} {'b/w':>10s} {'disp(proxy)':>12s}")
    for name in ("residual_raw", "residual", "geoae", "token_centred"):
        if name not in result["token_level"]:
            continue
        for variant in ("raw", "centred"):
            r = result["token_level"][name][variant]
            values = [r[key] for key in ("dispersion", "within", "between", "between_over_within")]
            values.append(result["doc_level"][name][variant]["dispersion"])
            rendered = ["null" if v is None else f"{v:.5f}" for v in values]
            print(f"{name:18s} {variant:8s} " + " ".join(f"{v:>10s}" for v in rendered))
    return result


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
