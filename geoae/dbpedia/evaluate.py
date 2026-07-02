"""
Evaluate how well GeoAE clusters match DBpedia-14 ground-truth classes.

Loads test-split activations + true labels, encodes through trained AE,
assigns cluster labels, then evaluates via:
  - NMI   (Normalized Mutual Information)
  - ARI   (Adjusted Rand Index)
  - Purity (fraction of points in their majority cluster class)
  - Accuracy with Hungarian matching  (best bijection cluster → class)
  - Per-cluster breakdown (which class dominates each cluster)
  - Confusion matrix

Usage:
    python -m geoae.dbpedia.evaluate \\
        --checkpoint dbpedia/checkpoints/unprompted/best_val.pt \\
        --mode unprompted

    python -m geoae.dbpedia.evaluate \\
        --checkpoint dbpedia/checkpoints/prompted/best_val.pt \\
        --mode prompted

    # AE + raw k-means baseline, side by side:
    python -m geoae.dbpedia.evaluate --compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


CLASSES = [
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork",
]

BASE = Path("dbpedia")
E2E_BASE = Path("e2e")   # end-to-end KL checkpoints live in a separate tree


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cluster_accuracy(true: np.ndarray, pred: np.ndarray, K: int) -> float:
    """Best accuracy via Hungarian matching (optimal bijection cluster→class)."""
    from scipy.optimize import linear_sum_assignment
    n_classes = len(np.unique(true))
    cost = np.zeros((K, n_classes), dtype=np.int64)
    for p, t in zip(pred, true):
        if p < K:
            cost[p, t] += 1
    row, col = linear_sum_assignment(-cost)
    correct = cost[row, col].sum()
    return float(correct / len(true))


def cluster_purity(true: np.ndarray, pred: np.ndarray) -> float:
    """Purity = fraction of points in the majority class of their cluster."""
    total = 0
    for k in np.unique(pred):
        mask = pred == k
        counts = np.bincount(true[mask], minlength=len(CLASSES))
        total += counts.max()
    return float(total / len(true))


def nmi(true: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(true, pred, average_method="arithmetic"))


def ari(true: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(true, pred))


def v_measure(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
    return (float(homogeneity_score(true, pred)),
            float(completeness_score(true, pred)),
            float(v_measure_score(true, pred)))


# ---------------------------------------------------------------------------
# Raw k-means baseline
# ---------------------------------------------------------------------------

def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project each row onto the unit sphere (for cosine / directional k-means)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def evaluate_baseline(act_dir: Path, layer: int, K: int, mode: str,
                      results_dir: Path | None = None,
                      metric: str = "euclidean") -> dict:
    """K-means on normalised raw activations — no AE involved.

    metric="euclidean": magnitude-based, on z-score-standardised activations.
    metric="cosine":    directional — rows are additionally L2-normalised so
                        Euclidean k-means is equivalent to spherical/cosine.
    """
    from sklearn.cluster import MiniBatchKMeans

    print(f"\n{'='*60}")
    print(f"  Mode: {mode} (raw k-means baseline, K={K}, metric={metric})")
    print(f"{'='*60}")

    train_acts = np.load(str(act_dir / f"layer_{layer}.npy")).astype(np.float32)
    mean = train_acts.mean(axis=0)
    std  = train_acts.std(axis=0) + 1e-8
    train_norm = (train_acts - mean) / std
    if metric == "cosine":
        train_norm = _l2_normalize_rows(train_norm)
    del train_acts

    print(f"  Fitting KMeans on {len(train_norm):,} train examples…")
    km = MiniBatchKMeans(n_clusters=K, n_init=5, random_state=42, batch_size=4096)
    km.fit(train_norm)
    del train_norm

    test_acts   = np.load(str(act_dir / f"layer_{layer}_test.npy")).astype(np.float32)
    true_labels = np.load(str(act_dir / "labels_test.npy")).astype(np.int32)
    test_norm   = (test_acts - mean) / std
    if metric == "cosine":
        test_norm = _l2_normalize_rows(test_norm)
    pred_labels = km.predict(test_norm).astype(np.int32)
    print(f"  Test examples: {len(true_labels):,}")

    acc        = cluster_accuracy(true_labels, pred_labels, K)
    pur        = cluster_purity(true_labels, pred_labels)
    nmi_       = nmi(true_labels, pred_labels)
    ari_       = ari(true_labels, pred_labels)
    hom, com, vm = v_measure(true_labels, pred_labels)

    print(f"\n  {'Metric':<30}  {'Value':>8}")
    print("  " + "-"*42)
    print(f"  {'Accuracy (Hungarian)':<30}  {acc*100:>7.2f}%")
    print(f"  {'Purity':<30}  {pur*100:>7.2f}%")
    print(f"  {'NMI':<30}  {nmi_:>8.4f}")
    print(f"  {'ARI':<30}  {ari_:>8.4f}")
    print(f"  {'Homogeneity':<30}  {hom:>8.4f}")
    print(f"  {'Completeness':<30}  {com:>8.4f}")
    print(f"  {'V-measure':<30}  {vm:>8.4f}")

    breakdown = cluster_breakdown(true_labels, pred_labels, K)
    print_breakdown(breakdown, K)

    results = {
        "mode":         f"{mode}_raw_kmeans_{metric}",
        "metric":       metric,
        "n_test":       len(true_labels),
        "K":            K,
        "accuracy":     acc,
        "purity":       pur,
        "nmi":          nmi_,
        "ari":          ari_,
        "homogeneity":  hom,
        "completeness": com,
        "v_measure":    vm,
        "breakdown":    breakdown,
    }

    out_dir = results_dir or Path("dbpedia/results")
    out = out_dir / f"{mode}_raw_kmeans_{metric}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out}")
    return results


# ---------------------------------------------------------------------------
# Encode test set
# ---------------------------------------------------------------------------

def encode_test(checkpoint: Path, act_dir: Path, device: torch.device):
    from geoae.checkpoint import load_ae_checkpoint
    ae, _, _, ckpt = load_ae_checkpoint(checkpoint, device)
    mc = ckpt["config"]["model"]

    norm_mean = ckpt["norm_mean"]
    norm_std  = ckpt["norm_std"]
    K         = mc["n_clusters"]

    # Load test activations
    layer      = ckpt["config"]["data"]["target_layer"]
    acts_path  = act_dir / f"layer_{layer}_test.npy"
    label_path = act_dir / "labels_test.npy"

    acts   = np.load(str(acts_path)).astype(np.float32)
    labels = np.load(str(label_path)).astype(np.int32)

    # Normalise
    acts_norm = (acts - norm_mean) / norm_std

    # Encode in batches
    batch = 512
    preds = []
    with torch.no_grad():
        for s in range(0, len(acts_norm), batch):
            x   = torch.from_numpy(acts_norm[s:s+batch]).to(device)
            out = ae(x)
            preds.append(out.Q.argmax(dim=1).cpu().numpy())

    pred_labels = np.concatenate(preds)
    return pred_labels, labels, K


# ---------------------------------------------------------------------------
# Per-cluster breakdown
# ---------------------------------------------------------------------------

def cluster_breakdown(true: np.ndarray, pred: np.ndarray, K: int) -> list[dict]:
    rows = []
    for k in range(K):
        mask   = pred == k
        n      = mask.sum()
        if n == 0:
            rows.append({"cluster": k, "n": 0, "dominant_class": None, "purity": 0.0})
            continue
        counts = np.bincount(true[mask], minlength=len(CLASSES))
        top_i  = counts.argmax()
        rows.append({
            "cluster":        k,
            "n":              int(n),
            "dominant_class": CLASSES[top_i],
            "dominant_label": int(top_i),
            "purity":         float(counts[top_i] / n),
            "class_dist":     {CLASSES[i]: int(c) for i, c in enumerate(counts) if c > 0},
        })
    return sorted(rows, key=lambda r: -r["purity"])


def print_breakdown(rows: list[dict], K: int) -> None:
    print(f"\n  {'Cluster':>8}  {'N':>6}  {'Dominant class':<25}  {'Purity':>7}")
    print("  " + "-"*55)
    for r in rows:
        if r["n"] == 0:
            continue
        print(f"  {r['cluster']:>8}  {r['n']:>6}  "
              f"{r['dominant_class']:<25}  {r['purity']*100:>6.1f}%")


def print_confusion(true: np.ndarray, pred: np.ndarray, K: int) -> None:
    from scipy.optimize import linear_sum_assignment
    C = len(CLASSES)
    mat = np.zeros((K, C), dtype=np.int32)
    for p, t in zip(pred, true):
        mat[p, t] += 1
    # Hungarian match for ordering
    row, col = linear_sum_assignment(-mat)
    ordering = list(zip(row.tolist(), col.tolist()))
    ordering += [(k, -1) for k in range(K) if k not in row]

    print("\n  Confusion (cluster rows, matched to class columns):")
    header = f"  {'Cluster':>8}" + "".join(f"  {CLASSES[c][:8]:>8}" for _, c in ordering if c >= 0)
    print(header[:120])
    for k in range(K):
        row_str = f"  {k:>8}"
        for r, c in ordering:
            if c < 0 or r != k:
                continue
            row_str += f"  {mat[k, c]:>8}"
        if mat[k].sum() > 0:
            print(row_str)


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint: Path, act_dir: Path, mode: str,
             device: torch.device, results_dir: Path | None = None) -> dict:
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"{'='*60}")

    pred, true, K = encode_test(checkpoint, act_dir, device)
    print(f"  Test examples: {len(true):,}   K={K}")

    acc   = cluster_accuracy(true, pred, K)
    pur   = cluster_purity(true, pred)
    nmi_  = nmi(true, pred)
    ari_  = ari(true, pred)
    hom, com, vm = v_measure(true, pred)

    print(f"\n  {'Metric':<30}  {'Value':>8}")
    print("  " + "-"*42)
    print(f"  {'Accuracy (Hungarian)':<30}  {acc*100:>7.2f}%")
    print(f"  {'Purity':<30}  {pur*100:>7.2f}%")
    print(f"  {'NMI':<30}  {nmi_:>8.4f}")
    print(f"  {'ARI':<30}  {ari_:>8.4f}")
    print(f"  {'Homogeneity':<30}  {hom:>8.4f}")
    print(f"  {'Completeness':<30}  {com:>8.4f}")
    print(f"  {'V-measure':<30}  {vm:>8.4f}")

    breakdown = cluster_breakdown(true, pred, K)
    print_breakdown(breakdown, K)
    print_confusion(true, pred, K)

    results = {
        "mode":         mode,
        "checkpoint":   str(checkpoint),
        "n_test":       len(true),
        "K":            K,
        "accuracy":     acc,
        "purity":       pur,
        "nmi":          nmi_,
        "ari":          ari_,
        "homogeneity":  hom,
        "completeness": com,
        "v_measure":    vm,
        "breakdown":    breakdown,
    }

    out_dir = results_dir or Path("dbpedia/results")
    out = out_dir / f"{mode}_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out}")
    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def compare(results: list[dict]) -> None:
    col = 20
    width = 24 + len(results) * (col + 2)
    print(f"\n{'='*width}")
    print(f"  {'Metric':<22}" +
          "".join(f"  {r['mode']:>{col}}" for r in results))
    print("  " + "-"*width)
    for key, label in [
        ("accuracy",    "Accuracy (Hungarian)"),
        ("purity",      "Purity"),
        ("nmi",         "NMI"),
        ("ari",         "ARI"),
        ("v_measure",   "V-measure"),
        ("homogeneity", "Homogeneity"),
        ("completeness","Completeness"),
    ]:
        vals = [r[key] for r in results]
        best = max(vals)
        row  = f"  {label:<22}"
        for v in vals:
            mark = "*" if abs(v - best) < 1e-9 else " "
            row += f"  {v*100:>{col-4}.2f}%  {mark} "
        print(row)
    print(f"{'='*width}")
    print("  * = best")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode",       default=None,
                        choices=["unprompted", "prompted"])
    parser.add_argument("--compare",    action="store_true",
                        help="Evaluate both modes and print side-by-side")
    parser.add_argument("--model",      default="llama3.2-3B",
                        help="Model name subdirectory (default: llama3.2-3B)")
    parser.add_argument("--layer",      type=int, default=27,
                        help="Target layer (default: 27)")
    parser.add_argument("--pooling",    default="last",
                        choices=["last", "mean"],
                        help="Token reduction tree to evaluate (default: last)")
    parser.add_argument("--metric",     default="euclidean",
                        choices=["euclidean", "cosine", "both"],
                        help="Raw k-means baseline geometry: euclidean (magnitude), "
                             "cosine (directional), or both")
    parser.add_argument("--e2e",        action="store_true",
                        help="Also include end-to-end KL checkpoints under "
                             "e2e/checkpoints/<model>/layer<N>/<mode>_*/best_val.pt "
                             "in the --compare table")
    args = parser.parse_args()
    baseline_metrics = ["euclidean", "cosine"] if args.metric == "both" else [args.metric]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pooling   = args.pooling
    act_base  = BASE / "activations" / args.model / pooling
    ckpt_base = BASE / "checkpoints" / args.model / f"layer{args.layer}"
    res_base  = BASE / "results" / args.model / f"layer{args.layer}"
    res_base.mkdir(parents=True, exist_ok=True)

    if args.compare:
        all_results = []
        for mode in ["unprompted", "prompted"]:
            act_dir = act_base / mode
            tag = f"{mode}_{pooling}"   # e.g. unprompted_mean
            # Raw k-means baseline (one per requested metric)
            if (act_dir / f"layer_{args.layer}.npy").exists():
                for m in baseline_metrics:
                    all_results.append(evaluate_baseline(act_dir, args.layer, 14, tag,
                                                         results_dir=res_base, metric=m))
            else:
                print(f"[eval] Skipping {tag} baseline — activations not found")
            # AE variants: checkpoint dir / result label = <mode>_<pooling>_<encoder>
            for enc in ["linear", "gelu", "semisup", "class_means"]:
                label = f"{tag}_{enc}"
                ckpt = ckpt_base / label / "best_val.pt"
                if ckpt.exists():
                    all_results.append(evaluate(ckpt, act_dir, label, device,
                                                results_dir=res_base))
                else:
                    print(f"[eval] Skipping {label} — checkpoint not found: {ckpt}")

            # End-to-end KL variants (separate checkpoints tree; auto-discovered)
            if args.e2e:
                e2e_dir = E2E_BASE / "checkpoints" / args.model / f"layer{args.layer}"
                if e2e_dir.exists():
                    # Restrict to the matching pooling so a last-token e2e model is
                    # never scored on mean-pooled test data (or vice versa).
                    for sub in sorted(e2e_dir.glob(f"{mode}_{pooling}_*")):
                        ckpt = sub / "best_val.pt"
                        if ckpt.exists():
                            all_results.append(evaluate(ckpt, act_dir, sub.name,
                                                        device, results_dir=res_base))
                        else:
                            print(f"[eval] Skipping {sub.name} — no best_val.pt")
                else:
                    print(f"[eval] No e2e checkpoints dir: {e2e_dir}")
        if len(all_results) > 1:
            compare(all_results)
    else:
        if not args.checkpoint or not args.mode:
            parser.error("Provide --checkpoint and --mode, or use --compare")
        ckpt    = Path(args.checkpoint)
        act_dir = act_base / args.mode
        evaluate(ckpt, act_dir, f"{args.mode}_{pooling}", device, results_dir=res_base)


if __name__ == "__main__":
    main()
