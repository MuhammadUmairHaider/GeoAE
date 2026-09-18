"""
Targeted Probe Perturbation (TPP): measure concept localization in
representation spaces.

Fit a linear probe, then replace ranked input coordinates with their training
means. k50 is the first sampled k at which target-class accuracy is at most half
its own baseline; k50 / representation width is the fraction removed. This is
fixed-probe vulnerability, not proof of disentanglement or information erasure.

Spaces compared:
  raw    — z-scored layer-27 residuals (baseline)
  geoae  — train-standardized GeoAE latent projections

Usage:
    python -m geoae.interp.probe_perturbation \
      --act_dir dbpedia/activations/llama3.2-3B/last/unprompted \
      --layer 27 \
      --ae_checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_vicreg/best_val.pt \
      --spaces raw geoae \
      --out results/tpp_dbpedia.json \
      --plot_dir plots/tpp

    # quick smoke test
    python -m geoae.interp.probe_perturbation \
      --act_dir dbpedia/activations/llama3.2-3B/last/unprompted \
      --layer 27 --spaces raw --smoke --out results/tpp_raw_smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from geoae.seeding import seed_everything

# numpy compat: trapz → trapezoid in numpy 2.0+
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, n_dims: int, n_classes: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_dims, n_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    lambda_l1: float = 0.,
    lambda_l2: float = 1e-4,
    lr: float = 3e-3,
    n_epochs: int = 100,
    batch_size: int = 2048,
    patience: int = 10,
    device: str = "cpu",
    bias: bool = True,
    seed: int = 42,
    return_info: bool = False,
):
    """Use the audited validation-CE trainer (mean L1 and AdamW decay).

    Lazy import avoids a cycle: BiasBios imports LinearProbe from this module.
    The old two-item return interface remains available to callers.
    """
    from geoae.bias.probe import train_probe as fit
    probe, info = fit(X_train, y_train, X_val, y_val, n_classes, seed=seed, lr=lr,
        n_epochs=n_epochs, patience=patience, batch_size=batch_size,
        lambda_l1=lambda_l1, lambda_l2=lambda_l2, device=device, bias=bias)
    return probe, info if return_info else info["stopped_epoch"]


# ---------------------------------------------------------------------------
# Activation transforms
# ---------------------------------------------------------------------------

def _clamp_infs(X: np.ndarray) -> np.ndarray:
    """Replace inf/-inf with the max/min finite value per column (fp16 overflow fix)."""
    inf_mask = np.isinf(X)
    if not inf_mask.any():
        return X
    n_inf = int(inf_mask.sum())
    n_dims = int(inf_mask.any(axis=0).sum())
    print(f"[TPP] Clamping {n_inf} inf values across {n_dims} dims (fp16 overflow)")
    X = X.copy()
    for d in np.where(inf_mask.any(axis=0))[0]:
        col = X[:, d]
        finite = col[np.isfinite(col)]
        if len(finite) > 0:
            col[col == np.inf] = finite.max()
            col[col == -np.inf] = finite.min()
        else:
            col[:] = 0.0
    return X


def load_activations(act_dir: Path, layer: int):
    """Load train/test activations and labels from the standard layout."""
    X_train = np.load(act_dir / f"layer_{layer}.npy").astype(np.float32)
    X_test = np.load(act_dir / f"layer_{layer}_test.npy").astype(np.float32)
    y_train = np.load(act_dir / "labels_train.npy").astype(np.int64)
    y_test = np.load(act_dir / "labels_test.npy").astype(np.int64)
    for X, y in [(X_train, y_train), (X_test, y_test)]:
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError("activation/label shapes do not match")
        if not np.isfinite(X).all():
            raise ValueError("non-finite cache: re-extract or explicitly repair before evaluation")
        if y.min() < 0: raise ValueError("class labels must be nonnegative")
    return X_train, y_train, X_test, y_test


def load_joint_correct_indices(
    json_path: str,
    act_dir: Path,
) -> np.ndarray:
    """Map joint-correct JSON texts to indices in the pre-extracted test activations.

    The extraction script (dbpedia/extract.py) shuffles the HF test split with
    seed=42 and takes the first N.  We reconstruct that ordering and match the
    joint-correct texts by their stripped content.
    """
    meta_path = act_dir / "meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    cached_labels = np.load(act_dir / "labels_test.npy")
    n_test = len(cached_labels)
    if meta.get("n_test", n_test) != n_test:
        raise ValueError("test metadata and label counts disagree")

    with open(json_path) as f:
        jc = json.load(f)
    jc_texts = {doc["text"].strip() for doc in jc["docs"]}

    from datasets import load_dataset
    ds = load_dataset("fancyzhx/dbpedia_14")
    test_list = list(
        ds["test"].shuffle(seed=meta.get("seed", 42)).select(range(min(n_test, len(ds["test"]))))
    )
    if not np.array_equal(cached_labels, [ex["label"] for ex in test_list]):
        raise ValueError("reconstructed DBpedia ordering does not match cached test labels")

    indices = []
    for i, ex in enumerate(test_list):
        if ex["content"].strip() in jc_texts:
            indices.append(i)

    return np.array(indices, dtype=np.int64)


def transform_raw(X_train: np.ndarray, X_test: np.ndarray, fit_indices=None):
    """Z-score using only the explicitly supplied probe-fitting rows."""
    fit = X_train if fit_indices is None else X_train[fit_indices]
    mean = fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = fit.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.
    return (X_train - mean) / std, (X_test - mean) / std


def transform_geoae(
    X_train: np.ndarray,
    X_test: np.ndarray,
    ckpt_path: str,
    device: str,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode through GeoAE encoder (includes GELU nonlinearity)."""
    from geoae.checkpoint import load_ae_checkpoint

    ae, norm_mean, norm_std, _ = load_ae_checkpoint(ckpt_path, device)
    norm_mean_np = norm_mean.cpu().numpy()
    norm_std_np = norm_std.cpu().numpy()

    def encode(X: np.ndarray) -> np.ndarray:
        X_norm = (X - norm_mean_np) / norm_std_np
        parts = []
        for i in range(0, len(X_norm), batch_size):
            xb = torch.from_numpy(X_norm[i:i + batch_size]).float().to(device)
            with torch.no_grad():
                zb = ae.encoder(xb)
            parts.append(zb.cpu().numpy())
        return np.concatenate(parts, axis=0)

    return encode(X_train), encode(X_test)


# ---------------------------------------------------------------------------
# Perturbation sweep
# ---------------------------------------------------------------------------

def rank_dims_global(W: np.ndarray) -> np.ndarray:
    """Rank softmax class contrasts, invariant to common weight offsets."""
    importance = np.abs(W - W.mean(axis=0, keepdims=True)).max(axis=0)
    return np.argsort(-importance, kind="stable")


def rank_dims_per_class(W: np.ndarray, class_idx: int) -> np.ndarray:
    """Rank the target-vs-mean-class weight contrast."""
    importance = np.abs(W[class_idx] - W.mean(axis=0))
    return np.argsort(-importance, kind="stable")


def summarize_curve(ks, target, complement, n_dims):
    """Literal 50% relative accuracy drop, using the first sampled crossing.

    Non-monotone curves are allowed. Unreached/undefined thresholds are censored,
    never replaced by n_dims. Collateral is evaluated at the crossing, not at D.
    """
    target, complement = list(target), list(complement)
    baseline = target[0]
    valid = baseline is not None and baseline > 0
    reached = np.flatnonzero(np.asarray(target, dtype=float) <= baseline * .5) if valid else []
    index = int(reached[0]) if len(reached) else None
    k50 = int(ks[index]) if index is not None else None
    collateral = (complement[0] - complement[index]
                  if index is not None and complement[0] is not None else None)
    selectivity = (baseline - target[index] - collateral if collateral is not None else None)
    return {
        "ks": [int(k) for k in ks], "n_dims": int(n_dims),
        "tgt_accs": target, "comp_accs": complement,
        "tgt_baseline": baseline, "comp_baseline": complement[0],
        "threshold": baseline * .5 if valid else None,
        "threshold_status": "reached" if k50 is not None else ("not_reached" if valid else "undefined_baseline"),
        "k_at_50pct_drop": k50,
        "k_at_50pct_drop_frac": k50 / n_dims if k50 is not None else None,
        "previous_tested_k": int(ks[index - 1]) if index is not None and index > 0 else None,
        "target_accuracy_at_k50": target[index] if index is not None else None,
        "complement_accuracy_at_k50": complement[index] if index is not None else None,
        "complement_drop_at_k50": collateral,
        "selectivity": selectivity,
        "selectivity_definition": "target drop minus complement drop at first sampled k50",
        "auc_tgt_normalized": float(_trapz(np.asarray(target) / baseline, np.asarray(ks) / n_dims)) if valid else None,
    }


@torch.inference_mode()
def perturbation_sweep(X_test, y_test, probe, dim_ranking, target_class,
                       n_steps=200, device="cpu"):
    """Zero standardized input coordinates for every row; keep the probe fixed."""
    n_dims = X_test.shape[1]
    if n_steps < 1 or n_dims < 1: raise ValueError("positive dimension/step counts required")
    if not np.array_equal(np.sort(dim_ranking), np.arange(n_dims)):
        raise ValueError("ranking must be a permutation of all input coordinates")
    ks = np.unique(np.linspace(0, n_dims, n_steps + 1, dtype=int))
    x = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_test, dtype=torch.long, device=device)
    target = y == target_class
    probe.eval()
    W = probe.linear.weight
    b = probe.linear.bias
    if b is None: b = torch.zeros(W.shape[0], device=device)
    logits = x @ W.T + b
    target_acc, complement_acc, previous = [], [], 0
    for k in ks:
        if k == n_dims:
            logits = b.expand(len(x), -1)  # exact bias-only endpoint
        elif k > previous:
            dims = torch.as_tensor(dim_ranking[previous:k], device=device)
            logits = logits - x[:, dims] @ W[:, dims].T
        correct = logits.argmax(1) == y
        target_acc.append(float(correct[target].float().mean()) if target.any() else None)
        complement_acc.append(float(correct[~target].float().mean()) if (~target).any() else None)
        previous = k
    return {**summarize_curve(ks, target_acc, complement_acc, n_dims),
            "n_target": int(target.sum()), "n_complement": int((~target).sum())}


def random_perturbation_sweep(X_test, y_test, probe, target_class, n_steps=200,
                              n_trials=5, seed=42, device="cpu"):
    """Keep trial thresholds separate from the threshold of the mean curve."""
    if n_trials < 1: raise ValueError("at least one random trial required")
    rng = np.random.RandomState(seed)
    trials = [perturbation_sweep(X_test, y_test, probe, rng.permutation(X_test.shape[1]),
                                target_class, n_steps, device) for _ in range(n_trials)]
    def mean_curve(key):
        values = [r[key] for r in trials]
        return [None] * len(values[0]) if values[0][0] is None else np.mean(values, axis=0).tolist()
    target, complement = mean_curve("tgt_accs"), mean_curve("comp_accs")
    summary = summarize_curve(trials[0]["ks"], target, complement, X_test.shape[1])
    ks = [r["k_at_50pct_drop"] for r in trials]
    return {**summary, "tgt_accs_mean": target, "comp_accs_mean": complement,
            "threshold_definition": "first crossing of the mean random curve; not mean trial k50",
            "n_trials": n_trials, "trial_k50": ks,
            "n_trials_reached": sum(k is not None for k in ks),
            "median_trial_k50": float(np.median(ks)) if all(k is not None for k in ks) else None,
            "trials": trials}


def validation_operating_point(validation, test):
    """Evaluate the validation-selected ablation budget on the common test cohort."""
    if validation["ks"] != test["ks"]: raise ValueError("validation/test grids differ")
    k = validation["k_at_50pct_drop"]
    if k is None: return {"k": None, "status": validation["threshold_status"], "test": None}
    i = test["ks"].index(k)
    t0, c0 = test["tgt_baseline"], test["comp_baseline"]
    t, c = test["tgt_accs"][i], test["comp_accs"][i]
    td = t0 - t if t0 is not None else None
    cd = c0 - c if c0 is not None else None
    return {"k": k, "fraction": k / test["n_dims"], "status": "selected_on_validation",
            "test": {"target_accuracy": t, "complement_accuracy": c,
                     "target_drop": td, "complement_drop": cd,
                     "target_relative_drop": td / t0 if t0 else None,
                     "selectivity": td - cd if td is not None and cd is not None else None}}


# ---------------------------------------------------------------------------
# Active dimensions (L1 sparsity metric)
# ---------------------------------------------------------------------------

def count_active_dims(W: np.ndarray, threshold: float = 1e-4) -> dict:
    """Count class-contrast weights above a numerical threshold, not exact sparsity."""
    W = W - W.mean(axis=0, keepdims=True)
    global_active = int((np.abs(W).max(axis=0) > threshold).sum())
    per_class = {}
    for c in range(W.shape[0]):
        per_class[c] = int((np.abs(W[c]) > threshold).sum())
    return {"global": global_active, "per_class": per_class}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_perturbation_curves(results, plot_dir):
    """Show fractions and censoring explicitly; no integer conversion of null k50."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    spaces = results["spaces"]
    names = list(spaces)
    concepts = sorted(set().union(*(s["concepts"] for s in spaces.values())))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = .8 / len(names)
    x = np.arange(len(concepts))
    for j, name in enumerate(names):
        values = [spaces[name]["concepts"].get(c, {}).get("importance", {}).get("k_at_50pct_drop_frac") for c in concepts]
        bars = axes[0].bar(x + j * width, [np.nan if v is None else 100*v for v in values], width, label=name)
        for bar, v in zip(bars, values):
            if v is None: axes[0].text(bar.get_x(), 1, "N/R", rotation=90, fontsize=7)
        for method, style in [("importance", "-"), ("random", "--")]:
            curves = [c[method] for c in spaces[name]["concepts"].values()
                      if method in c and c[method]["tgt_baseline"]]
            if curves:
                mean = np.mean([np.asarray(c["tgt_accs"]) / c["tgt_baseline"] for c in curves], axis=0)
                axes[1].plot(np.asarray(curves[0]["ks"]) / spaces[name]["n_dims"], mean, style, label=f"{name}: {method}")
    axes[0].set_xticks(x + width * (len(names)-1)/2, concepts, rotation=50, ha="right", fontsize=8)
    axes[0].set_ylabel("Input coordinates removed at sampled k50 (%)")
    axes[0].set_title("N/R = unreached or undefined; not 100%")
    axes[1].axhline(.5, color="gray", linestyle=":")
    axes[1].set_xlabel("Fraction of input coordinates replaced with training mean")
    axes[1].set_ylabel("Target accuracy / initial target accuracy")
    for ax in axes: ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Fixed-probe ablation on a common evaluation cohort")
    fig.tight_layout()
    fig.savefig(plot_dir / "k50_and_curves.png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run one space
# ---------------------------------------------------------------------------

def run_space(space_name, X_train, y_train, X_test, y_test, n_classes,
              class_names, args, device):
    """Shared fit/validation rows, identical test cohort, train-only scaling."""
    fit_idx, val_idx = args.fit_indices, args.val_indices
    X_train, X_test = transform_raw(X_train, X_test, fit_idx)
    print(f"[TPP] {space_name}: {X_train.shape[1]} dims, fitting probe", flush=True)
    probe, info = train_probe(X_train[fit_idx], y_train[fit_idx], X_train[val_idx], y_train[val_idx],
        n_classes, lambda_l1=args.lambda_l1, lambda_l2=args.lambda_l2, lr=args.probe_lr,
        n_epochs=args.probe_epochs, patience=args.patience, batch_size=args.batch_size,
        seed=args.seed, device=device, return_info=True)
    with torch.inference_mode():
        predictions = probe(torch.as_tensor(X_test, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
    W = probe.linear.weight.detach().cpu().numpy()
    active = count_active_dims(W)
    result = {"n_dims": int(X_train.shape[1]), "probe_accuracy": float(np.mean(predictions == y_test)),
              "n_test": len(y_test), "evaluation_cohort": "all common test rows; no per-space correctness filtering",
              "n_active_dims": active["global"], "probe_fitting": info, "concepts": {}}
    for c in range(n_classes):
        cname = class_names[c] if class_names else str(c)
        if args.ranking == "dprime":
            from geoae.interp.neuronlens import dprime_saliency
            membership = y_train[fit_idx] == c
            if not membership.any() or membership.all():
                result["concepts"][cname] = {"class_idx": c, "status": "missing_fitting_group"}
                continue
            ranking = np.argsort(-dprime_saliency(X_train[fit_idx], membership), kind="stable")
        else:
            ranking = rank_dims_per_class(W, c)
        validation = perturbation_sweep(X_train[val_idx], y_train[val_idx], probe, ranking, c, args.n_steps, device)
        sweep = perturbation_sweep(X_test, y_test, probe, ranking, c, args.n_steps, device)
        random = random_perturbation_sweep(X_test, y_test, probe, c, args.n_steps,
                                          args.n_random_trials, args.seed, device)
        k, rk = sweep["k_at_50pct_drop"], random["median_trial_k50"]
        ratio = rk / k if k is not None and k > 0 and rk is not None else None
        result["concepts"][cname] = {"class_idx": c, "status": "evaluated",
            "n_target": int(np.sum(y_test == c)), "n_complement": int(np.sum(y_test != c)),
            "importance": sweep, "validation": validation, "random": random,
            "validation_selected": validation_operating_point(validation, sweep),
            "localization_ratio": ratio, "dim_ranking": ranking.tolist()}
        print(f"[TPP] {space_name}/{cname}: k50={k} fraction={sweep['k_at_50pct_drop_frac']} "
              f"status={sweep['threshold_status']} collateral={sweep['complement_drop_at_k50']}", flush=True)
    evaluated = [r for r in result["concepts"].values() if r["status"] == "evaluated"]
    reached = [r for r in evaluated if r["importance"]["k_at_50pct_drop"] is not None]
    result["summary"] = {"n_classes": n_classes, "n_classes_evaluated": len(evaluated),
        "n_classes_reached": len(reached),
        "mean_k50_reached_only": float(np.mean([r["importance"]["k_at_50pct_drop"] for r in reached])) if reached else None,
        "mean_fraction_reached_only": float(np.mean([r["importance"]["k_at_50pct_drop_frac"] for r in reached])) if reached else None}
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--act_dir", required=True)
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--ae_checkpoint")
    ap.add_argument("--spaces", nargs="+", default=["raw", "geoae"], choices=["raw", "geoae"])
    ap.add_argument("--ranking", choices=["contrast", "dprime"], default="contrast")
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--n_random_trials", type=int, default=5)
    ap.add_argument("--lambda_l1", type=float, default=0., help="Mean absolute weight penalty")
    ap.add_argument("--lambda_l2", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--probe_lr", type=float, default=3e-3)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--max_train", type=int)
    ap.add_argument("--max_test", type=int)
    ap.add_argument("--val_frac", type=float, default=.1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot_dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--joint_correct", help="Optional common DBpedia LLM-correct cohort; not a per-probe filter")
    ap.add_argument("--source_hashes", help="Optional audited NPZ train/test text hashes for deduplication")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_steps, args.max_train, args.max_test = 20, 1000, 500
        args.n_random_trials, args.probe_epochs = 1, 20
    if "geoae" in args.spaces and not args.ae_checkpoint: ap.error("geoae requires --ae_checkpoint")
    if len(set(args.spaces)) != len(args.spaces): ap.error("duplicate spaces")
    if not 0 < args.val_frac < 1: ap.error("val_frac must be in (0, 1)")
    for key in ["n_steps", "n_random_trials", "probe_epochs", "patience", "batch_size", "max_train", "max_test"]:
        if getattr(args, key) is not None and getattr(args, key) < 1: ap.error(f"{key} must be positive")
    if min(args.lambda_l1, args.lambda_l2) < 0: ap.error("penalties must be nonnegative")
    out = Path(args.out)
    if out.exists(): ap.error(f"output exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    act_dir = Path(args.act_dir)
    metadata = json.loads((act_dir / "meta.json").read_text())
    if metadata["layer"] != args.layer: ap.error("activation metadata layer mismatch")
    X_train, y_train, X_test, y_test = load_activations(act_dir, args.layer)
    if X_train.shape[1] != X_test.shape[1]: ap.error("activation widths differ")
    class_names = metadata.get("classes")
    n_classes = len(class_names) if class_names else int(y_train.max()) + 1
    if max(y_train.max(), y_test.max()) >= n_classes: ap.error("label outside class vocabulary")
    train_rows, test_rows = np.arange(len(y_train)), np.arange(len(y_test))
    dedup = {"status": "not_checked_no_source_hashes"}
    if args.source_hashes:
        from geoae.bias.probe import deduplicate_indices
        hashes = np.load(args.source_hashes)
        if len(hashes["train"]) != len(y_train) or len(hashes["test"]) != len(y_test):
            ap.error("source hash row count mismatch")
        train_rows, test_rows = deduplicate_indices(hashes["train"], hashes["test"])
        dedup = {"status": "exact_text_hashes", "train_removed": len(y_train)-len(train_rows),
                 "test_removed": len(y_test)-len(test_rows)}
    if args.joint_correct:
        from geoae.dbpedia.extract import CLASSES
        if class_names != CLASSES: ap.error("joint_correct text reconstruction is DBpedia-specific")
        test_rows = np.intersect1d(test_rows, load_joint_correct_indices(args.joint_correct, act_dir))
    rng = np.random.RandomState(args.seed)
    if args.max_train: train_rows = rng.permutation(train_rows)[:args.max_train]
    if args.max_test: test_rows = rng.permutation(test_rows)[:args.max_test]
    X_train, y_train = X_train[train_rows], y_train[train_rows]
    X_test, y_test = X_test[test_rows], y_test[test_rows]
    if not len(y_test): ap.error("empty evaluation cohort")
    counts = np.unique(y_train, return_counts=True)[1]
    if counts.min() < 2: ap.error("each sampled training class needs at least two rows")
    fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=args.val_frac,
                                      random_state=args.seed, stratify=y_train)
    # Save serializable arguments before attaching the shared split to the namespace.
    result = {"schema_version": 2, "status": "running", "args": dict(vars(args)),
        "activation_metadata": metadata, "deduplication": dedup,
        "protocol": {"k50": "first sampled target accuracy <= half its own baseline; unreached is null",
                     "cohort": "same full test cohort in every space",
                     "scaling": "per-space probe-fitting mean/std; zero = training-mean replacement",
                     "selectivity": "target drop minus complement drop at sampled k50",
                     "heldout_collateral": "also reported at validation-selected k50",
                     "interpretation": "fixed-probe input-coordinate vulnerability; not information erasure"},
        "split": {"fit_source_rows": train_rows[fit_idx].tolist(), "validation_source_rows": train_rows[val_idx].tolist(),
                  "test_source_rows": test_rows.tolist(),
                  "sha256": hashlib.sha256(train_rows[fit_idx].tobytes()+train_rows[val_idx].tobytes()+test_rows.tobytes()).hexdigest()},
        "spaces": {}}
    args.fit_indices, args.val_indices = fit_idx, val_idx
    def save():
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        tmp.replace(out)
    save()
    for space in args.spaces:
        if space == "raw":
            Z_train, Z_test = X_train, X_test
        else:
            from geoae.bias.probe import encode_geoae
            (Z_train, Z_test), result["checkpoint"] = encode_geoae([X_train, X_test],
                args.ae_checkpoint, args.device, metadata, args.layer, args.batch_size)
        result["spaces"][space] = run_space(space, Z_train, y_train, Z_test, y_test,
            n_classes, class_names, args, args.device)
        del Z_train, Z_test
        save()
    common = set.intersection(*[
        {c for c, r in s["concepts"].items() if r.get("importance", {}).get("k_at_50pct_drop") is not None}
        for s in result["spaces"].values()])
    result["common_reached_classes"] = sorted(common)
    for s in result["spaces"].values():
        s["summary"]["mean_fraction_common_reached"] = float(np.mean([
            s["concepts"][c]["importance"]["k_at_50pct_drop_frac"] for c in common])) if common else None
    result["status"] = "complete"
    save()
    if args.plot_dir: plot_perturbation_curves(result, Path(args.plot_dir))
    print(f"[TPP] Complete: {out}", flush=True)


if __name__ == "__main__":
    main()
