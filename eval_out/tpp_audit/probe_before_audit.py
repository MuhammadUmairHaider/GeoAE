"""
Targeted Probe Perturbation (TPP): measure concept localization in
representation spaces.

Train an L1-regularized linear probe on classification labels, then
progressively zero out the most important dimensions and measure accuracy
drop.  Fewer dims to degrade = more localized = better disentangled.

Spaces compared:
  raw    — z-scored layer-27 residuals (baseline)
  geoae  — GeoAE latent projections (ae.encoder, post-GELU)
  sae    — SAE latent projections (future, optional)

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
      --layer 27 --spaces raw --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
    lambda_l1: float = 1e-1,
    lambda_l2: float = 1e-2,
    lr: float = 1e-3,
    n_epochs: int = 100,
    batch_size: int = 2048,
    patience: int = 10,
    device: str = "cpu",
    bias: bool = True,
) -> tuple[LinearProbe, int]:
    """Train an elastic-net-regularized linear probe.  Returns (probe, epochs_trained).

    Elastic net = λ₁·||W||₁ + λ₂·||W||₂²
      L1 drives weights to exact zero (feature selection / sparsity).
      L2 stabilises correlated features (grouping effect) and improves
      generalisation when n_features >> n_samples.
    """
    probe = LinearProbe(X_train.shape[1], n_classes, bias=bias).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    best_val_acc = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, n_epochs + 1):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = probe(xb)
            W = probe.linear.weight
            loss = (ce_loss(logits, yb)
                    + lambda_l1 * W.abs().sum()
                    + lambda_l2 * (W * W).sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        probe.eval()
        with torch.no_grad():
            val_acc = (probe(X_val_t).argmax(1) == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    probe.load_state_dict(best_state)
    probe.eval()
    return probe, epoch


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
    X_train = _clamp_infs(np.load(act_dir / f"layer_{layer}.npy").astype(np.float32))
    X_test = _clamp_infs(np.load(act_dir / f"layer_{layer}_test.npy").astype(np.float32))
    y_train = np.load(act_dir / "labels_train.npy").astype(np.int64)
    y_test = np.load(act_dir / "labels_test.npy").astype(np.int64)
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
    n_test = meta.get("n_test", 10_000)

    with open(json_path) as f:
        jc = json.load(f)
    jc_texts = {doc["text"] for doc in jc["docs"]}

    from datasets import load_dataset
    ds = load_dataset("fancyzhx/dbpedia_14")
    test_list = list(
        ds["test"].shuffle(seed=42).select(range(min(n_test, len(ds["test"]))))
    )

    indices = []
    for i, ex in enumerate(test_list):
        if ex["content"].strip() in jc_texts:
            indices.append(i)

    return np.array(indices, dtype=np.int64)


def transform_raw(X_train: np.ndarray, X_test: np.ndarray):
    """Z-score using train statistics."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
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
    """Rank dims by max absolute weight across classes. Returns indices descending."""
    importance = np.abs(W).max(axis=0)
    return np.argsort(-importance)


def rank_dims_per_class(W: np.ndarray, class_idx: int) -> np.ndarray:
    """Rank dims by absolute weight for a specific class. Returns indices descending."""
    importance = np.abs(W[class_idx])
    return np.argsort(-importance)


def perturbation_sweep(
    X_test: np.ndarray,
    y_test: np.ndarray,
    probe: LinearProbe,
    dim_ranking: np.ndarray,
    target_class: int,
    n_steps: int = 200,
    device: str = "cpu",
) -> dict:
    """Zero out top-k dims ranked for target_class, measure target vs complement acc.

    Like the steering experiments: perturbing dims important for concept c
    should drop target (class c) accuracy while preserving complement (non-c).
    """
    n_dims = X_test.shape[1]
    ks = np.unique(np.linspace(0, n_dims, n_steps + 1, dtype=int))

    X_t = torch.from_numpy(X_test).float().to(device)
    y_t = torch.from_numpy(y_test).long().to(device)
    tgt_mask = y_t == target_class
    comp_mask = ~tgt_mask

    probe.eval()
    tgt_accs, comp_accs = [], []
    with torch.no_grad():
        for k in ks:
            X_p = X_t.clone()
            if k > 0:
                X_p[:, dim_ranking[:k]] = 0.0
            preds = probe(X_p).argmax(1)
            tgt_accs.append((preds[tgt_mask] == y_t[tgt_mask]).float().mean().item()
                            if tgt_mask.any() else 0.0)
            comp_accs.append((preds[comp_mask] == y_t[comp_mask]).float().mean().item()
                             if comp_mask.any() else 0.0)

    tgt_accs = np.array(tgt_accs)
    comp_accs = np.array(comp_accs)
    tgt_base = tgt_accs[0]
    comp_base = comp_accs[0]

    # k@50% drop on target accuracy
    half_drop = tgt_base * 0.5
    below = np.where(tgt_accs <= half_drop)[0]
    k_50 = int(ks[below[0]]) if len(below) > 0 else int(n_dims)

    # Selectivity: target drops more than complement
    tgt_delta = tgt_accs - tgt_base      # negative = good (target degraded)
    comp_delta = comp_accs - comp_base    # should stay near 0
    selectivity = float((-tgt_delta[-1]) - (-comp_delta[-1]))  # positive = selective

    auc_tgt = float(_trapz(tgt_accs / max(tgt_base, 1e-8), ks / max(n_dims, 1)))

    return {
        "ks": ks.tolist(),
        "tgt_accs": [float(a) for a in tgt_accs],
        "comp_accs": [float(a) for a in comp_accs],
        "tgt_baseline": float(tgt_base),
        "comp_baseline": float(comp_base),
        "k_at_50pct_drop": k_50,
        "k_at_50pct_drop_frac": k_50 / n_dims,
        "auc_tgt_normalized": auc_tgt,
        "selectivity": selectivity,
        "n_dims": int(n_dims),
    }


def random_perturbation_sweep(
    X_test: np.ndarray,
    y_test: np.ndarray,
    probe: LinearProbe,
    target_class: int,
    n_steps: int = 200,
    n_trials: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Average perturbation sweep over random dim orderings (control)."""
    rng = np.random.RandomState(seed)
    n_dims = X_test.shape[1]
    all_tgt, all_comp = [], []

    for t in range(n_trials):
        ranking = rng.permutation(n_dims)
        result = perturbation_sweep(
            X_test, y_test, probe, ranking, target_class, n_steps, device,
        )
        all_tgt.append(result["tgt_accs"])
        all_comp.append(result["comp_accs"])

    tgt_mean = np.array(all_tgt).mean(axis=0)
    comp_mean = np.array(all_comp).mean(axis=0)
    tgt_base = tgt_mean[0]
    ks = np.array(result["ks"])

    half_drop = tgt_base * 0.5
    below = np.where(tgt_mean <= half_drop)[0]
    k_50 = int(ks[below[0]]) if len(below) > 0 else int(n_dims)

    auc = float(_trapz(tgt_mean / max(tgt_base, 1e-8), ks / max(n_dims, 1)))

    return {
        "ks": ks.tolist(),
        "tgt_accs_mean": [float(a) for a in tgt_mean],
        "comp_accs_mean": [float(a) for a in comp_mean],
        "k_at_50pct_drop": k_50,
        "auc_tgt_normalized": auc,
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# Active dimensions (L1 sparsity metric)
# ---------------------------------------------------------------------------

def count_active_dims(W: np.ndarray, threshold: float = 1e-4) -> dict:
    """Count dimensions with |W| > threshold, globally and per-class."""
    global_active = int((np.abs(W).max(axis=0) > threshold).sum())
    per_class = {}
    for c in range(W.shape[0]):
        per_class[c] = int((np.abs(W[c]) > threshold).sum())
    return {"global": global_active, "per_class": per_class}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_perturbation_curves(results: dict, plot_dir: Path) -> None:
    """Generate all PNG plots from TPP results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    spaces = results["spaces"]
    space_names = list(spaces.keys())
    colors = {"raw": "#1f77b4", "geoae": "#ff7f0e", "sae": "#2ca02c"}

    # Collect concept names from first space
    first_space = spaces[space_names[0]]
    concept_names = list(first_space["concepts"].keys())

    # 1. Per-concept perturbation curves (averaged across concepts)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for name in space_names:
        s = spaces[name]
        n_dims = s["n_dims"]
        # Average target and complement curves across concepts
        all_tgt, all_comp = [], []
        for cname in concept_names:
            cd = s["concepts"][cname]["importance"]
            ks_frac = np.array(cd["ks"]) / n_dims
            all_tgt.append(np.array(cd["tgt_accs"]) / max(cd["tgt_baseline"], 1e-8))
            all_comp.append(np.array(cd["comp_accs"]) / max(cd["comp_baseline"], 1e-8))
        tgt_mean = np.mean(all_tgt, axis=0)
        comp_mean = np.mean(all_comp, axis=0)
        c = colors.get(name, None)
        axes[0].plot(ks_frac, tgt_mean, label=name, color=c, linewidth=2)
        axes[1].plot(ks_frac, comp_mean, label=name, color=c, linewidth=2)

    axes[0].set_title("Target accuracy (concept c)")
    axes[1].set_title("Complement accuracy (non-c)")
    for ax in axes:
        ax.set_xlabel("Fraction of concept-c dims zeroed")
        ax.set_ylabel("Normalized accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Targeted Probe Perturbation (mean across concepts)", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "perturbation_curve.png", dpi=150)
    plt.close(fig)

    # 2. k@50% bar chart (mean across concepts)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(space_names))
    k50_vals = [spaces[n]["summary"]["mean_k_at_50pct_drop"] for n in space_names]
    bars = ax.bar(x_pos, k50_vals, color=[colors.get(n, "#999") for n in space_names])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(space_names)
    ax.set_ylabel("Mean k @ 50% target accuracy drop")
    ax.set_title("Dimensions to halve concept accuracy (fewer = more localized)")
    for bar, v in zip(bars, k50_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_dir / "k_at_50pct.png", dpi=150)
    plt.close(fig)

    # 3. Per-concept heatmap: k@50% (concepts × spaces)
    matrix = np.zeros((len(concept_names), len(space_names)))
    for j, name in enumerate(space_names):
        for i, cname in enumerate(concept_names):
            matrix[i, j] = spaces[name]["concepts"][cname]["importance"]["k_at_50pct_drop"]

    fig, ax = plt.subplots(
        figsize=(max(6, len(space_names) * 2), max(6, len(concept_names) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(space_names)))
    ax.set_xticklabels(space_names)
    ax.set_yticks(range(len(concept_names)))
    ax.set_yticklabels(concept_names, fontsize=8)
    ax.set_title("k @ 50% target drop per concept (fewer = more localized)")
    for i in range(len(concept_names)):
        for j in range(len(space_names)):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "per_class_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. Importance decay
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in space_names:
        W = np.array(spaces[name]["_probe_weights"])
        global_imp = np.abs(W).max(axis=0)
        sorted_imp = np.sort(global_imp)[::-1]
        ax.plot(np.arange(len(sorted_imp)) / len(sorted_imp), sorted_imp,
                label=name, color=colors.get(name, None), linewidth=2)
    ax.set_xlabel("Fraction of dimensions (sorted by importance)")
    ax.set_ylabel("|W| (max across classes)")
    ax.set_title("Probe weight importance decay")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "importance_decay.png", dpi=150)
    plt.close(fig)

    # 5. Selectivity bar chart per concept
    fig, ax = plt.subplots(figsize=(max(10, len(concept_names) * 0.8), 6))
    x_pos = np.arange(len(concept_names))
    width = 0.8 / len(space_names)
    for j, name in enumerate(space_names):
        sels = [spaces[name]["concepts"][cn]["importance"]["selectivity"]
                for cn in concept_names]
        ax.bar(x_pos + j * width, sels, width, label=name,
               color=colors.get(name, None))
    ax.set_xticks(x_pos + width * (len(space_names) - 1) / 2)
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Selectivity (target drop − complement drop)")
    ax.set_title("Perturbation selectivity per concept (higher = more selective)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "selectivity.png", dpi=150)
    plt.close(fig)

    print(f"[TPP] Plots saved to {plot_dir}/")


# ---------------------------------------------------------------------------
# Run one space
# ---------------------------------------------------------------------------

def run_space(
    space_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    class_names: list[str] | None,
    args,
    device: str,
) -> dict:
    """Full TPP pipeline for one representation space.

    Per-concept evaluation (like steering experiments): for each concept c,
    rank dims by importance for c, perturb, and measure target (class c)
    accuracy vs complement (non-c) accuracy.
    """
    print(f"\n{'='*50}")
    print(f"  Space: {space_name}  ({X_train.shape[1]} dims)")
    print(f"{'='*50}")

    # Train/val split from training data
    n = len(X_train)
    n_val = max(1, int(n * 0.1))
    idx = np.random.permutation(n)
    X_tr, y_tr = X_train[idx[n_val:]], y_train[idx[n_val:]]
    X_va, y_va = X_train[idx[:n_val]], y_train[idx[:n_val]]

    print(f"[TPP] Training probe: {X_tr.shape[0]} train, {X_va.shape[0]} val, "
          f"lambda_l1={args.lambda_l1}, lambda_l2={args.lambda_l2}")
    probe, epochs = train_probe(
        X_tr, y_tr, X_va, y_va, n_classes,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        lr=args.probe_lr,
        n_epochs=args.probe_epochs,
        patience=args.patience,
        device=device,
    )

    # Test accuracy (before filtering)
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).float().to(device)
        y_test_t = torch.from_numpy(y_test).long().to(device)
        preds = probe(X_test_t).argmax(1)
        test_acc = (preds == y_test_t).float().mean().item()
    print(f"[TPP] Probe accuracy: {test_acc:.4f}  (trained {epochs} epochs)")

    # Filter to probe true positives — only perturb samples the probe gets right
    tp_mask = (preds == y_test_t).cpu().numpy()
    n_before = len(X_test)
    X_test = X_test[tp_mask]
    y_test = y_test[tp_mask]
    print(f"[TPP] Probe TP filter: {n_before} → {len(X_test)} test samples")

    W = probe.linear.weight.detach().cpu().numpy()  # (n_classes, n_dims)
    active = count_active_dims(W)

    print(f"[TPP] Active dims (global): {active['global']} / {X_train.shape[1]}")

    result = {
        "n_dims": int(X_train.shape[1]),
        "probe_accuracy": float(test_acc),
        "n_test_before_tp": n_before,
        "n_test_after_tp": len(X_test),
        "n_active_dims": active["global"],
        "n_active_dims_per_class": {str(k): v for k, v in active["per_class"].items()},
        "probe_train_epochs": epochs,
        "_probe_weights": W.tolist(),
        "concepts": {},
    }

    # Per-concept perturbation (one at a time, like steering)
    class_labels = sorted(np.unique(y_test))
    for c in class_labels:
        cname = class_names[c] if class_names else str(c)
        n_tgt = int((y_test == c).sum())
        n_comp = int((y_test != c).sum())
        print(f"\n--- Concept {c} ({cname})  tgt={n_tgt}  comp={n_comp} ---")

        ranking_c = rank_dims_per_class(W, c)

        # Importance-ranked sweep
        sweep = perturbation_sweep(
            X_test, y_test, probe, ranking_c, c, args.n_steps, device,
        )

        # Random control
        rand = random_perturbation_sweep(
            X_test, y_test, probe, c, args.n_steps, args.n_random_trials,
            args.seed, device,
        )

        loc_ratio = rand["k_at_50pct_drop"] / max(sweep["k_at_50pct_drop"], 1)

        print(f"  tgt_base={sweep['tgt_baseline']:.3f}  "
              f"comp_base={sweep['comp_baseline']:.3f}  "
              f"k@50%={sweep['k_at_50pct_drop']}  "
              f"sel={sweep['selectivity']:.3f}  "
              f"loc_ratio={loc_ratio:.1f}")

        result["concepts"][cname] = {
            "class_idx": int(c),
            "n_target": n_tgt,
            "n_complement": n_comp,
            "n_active_dims": active["per_class"][c],
            "importance": sweep,
            "random": rand,
            "localization_ratio": loc_ratio,
        }

    # Summary across concepts
    k50s = [result["concepts"][cn]["importance"]["k_at_50pct_drop"]
            for cn in result["concepts"]]
    sels = [result["concepts"][cn]["importance"]["selectivity"]
            for cn in result["concepts"]]
    lrs = [result["concepts"][cn]["localization_ratio"]
           for cn in result["concepts"]]
    result["summary"] = {
        "mean_k_at_50pct_drop": float(np.mean(k50s)),
        "mean_selectivity": float(np.mean(sels)),
        "mean_localization_ratio": float(np.mean(lrs)),
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Targeted Probe Perturbation: measure concept localization"
    )
    ap.add_argument("--act_dir", required=True,
                    help="Dir with layer_N.npy, labels_train.npy, etc.")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--ae_checkpoint", default=None,
                    help="GeoAE checkpoint .pt (omit to skip geoae space)")
    ap.add_argument("--sae_checkpoint", default=None,
                    help="SAE checkpoint .pt (future, omit to skip)")
    ap.add_argument("--spaces", nargs="+", default=["raw", "geoae"],
                    choices=["raw", "geoae", "sae"])
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--n_random_trials", type=int, default=5)
    ap.add_argument("--lambda_l1", type=float, default=1e-1,
                    help="L1 regularization strength for probe (higher = sparser)")
    ap.add_argument("--lambda_l2", type=float, default=1e-2,
                    help="L2 regularization strength for probe (elastic net)")
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--out", default="results_tpp.json")
    ap.add_argument("--plot_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--joint_correct", default=None,
                    help="Path to joint-correct JSON (filter test set to LLM+AE "
                         "true positives only)")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick test: 1000 train, 20 steps, 1 trial")
    args = ap.parse_args()

    if args.smoke:
        args.n_steps = 20
        args.max_train = 1000
        args.n_random_trials = 1
        args.probe_epochs = 20
        if args.plot_dir is None:
            args.plot_dir = "plots/tpp_smoke"

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load raw activations
    act_dir = Path(args.act_dir)
    print(f"[TPP] Loading activations from {act_dir}")
    X_train_raw, y_train, X_test_raw, y_test = load_activations(act_dir, args.layer)

    # Filter test set to joint-correct samples (LLM + AE splice both correct)
    if args.joint_correct:
        jc_idx = load_joint_correct_indices(args.joint_correct, act_dir)
        print(f"[TPP] Joint-correct filter: {len(jc_idx)} / {len(X_test_raw)} "
              f"test samples retained")
        X_test_raw, y_test = X_test_raw[jc_idx], y_test[jc_idx]

    if args.max_train is not None and args.max_train < len(X_train_raw):
        idx = np.random.permutation(len(X_train_raw))[:args.max_train]
        X_train_raw, y_train = X_train_raw[idx], y_train[idx]

    n_classes = len(np.unique(y_train))

    # Load class names from joint-correct JSON or meta.json if available
    class_names = None
    if args.joint_correct:
        with open(args.joint_correct) as f:
            class_names = json.load(f).get("classes")
    if class_names is None:
        meta_path = act_dir / "meta.json"
        if meta_path.exists():
            class_names = json.load(open(meta_path)).get("classes")

    print(f"[TPP] Train: {X_train_raw.shape}  Test: {X_test_raw.shape}  "
          f"Classes: {n_classes}")

    # Validate space requirements
    if "geoae" in args.spaces and args.ae_checkpoint is None:
        print("[TPP] WARNING: --ae_checkpoint required for geoae space, skipping")
        args.spaces = [s for s in args.spaces if s != "geoae"]
    if "sae" in args.spaces and args.sae_checkpoint is None:
        print("[TPP] WARNING: --sae_checkpoint required for sae space, skipping")
        args.spaces = [s for s in args.spaces if s != "sae"]

    results = {
        "layer": args.layer,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "n_train": len(X_train_raw),
        "n_test": len(X_test_raw),
        "n_classes": n_classes,
        "joint_correct": args.joint_correct,
        "seed": args.seed,
        "spaces": {},
    }

    for space in args.spaces:
        if space == "raw":
            X_tr, X_te = transform_raw(X_train_raw, X_test_raw)
        elif space == "geoae":
            X_tr, X_te = transform_geoae(
                X_train_raw, X_test_raw, args.ae_checkpoint, device,
            )
        elif space == "sae":
            raise NotImplementedError("SAE space not yet implemented")
        else:
            raise ValueError(f"Unknown space: {space}")

        results["spaces"][space] = run_space(
            space, X_tr, y_train, X_te, y_test, n_classes,
            class_names, args, device,
        )

    # Remove probe weights from JSON output (large), keep for plotting
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot before stripping weights
    if args.plot_dir:
        plot_perturbation_curves(results, Path(args.plot_dir))

    # Strip weights for JSON
    results_json = json.loads(json.dumps(results))  # deep copy
    for s in results_json["spaces"].values():
        s.pop("_probe_weights", None)

    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[TPP] Results saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY  (per-concept means)   lambda_l1={args.lambda_l1}  lambda_l2={args.lambda_l2}")
    print(f"{'='*80}")
    print(f"{'Space':<10} {'Dims':>6} {'Acc':>7} {'Active':>7} {'Act%':>5} "
          f"{'k@50%':>7} {'k@50%%':>6} {'Select':>7} {'LocR':>6}")
    print(f"{'-'*80}")
    for name, s in results["spaces"].items():
        sm = s["summary"]
        ndims = s["n_dims"]
        act_pct = 100.0 * s["n_active_dims"] / max(ndims, 1)
        k50_pct = 100.0 * sm["mean_k_at_50pct_drop"] / max(ndims, 1)
        print(f"{name:<10} {ndims:>6} {s['probe_accuracy']:>7.3f} "
              f"{s['n_active_dims']:>7} {act_pct:>4.1f}% "
              f"{sm['mean_k_at_50pct_drop']:>7.0f} {k50_pct:>5.1f}% "
              f"{sm['mean_selectivity']:>7.3f} "
              f"{sm['mean_localization_ratio']:>6.1f}")

    # Per-concept detail table
    if class_names:
        print(f"\n{'='*90}")
        print("  PER-CONCEPT DETAIL")
        print(f"{'='*90}")
        for name, s in results["spaces"].items():
            ndims = s["n_dims"]
            print(f"\n  [{name}]")
            print(f"  {'Concept':<25} {'k@50%':>6} {'k%':>5} {'Select':>7} "
                  f"{'LocR':>6} {'Active':>7} {'Act%':>5} {'Tgt0':>6} {'Comp0':>6}")
            print(f"  {'-'*85}")
            row_k50, row_sel, row_lr, row_act, row_tgt, row_comp = [], [], [], [], [], []
            for cname, cd in s["concepts"].items():
                imp = cd["importance"]
                k50 = imp["k_at_50pct_drop"]
                k50_pct = 100.0 * k50 / max(ndims, 1)
                act_pct = 100.0 * cd["n_active_dims"] / max(ndims, 1)
                print(f"  {cname:<25} {k50:>6} {k50_pct:>4.1f}% "
                      f"{imp['selectivity']:>7.3f} "
                      f"{cd['localization_ratio']:>6.1f} "
                      f"{cd['n_active_dims']:>7} {act_pct:>4.1f}% "
                      f"{imp['tgt_baseline']:>6.3f} "
                      f"{imp['comp_baseline']:>6.3f}")
                row_k50.append(k50)
                row_sel.append(imp["selectivity"])
                row_lr.append(cd["localization_ratio"])
                row_act.append(cd["n_active_dims"])
                row_tgt.append(imp["tgt_baseline"])
                row_comp.append(imp["comp_baseline"])
            # AVG row
            print(f"  {'-'*85}")
            avg_k50 = np.mean(row_k50)
            avg_k50_pct = 100.0 * avg_k50 / max(ndims, 1)
            avg_act = np.mean(row_act)
            avg_act_pct = 100.0 * avg_act / max(ndims, 1)
            print(f"  {'AVG':<25} {avg_k50:>6.0f} {avg_k50_pct:>4.1f}% "
                  f"{np.mean(row_sel):>7.3f} "
                  f"{np.mean(row_lr):>6.1f} "
                  f"{avg_act:>7.0f} {avg_act_pct:>4.1f}% "
                  f"{np.mean(row_tgt):>6.3f} "
                  f"{np.mean(row_comp):>6.3f}")


if __name__ == "__main__":
    main()
