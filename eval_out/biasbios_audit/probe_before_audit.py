"""
Bias-controlled dual-probe perturbation analysis.

Train separate gender (2-class) and profession (28-class) linear probes on
BiasBios activations, then perform cross-attribute perturbation: rank dims by
gender probe importance, zero them out, and measure whether profession accuracy
is preserved or improves (removing the gender confound).

Compare raw layer-27 residuals vs GeoAE latent projections.

Usage:
    python -u -m geoae.bias.probe \
      --act_dir biasbios/activations/llama3.2-3B/last \
      --layer 27 \
      --ae_checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu_k2000_vicreg/best_val.pt \
      --spaces raw geoae \
      --out results/bias_probe.json \
      --plot_dir plots/bias_probe

    # quick smoke test
    python -m geoae.bias.probe \
      --act_dir biasbios/activations/llama3.2-3B/last \
      --layer 27 --spaces raw --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geoae.seeding import seed_everything
from geoae.interp.probe_perturbation import (
    LinearProbe,
    train_probe,
    load_activations,
    transform_raw,
    transform_geoae,
    rank_dims_global,
    count_active_dims,
)

from geoae.bias.extract import PROFESSIONS

# numpy compat: trapz → trapezoid in numpy 2.0+
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dual_labels(act_dir: Path):
    """Load profession + gender label arrays."""
    prof_train = np.load(act_dir / "labels_profession_train.npy").astype(np.int64)
    prof_test = np.load(act_dir / "labels_profession_test.npy").astype(np.int64)
    gender_train = np.load(act_dir / "labels_gender_train.npy").astype(np.int64)
    gender_test = np.load(act_dir / "labels_gender_test.npy").astype(np.int64)
    return prof_train, prof_test, gender_train, gender_test


# ---------------------------------------------------------------------------
# Cross-perturbation sweep
# ---------------------------------------------------------------------------

def cross_perturbation_sweep(
    X_test: np.ndarray,
    y_task: np.ndarray,
    probe_task: LinearProbe,
    y_spurious: np.ndarray,
    probe_spurious: LinearProbe,
    dim_ranking: np.ndarray,
    n_steps: int = 200,
    device: str = "cpu",
) -> dict:
    """Zero out dims ranked by spurious attribute, measure both task and spurious accuracy.

    Returns accuracy curves for both probes as dims are ablated.
    """
    n_dims = X_test.shape[1]
    ks = np.unique(np.linspace(0, n_dims, n_steps + 1, dtype=int))

    X_t = torch.from_numpy(X_test).float().to(device)
    y_task_t = torch.from_numpy(y_task).long().to(device)
    y_spur_t = torch.from_numpy(y_spurious).long().to(device)

    probe_task.eval()
    probe_spurious.eval()
    task_accs, spur_accs = [], []

    with torch.no_grad():
        for k in ks:
            X_p = X_t.clone()
            if k > 0:
                X_p[:, dim_ranking[:k]] = 0.0
            task_accs.append(
                (probe_task(X_p).argmax(1) == y_task_t).float().mean().item())
            spur_accs.append(
                (probe_spurious(X_p).argmax(1) == y_spur_t).float().mean().item())

    task_accs = np.array(task_accs)
    spur_accs = np.array(spur_accs)

    # k@50% for spurious accuracy
    spur_base = spur_accs[0]
    half_drop = spur_base * 0.5
    below = np.where(spur_accs <= half_drop)[0]
    spur_k50 = int(ks[below[0]]) if len(below) > 0 else int(n_dims)

    # k@chance for spurious (below 50% for binary = random)
    below_chance = np.where(spur_accs <= 0.5)[0]
    spur_k_chance = int(ks[below_chance[0]]) if len(below_chance) > 0 else int(n_dims)

    # Task accuracy delta at the spurious k@chance point (primary metric)
    task_base = task_accs[0]
    kch_idx = np.searchsorted(ks, spur_k_chance)
    kch_idx = min(kch_idx, len(task_accs) - 1)
    task_at_kch = task_accs[kch_idx]
    task_delta_at_kch = float(task_at_kch - task_base)

    # Also at k@50% (may saturate for binary)
    k50_idx = np.searchsorted(ks, spur_k50)
    k50_idx = min(k50_idx, len(task_accs) - 1)
    task_at_k50 = task_accs[k50_idx]
    task_delta_at_k50 = float(task_at_k50 - task_base)

    return {
        "ks": ks.tolist(),
        "task_accs": [float(a) for a in task_accs],
        "spur_accs": [float(a) for a in spur_accs],
        "task_baseline": float(task_base),
        "spur_baseline": float(spur_base),
        "spur_k_at_50pct": spur_k50,
        "spur_k_at_50pct_frac": spur_k50 / n_dims,
        "spur_k_at_chance": spur_k_chance,
        "spur_k_at_chance_frac": spur_k_chance / n_dims,
        "task_delta_at_spur_k50": task_delta_at_k50,
        "task_at_spur_k50": float(task_at_k50),
        "task_delta_at_spur_kch": task_delta_at_kch,
        "task_at_spur_kch": float(task_at_kch),
        "n_dims": int(n_dims),
    }


def random_cross_sweep(
    X_test: np.ndarray,
    y_task: np.ndarray,
    probe_task: LinearProbe,
    y_spurious: np.ndarray,
    probe_spurious: LinearProbe,
    n_steps: int = 200,
    n_trials: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Average cross-perturbation sweep over random dim orderings (control)."""
    rng = np.random.RandomState(seed)
    n_dims = X_test.shape[1]
    all_task, all_spur = [], []

    for _ in range(n_trials):
        ranking = rng.permutation(n_dims)
        result = cross_perturbation_sweep(
            X_test, y_task, probe_task, y_spurious, probe_spurious,
            ranking, n_steps, device,
        )
        all_task.append(result["task_accs"])
        all_spur.append(result["spur_accs"])

    task_mean = np.array(all_task).mean(axis=0)
    spur_mean = np.array(all_spur).mean(axis=0)
    ks = np.array(result["ks"])

    spur_base = spur_mean[0]
    half_drop = spur_base * 0.5
    below = np.where(spur_mean <= half_drop)[0]
    spur_k50 = int(ks[below[0]]) if len(below) > 0 else int(n_dims)

    return {
        "ks": ks.tolist(),
        "task_accs_mean": [float(a) for a in task_mean],
        "spur_accs_mean": [float(a) for a in spur_mean],
        "spur_k_at_50pct": spur_k50,
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# Per-profession debiasing breakdown
# ---------------------------------------------------------------------------

def per_profession_delta(
    X_test: np.ndarray,
    y_prof: np.ndarray,
    probe_prof: LinearProbe,
    dim_ranking: np.ndarray,
    k: int,
    device: str = "cpu",
) -> dict:
    """Measure per-profession accuracy before/after zeroing top-k dims."""
    X_t = torch.from_numpy(X_test).float().to(device)
    y_t = torch.from_numpy(y_prof).long().to(device)

    probe_prof.eval()
    with torch.no_grad():
        preds_before = probe_prof(X_t).argmax(1)
        X_p = X_t.clone()
        X_p[:, dim_ranking[:k]] = 0.0
        preds_after = probe_prof(X_p).argmax(1)

    result = {}
    for c in sorted(np.unique(y_prof)):
        mask = y_t == c
        n = int(mask.sum())
        if n == 0:
            continue
        acc_before = float((preds_before[mask] == c).float().mean())
        acc_after = float((preds_after[mask] == c).float().mean())
        name = PROFESSIONS[c] if c < len(PROFESSIONS) else str(c)
        result[name] = {
            "n": n,
            "acc_before": acc_before,
            "acc_after": acc_after,
            "delta": acc_after - acc_before,
        }
    return result


# ---------------------------------------------------------------------------
# Dimension overlap
# ---------------------------------------------------------------------------

def compute_dim_overlap(
    ranking_a: np.ndarray,
    ranking_b: np.ndarray,
    ks: list[int],
) -> dict:
    """Fraction of top-k(a) ∩ top-k(b) at each k."""
    overlaps = {}
    for k in ks:
        if k <= 0:
            continue
        set_a = set(ranking_a[:k].tolist())
        set_b = set(ranking_b[:k].tolist())
        overlaps[k] = len(set_a & set_b) / k
    return overlaps


# ---------------------------------------------------------------------------
# Run one space
# ---------------------------------------------------------------------------

def run_bias_space(
    space_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    prof_train: np.ndarray,
    prof_test: np.ndarray,
    gender_train: np.ndarray,
    gender_test: np.ndarray,
    args,
    device: str,
) -> dict:
    """Full dual-probe pipeline for one representation space."""
    n_professions = len(np.unique(prof_train))
    n_genders = 2

    print(f"\n{'='*60}")
    print(f"  Space: {space_name}  ({X_train.shape[1]} dims)")
    print(f"{'='*60}")

    # Train/val split
    n = len(X_train)
    n_val = max(1, int(n * 0.1))
    idx = np.random.permutation(n)
    X_tr, X_va = X_train[idx[n_val:]], X_train[idx[:n_val]]
    p_tr, p_va = prof_train[idx[n_val:]], prof_train[idx[:n_val]]
    g_tr, g_va = gender_train[idx[n_val:]], gender_train[idx[:n_val]]

    # --- Train gender probe ---
    use_bias = not args.no_probe_bias
    print(f"\n[bias] Training GENDER probe (2-class, L1={args.gender_lambda_l1}, "
          f"L2={args.gender_lambda_l2}, bias={use_bias}) ...")
    gender_probe, g_epochs = train_probe(
        X_tr, g_tr, X_va, g_va, n_genders,
        lambda_l1=args.gender_lambda_l1, lambda_l2=args.gender_lambda_l2,
        lr=args.probe_lr, n_epochs=args.probe_epochs,
        patience=args.patience, device=device, bias=use_bias,
    )
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_test).float().to(device)
        gender_acc = (gender_probe(X_te_t).argmax(1) ==
                      torch.from_numpy(gender_test).long().to(device)).float().mean().item()
    print(f"[bias] Gender probe accuracy: {gender_acc:.4f}  ({g_epochs} epochs)")

    # --- Train profession probe ---
    print(f"[bias] Training PROFESSION probe ({n_professions}-class, L1={args.prof_lambda_l1}, "
          f"L2={args.prof_lambda_l2}, bias={use_bias}) ...")
    prof_probe, p_epochs = train_probe(
        X_tr, p_tr, X_va, p_va, n_professions,
        lambda_l1=args.prof_lambda_l1, lambda_l2=args.prof_lambda_l2,
        lr=args.probe_lr, n_epochs=args.probe_epochs,
        patience=args.patience, device=device, bias=use_bias,
    )
    with torch.no_grad():
        prof_acc = (prof_probe(X_te_t).argmax(1) ==
                    torch.from_numpy(prof_test).long().to(device)).float().mean().item()
    print(f"[bias] Profession probe accuracy: {prof_acc:.4f}  ({p_epochs} epochs)")

    # --- Dimension rankings ---
    W_gender = gender_probe.linear.weight.detach().cpu().numpy()
    W_prof = prof_probe.linear.weight.detach().cpu().numpy()
    gender_ranking = rank_dims_global(W_gender)
    prof_ranking = rank_dims_global(W_prof)

    gender_active = count_active_dims(W_gender)
    prof_active = count_active_dims(W_prof)
    print(f"[bias] Gender active dims: {gender_active['global']} / {X_train.shape[1]}")
    print(f"[bias] Prof active dims:   {prof_active['global']} / {X_train.shape[1]}")

    # --- Cross-perturbation: zero gender dims, measure both ---
    print("\n[bias] Cross-perturbation: zeroing GENDER-ranked dims ...")
    debiasing = cross_perturbation_sweep(
        X_test, prof_test, prof_probe,
        gender_test, gender_probe,
        gender_ranking, args.n_steps, device,
    )
    print(f"  gender k@chance={debiasing['spur_k_at_chance']} ({debiasing['spur_k_at_chance_frac']:.1%})  "
          f"k@50%={debiasing['spur_k_at_50pct']}  "
          f"prof Δ@kch={debiasing['task_delta_at_spur_kch']:+.4f}")

    # --- Reverse cross: zero profession dims, measure both ---
    print("[bias] Reverse cross: zeroing PROFESSION-ranked dims ...")
    leakage = cross_perturbation_sweep(
        X_test, prof_test, prof_probe,
        gender_test, gender_probe,
        prof_ranking, args.n_steps, device,
    )
    print(f"  prof k@50%: gender drops to {leakage['spur_accs'][-1]:.3f} "
          f"when all prof dims zeroed")

    # --- Random control ---
    print("[bias] Random control sweep ...")
    random_ctrl = random_cross_sweep(
        X_test, prof_test, prof_probe,
        gender_test, gender_probe,
        args.n_steps, args.n_random_trials, args.seed, device,
    )

    # --- Per-profession debiasing at gender k@chance ---
    k_debias = debiasing["spur_k_at_chance"]
    print(f"\n[bias] Per-profession delta at k={k_debias} (gender k@chance) ...")
    per_prof = per_profession_delta(
        X_test, prof_test, prof_probe,
        gender_ranking, k_debias, device,
    )
    # Print top gainers/losers
    sorted_profs = sorted(per_prof.items(), key=lambda x: x[1]["delta"], reverse=True)
    print(f"  {'Profession':<22} {'N':>6} {'Before':>7} {'After':>7} {'Delta':>7}")
    print(f"  {'-'*55}")
    for name, d in sorted_profs[:5]:
        print(f"  {name:<22} {d['n']:>6} {d['acc_before']:>7.3f} "
              f"{d['acc_after']:>7.3f} {d['delta']:>+7.3f}")
    print("  ...")
    for name, d in sorted_profs[-5:]:
        print(f"  {name:<22} {d['n']:>6} {d['acc_before']:>7.3f} "
              f"{d['acc_after']:>7.3f} {d['delta']:>+7.3f}")

    # --- Dimension overlap ---
    overlap_ks = [k for k in [50, 100, 200, 500] if k <= X_train.shape[1]]
    overlaps = compute_dim_overlap(gender_ranking, prof_ranking, overlap_ks)
    print("\n[bias] Dim overlap (gender ∩ prof):  "
          + "  ".join(f"@{k}={v:.2f}" for k, v in overlaps.items()))

    # --- Debiasing efficiency (at k@chance) ---
    prof_drop = debiasing["task_baseline"] - debiasing["task_at_spur_kch"]
    gender_drop = debiasing["spur_baseline"] - 0.5  # dropped to chance
    efficiency = gender_drop / max(abs(prof_drop), 1e-8)
    # positive prof_drop = profession hurt; negative = profession improved

    return {
        "space": space_name,
        "n_dims": int(X_train.shape[1]),
        "gender_probe_accuracy": float(gender_acc),
        "profession_probe_accuracy": float(prof_acc),
        "gender_active_dims": gender_active["global"],
        "profession_active_dims": prof_active["global"],
        "gender_probe_epochs": g_epochs,
        "profession_probe_epochs": p_epochs,
        "debiasing_curve": debiasing,
        "leakage_curve": leakage,
        "random_control": random_ctrl,
        "per_profession_debiasing": per_prof,
        "dim_overlap": overlaps,
        "summary": {
            "gender_k_at_chance": debiasing["spur_k_at_chance"],
            "gender_k_at_chance_frac": debiasing["spur_k_at_chance_frac"],
            "gender_k_at_50pct": debiasing["spur_k_at_50pct"],
            "profession_delta_at_gender_kch": debiasing["task_delta_at_spur_kch"],
            "profession_at_gender_kch": debiasing["task_at_spur_kch"],
            "debiasing_efficiency": float(efficiency),
            "dim_overlap_at_100": overlaps.get(100, None),
        },
        "_gender_probe_weights": W_gender.tolist(),
        "_profession_probe_weights": W_prof.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bias_curves(results: dict, plot_dir: Path) -> None:
    """Generate all plots from bias probe results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    spaces = results["spaces"]
    space_names = list(spaces.keys())
    colors = {"raw": "#1f77b4", "geoae": "#ff7f0e", "sae": "#2ca02c"}

    # 1. Debiasing curve: gender + profession acc vs gender dims zeroed
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for name in space_names:
        s = spaces[name]
        d = s["debiasing_curve"]
        n_dims = d["n_dims"]
        ks_frac = np.array(d["ks"]) / n_dims
        c = colors.get(name, None)
        axes[0].plot(ks_frac, d["spur_accs"], label=name, color=c, linewidth=2)
        axes[1].plot(ks_frac, d["task_accs"], label=name, color=c, linewidth=2)

    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    axes[0].set_title("Gender accuracy (should drop)")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Profession accuracy (should stay / improve)")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.set_xlabel("Fraction of gender-ranked dims zeroed")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Debiasing curve: zero gender-important dims", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "debiasing_curve.png", dpi=150)
    plt.close(fig)

    # 2. Cross-perturbation matrix: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    titles = [
        ["Gender acc vs gender dims zeroed", "Prof acc vs gender dims zeroed"],
        ["Gender acc vs prof dims zeroed", "Prof acc vs prof dims zeroed"],
    ]
    for name in space_names:
        s = spaces[name]
        c = colors.get(name, None)
        d = s["debiasing_curve"]
        l = s["leakage_curve"]
        n_d = d["n_dims"]
        n_l = l["n_dims"]
        ks_d = np.array(d["ks"]) / n_d
        ks_l = np.array(l["ks"]) / n_l
        axes[0, 0].plot(ks_d, d["spur_accs"], label=name, color=c, linewidth=2)
        axes[0, 1].plot(ks_d, d["task_accs"], label=name, color=c, linewidth=2)
        axes[1, 0].plot(ks_l, l["spur_accs"], label=name, color=c, linewidth=2)
        axes[1, 1].plot(ks_l, l["task_accs"], label=name, color=c, linewidth=2)
    for i in range(2):
        for j in range(2):
            axes[i, j].set_title(titles[i][j])
            axes[i, j].set_xlabel("Fraction of dims zeroed")
            axes[i, j].set_ylabel("Accuracy")
            axes[i, j].legend()
            axes[i, j].grid(True, alpha=0.3)
    fig.suptitle("Cross-perturbation matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(plot_dir / "cross_perturbation_matrix.png", dpi=150)
    plt.close(fig)

    # 3. gender_k@chance bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(space_names))
    kch_vals = [spaces[n]["summary"]["gender_k_at_chance"] for n in space_names]
    bars = ax.bar(x_pos, kch_vals, color=[colors.get(n, "#999") for n in space_names])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(space_names)
    ax.set_ylabel("Dims to reduce gender to chance")
    ax.set_title("Gender k@chance (fewer = more localized)")
    for bar, v in zip(bars, kch_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{v}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_dir / "gender_k_at_chance.png", dpi=150)
    plt.close(fig)

    # 4. Debiasing efficiency bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    eff_vals = [spaces[n]["summary"]["debiasing_efficiency"] for n in space_names]
    eff_vals_clipped = [min(v, 100) for v in eff_vals]  # clip for display
    bars = ax.bar(x_pos, eff_vals_clipped,
                  color=[colors.get(n, "#999") for n in space_names])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(space_names)
    ax.set_ylabel("Gender drop / Profession drop")
    ax.set_title("Debiasing efficiency (higher = more surgical)")
    for bar, v in zip(bars, eff_vals):
        label = f"{v:.1f}" if v < 100 else f"{v:.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                label, ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(plot_dir / "debiasing_efficiency.png", dpi=150)
    plt.close(fig)

    # 5. Dimension overlap curve
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in space_names:
        s = spaces[name]
        ov = s["dim_overlap"]
        ks_ov = sorted(int(k) for k in ov.keys())
        vals = [ov[k] if isinstance(k, int) else ov[str(k)] for k in ks_ov]
        ax.plot(ks_ov, vals, "o-", label=name, color=colors.get(name, None),
                linewidth=2, markersize=8)
    ax.set_xlabel("k (top-k dims)")
    ax.set_ylabel("Overlap fraction (gender ∩ profession)")
    ax.set_title("Dimension overlap (lower = more disentangled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "dim_overlap.png", dpi=150)
    plt.close(fig)

    # 6. Per-profession heatmap
    # Collect all profession names across spaces
    all_prof_names = set()
    for name in space_names:
        all_prof_names.update(spaces[name]["per_profession_debiasing"].keys())
    prof_names = sorted(all_prof_names)

    matrix = np.zeros((len(prof_names), len(space_names)))
    for j, name in enumerate(space_names):
        ppd = spaces[name]["per_profession_debiasing"]
        for i, pname in enumerate(prof_names):
            if pname in ppd:
                matrix[i, j] = ppd[pname]["delta"]

    fig, ax = plt.subplots(
        figsize=(max(6, len(space_names) * 2.5), max(8, len(prof_names) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(space_names)))
    ax.set_xticklabels(space_names)
    ax.set_yticks(range(len(prof_names)))
    ax.set_yticklabels(prof_names, fontsize=8)
    ax.set_title("Per-profession accuracy delta after debiasing\n(green=improved, red=hurt)")
    for i in range(len(prof_names)):
        for j in range(len(space_names)):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(plot_dir / "per_profession_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"[bias] Plots saved to {plot_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Bias-controlled dual-probe perturbation analysis"
    )
    ap.add_argument("--act_dir", required=True,
                    help="Dir with layer_N.npy, labels_profession_*.npy, labels_gender_*.npy")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--ae_checkpoint", default=None)
    ap.add_argument("--spaces", nargs="+", default=["raw", "geoae"],
                    choices=["raw", "geoae"])
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--n_random_trials", type=int, default=5)
    ap.add_argument("--lambda_l1", type=float, default=1e-1,
                    help="Default L1 regularization for both probes")
    ap.add_argument("--lambda_l2", type=float, default=1e-2,
                    help="Default L2 regularization for both probes")
    ap.add_argument("--gender_lambda_l1", type=float, default=None,
                    help="L1 for gender probe (overrides --lambda_l1)")
    ap.add_argument("--prof_lambda_l1", type=float, default=None,
                    help="L1 for profession probe (overrides --lambda_l1)")
    ap.add_argument("--gender_lambda_l2", type=float, default=None,
                    help="L2 for gender probe (overrides --lambda_l2)")
    ap.add_argument("--prof_lambda_l2", type=float, default=None,
                    help="L2 for profession probe (overrides --lambda_l2)")
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--out", default="results_bias_probe.json")
    ap.add_argument("--plot_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zscore_latents", action="store_true",
                    help="Z-score AE latents before probing (handles GELU non-negativity)")
    ap.add_argument("--no_probe_bias", action="store_true",
                    help="Train probes without bias term (removes prior leakage in k@chance)")
    ap.add_argument("--smoke", action="store_true",
                    help="Quick test: 1000 train, 20 steps, 1 trial")
    args = ap.parse_args()

    # Resolve per-probe L1/L2 (fall back to --lambda_l1/l2)
    if args.gender_lambda_l1 is None:
        args.gender_lambda_l1 = args.lambda_l1
    if args.prof_lambda_l1 is None:
        args.prof_lambda_l1 = args.lambda_l1
    if args.gender_lambda_l2 is None:
        args.gender_lambda_l2 = args.lambda_l2
    if args.prof_lambda_l2 is None:
        args.prof_lambda_l2 = args.lambda_l2

    if args.smoke:
        args.n_steps = 20
        args.max_train = 1000
        args.n_random_trials = 1
        args.probe_epochs = 20
        if args.plot_dir is None:
            args.plot_dir = "plots/bias_probe_smoke"

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load raw activations
    act_dir = Path(args.act_dir)
    print(f"[bias] Loading activations from {act_dir}")
    X_train_raw, _, X_test_raw, _ = load_activations(act_dir, args.layer)
    prof_train, prof_test, gender_train, gender_test = load_dual_labels(act_dir)

    if args.max_train is not None and args.max_train < len(X_train_raw):
        idx = np.random.permutation(len(X_train_raw))[:args.max_train]
        X_train_raw = X_train_raw[idx]
        prof_train = prof_train[idx]
        gender_train = gender_train[idx]

    n_professions = len(np.unique(prof_train))
    print(f"[bias] Train: {X_train_raw.shape}  Test: {X_test_raw.shape}  "
          f"Professions: {n_professions}  Genders: 2")
    print(f"[bias] Gender balance (train): "
          f"male={int((gender_train==0).sum())} "
          f"female={int((gender_train==1).sum())}")

    # Validate space requirements
    if "geoae" in args.spaces and args.ae_checkpoint is None:
        print("[bias] WARNING: --ae_checkpoint required for geoae space, skipping")
        args.spaces = [s for s in args.spaces if s != "geoae"]

    results = {
        "layer": args.layer,
        "gender_lambda_l1": args.gender_lambda_l1,
        "gender_lambda_l2": args.gender_lambda_l2,
        "prof_lambda_l1": args.prof_lambda_l1,
        "prof_lambda_l2": args.prof_lambda_l2,
        "n_train": len(X_train_raw),
        "n_test": len(X_test_raw),
        "n_professions": n_professions,
        "seed": args.seed,
        "zscore_latents": args.zscore_latents,
        "spaces": {},
    }

    for space in args.spaces:
        if space == "raw":
            X_tr, X_te = transform_raw(X_train_raw, X_test_raw)
        elif space == "geoae":
            X_tr, X_te = transform_geoae(
                X_train_raw, X_test_raw, args.ae_checkpoint, device,
            )
            if args.zscore_latents:
                mean = X_tr.mean(axis=0)
                std = X_tr.std(axis=0) + 1e-8
                X_tr = (X_tr - mean) / std
                X_te = (X_te - mean) / std
        else:
            raise ValueError(f"Unknown space: {space}")

        results["spaces"][space] = run_bias_space(
            space, X_tr, X_te,
            prof_train, prof_test,
            gender_train, gender_test,
            args, device,
        )

    # Plot before stripping weights
    if args.plot_dir:
        plot_bias_curves(results, Path(args.plot_dir))

    # Strip probe weights for JSON output (large)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_json = json.loads(json.dumps(results))
    for s in results_json["spaces"].values():
        s.pop("_gender_probe_weights", None)
        s.pop("_profession_probe_weights", None)

    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[bias] Results saved to {out_path}")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"  SUMMARY   gender_l1={args.gender_lambda_l1} gender_l2={args.gender_lambda_l2}  "
          f"prof_l1={args.prof_lambda_l1} prof_l2={args.prof_lambda_l2}")
    print(f"{'='*90}")
    print(f"{'Space':<10} {'Dims':>6} {'GendAcc':>8} {'ProfAcc':>8} "
          f"{'G_k@ch':>7} {'G_k@ch%':>7} "
          f"{'P@kch':>7} {'P_Δ@kch':>8} {'Effic':>7} {'Ovlp@100':>9}")
    print(f"{'-'*90}")
    for name, s in results["spaces"].items():
        sm = s["summary"]
        ndims = s["n_dims"]
        kch_pct = 100.0 * sm["gender_k_at_chance"] / max(ndims, 1)
        ovlp = sm.get("dim_overlap_at_100")
        ovlp_str = f"{ovlp:.3f}" if ovlp is not None else "N/A"
        print(f"{name:<10} {ndims:>6} {s['gender_probe_accuracy']:>8.4f} "
              f"{s['profession_probe_accuracy']:>8.4f} "
              f"{sm['gender_k_at_chance']:>7} {kch_pct:>6.1f}% "
              f"{sm['profession_at_gender_kch']:>7.4f} "
              f"{sm['profession_delta_at_gender_kch']:>+8.4f} "
              f"{sm['debiasing_efficiency']:>7.1f} "
              f"{ovlp_str:>9}")


if __name__ == "__main__":
    main()
