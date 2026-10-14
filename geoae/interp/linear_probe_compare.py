"""Compare probe decodability across raw and GeoAE representation spaces.

The utility is deliberately narrower than ``probe_perturbation``: it focuses on
whether a linear or one-hidden-layer classifier was optimized well enough. It
reports the best epoch and final train/validation/test cross-entropy, accuracy,
balanced accuracy, and macro-F1, together with the full learning curve.

Example (BiasBios profession, 28 classes)::

    python -m geoae.interp.linear_probe_compare \
      --act_dir biasbios/activations/llama3.2-3B/last --layer 27 \
      --checkpoint unsupervised=checkpoints/.../step_0014200.pt \
      --checkpoint seeded_atlas=checkpoints/.../step_0014200.pt \
      --out results/linear_probe_biasbios_profession_l27.json
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from geoae.checkpoint import load_ae_checkpoint
from geoae.seeding import seed_everything


def _load_data(act_dir: Path, layer: int, label: str):
    train_path = act_dir / f"layer_{layer}.npy"
    test_path = act_dir / f"layer_{layer}_test.npy"
    if label == "profession":
        y_train_path = act_dir / "labels_profession_train.npy"
        y_test_path = act_dir / "labels_profession_test.npy"
    else:
        y_train_path = act_dir / "labels_train.npy"
        y_test_path = act_dir / "labels_test.npy"

    for path in (train_path, test_path, y_train_path, y_test_path):
        if not path.exists():
            raise FileNotFoundError(path)

    X_train = np.load(train_path).astype(np.float32)
    X_test = np.load(test_path).astype(np.float32)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("activation and label row counts do not match")

    # Multi-label datasets such as GoEmotions use the official validation split
    # rather than making a random split whose label combinations are difficult
    # to stratify faithfully.
    val_path = act_dir / f"layer_{layer}_validation.npy"
    y_val_path = act_dir / "labels_validation.npy"
    if val_path.exists() != y_val_path.exists():
        raise FileNotFoundError("validation activations and labels must both exist")
    if val_path.exists():
        X_val = np.load(val_path).astype(np.float32)
        y_val = np.load(y_val_path)
        if len(X_val) != len(y_val):
            raise ValueError("validation activation and label row counts do not match")
    else:
        X_val = y_val = None
    return X_train, y_train, X_val, y_val, X_test, y_test


@torch.inference_mode()
def _encode_geoae(
    arrays: list[np.ndarray],
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
):
    ae, norm_mean, norm_std, _ = load_ae_checkpoint(checkpoint, device)
    ae.eval()

    def encode(X: np.ndarray):
        parts = []
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start : start + batch_size]).to(device)
            z = ae.encoder((xb - norm_mean) / norm_std)
            parts.append(z.cpu().numpy())
        return np.concatenate(parts)

    encoded = [encode(X) for X in arrays]
    del ae
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return encoded


@torch.inference_mode()
def _evaluate(
    probe: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    task: str = "single_label",
    threshold: float = 0.5,
    detailed: bool = True,
):
    probe.eval()
    ce_sum = 0.0
    score_parts = []
    for start in range(0, len(X), batch_size):
        xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = probe(xb)
        if task == "multi_label":
            # Sum over examples and labels here, then divide by N*C below so
            # this matches PyTorch's usual mean BCE scale.
            ce_sum += nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="sum"
            ).item()
            score_parts.append(logits.sigmoid().cpu())
        else:
            ce_sum += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
            score_parts.append(logits.argmax(1).cpu())
    scores = torch.cat(score_parts).numpy()
    target = y.cpu().numpy()
    if task == "multi_label":
        pred = scores >= threshold
        target_bool = target.astype(bool)
        basic = {
            "cross_entropy": ce_sum / target.size,
            "accuracy": float(np.all(pred == target_bool, axis=1).mean()),
        }
        if not detailed:
            return basic
        per_label_balanced = [
            balanced_accuracy_score(target_bool[:, j], pred[:, j])
            for j in range(target.shape[1])
        ]
        return {
            **basic,
            "balanced_accuracy": float(np.mean(per_label_balanced)),
            "macro_f1": float(f1_score(target_bool, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(target_bool, pred, average="micro", zero_division=0)),
            "samples_f1": float(f1_score(target_bool, pred, average="samples", zero_division=0)),
            "hamming_accuracy": float((pred == target_bool).mean()),
            "micro_average_precision": float(
                average_precision_score(target_bool, scores, average="micro")
            ),
            "macro_average_precision": float(
                average_precision_score(target_bool, scores, average="macro")
            ),
        }
    pred = scores
    return {
        "cross_entropy": ce_sum / len(X),
        "accuracy": float((pred == target).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(target, pred)),
        "macro_f1": float(f1_score(target, pred, average="macro", zero_division=0)),
    }


def _fit_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    task: str,
    seed: int,
    val_frac: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
):
    seed_everything(seed)
    if X_val is None:
        indices = np.arange(len(y_train))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_frac,
            random_state=seed,
            stratify=y_train if task == "single_label" else None,
        )
        X_fit, y_fit = X_train[train_idx], y_train[train_idx]
        X_holdout, y_holdout = X_train[val_idx], y_train[val_idx]
    else:
        X_fit, y_fit = X_train, y_train
        X_holdout, y_holdout = X_val, y_val

    # Feature scaling is fit on probe-training rows only.  This avoids leakage
    # from validation/test data and makes optimization comparable across spaces.
    mean = X_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_fit.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    Xtr = torch.from_numpy((X_fit - mean) / std).to(device)
    Xva = torch.from_numpy((X_holdout - mean) / std).to(device)
    Xte = torch.from_numpy((X_test - mean) / std).to(device)
    target_dtype = torch.float32 if task == "multi_label" else torch.int64
    ytr = torch.as_tensor(y_fit, dtype=target_dtype, device=device)
    yva = torch.as_tensor(y_holdout, dtype=target_dtype, device=device)
    yte = torch.as_tensor(y_test, dtype=target_dtype, device=device)

    n_classes = (
        int(y_train.shape[1]) if task == "multi_label"
        else int(max(y_train.max(), y_test.max()) + 1)
    )
    if hidden_dim > 0:
        probe = nn.Sequential(
            nn.Linear(X_train.shape[1], hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes, bias=True),
        ).to(device)
    else:
        probe = nn.Linear(X_train.shape[1], n_classes, bias=True).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    if task == "multi_label":
        # GoEmotions has only about 4% positive label slots. Starting each
        # output at its empirical prior lets optimization focus immediately on
        # representation-dependent signal rather than relearning prevalence.
        prior = ytr.mean(dim=0).clamp(1e-5, 1 - 1e-5)
        output_layer = probe[-1] if hidden_dim > 0 else probe
        with torch.no_grad():
            output_layer.bias.copy_(torch.logit(prior))

    history = []
    best_epoch = 0
    best_val_ce = float("inf")
    best_state = None
    wait = 0
    generator = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(1, epochs + 1):
        probe.train()
        perm = torch.randperm(len(Xtr), generator=generator, device=device)
        for start in range(0, len(perm), batch_size):
            take = perm[start : start + batch_size]
            logits = probe(Xtr[take])
            if task == "multi_label":
                loss = nn.functional.binary_cross_entropy_with_logits(logits, ytr[take])
            else:
                loss = nn.functional.cross_entropy(logits, ytr[take])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_metrics = _evaluate(
            probe, Xtr, ytr, batch_size, task=task, detailed=False
        )
        val_metrics = _evaluate(
            probe, Xva, yva, batch_size, task=task, detailed=False
        )
        history.append({
            "epoch": epoch,
            "train_cross_entropy": train_metrics["cross_entropy"],
            "train_accuracy": train_metrics["accuracy"],
            "val_cross_entropy": val_metrics["cross_entropy"],
            "val_accuracy": val_metrics["accuracy"],
        })
        print(
            f"    epoch {epoch:3d}  train CE={train_metrics['cross_entropy']:.4f} "
            f"acc={train_metrics['accuracy']:.4f}  val CE={val_metrics['cross_entropy']:.4f} "
            f"acc={val_metrics['accuracy']:.4f}",
            flush=True,
        )

        # Cross-entropy is smoother than accuracy and detects continued fitting
        # even when the discrete validation predictions do not change.
        if val_metrics["cross_entropy"] < best_val_ce - 1e-5:
            best_val_ce = val_metrics["cross_entropy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("probe training produced no checkpoint")
    probe.load_state_dict(best_state)
    threshold = 0.5
    if task == "multi_label":
        # A single validation-calibrated threshold is more meaningful than 0.5
        # for sparse labels, while avoiding 28 separately overfit thresholds.
        candidates = np.arange(0.05, 0.951, 0.01)
        score_parts = []
        with torch.inference_mode():
            for start in range(0, len(Xva), batch_size):
                score_parts.append(probe(Xva[start:start + batch_size]).sigmoid().cpu())
        val_scores = torch.cat(score_parts).numpy()
        val_target = yva.cpu().numpy().astype(bool)
        threshold = max(
            candidates,
            key=lambda t: f1_score(
                val_target, val_scores >= t, average="micro", zero_division=0
            ),
        )
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "threshold": float(threshold),
        "train": _evaluate(probe, Xtr, ytr, batch_size, task=task, threshold=threshold),
        "validation": _evaluate(probe, Xva, yva, batch_size, task=task, threshold=threshold),
        "test": _evaluate(probe, Xte, yte, batch_size, task=task, threshold=threshold),
        "history": history,
    }
    result["train_test_accuracy_gap"] = (
        result["train"]["accuracy"] - result["test"]["accuracy"]
    )
    return result


def _aggregate(runs: list[dict], task: str):
    summary = {}
    for split in ("train", "validation", "test"):
        summary[split] = {}
        metrics = ["cross_entropy", "accuracy", "balanced_accuracy", "macro_f1"]
        if task == "multi_label":
            metrics += [
                "micro_f1", "samples_f1", "hamming_accuracy",
                "micro_average_precision", "macro_average_precision",
            ]
        for metric in metrics:
            values = np.array([r[split][metric] for r in runs], dtype=np.float64)
            summary[split][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    summary["best_epoch_mean"] = float(np.mean([r["best_epoch"] for r in runs]))
    if task == "multi_label":
        thresholds = np.array([r["threshold"] for r in runs], dtype=np.float64)
        summary["threshold_mean"] = float(thresholds.mean())
        summary["threshold_std"] = (
            float(thresholds.std(ddof=1)) if len(thresholds) > 1 else 0.0
        )
    summary["train_test_accuracy_gap_mean"] = float(
        np.mean([r["train_test_accuracy_gap"] for r in runs])
    )
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--act_dir", required=True)
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--label", choices=["profession", "default"], default="profession")
    ap.add_argument(
        "--task", choices=["single_label", "multi_label"], default="single_label",
        help="classification target type (multi_label expects an N x C label matrix)",
    )
    ap.add_argument(
        "--checkpoint", action="append", default=[], metavar="NAME=PATH",
        help="GeoAE space to add; may be supplied multiple times",
    )
    ap.add_argument("--seeds", default="42", help="comma-separated probe seeds")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--encode_batch_size", type=int, default=4096)
    ap.add_argument(
        "--hidden_dim", type=int, default=0,
        help="MLP hidden width; 0 retains the original linear probe",
    )
    ap.add_argument(
        "--dropout", type=float, default=0.0,
        help="dropout after the hidden GELU (only used when --hidden_dim > 0)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.hidden_dim < 0:
        ap.error("--hidden_dim must be non-negative")
    if not 0.0 <= args.dropout < 1.0:
        ap.error("--dropout must be in [0, 1)")

    checkpoints = {}
    for spec in args.checkpoint:
        if "=" not in spec:
            ap.error(f"--checkpoint must be NAME=PATH, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not name or not path.exists():
            ap.error(f"invalid checkpoint specification: {spec!r}")
        checkpoints[name] = path

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_val, y_val, X_test, y_test = _load_data(
        Path(args.act_dir), args.layer, args.label
    )
    if args.task == "single_label":
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
        if y_val is not None:
            y_val = y_val.astype(np.int64)
        n_classes = int(len(np.unique(y_train)))
    else:
        if y_train.ndim != 2 or y_test.ndim != 2:
            ap.error("--task multi_label requires N x C label matrices")
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        if y_val is not None:
            y_val = y_val.astype(np.float32)
        n_classes = int(y_train.shape[1])
    print(
        f"[probe] train={X_train.shape} "
        f"validation={None if X_val is None else X_val.shape} test={X_test.shape} "
        f"classes={n_classes} task={args.task} device={device}"
    )

    output = {
        "dataset": str(args.act_dir),
        "label": args.label,
        "layer": args.layer,
        "n_train_total": len(X_train),
        "n_validation": 0 if X_val is None else len(X_val),
        "n_test": len(X_test),
        "n_classes": n_classes,
        "probe": {
            "seeds": seeds,
            "val_frac": args.val_frac,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "type": "mlp" if args.hidden_dim > 0 else "linear",
            "hidden_dim": args.hidden_dim,
            "activation": "gelu" if args.hidden_dim > 0 else None,
            "dropout": args.dropout if args.hidden_dim > 0 else 0.0,
            "feature_standardization": "probe-train statistics per space",
            "selection": "minimum validation cross-entropy",
            "task": args.task,
            "validation_source": "official" if X_val is not None else "train split",
            "threshold_selection": (
                "global threshold maximizing validation micro-F1 on 0.05:0.01:0.95"
                if args.task == "multi_label" else None
            ),
            "output_bias_initialization": (
                "logit of per-label probe-training prevalence"
                if args.task == "multi_label" else "framework default"
            ),
        },
        "spaces": {},
    }

    spaces = [("raw_base", None), *checkpoints.items()]
    for name, checkpoint in spaces:
        print(f"\n[probe] Representation: {name}", flush=True)
        if checkpoint is None:
            Z_train, Z_val, Z_test = X_train, X_val, X_test
        else:
            source_arrays = [X_train, *([] if X_val is None else [X_val]), X_test]
            encoded = _encode_geoae(
                source_arrays, checkpoint, device, args.encode_batch_size
            )
            if X_val is None:
                Z_train, Z_test = encoded
                Z_val = None
            else:
                Z_train, Z_val, Z_test = encoded

        runs = []
        for seed in seeds:
            print(f"  [probe] seed={seed}", flush=True)
            runs.append(_fit_one(
                Z_train, y_train, Z_test, y_test,
                X_val=Z_val, y_val=y_val, task=args.task,
                seed=seed, val_frac=args.val_frac, lr=args.lr,
                weight_decay=args.weight_decay, epochs=args.epochs,
                patience=args.patience, batch_size=args.batch_size,
                hidden_dim=args.hidden_dim, dropout=args.dropout, device=device,
            ))
        if args.hidden_dim > 0:
            n_probe_parameters = (
                Z_train.shape[1] * args.hidden_dim + args.hidden_dim
                + args.hidden_dim * output["n_classes"] + output["n_classes"]
            )
        else:
            n_probe_parameters = Z_train.shape[1] * output["n_classes"] + output["n_classes"]
        output["spaces"][name] = {
            "n_dims": int(Z_train.shape[1]),
            "n_probe_parameters": int(n_probe_parameters),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "runs": runs,
            "summary": _aggregate(runs, args.task),
        }
        if checkpoint is not None:
            del Z_train, Z_test
            if Z_val is not None:
                del Z_val
            gc.collect()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)

    print("\n[probe] Final test results")
    for name, space in output["spaces"].items():
        test = space["summary"]["test"]
        train = space["summary"]["train"]
        if args.task == "multi_label":
            print(
                f"  {name:<20} train_micro_f1={train['micro_f1']['mean']:.4f}  "
                f"test_micro_f1={test['micro_f1']['mean']:.4f}  "
                f"test_macro_f1={test['macro_f1']['mean']:.4f}  "
                f"test_micro_AP={test['micro_average_precision']['mean']:.4f}"
            )
        else:
            print(
                f"  {name:<20} train_acc={train['accuracy']['mean']:.4f}  "
                f"test_acc={test['accuracy']['mean']:.4f}  "
                f"test_bal_acc={test['balanced_accuracy']['mean']:.4f}  "
                f"test_macro_f1={test['macro_f1']['mean']:.4f}"
            )
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
