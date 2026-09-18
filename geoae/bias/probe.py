"""BiasBios fixed-probe ablation with validation-selected coordinate budgets.

This measures accessibility to frozen linear probes, not LM behavior or gender
information erasure. Both spaces use the same split and train-only scaling;
zeroing standardized coordinates replaces them with their training means.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from geoae.bias.extract import PROFESSIONS
from geoae.checkpoint import load_ae_checkpoint
from geoae.interp.neuronlens import dprime_saliency
from geoae.interp.probe_perturbation import LinearProbe
from geoae.seeding import seed_everything


def load_dual_labels(act_dir):
    return tuple(np.load(Path(act_dir) / f"labels_{field}_{split}.npy").astype(np.int64)
                 for field in ["profession", "gender"] for split in ["train", "test"])


def deduplicate_indices(train_hashes, test_hashes):
    """Keep first train occurrence and unique test texts absent from training."""
    _, train_idx = np.unique(train_hashes, return_index=True)
    _, test_idx = np.unique(test_hashes, return_index=True)
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)
    test_idx = test_idx[~np.isin(test_hashes[test_idx], train_hashes[train_idx])]
    return train_idx, test_idx


def shared_split(profession, gender, seed, val_frac=.1):
    n_val = int(np.ceil(len(profession) * val_frac))
    for name, strata in [("profession_gender", 2 * profession + gender),
                         ("profession", profession), ("gender", gender)]:
        counts = np.unique(strata, return_counts=True)[1]
        if counts.min() >= 2 and len(counts) <= min(n_val, len(profession) - n_val):
            fit, val = train_test_split(np.arange(len(profession)), test_size=val_frac,
                                        random_state=seed, stratify=strata)
            return fit, val, name
    raise ValueError("too few examples for a stratified probe split")


def standardize(arrays, fit_idx):
    fit = arrays[0][fit_idx]
    mean = fit.mean(0, dtype=np.float64).astype(np.float32)
    std = fit.std(0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return [(x - mean) / std for x in arrays]


def rank_dims_global(weights):
    """Remove softmax's arbitrary common weight row before ranking."""
    contrasts = weights - weights.mean(0, keepdims=True)
    return np.argsort(-np.abs(contrasts).max(0), kind="stable")


@torch.inference_mode()
def classification_metrics(logits, labels, n_classes):
    pred = logits.argmax(1)
    counts = torch.bincount(labels, minlength=n_classes)
    hits = torch.bincount(labels[pred == labels], minlength=n_classes)
    present = counts > 0
    return {"accuracy": float((pred == labels).float().mean()),
            "balanced_accuracy": float((hits[present] / counts[present]).mean()),
            "cross_entropy": float(torch.nn.functional.cross_entropy(logits, labels))}


def train_probe(X_train, y_train, X_val, y_val, n_classes, *, seed, lr,
                n_epochs, patience, batch_size, lambda_l1, lambda_l2, device, bias=True):
    seed_everything(seed)
    probe = LinearProbe(X_train.shape[1], n_classes, bias=bias).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=lambda_l2)
    x = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.long, device=device)
    best, best_epoch, state, wait, history = float("inf"), 0, None, 0, []
    generator = torch.Generator(device=device).manual_seed(seed)
    for epoch in range(1, n_epochs + 1):
        probe.train()
        order = torch.randperm(len(x), generator=generator, device=device)
        for start in range(0, len(x), batch_size):
            take = order[start:start + batch_size]
            loss = torch.nn.functional.cross_entropy(probe(x[take]), y[take])
            loss = loss + lambda_l1 * probe.linear.weight.abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        probe.eval()
        with torch.inference_mode():
            validation = classification_metrics(probe(xv), yv, n_classes)
        history.append({"epoch": epoch, "validation": validation})
        if validation["cross_entropy"] < best - 1e-5:
            best, best_epoch, wait = validation["cross_entropy"], epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch={epoch} val CE={validation['cross_entropy']:.4f} "
                  f"acc={validation['accuracy']:.4f}", flush=True)
        if wait >= patience: break
    if state is None: raise RuntimeError("probe produced no finite validation checkpoint")
    probe.load_state_dict(state)
    with torch.inference_mode():
        train = classification_metrics(probe(x), y, n_classes)
        validation = classification_metrics(probe(xv), yv, n_classes)
    return probe, {"best_epoch": best_epoch, "stopped_epoch": epoch,
                   "epoch_budget_reached": epoch == n_epochs,
                   "train": train, "validation": validation, "history": history}


@torch.inference_mode()
def cross_perturbation_sweep(X_test, y_task, probe_task, y_spurious, probe_spurious,
                             dim_ranking, n_steps=200, device="cpu"):
    """Subtract ablated linear contributions incrementally; verify against direct edits in tests."""
    n_dims = X_test.shape[1]
    if not np.array_equal(np.sort(dim_ranking), np.arange(n_dims)):
        raise ValueError("dimension ranking must be a permutation")
    if set(np.unique(y_spurious)) != {0, 1}:
        raise ValueError("balanced gender evaluation requires both labels")
    ks = np.unique(np.linspace(0, n_dims, n_steps + 1, dtype=int))
    x = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_task, dtype=torch.long, device=device)
    yg = torch.as_tensor(y_spurious, dtype=torch.long, device=device)
    probes = [probe_task, probe_spurious]
    widths = [p.linear.out_features for p in probes]
    W = torch.cat([p.linear.weight for p in probes])
    b = torch.cat([p.linear.bias if p.linear.bias is not None else
                   torch.zeros(p.linear.out_features, device=device) for p in probes])
    logits = x @ W.T + b
    reference = logits[:, :widths[0]].clone()
    ref_logp, ref_pred = reference.log_softmax(1), reference.argmax(1)
    curves = {"ks": ks.tolist(), "n_dims": n_dims,
              "task_accs": [], "task_balanced_accs": [], "spur_accs": [],
              "spur_balanced_accs": [], "task_kl": [], "task_top1_change": []}
    previous = 0
    for k in ks:
        if k == n_dims:
            logits = b.expand(len(x), -1)  # exact endpoint, avoiding accumulated roundoff
        elif k > previous:
            dims = torch.as_tensor(dim_ranking[previous:k], device=device)
            logits = logits - x[:, dims] @ W[:, dims].T
        task, gender = logits.split(widths, dim=1)
        tm = classification_metrics(task, yt, widths[0])
        gm = classification_metrics(gender, yg, widths[1])
        curves["task_accs"].append(tm["accuracy"])
        curves["task_balanced_accs"].append(tm["balanced_accuracy"])
        curves["spur_accs"].append(gm["accuracy"])
        curves["spur_balanced_accs"].append(gm["balanced_accuracy"])
        curves["task_kl"].append(float((ref_logp.exp() * (ref_logp - task.log_softmax(1))).sum(1).mean()))
        curves["task_top1_change"].append(float((task.argmax(1) != ref_pred).float().mean()))
        previous = k
    return curves


def select_operating_points(validation, test, chance_tolerance=.01):
    """Choose k from validation only; unreachable thresholds remain null."""
    if validation["ks"] != test["ks"]: raise ValueError("validation/test grids differ")
    gender = np.asarray(validation["spur_balanced_accs"])
    half = .5 + max(0., gender[0] - .5) / 2
    conditions = {"half_excess": gender <= half,
                  "near_chance": np.abs(gender - .5) <= chance_tolerance}
    result = {}
    for name, condition in conditions.items():
        found = np.flatnonzero(condition)
        if not len(found):
            result[name] = {"reached_on_validation": False, "k": None, "test": None}
            continue
        index = int(found[0])
        result[name] = {
            "reached_on_validation": True, "k": validation["ks"][index],
            "fraction": validation["ks"][index] / validation["n_dims"],
            "validation_gender_balanced_accuracy": float(gender[index]),
            "test": {key: test[key][index] for key in test if key not in {"ks", "n_dims"}},
            "test_profession_accuracy_delta": test["task_accs"][index] - test["task_accs"][0],
            "test_profession_balanced_accuracy_delta": test["task_balanced_accs"][index] - test["task_balanced_accs"][0],
        }
    return result


@torch.inference_mode()
def per_profession_delta(X_test, y_prof, gender, probe_prof, dim_ranking, k, device="cpu"):
    x = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    before = probe_prof(x).argmax(1).cpu().numpy()
    changed = x.clone()
    changed[:, dim_ranking[:k]] = 0
    after = probe_prof(changed).argmax(1).cpu().numpy()
    result = {}
    for c in np.unique(y_prof):
        mask = y_prof == c
        row = {"n": int(mask.sum()), "accuracy_before": float((before[mask] == c).mean()),
               "accuracy_after": float((after[mask] == c).mean())}
        for g in [0, 1]:
            group = mask & (gender == g)
            row[f"gender_{g}"] = {"n": int(group.sum()),
                "recall_before": float((before[group] == c).mean()) if group.any() else None,
                "recall_after": float((after[group] == c).mean()) if group.any() else None}
        if row["gender_0"]["n"] and row["gender_1"]["n"]:
            for when in ["before", "after"]:
                row[f"gender_recall_gap_{when}"] = row["gender_1"][f"recall_{when}"] - row["gender_0"][f"recall_{when}"]
        result[PROFESSIONS[c]] = row
    return result


def run_bias_space(space, X_train, X_test, prof_train, prof_test, gender_train,
                   gender_test, fit_idx, val_idx, args, device):
    X_train, X_test = standardize([X_train, X_test], fit_idx)
    print(f"[bias] {space}: {X_train.shape[1]} dims", flush=True)
    probes, fitting = {}, {}
    for offset, (name, labels, n_classes) in enumerate([
            ("gender", gender_train, 2), ("profession", prof_train, len(PROFESSIONS))]):
        print(f"[bias] fitting {name}", flush=True)
        prefix = "gender" if name == "gender" else "prof"
        probes[name], fitting[name] = train_probe(
            X_train[fit_idx], labels[fit_idx], X_train[val_idx], labels[val_idx], n_classes,
            seed=args.seed + offset, lr=args.probe_lr, n_epochs=args.probe_epochs,
            patience=args.patience, batch_size=args.batch_size,
            lambda_l1=getattr(args, f"{prefix}_lambda_l1"), lambda_l2=getattr(args, f"{prefix}_lambda_l2"),
            device=device, bias=not args.no_probe_bias)
    rankings = {
        "gender_probe": rank_dims_global(probes["gender"].linear.weight.detach().cpu().numpy()),
        "gender_dprime": np.argsort(-dprime_saliency(X_train[fit_idx], gender_train[fit_idx] == 1), kind="stable"),
        "profession_probe": rank_dims_global(probes["profession"].linear.weight.detach().cpu().numpy()),
    }
    rng = np.random.RandomState(args.seed)
    for trial in range(args.n_random_trials): rankings[f"random_{trial}"] = rng.permutation(X_train.shape[1])
    result = {"n_dims": X_train.shape[1], "probe_fitting": fitting, "ablations": {}}
    for name, ranking in rankings.items():
        print(f"[bias] {space} sweep {name}", flush=True)
        curves = {}
        for split, X, p, g in [("validation", X_train[val_idx], prof_train[val_idx], gender_train[val_idx]),
                                ("test", X_test, prof_test, gender_test)]:
            curves[split] = cross_perturbation_sweep(X, p, probes["profession"], g,
                probes["gender"], ranking, args.n_steps, device)
        points = select_operating_points(curves["validation"], curves["test"], args.chance_tolerance)
        result["ablations"][name] = {**curves, "operating_points": points, "dim_ranking": ranking.tolist()}
        if name.startswith("gender_"):
            result["ablations"][name]["per_profession"] = {
                point: per_profession_delta(X_test, prof_test, gender_test, probes["profession"],
                    ranking, data["k"], device) if data["k"] is not None else None
                for point, data in points.items()}
        half = points["half_excess"]
        print(f"[bias] {space}/{name}: val-selected half-excess k={half['k']} test={half['test']}", flush=True)
    baseline = result["ablations"]["gender_probe"]["test"]
    result["test_baseline"] = {key: value[0] for key, value in baseline.items() if key not in {"ks", "n_dims"}}
    return result


@torch.inference_mode()
def encode_geoae(arrays, path, device, metadata, layer, batch_size):
    ae, mean, std, ck = load_ae_checkpoint(path, device)
    cfg = ck["config"]
    if cfg["data"]["target_layer"] != layer or cfg["extraction"]["model_name"] != metadata["model"]:
        raise ValueError("AE checkpoint layer/model does not match activation metadata")
    if arrays[0].shape[1] != ae.decoder.out_features or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("invalid AE input dimension or normalization statistics")
    from geoae.interp._shared import ae_fingerprint
    info = {"path": str(path), "epoch": ck.get("epoch"), "ae_sha": ae_fingerprint(ae)}
    del ck
    encoded = []
    for X in arrays:
        chunks = []
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            chunks.append(ae.encoder((xb - mean) / std).cpu().numpy())
        z = np.concatenate(chunks)
        if not np.isfinite(z).all(): raise ValueError("non-finite encoded activations")
        encoded.append(z)
    return encoded, info


def plot_bias_curves(results, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for space, result in results["spaces"].items():
        for ranking, style in [("gender_probe", "-"), ("gender_dprime", "--")]:
            curve = result["ablations"][ranking]["test"]
            label = f"{space}: {ranking}"
            x = np.array(curve["ks"]) / result["n_dims"]
            axes[0].plot(x, curve["spur_balanced_accs"], style, label=label)
            axes[1].plot(x, curve["task_balanced_accs"], style, label=label)
            axes[2].plot(curve["spur_balanced_accs"], curve["task_balanced_accs"], style, label=label)
    axes[0].axhline(.5, color="gray", linestyle=":")
    for ax in axes[:2]: ax.set_xlabel("Fraction of coordinates replaced with training mean")
    axes[0].set_ylabel("Gender balanced accuracy")
    axes[1].set_ylabel("Profession balanced accuracy")
    axes[2].set_xlabel("Gender balanced accuracy")
    axes[2].set_ylabel("Profession balanced accuracy")
    for ax in axes: ax.grid(alpha=.2)
    axes[2].legend(fontsize=8)
    fig.suptitle("BiasBios fixed-probe ablation (not LM steering or proof of information erasure)")
    fig.tight_layout()
    fig.savefig(plot_dir / "fixed_probe_ablation.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--act_dir", required=True)
    ap.add_argument("--source_hashes", required=True, help="NPZ train/test text SHA256 arrays audited against source labels")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--ae_checkpoint")
    ap.add_argument("--spaces", nargs="+", choices=["raw", "geoae"], default=["raw", "geoae"])
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--n_random_trials", type=int, default=5)
    ap.add_argument("--lambda_l1", type=float, default=0., help="Mean absolute weight penalty")
    ap.add_argument("--lambda_l2", type=float, default=1e-4, help="AdamW weight decay")
    for target in ["gender", "prof"]:
        for penalty in ["l1", "l2"]: ap.add_argument(f"--{target}_lambda_{penalty}", type=float)
    ap.add_argument("--probe_lr", type=float, default=3e-3)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--encode_batch_size", type=int, default=2048)
    ap.add_argument("--max_train", type=int)
    ap.add_argument("--max_test", type=int)
    ap.add_argument("--val_frac", type=float, default=.1)
    ap.add_argument("--chance_tolerance", type=float, default=.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_probe_bias", action="store_true")
    ap.add_argument("--zscore_latents", action="store_true", help="Compatibility flag; both spaces always standardized")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot_dir")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.max_train, args.max_test, args.n_steps = 1000, 500, 20
        args.n_random_trials, args.probe_epochs = 1, 20
    if "geoae" in args.spaces and not args.ae_checkpoint: ap.error("geoae requires --ae_checkpoint")
    if not 0 < args.val_frac < 1 or not 0 <= args.chance_tolerance < .5: ap.error("invalid fraction/tolerance")
    for key in ["n_steps", "n_random_trials", "probe_epochs", "patience", "batch_size", "encode_batch_size"]:
        if getattr(args, key) < 1: ap.error(f"{key} must be positive")
    for target in ["gender", "prof"]:
        for penalty in ["l1", "l2"]:
            key = f"{target}_lambda_{penalty}"
            if getattr(args, key) is None: setattr(args, key, getattr(args, f"lambda_{penalty}"))
            if getattr(args, key) < 0: ap.error("regularization must be nonnegative")
    out = Path(args.out)
    if out.exists(): ap.error(f"output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    p = Path(args.act_dir)
    metadata = json.loads((p / "meta.json").read_text())
    if metadata["layer"] != args.layer: ap.error("activation metadata layer mismatch")
    arrays = [np.load(p / name).astype(np.float32)
              for name in [f"layer_{args.layer}.npy", f"layer_{args.layer}_test.npy"]]
    prof_train, prof_test, gender_train, gender_test = load_dual_labels(p)
    hashes = np.load(args.source_hashes)
    for split, X, prof, gender in [("train", arrays[0], prof_train, gender_train),
                                  ("test", arrays[1], prof_test, gender_test)]:
        if not len(X) == len(prof) == len(gender) == len(hashes[split]) == metadata[f"n_{split}"]:
            raise ValueError(f"{split} row alignment mismatch")
        if X.ndim != 2 or not np.isfinite(X).all(): raise ValueError("invalid/non-finite activation cache")
        if prof.min() < 0 or prof.max() >= len(PROFESSIONS) or set(np.unique(gender)) != {0, 1}:
            raise ValueError("invalid labels")
    keep_train, keep_test = deduplicate_indices(hashes["train"], hashes["test"])
    dedup = {"train_rows_removed": len(arrays[0]) - len(keep_train),
             "test_rows_removed": len(arrays[1]) - len(keep_test)}
    rng = np.random.RandomState(args.seed)
    if args.max_train is not None: keep_train = rng.permutation(keep_train)[:args.max_train]
    if args.max_test is not None: keep_test = rng.permutation(keep_test)[:args.max_test]
    raw_train, raw_test = arrays[0][keep_train], arrays[1][keep_test]
    del arrays
    prof_train, gender_train = prof_train[keep_train], gender_train[keep_train]
    prof_test, gender_test = prof_test[keep_test], gender_test[keep_test]
    fit_idx, val_idx, stratification = shared_split(prof_train, gender_train, args.seed, args.val_frac)
    split_hash = hashlib.sha256(keep_train[fit_idx].tobytes() + keep_train[val_idx].tobytes() + keep_test.tobytes()).hexdigest()
    result = {"schema_version": 2, "status": "running", "args": vars(args),
              "activation_metadata": metadata, "deduplication": dedup,
              "split": {"fit_source_rows": keep_train[fit_idx].tolist(),
                        "validation_source_rows": keep_train[val_idx].tolist(),
                        "test_source_rows": keep_test.tolist(), "sha256": split_hash, "stratification": stratification},
              "protocol": {"standardization": "per-space probe-training mean/std",
                           "ablation": "replace selected coordinates with probe-training mean",
                           "selection": "validation-only budgets; test curves descriptive",
                           "ranking": "softmax weight contrasts and train-only d-prime",
                           "interpretation": "fixed-probe accessibility; not LM behavior or information erasure"},
              "spaces": {}}
    def save():
        temp = out.with_suffix(out.suffix + ".tmp")
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temp.replace(out)
    save()
    print(f"[bias] fit={len(fit_idx)} val={len(val_idx)} test={len(raw_test)} dedup={dedup}", flush=True)
    for space in args.spaces:
        if space == "raw":
            X_train, X_test = raw_train, raw_test
        else:
            print("[bias] Encoding AE latents", flush=True)
            (X_train, X_test), result["checkpoint"] = encode_geoae(
                [raw_train, raw_test], args.ae_checkpoint, args.device, metadata, args.layer, args.encode_batch_size)
        result["spaces"][space] = run_bias_space(space, X_train, X_test, prof_train, prof_test,
            gender_train, gender_test, fit_idx, val_idx, args, args.device)
        del X_train, X_test
        gc.collect()
        save()
    result["status"] = "complete"
    save()
    if args.plot_dir: plot_bias_curves(result, args.plot_dir)
    print(f"[bias] Complete: {out}", flush=True)


if __name__ == "__main__":
    main()
