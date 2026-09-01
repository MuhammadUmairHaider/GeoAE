"""
Gradient audit: what each loss term actually contributes to the AE update.

The e2e objective is a weighted sum with KL pinned at weight 1,

    L = KL + λ_cluster·L_cluster + λ_sep·L_sep + λ_mse·MSE
           + λ_var·L_var + λ_cov·L_cov + λ_unif·L_unif

and the λ's have historically been set by eye. Loss *values* are a poor guide:
the terms have unrelated units and curvatures, so a term worth 20% of the loss
may drive 2% or 80% of the parameter update. What matters is the gradient each
term induces on the AE parameters. This tool measures that directly.

For a fixed checkpoint and a fixed batch it reports, per term:

  ||grad||        the UNWEIGHTED gradient norm ||∇_θ L_i||. This is the quantity
                  to calibrate against, because it is independent of the λ you
                  happen to be running.
  weighted        ||λ_i ∇_θ L_i|| at the checkpoint's current λ.
  share           that term's fraction of Σ_j ||λ_j ∇_θ L_j||.
  cos vs KL       cosine between ∇_θ L_i and ∇_θ KL.

The cosine is the part naive norm-balancing misses, and it is usually the
decision-relevant number:

    cos < 0   the term FIGHTS the KL. Every unit of λ buys its objective
              strictly at KL's expense; the λ choice is a real trade-off.
    cos ≈ 0   the term is orthogonal — it acts in directions KL does not
              constrain, so it is close to free.
    cos ≫ 0   the term is largely REDUNDANT with KL: it pushes where KL already
              pushes. Raising λ then improves both metrics at once (this is why
              the layer-47 λ_mse sweep improved val_mse and val_kl together),
              and it is also the signature that the KL term may be dispensable.

`--target` inverts the measurement: for each ratio r it prints the λ that would
give the term a gradient norm r× the KL's, i.e. λ* = r·||∇KL|| / ||∇L_i||.

Usage:
    python -m geoae.grad_audit \
      --checkpoint e2e/checkpoints/.../best_val.pt --n_docs 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.e2e.losses import kl_loss
from geoae.losses import (
    cluster_loss, sep_loss, recon_loss,
    variance_loss, covariance_loss, uniformity_loss,
)

# Terms in report order. KL is first and is the reference for shares/cosines.
TERM_ORDER = ["kl", "mse", "cluster", "sep", "var", "cov", "unif"]
LAMBDA_KEY = {
    "kl": None,               # pinned at 1.0 by construction
    "mse": "lambda_mse",
    "cluster": "lambda_cluster",
    "sep": "lambda_sep",
    "var": "lambda_var",
    "cov": "lambda_cov",
    "unif": "lambda_unif",
}


def flat_grad(params) -> torch.Tensor:
    """Concatenate .grad over params, treating None as zeros."""
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for p in params
    ])


def compute_terms(ae, out, x, teacher, student, metric) -> dict[str, torch.Tensor]:
    """Every loss term for one document, unweighted."""
    terms = {
        "kl": kl_loss(teacher, student),
        "mse": recon_loss(x, out.x_hat)[0],
        "sep": sep_loss(out.z, out.Q, metric=metric),
        "var": variance_loss(out.z),
        "cov": covariance_loss(out.z),
    }
    if bool(ae.centroids_initialized.item()):
        terms["cluster"] = cluster_loss(
            out.z, ae.centroids.detach(), out.Q, metric=metric
        )
    if metric == "cosine":
        terms["unif"] = uniformity_loss(out.z)
    return terms


@torch.enable_grad()
def audit(ckpt_path: Path, n_docs: int, max_doc_tokens: int, skip_leading: int,
          min_doc_len: int, seed: int, device) -> dict:
    from transformers import AutoTokenizer
    from geoae.e2e.train import load_frozen_lm
    from geoae.e2e.train_stream import StreamLogits
    from geoae.extract import DEFAULT_SOURCES, open_sources, stream_docs

    ae, mean_t, std_t, ckpt = load_ae_checkpoint(ckpt_path, device)
    ae.train()
    cfg = ckpt["config"]
    layer = cfg["data"]["target_layer"]
    model_name = cfg["extraction"]["model_name"]
    metric = cfg["model"].get("metric", "euclidean")
    lam = cfg["loss"]

    def _fmt(key, spec):
        """Checkpoints differ by trainer: geoae.train saves val_mse but no
        val_kl, the e2e trainers save both. Missing keys print as '-'."""
        v = ckpt.get(key)
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    print(f"[audit] {ckpt_path}")
    print(f"[audit] epoch={ckpt.get('epoch')} val_kl={_fmt('val_kl', '.5f')} "
          f"val_mse={_fmt('val_mse', '.5f')} tau={_fmt('tau', '.3f')}")
    print(f"[audit] {model_name} layer {layer} | K={cfg['model']['n_clusters']} "
          f"metric={metric} | centroids_init={bool(ae.centroids_initialized.item())}")

    lm = load_frozen_lm(model_name, device)
    sl = StreamLogits(lm, layer)
    print(f"[audit] path={'head-only' if sl.is_last else 'full-forward splice'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    srcs = open_sources([dict(s) for s in DEFAULT_SOURCES])
    docs = []
    for ids, _dom in stream_docs(srcs, n_docs * max_doc_tokens * 4, tokenizer,
                                 min_doc_len, max_doc_tokens):
        docs.append(ids)
        if len(docs) >= n_docs:
            break
    print(f"[audit] {len(docs)} docs, <={max_doc_tokens} tokens each")

    params = [p for p in ae.parameters() if p.requires_grad]
    n_param = sum(p.numel() for p in params)
    accum: dict[str, torch.Tensor] = {}
    values: dict[str, float] = {}
    n_used = 0

    for i, ids in enumerate(docs):
        ids = ids.to(device)
        T = ids.shape[1]
        if T <= skip_leading:
            continue
        residual, teacher = sl.teacher_and_residual(ids)
        x = (residual - mean_t) / std_t
        out = ae(x)
        recon_raw = out.x_hat * std_t + mean_t
        student = sl.student(ids, recon_raw)
        v = slice(skip_leading, T)

        class _Sliced:                      # out.* restricted to scored positions
            pass
        o = _Sliced()
        o.x_hat, o.z, o.Q = out.x_hat[v], out.z[v], out.Q[v]

        terms = compute_terms(ae, o, x[v], teacher[v], student[v], metric)
        names = [t for t in TERM_ORDER if t in terms]
        for j, name in enumerate(names):
            ae.zero_grad(set_to_none=True)
            terms[name].backward(retain_graph=(j < len(names) - 1))
            g = flat_grad(params).detach()
            accum[name] = g.clone() if name not in accum else accum[name] + g
            values[name] = values.get(name, 0.0) + float(terms[name].detach())
        n_used += 1
        if (i + 1) % 8 == 0:
            print(f"[audit]   {i + 1}/{len(docs)} docs")

    ae.zero_grad(set_to_none=True)
    if n_used == 0:
        raise RuntimeError("No usable documents — all shorter than skip_leading.")

    grads = {k: v / n_used for k, v in accum.items()}
    values = {k: v / n_used for k, v in values.items()}
    g_kl = grads["kl"]
    kl_norm = float(g_kl.norm())
    mse_is_primary = "val_kl" not in ckpt      # written by geoae.train, not e2e
    if mse_is_primary:
        print("[audit] base-pipeline checkpoint (no val_kl): reporting MSE as the "
              "weight-1 faithfulness term; KL was never in this model's objective.")

    rows = []
    for name in TERM_ORDER:
        if name not in grads:
            continue
        g = grads[name]
        raw = float(g.norm())
        # Which term carries weight 1 depends on the trainer that wrote the
        # checkpoint: the e2e loss pins KL at 1 and weights MSE by lambda_mse,
        # while geoae.losses.total_loss pins RECON at 1 and has no lambda_mse at
        # all. Base-pipeline checkpoints are identified by the absence of val_kl
        # (geoae.train never computes one), else MSE is reported at lambda 0.
        if name == "kl":
            lam_i = 0.0 if mse_is_primary else 1.0
        elif name == "mse" and mse_is_primary:
            lam_i = 1.0
        else:
            lam_i = float(lam.get(LAMBDA_KEY[name], 0.0) or 0.0)
        cos = float(torch.nn.functional.cosine_similarity(
            g.unsqueeze(0), g_kl.unsqueeze(0)).item())
        rows.append({
            "term": name, "value": values[name], "lambda": lam_i,
            "grad_norm": raw, "weighted": lam_i * raw,
            "cos_vs_kl": cos,
            "ratio_to_kl": raw / kl_norm if kl_norm > 0 else float("nan"),
        })
    total_w = sum(r["weighted"] for r in rows) or 1.0
    for r in rows:
        r["share"] = r["weighted"] / total_w

    return {
        "checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch"),
        "val_kl": ckpt.get("val_kl"), "val_mse": ckpt.get("val_mse"),
        "model": model_name, "layer": layer, "metric": metric,
        "n_docs": n_used, "n_params": n_param, "kl_grad_norm": kl_norm,
        "terms": rows,
    }


def report(res: dict, targets: list[float]) -> None:
    print(f"\n[audit] gradient norms over {res['n_params']:,} AE params, "
          f"averaged over {res['n_docs']} docs\n")
    print(f"{'term':<9} {'value':>10} {'lambda':>8} {'||grad||':>11} "
          f"{'weighted':>11} {'share':>7} {'cos vs KL':>10}")
    print("-" * 72)
    for r in res["terms"]:
        print(f"{r['term']:<9} {r['value']:>10.4f} {r['lambda']:>8.4g} "
              f"{r['grad_norm']:>11.4g} {r['weighted']:>11.4g} "
              f"{100 * r['share']:>6.1f}% {r['cos_vs_kl']:>10.3f}")

    print(f"\n[audit] lambda that would give each term a gradient norm r x the KL's "
          f"(||grad KL|| = {res['kl_grad_norm']:.4g}):\n")
    head = "  ".join(f"r={t:g}".rjust(10) for t in targets)
    print(f"{'term':<9} {'current':>9}  {head}")
    print("-" * (20 + 12 * len(targets)))
    for r in res["terms"]:
        if r["term"] == "kl":
            continue
        cells = "  ".join(
            f"{t * res['kl_grad_norm'] / r['grad_norm']:10.4g}" if r["grad_norm"] > 0
            else f"{'--':>10}" for t in targets
        )
        print(f"{r['term']:<9} {r['lambda']:>9.4g}  {cells}")

    print("\n[audit] reading the cosines:")
    for r in res["terms"]:
        if r["term"] == "kl":
            continue
        c = r["cos_vs_kl"]
        if c < -0.05:
            verdict = "FIGHTS KL — raising lambda costs faithfulness"
        elif c < 0.05:
            verdict = "orthogonal to KL — nearly free"
        elif c < 0.3:
            verdict = "mildly aligned with KL"
        else:
            verdict = "STRONGLY aligned — largely redundant with KL"
        print(f"  {r['term']:<9} cos={c:+.3f}  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n_docs", type=int, default=32,
                    help="Documents to average over. More = less noise, linear cost.")
    ap.add_argument("--max_doc_tokens", type=int, default=128)
    ap.add_argument("--skip_leading", type=int, default=4)
    ap.add_argument("--min_doc_len", type=int, default=10)
    ap.add_argument("--target", type=float, nargs="+", default=[0.05, 0.25, 0.5, 1.0],
                    help="Gradient-norm ratios to KL to solve lambda for")
    ap.add_argument("--out", default=None, help="Write the full result as JSON")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from geoae.seeding import seed_everything
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = audit(Path(args.checkpoint), args.n_docs, args.max_doc_tokens,
                args.skip_leading, args.min_doc_len, args.seed, device)
    report(res, args.target)

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"\n[audit] wrote {args.out}")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)   # HF streaming threads don't join cleanly; all output is written above
