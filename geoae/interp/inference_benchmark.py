"""
Splice any trained AE into Llama during inference and measure reconstruction
fidelity on a standard benchmark dataset.

Three forward-pass modes compared side by side:
  original     — no intervention, baseline perplexity
  ae_spliced   — AE reconstruction replaces the residual stream at target layer
  mean_ablated — per-position mean replaces the residual stream (information floor)

Metrics reported:
  perplexity          exp(mean cross-entropy)
  bits_per_token      mean cross-entropy in bits (= CE / ln 2)
  top1_accuracy       fraction of tokens where argmax(logits) == true next token
  loss_recovered      (CE_ablated - CE_spliced) / (CE_ablated - CE_orig)
  kl_from_original    mean KL(original || spliced) per token

Usage:
    python -m geoae.interp.inference_benchmark \\
        --checkpoint checkpoints/layer27_k128_d2048_linear/best_val.pt \\
        --layer 27

    # Compare multiple checkpoints at once:
    python -m geoae.interp.inference_benchmark \\
        --checkpoint checkpoints/layer27_k128_d2048_linear/best_val.pt \\
                     checkpoints/layer27_k128_d2048_relu/best_val.pt \\
                     checkpoints/layer27_k128_d3072_linear/best_val.pt \\
        --layer 27 --n_tokens 20000

    # No checkpoint — just run original vs mean-ablated as a sanity check:
    python -m geoae.interp.inference_benchmark --layer 27 --no_ae
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer


from geoae.checkpoint import ae_label, load_ae_checkpoint, load_lm
from geoae.hooks import SplicingHook


# ---------------------------------------------------------------------------
# AE loader
# ---------------------------------------------------------------------------

def load_ae(ckpt_path: Path, device: torch.device):
    """Load GeoAE from checkpoint. Returns (ae, norm_mean, norm_std, label)."""
    ae, norm_mean, norm_std, ckpt = load_ae_checkpoint(ckpt_path, device)
    return ae, norm_mean, norm_std, ae_label(ckpt)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_tokens(dataset: str, tokenizer, n_tokens: int) -> Tensor:
    """Returns a flat (1, N) token tensor for rolling-window evaluation."""
    from datasets import load_dataset

    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(ds["text"])
    elif dataset == "ptb":
        ds = load_dataset("ptb_text_only", split="test")
        text = "\n".join(ds["sentence"])
    elif dataset == "c4":
        ds = load_dataset("allenai/c4", name="en", split="validation", streaming=True)
        chunks = []
        for ex in ds:
            chunks.append(ex["text"])
            if sum(len(c) for c in chunks) > n_tokens * 6:
                break
        text = "\n".join(chunks)
    else:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose: wikitext | ptb | c4")

    ids = tokenizer.encode(text)
    ids = ids[:n_tokens + 512]   # grab a little extra for the rolling window
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)   # (1, N)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mode(
    lm,
    hook: SplicingHook,
    token_ids: Tensor,
    device: torch.device,
    stride: int,
    seq_len: int,
    mode: str,                      # "original" | "ae_spliced" | "mean_ablated"
    ae=None,
    norm_mean: Tensor | None = None,
    norm_std:  Tensor | None = None,
    captured_means: dict | None = None,  # populated on first "original" pass
) -> dict:
    """
    Sliding-window evaluation.  Returns dict of metrics.

    captured_means: dict mapping position → mean activation (for mean_ablated mode).
    For mean_ablated we compute the per-sequence mean on the fly.
    """
    N = token_ids.shape[1]
    total_ce = 0.0
    total_top1 = 0
    total_tokens = 0
    all_ce_tokens: list[float] = []

    ae_device = next(ae.parameters()).device if ae is not None else device

    for begin in range(0, N - 1, stride):
        end = min(begin + seq_len, N)
        ids = token_ids[:, begin:end].to(device)
        T = ids.shape[1]
        if T < 2:
            continue

        # ----- build the splice function for this window -----
        if mode == "original":
            splice_fn = None

        elif mode == "mean_ablated":
            captured = []
            hook.activate(lambda hs, c=captured: (c.append(hs.detach()) or hs))
            lm(input_ids=ids)
            hook.deactivate()
            act_mean = captured[0].mean(dim=1, keepdim=True).expand_as(captured[0])
            def splice_fn(hs, m=act_mean):
                return m.to(hs.device)

        elif mode == "ae_spliced":
            def splice_fn(hs):
                B, T_, D = hs.shape
                flat = hs.reshape(B * T_, D).float().to(ae_device)
                mean = norm_mean.to(ae_device)
                std  = norm_std.to(ae_device)
                normed = (flat - mean) / std
                recon = ae(normed).x_hat
                raw = recon * std + mean
                return raw.reshape(B, T_, D).to(hs.device).to(hs.dtype)

        # ----- forward pass -----
        if splice_fn is not None:
            hook.activate(splice_fn)

        out = lm(input_ids=ids)

        if splice_fn is not None:
            hook.deactivate()

        # ----- metrics -----
        shift_logits = out.logits[0, :-1].float()    # (T-1, V)
        shift_labels = ids[0, 1:]                     # (T-1,)

        # Only score the non-overlapping suffix to avoid double-counting
        score_start = min(stride, T - 1) if begin > 0 else 0
        sl = shift_logits[score_start:]
        lb = shift_labels[score_start:]
        if lb.numel() == 0:
            continue

        ce = F.cross_entropy(sl, lb, reduction="none")   # (n,)
        total_ce   += ce.sum().item()
        total_top1 += (sl.argmax(dim=-1) == lb).sum().item()
        total_tokens += lb.numel()
        all_ce_tokens.extend(ce.tolist())

    mean_ce = total_ce / max(total_tokens, 1)
    return {
        "perplexity":    math.exp(min(mean_ce, 20)),   # cap to avoid overflow display
        "bits_per_token": mean_ce / math.log(2),
        "top1_accuracy": total_top1 / max(total_tokens, 1),
        "mean_ce":       mean_ce,
        "n_tokens":      total_tokens,
        "all_ce":        all_ce_tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Splice AE into Llama and benchmark")
    parser.add_argument("--checkpoint", nargs="*", default=None,
                        help="Path(s) to AE checkpoint(s). Multiple → compare all.")
    parser.add_argument("--layer",      type=int, default=27)
    parser.add_argument("--dataset",    default="wikitext",
                        choices=["wikitext", "ptb", "c4"])
    parser.add_argument("--n_tokens",   type=int, default=10_000,
                        help="Approximate number of tokens to evaluate on")
    parser.add_argument("--seq_len",    type=int, default=512)
    parser.add_argument("--stride",     type=int, default=256,
                        help="Sliding-window stride (overlap = seq_len - stride)")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--no_ae",      action="store_true",
                        help="Only run original + mean_ablated (no AE needed)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load LM
    print(f"[bench] Loading {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    lm = load_lm(args.model_name, device_map="auto")

    hook = SplicingHook(lm, args.layer)

    # Load tokens
    print(f"[bench] Loading dataset: {args.dataset} (~{args.n_tokens:,} tokens) …")
    token_ids = get_tokens(args.dataset, tokenizer, args.n_tokens)
    print(f"[bench] Tokens available: {token_ids.shape[1]:,}")

    eval_kwargs = dict(
        lm=lm, hook=hook, token_ids=token_ids, device=device,
        stride=args.stride, seq_len=args.seq_len,
    )

    results = {}

    # --- Baseline: original ---
    print("\n[bench] Evaluating: original (no intervention) …")
    results["original"] = evaluate_mode(mode="original", **eval_kwargs)

    # --- Mean ablated ---
    print("[bench] Evaluating: mean_ablated …")
    results["mean_ablated"] = evaluate_mode(mode="mean_ablated", **eval_kwargs)

    # --- AE checkpoints ---
    checkpoints = args.checkpoint or []
    ae_results = {}
    for ckpt_path in checkpoints:
        print(f"\n[bench] Loading AE: {ckpt_path}")
        ae, norm_mean, norm_std, label = load_ae(Path(ckpt_path), device)
        print(f"[bench] Evaluating: {label} …")
        res = evaluate_mode(
            mode="ae_spliced", ae=ae,
            norm_mean=norm_mean, norm_std=norm_std,
            **eval_kwargs,
        )
        ae_results[label] = (ckpt_path, res)

    # --- Print table ---
    orig_ce    = results["original"]["mean_ce"]
    ablated_ce = results["mean_ablated"]["mean_ce"]

    print("\n" + "="*90)
    print(f"  Dataset: {args.dataset}  |  Layer: {args.layer}  |  Tokens: {results['original']['n_tokens']:,}")
    print("="*90)
    print(f"  {'Mode':<45} {'PPL':>8} {'BPT':>8} {'Top-1%':>8} {'LossRec':>9}")
    print("-"*90)

    def loss_recovered(ce):
        return (ablated_ce - ce) / max(ablated_ce - orig_ce, 1e-8)

    def print_row(label, res):
        lr = loss_recovered(res["mean_ce"])
        print(f"  {label:<45} {res['perplexity']:>8.2f} {res['bits_per_token']:>8.4f} "
              f"{res['top1_accuracy']*100:>7.2f}% {lr:>9.4f}")

    print_row("original (no hook)", results["original"])
    print_row("mean_ablated", results["mean_ablated"])
    for label, (path, res) in ae_results.items():
        print_row(label, res)

    print("="*90)
    print("  loss_recovered = (CE_ablated - CE_mode) / (CE_ablated - CE_orig)")
    print("  1.0 = perfect reconstruction, 0.0 = no better than mean ablation")

    # Sanity: mean_ablated loss_recovered should be ~0
    lr_ablated = loss_recovered(ablated_ce)
    if abs(lr_ablated) > 0.05:
        print(f"\n  WARNING: mean_ablated loss_recovered = {lr_ablated:.4f} (expected ~0)")


if __name__ == "__main__":
    main()
