"""
Distinct-live feature inventory for GeoAE, built on Direct Logit Attribution.

Combines three signals to cut K nominal clusters down to the features actually
worth inspecting:
  1. DLA (logit_attribution.compute_dla)  -> each cluster's output-space effect.
  2. Hard nearest-centroid usage over real tokens -> drops DEAD clusters.
  3. Greedy dedup on the *distinctive* (baseline-removed) logit cosine -> drops
     functional DUPLICATES, keeping the highest-usage member of each clique.

The survivors are the distinct, used features. They are tagged with a coarse
category (numbers / discourse / caps-word / format / non-latin / mixed) and
ranked by usage so the interesting ones float to the top.

Usage:
    python -m geoae.interp.dla_inventory \
      --checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu/best_val.pt
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from geoae.checkpoint import load_lm
from geoae.seeding import seed_everything
from geoae.interp.logit_attribution import load_ae_and_norm, get_unembedding, compute_dla, topk_tokens


# ---------------------------------------------------------------------------
# Hard nearest-centroid usage from extracted activations (no LM needed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def nearest_centroid_usage(ae, norm_mean, norm_std, act_path: Path,
                           n_sample: int, device, batch: int = 16384,
                           seed: int = 42) -> np.ndarray:
    mmap = np.load(str(act_path), mmap_mode="r")
    N = mmap.shape[0]
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(N, min(n_sample, N), replace=False))
    counts = torch.zeros(ae.n_clusters, dtype=torch.long, device=device)
    for s in range(0, len(idx), batch):
        chunk = np.ascontiguousarray(mmap[idx[s:s+batch]]).astype(np.float32)
        x = (torch.from_numpy(chunk).to(device) - norm_mean) / norm_std
        counts += torch.bincount(ae(x).dist2.argmin(1), minlength=ae.n_clusters)
    return counts.cpu().numpy()


# ---------------------------------------------------------------------------
# Coarse category tag from promoted tokens
# ---------------------------------------------------------------------------

def categorize(promoted_tokens: list[str]) -> str:
    toks = [t for t, _ in promoted_tokens][:8]
    def frac(pred): return sum(pred(t) for t in toks) / max(len(toks), 1)
    DISCOURSE = {"however", "although", "since", "because", "therefore", "while",
                 "whereas", "unlike", "despite", "though", "moreover", "thus",
                 "nevertheless", "finally", "after", "before", "during", "based"}
    if frac(lambda t: bool(re.fullmatch(r"\s*\d[\d.,]*", t))) >= 0.6:
        return "numbers"
    if frac(lambda t: t.strip().lower() in DISCOURSE) >= 0.4:
        return "discourse"
    if frac(lambda t: bool(re.search(r"[\n\t]", t)) or not t.strip()) >= 0.5:
        return "format/ws"
    if frac(lambda t: bool(t) and not t.isascii()) >= 0.5:
        return "non-latin"
    if frac(lambda t: bool(re.fullmatch(r"\s*[A-Z][a-zA-Z]+", t))) >= 0.6:
        return "caps-word"
    if frac(lambda t: bool(re.fullmatch(r"\s*[a-z][a-zA-Z]+", t))) >= 0.6:
        return "lower-word"
    return "mixed"


# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--activations", default="activations/layer_27.npy")
    ap.add_argument("--n_sample", type=int, default=1_000_000)
    ap.add_argument("--live_min", type=int, default=1, help="Min nearest-centroid tokens to count as live")
    ap.add_argument("--dedup_cos", type=float, default=0.95,
                    help="Merge live clusters whose distinctive-effect cosine >= this")
    ap.add_argument("--top_k", type=int, default=12, help="Promoted/suppressed tokens shown per feature")
    ap.add_argument("--top_n", type=int, default=30, help="Distinct features to print")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    ae, norm_mean, norm_std, cfg = load_ae_and_norm(ckpt, device)
    K = ae.n_clusters
    model_name = cfg["extraction"]["model_name"]

    print(f"[inv] Loading LM unembedding: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = load_lm(model_name)
    W_U, final_norm_w, _ = get_unembedding(lm)
    W_U = W_U.to(device); final_norm_w = final_norm_w.to(device); del lm
    if device.type == "cuda": torch.cuda.empty_cache()

    logits = compute_dla(ae, norm_std, W_U, final_norm_w, apply_final_norm=True)   # (K, V)

    print(f"[inv] Counting nearest-centroid usage over {args.n_sample:,} tokens …")
    usage = nearest_centroid_usage(ae, norm_mean, norm_std, Path(args.activations),
                                   args.n_sample, device, seed=args.seed)
    total = usage.sum()
    live = np.where(usage >= args.live_min)[0]
    print(f"[inv] live (>= {args.live_min} tok): {len(live)}/{K}   dead: {K-len(live)}/{K}")

    # Distinctive (baseline-removed) cosine for dedup
    resid = logits - logits.mean(dim=0, keepdim=True)
    dirn = torch.nn.functional.normalize(resid, dim=1)            # (K, V)

    # Greedy dedup among LIVE clusters, highest-usage first
    order = sorted(live.tolist(), key=lambda k: -usage[k])
    reps: list[int] = []
    clique: dict[int, list[int]] = {}
    rep_dirs = None
    for k in order:
        v = dirn[k:k+1]                                           # (1, V)
        if reps:
            sims = (rep_dirs @ v.T).squeeze(1)                    # (n_reps,)
            j = int(sims.argmax())
            if float(sims[j]) >= args.dedup_cos:
                clique[reps[j]].append(k)
                continue
        reps.append(k); clique[k] = [k]
        rep_dirs = v if rep_dirs is None else torch.cat([rep_dirs, v], 0)

    print(f"[inv] distinct live features after dedup (cos>={args.dedup_cos}): {len(reps)}/{K}")

    # Build records
    records = []
    for k in reps:
        promoted, suppressed = topk_tokens(logits[k], tokenizer, args.top_k)
        records.append({
            "cluster": k,
            "usage_tokens": int(usage[k]),
            "usage_pct": round(100 * usage[k] / total, 3),
            "clique_size_live": len(clique[k]),
            "clique_members": sorted(clique[k]),
            "category": categorize(promoted),
            "logit_norm": round(float(logits[k].norm()), 2),
            "promoted": promoted,
            "suppressed": suppressed,
        })
    records.sort(key=lambda r: -r["usage_tokens"])

    out_path = Path(args.out) if args.out else ckpt.parent / "dla_distinct.json"
    with open(out_path, "w") as f:
        json.dump({
            "meta": {"checkpoint": str(ckpt), "n_clusters": K, "n_live": len(live),
                     "n_dead": K - len(live), "n_distinct_live": len(reps),
                     "dedup_cos": args.dedup_cos, "n_sample": args.n_sample},
            "features": records,
        }, f, indent=2)
    print(f"[inv] Saved {len(records)} distinct features -> {out_path}\n")

    # Print inventory
    print("=" * 92)
    print(f"  DISTINCT LIVE FEATURES (top {args.top_n} by usage)   "
          f"[1000 nominal -> {len(live)} live -> {len(reps)} distinct]")
    print("=" * 92)
    for r in records[:args.top_n]:
        prom = ", ".join(f"{t!r}" for t, _ in r["promoted"][:8])
        dup = f" +{r['clique_size_live']-1}dup" if r["clique_size_live"] > 1 else ""
        print(f"\ncluster {r['cluster']:4d}  {r['usage_pct']:>5.2f}% use  [{r['category']:<10}]{dup}")
        print(f"   promotes: {prom}")

    # Category breakdown
    from collections import Counter
    cats = Counter(r["category"] for r in records)
    print("\n" + "-" * 92)
    print("  category breakdown of distinct live features:  " +
          "  ".join(f"{c}={n}" for c, n in cats.most_common()))


if __name__ == "__main__":
    main()
