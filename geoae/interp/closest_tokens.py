"""
Max-activating-example collector for GeoAE clusters (v2).

For each LIVE cluster, gathers the tokens whose layer-N activation lands closest
to that centroid, drawn from a DIVERSE multi-domain corpus, and applies the
standard interpretability hygiene used in SAE feature analysis:

  * live-only        : nearest-centroid usage is computed first; dead clusters
                       (no token assigns to them) are skipped entirely.
  * diversified data : round-robins over web / encyclopedic / code / math so a
                       cluster that fires on code or non-English text still gets
                       representative contexts. Each example records its domain.
  * spectrum examples: besides the top (closest) examples, samples contexts at
                       distance percentiles of the cluster's assigned tokens, so
                       you see the cluster's whole range — not just its extreme,
                       which guards against the "interpretability illusion".
  * context dedup    : near-identical contexts (same token + left window) are
                       collapsed, so the BOS flood and repeats don't fill a slot.
  * token mix        : per cluster, the distribution of actual token strings and
                       a monosemanticity score (1 - normalised token entropy).
  * stats            : assigned count, usage %, distance min/median, per-domain
                       firing counts.

Output schema (per cluster key):
  { n_assigned, usage_pct, dist_min, dist_p50, monosemanticity, domains,
    token_distribution, top:[{dist,token,context,domain}], spectrum:[...] }

Usage:
    python -m geoae.interp.closest_tokens \
      --checkpoint e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu/best_val.pt \
      --n_tokens 500000 \
      --out e2e/checkpoints/general/llama3.2-3B/layer27/kl_gelu/closest_tokens.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


from geoae.checkpoint import load_ae_checkpoint, load_lm
from geoae.hooks import SplicingHook
from geoae.losses import pairwise_sq_dist


# ---------------------------------------------------------------------------
# Diverse corpus. Each source is tried independently; any that fails to load
# (auth, network, schema change) is skipped, and the token budget is spread
# round-robin over whatever loads — so the run is robust to one source breaking.
# ---------------------------------------------------------------------------

DEFAULT_SOURCES = [
    dict(domain="web",  name="allenai/c4",                        config="en",          split="validation", field="text"),
    dict(domain="wiki", name="wikimedia/wikipedia",               config="20231101.en", split="train",      field="text"),
    dict(domain="code", name="codeparrot/codeparrot-clean-valid", config=None,          split="train",      field="content"),
    dict(domain="math", name="open-web-math/open-web-math",       config=None,          split="train",      field="text"),
    dict(domain="pile", name="monology/pile-uncopyrighted",       config=None,          split="train",      field="text"),
]


def open_sources(sources: list[dict]) -> list[dict]:
    live = []
    for s in sources:
        try:
            ds = load_dataset(s["name"], name=s["config"], split=s["split"], streaming=True)
            it = iter(ds)
            # Probe one example so a broken schema fails here, not mid-run.
            first = next(it)
            if s["field"] not in first:
                print(f"[collect]   ! {s['domain']}: no field '{s['field']}' — skipping")
                continue
            live.append({**s, "iter": it, "primed": first})
            print(f"[collect]   + {s['domain']:<5} {s['name']}")
        except Exception as e:
            print(f"[collect]   ! {s['domain']:<5} {s['name']}: {type(e).__name__} — skipping")
    if not live:
        raise RuntimeError("No data sources could be loaded.")
    return live


def stream_docs(sources: list[dict], n_tokens: int, tokenizer, min_len: int, max_len: int):
    """Round-robin over sources, yielding (input_ids[T], domain) until budget met.

    Each document is truncated to `max_len` tokens (one chunk per doc) so that
    domains with long documents (e.g. Wikipedia articles) don't flood the budget,
    and a fixed token budget buys the maximum number of DISTINCT contexts.
    """
    pbar = tqdm(total=n_tokens, desc="[collect] Encoding tokens")
    seen = 0
    exhausted = set()
    while seen < n_tokens and len(exhausted) < len(sources):
        for si, s in enumerate(sources):
            if si in exhausted or seen >= n_tokens:
                continue
            try:
                ex = s.pop("primed", None) or next(s["iter"])
            except StopIteration:
                exhausted.add(si)
                continue
            text = ex.get(s["field"]) or ""
            if not text:
                continue
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_len)["input_ids"]
            if ids.shape[1] < min_len:
                continue
            yield ids, s["domain"]
            seen += ids.shape[1]
            pbar.update(ids.shape[1])
    pbar.close()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_baseline_kmeans(npz_path: Path, device: torch.device):
    """Raw k-means baseline: centroids live directly in the normalised activation
    space (no encoder). Returns (centroids, mean, std, K, target_layer, model_name)."""
    print(f"[collect] Loading baseline k-means centroids: {npz_path}")
    d = np.load(str(npz_path), allow_pickle=True)
    centroids = torch.as_tensor(d["centroids"], dtype=torch.float32, device=device)
    mean = torch.as_tensor(d["norm_mean"], dtype=torch.float32, device=device)
    std  = torch.as_tensor(d["norm_std"],  dtype=torch.float32, device=device)
    K = centroids.shape[0]
    target_layer = int(d["layer"]) if "layer" in d.files else int(d["target_layer"])
    model_name = str(d["model_name"]) if "model_name" in d.files else "meta-llama/Llama-3.2-3B"
    return centroids, mean, std, K, target_layer, model_name


def load_ae_and_norm(ckpt_path: Path, device: torch.device):
    print(f"[collect] Loading AE checkpoint: {ckpt_path}")
    ae, mean, std, ckpt = load_ae_checkpoint(ckpt_path, device)
    return ae, mean, std, ckpt["config"]


# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------

def render_context(tokens: list[int], idx: int, tokenizer, window: int) -> str:
    start, end = max(0, idx - window), min(len(tokens), idx + window + 1)
    left   = tokenizer.decode(tokens[start:idx])
    target = tokenizer.decode([tokens[idx]])
    right  = tokenizer.decode(tokens[idx + 1:end])
    return f"{left}«{target}»{right}".replace("\n", "\\n")


def context_signature(tokens: list[int], idx: int, window: int = 4):
    """Dedup key: target token + the few preceding tokens (collapses BOS flood)."""
    return (tokens[idx], tuple(tokens[max(0, idx - window):idx]))


# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="AE checkpoint .pt (learned clusters)")
    ap.add_argument("--baseline_kmeans", default=None,
                    help="Raw k-means .npz instead of an AE (clusters in activation space)")
    ap.add_argument("--n_tokens", type=int, default=500_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--live_min", type=int, default=1,
                    help="Min hard-assigned tokens for a cluster to be reported (skip dead)")
    ap.add_argument("--top_n", type=int, default=30, help="Closest deduped examples per cluster")
    ap.add_argument("--spectrum_n", type=int, default=6, help="Percentile-sampled examples per cluster")
    ap.add_argument("--context_window", type=int, default=6)
    ap.add_argument("--min_doc_len", type=int, default=10)
    ap.add_argument("--max_doc_tokens", type=int, default=256,
                    help="Truncate each doc to this many tokens (balances domains, maximises distinct contexts)")
    ap.add_argument("--skip_leading", type=int, default=4,
                    help="Skip the first N tokens of each doc as candidates (BOS / doc-start flood)")
    ap.add_argument("--print_top_n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from geoae.seeding import seed_everything
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[collect] Device: {device}")

    if bool(args.checkpoint) == bool(args.baseline_kmeans):
        ap.error("Provide exactly one of --checkpoint or --baseline_kmeans")

    use_ae = args.checkpoint is not None
    if use_ae:
        ckpt_path = Path(args.checkpoint)
        ae, mean, std, cfg = load_ae_and_norm(ckpt_path, device)
        centroids = ae.centroids
        metric = ae.metric
        K = ae.n_clusters
        target_layer = cfg["data"]["target_layer"]
        model_name = cfg["extraction"]["model_name"]
        source_path = ckpt_path
    else:
        ckpt_path = Path(args.baseline_kmeans)
        centroids, mean, std, K, target_layer, model_name = load_baseline_kmeans(ckpt_path, device)
        ae, metric = None, "euclidean"
        source_path = ckpt_path
    print(f"[collect] Mode: {'learned AE' if use_ae else 'raw k-means baseline'}  K={K}  layer={target_layer}")

    print(f"[collect] Loading LM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = load_lm(model_name, device_map="auto")

    captured = []
    hook = SplicingHook(lm, target_layer)
    hook.activate(lambda hs: (captured.append(hs.detach()) or hs))

    print("[collect] Opening data sources:")
    sources = open_sources(DEFAULT_SOURCES)

    # --- Pass 1: encode; store only each token's NEAREST centroid + distance ----
    # Keeping (label, min_dist) per token instead of the full (N, K) matrix makes
    # the memory cost ~8 bytes/token, so this scales to many millions of tokens.
    # Leading tokens of each doc are skipped as candidates (BOS / document-start
    # positions otherwise flood the boundary clusters), but the full doc is kept
    # so left-context still renders correctly.
    label_rows: list[torch.Tensor] = []
    mind_rows:  list[torch.Tensor] = []
    docs: list[list[int]] = []
    domains: list[str] = []
    tok_index: list[tuple[int, int]] = []  # candidate token -> (doc_idx, pos)
    skip = max(0, args.skip_leading)

    for ids, domain in stream_docs(sources, args.n_tokens, tokenizer,
                                   args.min_doc_len, args.max_doc_tokens):
        T = ids.shape[1]
        if T <= skip:
            continue
        captured.clear()
        lm(input_ids=ids.to(device))
        if not captured:
            continue
        hs = captured[0][0].float()                        # (T, D)
        x_norm = (hs - mean) / std
        # AE: distance in the LEARNED latent space; baseline: in the raw (normalised)
        # activation space — the only difference between the two runs.
        q = ae.encoder(x_norm) if use_ae else x_norm
        d2 = pairwise_sq_dist(q, centroids, metric=metric)[skip:]   # (T-skip, K)
        mind, lab = d2.min(dim=1)                           # nearest centroid + dist
        mind_rows.append(mind.cpu())
        label_rows.append(lab.cpu())
        di = len(docs)
        docs.append(ids[0].tolist())
        domains.append(domain)
        tok_index.extend((di, t) for t in range(skip, T))

    hook.deactivate()
    del lm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    labels   = torch.cat(label_rows, dim=0)                 # (N,) nearest centroid id
    min_dist = torch.cat(mind_rows,  dim=0)                 # (N,) dist to it
    N = labels.shape[0]
    print(f"[collect] Encoded {N:,} candidate tokens across {len(docs):,} docs "
          f"(skipped first {skip} tok/doc).")

    # --- Usage (the live/dead determination) -----------------------------------
    usage = torch.bincount(labels, minlength=K)            # (K,)
    live = (usage >= args.live_min).nonzero(as_tuple=True)[0].tolist()
    dom_per_tok = np.array([domains[di] for di, _ in tok_index])
    print(f"[collect] live clusters (>= {args.live_min} tok): {len(live)}/{K}  "
          f"(skipping {K - len(live)} dead)")

    # --- Pass 2: build per-cluster records (live only) -------------------------
    decoded_cache: dict[int, str] = {}
    def dec(tid: int) -> str:
        if tid not in decoded_cache:
            decoded_cache[tid] = tokenizer.decode([tid])
        return decoded_cache[tid]

    labels_gpu   = labels.to(device)
    min_dist_gpu = min_dist.to(device)

    out_clusters = {}
    for k in tqdm(live, desc="[collect] Building cluster records"):
        n_assigned = int(usage[k].item())

        # The cluster's assigned tokens, sorted by distance to centroid k (closest
        # first). "top" = closest deduped contexts; both top and spectrum draw from
        # this set, so examples are tokens that actually belong to the cluster.
        assigned = (labels_gpu == k).nonzero(as_tuple=True)[0]
        a_d = min_dist_gpu[assigned]
        order = torch.argsort(a_d)
        sorted_idx = assigned[order].cpu().tolist()
        sorted_d   = a_d[order].cpu().tolist()

        top, seen = [], set()
        for gi, d in zip(sorted_idx, sorted_d):
            di, pos = tok_index[gi]
            sig = context_signature(docs[di], pos)
            if sig in seen:
                continue
            seen.add(sig)
            top.append({"dist": round(d, 4), "token": dec(docs[di][pos]),
                        "context": render_context(docs[di], pos, tokenizer, args.context_window),
                        "domain": domains[di]})
            if len(top) >= args.top_n:
                break

        # Spectrum: percentile-sampled examples across the assigned distance range.
        spectrum = []
        M = len(sorted_idx)
        if M > 0:
            for p in np.linspace(0, 100, args.spectrum_n):
                j = int(round((p / 100) * (M - 1)))
                di, pos = tok_index[sorted_idx[j]]
                spectrum.append({"pct": int(p), "dist": round(sorted_d[j], 4),
                                 "token": dec(docs[di][pos]),
                                 "context": render_context(docs[di], pos, tokenizer, args.context_window),
                                 "domain": domains[di]})

        # Token-string distribution + monosemanticity over the closest assigned core.
        core = sorted_idx[:min(200, M)]
        tok_counts = Counter(dec(docs[tok_index[gi][0]][tok_index[gi][1]]) for gi in core)
        n_types = len(tok_counts)
        if core:
            ps = np.array(list(tok_counts.values()), dtype=float); ps /= ps.sum()
            H = -(ps * np.log(ps + 1e-12)).sum()
            mono = 1.0 - (H / math.log(n_types)) if n_types > 1 else 1.0
        else:
            mono = 0.0
        dom_counts = Counter(dom_per_tok[assigned.cpu().numpy()].tolist()) if n_assigned else {}

        out_clusters[str(k)] = {
            "n_assigned": n_assigned,
            "usage_pct": round(100 * n_assigned / N, 3),
            "dist_min": round(float(a_d.min().item()), 4) if n_assigned else None,
            "dist_p50": round(float(a_d.median().item()), 4) if n_assigned else None,
            "monosemanticity": round(float(mono), 3),
            "domains": dict(dom_counts),
            "token_distribution": tok_counts.most_common(15),
            "top": top,
            "spectrum": spectrum,
        }

    # --- Save ------------------------------------------------------------------
    out_path = Path(args.out) if args.out else ckpt_path.parent / "closest_tokens.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "mode": "learned_ae" if use_ae else "raw_kmeans_baseline",
            "checkpoint": str(source_path), "model_name": model_name,
            "target_layer": target_layer, "n_clusters": K,
            "n_tokens": N, "n_docs": len(docs),
            "sources": [{"domain": s["domain"], "name": s["name"]} for s in sources],
            "n_live": len(live), "n_dead": K - len(live), "live_min": args.live_min,
            "top_n": args.top_n, "spectrum_n": args.spectrum_n,
            "context_window": args.context_window,
        },
        "clusters": out_clusters,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[collect] Saved {len(live)} live clusters -> {out_path}")

    # --- Preview ---------------------------------------------------------------
    ranked = sorted(live, key=lambda k: -usage[k].item())
    print("\n=== Most-used live clusters (mono = monosemanticity 0..1) ===")
    for k in ranked[:args.print_top_n]:
        r = out_clusters[str(k)]
        doms = " ".join(f"{d}:{c}" for d, c in r["domains"].items())
        print(f"\ncluster {k}  use={r['usage_pct']:.2f}%  mono={r['monosemanticity']:.2f}  [{doms}]")
        toks = ", ".join(f"{t!r}×{c}" for t, c in r["token_distribution"][:6])
        print(f"  tokens: {toks}")
        for item in r["top"][:5]:
            print(f"    d={item['dist']:.3f} [{item['domain']}] {item['context']}")


if __name__ == "__main__":
    main()
    # HF streaming spawns background parquet-prefetch threads that don't join
    # cleanly and crash the interpreter at finalization. All work is already
    # saved by here, so hard-exit to skip the broken teardown.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
