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

Two data modes:
  * default         : the LEGACY recipe (DEFAULT_SOURCES below, positions 4..255,
                      unshuffled round-robin) -- the activations_diverse_10M mix.
  * --sampled_from  : a sampled-position dump (e.g. activations_sampled_10M). Reads
                      its meta.json for the frozen corpus, source weights,
                      context_len, positions_per_doc and skip_leading, and draws
                      HELD-OUT documents: the corpus margin that build_corpus wrote
                      but extract_sampled never consumed. Candidates are the same
                      uniform random positions per doc as the dump, so the clusters
                      are profiled on the distribution the AE was trained on,
                      without reusing a training document.
"""
from __future__ import annotations

import argparse
import gzip
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


def _corpus_tail(path: Path, skip: int):
    """Documents of a build_corpus jsonl.gz after the first `skip` lines. One doc
    per line (json.dumps escapes newlines), so skipped lines need no parsing."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for n in range(skip):
            if not f.readline():
                raise ValueError(f"{path} has only {n} docs, fewer than the {skip} the "
                                 f"dump consumed — wrong corpus for this dump?")
        for line in f:
            yield json.loads(line)["text"]


def sampled_sources(acts_dir: Path, corpus_dir: Path | None = None) -> tuple[dict, list[dict]]:
    """-> (dump meta, [{domain, weight, file, offset}]) for a sampled-position dump.

    `offset` is how many docs of that source's corpus file extract_sampled consumed
    (kept + skipped as short). They were read in file order, so every doc from
    `offset` on is held out of both the train and the val split."""
    meta = json.load(open(acts_dir / "meta.json"))
    if meta.get("mode") != "sampled":
        raise ValueError(f"{acts_dir} is not a sampled-position dump (mode={meta.get('mode')!r})")
    corpus_dir = Path(corpus_dir or meta["corpus_dir"])
    if not (corpus_dir / "manifest.json").exists():
        raise FileNotFoundError(f"no corpus at {corpus_dir} (path from meta.json) — "
                                f"pass --corpus_dir")
    files = {r["domain"]: r["file"] for r in json.load(open(corpus_dir / "manifest.json"))["sources"]}
    out = []
    for s in meta["sources"]:
        if s.get("kind", "hf") != "hf" or s["domain"] not in files:
            raise ValueError(f"source {s['domain']!r} has no corpus file (kind "
                             f"{s.get('kind', 'hf')!r}); held-out sampling needs one")
        out.append({"domain": s["domain"], "weight": s["weight"],
                    "file": corpus_dir / files[s["domain"]],
                    "offset": s["docs"] + s.get("skipped_short", 0)})
    return meta, out


def stream_sampled_docs(sources: list[dict], meta: dict, n_tokens: int, tokenizer,
                        min_len: int, seed: int):
    """Yield (input_ids[1,T], domain, candidate positions) in the dump's distribution:
    next doc from the source furthest below its weight's row quota, text cut to
    context_len, `positions_per_doc` uniform positions from [skip_leading, T)."""
    from geoae.extract_sampled import QuotaScheduler, sample_positions
    ctx, k, skip = meta["context_len"], meta["positions_per_doc"], meta["skip_leading"]
    iters = [_corpus_tail(s["file"], s["offset"]) for s in sources]
    sched = QuotaScheduler([s["weight"] for s in sources], n_tokens)
    rng = np.random.default_rng(seed)
    planned = 0
    pbar = tqdm(total=n_tokens, desc="[collect] Encoding sampled positions")
    while planned < n_tokens:
        i = sched.next()
        if i is None:
            break
        text = next(iters[i], None)
        if text is None:
            print(f"[collect]   ! {sources[i]['domain']}: held-out docs exhausted at "
                  f"{sched.got[i]:,.0f}/{sched.target[i]:,.0f} rows — the mix drifts from here")
            sched.kill(i)
            continue
        ids = tokenizer(text[:ctx * 16], return_tensors="pt", truncation=True,
                        max_length=ctx)["input_ids"]
        if ids.shape[1] < max(min_len, skip + 1):
            continue
        pos = sample_positions(ids.shape[1], skip, k, rng)[:n_tokens - planned]
        planned += len(pos)
        sched.add(i, len(pos))
        pbar.update(len(pos))
        yield ids, sources[i]["domain"], pos
    pbar.close()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_baseline_kmeans(npz_path: Path, device: torch.device, allow_erasure: bool = False):
    """Raw k-means baseline: centroids live directly in the normalised activation
    space (no encoder). Returns (centroids, mean, std, K, target_layer, model_name).

    A codebook fit with fit_balanced_kmeans --erase lives in a PROJECTED space;
    assigning un-projected activations to it is silently wrong, so callers that do
    not apply `erase_U` themselves (allow_erasure=False) are refused."""
    print(f"[collect] Loading baseline k-means centroids: {npz_path}")
    d = np.load(str(npz_path), allow_pickle=True)
    if "token_bias" in d.files and not allow_erasure:
        raise SystemExit(f"{npz_path} was fit on x - b[token] ({d['token_bias']}); this tool does "
                         f"not apply the token bias. Use concept_probe / probe_chance_corrected.")
    if "erase_U" in d.files and not allow_erasure:
        raise SystemExit(f"{npz_path} was fit with {int(d['erase_rank'])} {d['erase_basis']} directions "
                         f"erased; this tool does not apply the projection. Use concept_probe.")
    centroids = torch.as_tensor(d["centroids"], dtype=torch.float32, device=device)
    mean = torch.as_tensor(d["norm_mean"], dtype=torch.float32, device=device)
    std  = torch.as_tensor(d["norm_std"],  dtype=torch.float32, device=device)
    K = centroids.shape[0]
    target_layer = int(d["layer"]) if "layer" in d.files else int(d["target_layer"])
    model_name = str(d["model_name"]) if "model_name" in d.files else "meta-llama/Llama-3.2-3B"
    return centroids, mean, std, K, target_layer, model_name


def load_ae_and_norm(ckpt_path: Path, device: torch.device):
    print(f"[collect] Loading AE checkpoint: {ckpt_path}")
    ae, mean, std, ckpt = load_ae_checkpoint(ckpt_path, device, allow_token_bias=True)
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
# Document view (--doc_view): clusters that group by document/topic rather than
# by token (e.g. a token-bypass AE) look like noise at +-6 tokens of one row.
# These helpers show which DOCUMENTS a cluster draws from.
# ---------------------------------------------------------------------------

def doc_title(tokens: list[int], tokenizer, n_tokens: int = 64, width: int = 80) -> str:
    """First non-empty line of a document, as its title."""
    text = tokenizer.decode(tokens[:n_tokens], skip_special_tokens=True)
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line[:width] + ("…" if len(line) > width else "")


def doc_stats(doc_ids) -> dict:
    """Document spread of a cluster: distinct docs, effective docs (exp entropy), largest-doc share."""
    _, c = np.unique(np.asarray(doc_ids), return_counts=True)
    if len(c) == 0:
        return {"n_docs": 0, "eff_docs": 0.0, "top_doc_share": 0.0}
    p = c / c.sum()
    return {"n_docs": int(len(c)), "eff_docs": round(float(np.exp(-(p * np.log(p)).sum())), 2),
            "top_doc_share": round(float(p.max()), 3)}


def pick_top(sorted_idx, tok_index, docs, top_n: int, per_doc: int | None = None) -> list[int]:
    """Closest examples, deduped by context signature and capped at `per_doc` per document."""
    picked, seen, per = [], set(), Counter()
    for gi in sorted_idx:
        di, pos = tok_index[gi]
        sig = context_signature(docs[di], pos)
        if sig in seen or (per_doc is not None and per[di] >= per_doc):
            continue
        seen.add(sig)
        per[di] += 1
        picked.append(gi)
        if len(picked) >= top_n:
            break
    return picked


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
    ap.add_argument("--sampled_from", default=None,
                    help="Sampled-position dump dir (e.g. activations_sampled_10M): profile on "
                         "HELD-OUT docs of its corpus with its mix, context_len and "
                         "positions_per_doc, instead of the legacy recipe. --max_doc_tokens and "
                         "--skip_leading are then taken from the dump's meta.json.")
    ap.add_argument("--corpus_dir", default=None,
                    help="With --sampled_from: override the corpus path stored in meta.json "
                         "(needed when the repo moved, e.g. on Delta)")
    ap.add_argument("--doc_view", action="store_true",
                    help="Document-level view for clusters that group by document/topic: wider "
                         "context, <= --per_doc examples per document, each tagged with its "
                         "document's title, and per-cluster document spread, model confidence "
                         "and a hub flag.")
    ap.add_argument("--doc_context", type=int, default=24, help="--doc_view: context tokens each side")
    ap.add_argument("--per_doc", type=int, default=2, help="--doc_view: max top examples per document")
    ap.add_argument("--hub_usage_x", type=float, default=3.0,
                    help="--doc_view: a cluster is flagged `hub` if its usage is >= this x uniform "
                         "AND its median next-token confidence is in the top quartile")
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

    if args.sampled_from:
        from geoae.paths import resolve_path
        acts_dir = resolve_path(args.sampled_from)
        dump_meta, sources = sampled_sources(
            acts_dir, resolve_path(args.corpus_dir) if args.corpus_dir else None)
        if dump_meta.get("target_layer") != target_layer or dump_meta.get("model") != model_name:
            print(f"[collect] ! dump is {dump_meta.get('model')} L{dump_meta.get('target_layer')}, "
                  f"clusters are {model_name} L{target_layer} — only the TEXT is used, so this "
                  f"runs, but it is not the AE's own training distribution")
        print(f"[collect] Held-out docs from {acts_dir.name} "
              f"(context {dump_meta['context_len']}, {dump_meta['positions_per_doc']} pos/doc):")
        for s in sources:
            print(f"[collect]   + {s['domain']:<10} w={s['weight']:<4} skip first {s['offset']:,} docs")
        doc_stream = stream_sampled_docs(sources, dump_meta, args.n_tokens, tokenizer,
                                         args.min_doc_len, args.seed)
    else:
        print("[collect] Opening data sources:")
        sources = open_sources(DEFAULT_SOURCES)
        doc_stream = ((ids, dom, None) for ids, dom in
                      stream_docs(sources, args.n_tokens, tokenizer,
                                  args.min_doc_len, args.max_doc_tokens))

    # --- Pass 1: encode; store only each token's NEAREST centroid + distance ----
    # Keeping (label, min_dist) per token instead of the full (N, K) matrix makes
    # the memory cost ~8 bytes/token, so this scales to many millions of tokens.
    # Leading tokens of each doc are skipped as candidates (BOS / document-start
    # positions otherwise flood the boundary clusters), but the full doc is kept
    # so left-context still renders correctly.
    label_rows: list[torch.Tensor] = []
    mind_rows:  list[torch.Tensor] = []
    conf_rows:  list[torch.Tensor] = []   # --doc_view: the LM's top next-token probability
    docs: list[list[int]] = []
    domains: list[str] = []
    tok_index: list[tuple[int, int]] = []  # candidate token -> (doc_idx, pos)
    skip = dump_meta["skip_leading"] if args.sampled_from else max(0, args.skip_leading)

    for ids, domain, pos in doc_stream:
        T = ids.shape[1]
        if T <= skip:
            continue
        if pos is None:                                     # legacy: every position
            pos = np.arange(skip, T)
        captured.clear()
        lm_out = lm(input_ids=ids.to(device))
        if not captured:
            continue
        if args.doc_view:
            lg = lm_out.logits[0]
            conf_rows.append(lg.index_select(0, torch.as_tensor(pos, device=lg.device))
                             .float().softmax(-1).max(-1).values.cpu())
        del lm_out
        hs = captured[0][0].float()                        # (T, D)
        hs = hs.index_select(0, torch.as_tensor(pos, device=hs.device))
        x_norm = (hs - mean) / std
        # AE: distance in the LEARNED latent space; baseline: in the raw (normalised)
        # activation space — the only difference between the two runs.
        if use_ae:
            # token-bypass AE: encode the deviation from each row's current-token mean
            tok = (ids[0].to(x_norm.device)[torch.as_tensor(pos, device=x_norm.device)]
                   if ae.has_token_bias else None)
            q = ae.encode(x_norm, tok)
        else:
            q = x_norm
        d2 = pairwise_sq_dist(q, centroids, metric=metric)  # (n_pos, K)
        mind, lab = d2.min(dim=1)                           # nearest centroid + dist
        mind_rows.append(mind.cpu())
        label_rows.append(lab.cpu())
        di = len(docs)
        docs.append(ids[0].tolist())
        domains.append(domain)
        tok_index.extend((di, int(t)) for t in pos)

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

    window = args.doc_context if args.doc_view else args.context_window
    doc_meta = None
    if args.doc_view:
        conf = torch.cat(conf_rows).numpy()
        conf_q75 = float(np.quantile(conf, 0.75))
        tok_doc = np.array([di for di, _ in tok_index])
        titles: dict[int, str] = {}
        def title(di: int) -> str:
            if di not in titles:
                titles[di] = doc_title(docs[di], tokenizer)
            return titles[di]
        # "No context" reference point: the latent of a zero normalised input — for a
        # token-bypass AE that is any row sitting exactly on its token's mean. Hubs of
        # low-information rows tend to sit near it.
        with torch.no_grad():
            null = (ae.encoder(torch.zeros(1, mean.shape[0], device=centroids.device)) if use_ae
                    else torch.zeros(1, centroids.shape[1], device=centroids.device))
        null_d = (centroids - null).norm(dim=1).cpu().numpy()
        null_rank = np.empty(K); null_rank[np.argsort(null_d)] = np.arange(K) / max(K - 1, 1)
        hub_min = args.hub_usage_x * N / K
        doc_meta = {"doc_context": args.doc_context, "per_doc": args.per_doc,
                    "hub_rule": f"usage >= {args.hub_usage_x}x uniform and median confidence >= "
                                f"global 75th pct ({conf_q75:.3f})",
                    "conf_quantiles": {q: round(float(np.quantile(conf, q)), 4) for q in (0.25, 0.5, 0.75)}}

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

        dist_of = dict(zip(sorted_idx, sorted_d))

        def example(gi: int) -> dict:
            di, pos = tok_index[gi]
            e = {"dist": round(dist_of[gi], 4), "token": dec(docs[di][pos]),
                 "context": render_context(docs[di], pos, tokenizer, window),
                 "domain": domains[di]}
            if args.doc_view:
                e.update(doc=di, title=title(di), conf=round(float(conf[gi]), 3))
            return e

        top = [example(gi) for gi in pick_top(sorted_idx, tok_index, docs, args.top_n,
                                              args.per_doc if args.doc_view else None)]

        # Spectrum: percentile-sampled examples across the assigned distance range.
        spectrum = []
        M = len(sorted_idx)
        if M > 0:
            for p in np.linspace(0, 100, args.spectrum_n):
                j = int(round((p / 100) * (M - 1)))
                spectrum.append({"pct": int(p), **example(sorted_idx[j])})

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
        if args.doc_view:
            a_np = assigned.cpu().numpy()
            dids = tok_doc[a_np]
            dc = Counter(dids.tolist()).most_common(5)
            cm = float(np.median(conf[a_np])) if n_assigned else 0.0
            out_clusters[str(k)].update(
                **doc_stats(dids),
                top_docs=[{"title": title(di), "domain": domains[di], "n": n} for di, n in dc],
                conf_median=round(cm, 3),
                null_dist=round(float(null_d[k]), 3), null_dist_pct=round(float(null_rank[k]), 3),
                hub=bool(n_assigned >= hub_min and cm >= conf_q75))

    # --- Save ------------------------------------------------------------------
    out_path = Path(args.out) if args.out else ckpt_path.parent / "closest_tokens.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "mode": "learned_ae" if use_ae else "raw_kmeans_baseline",
            "checkpoint": str(source_path), "model_name": model_name,
            "target_layer": target_layer, "n_clusters": K,
            "n_tokens": N, "n_docs": len(docs),
            "sources": [{"domain": s["domain"], "name": s.get("name", str(s.get("file")))}
                        for s in sources],
            "data": ({"kind": "sampled_held_out", "activations_dir": str(args.sampled_from),
                      "corpus_files": {s["domain"]: str(s["file"]) for s in sources},
                      "doc_offsets": {s["domain"]: s["offset"] for s in sources},
                      "weights": {s["domain"]: s["weight"] for s in sources},
                      "context_len": dump_meta["context_len"],
                      "positions_per_doc": dump_meta["positions_per_doc"],
                      "skip_leading": skip, "seed": args.seed}
                     if args.sampled_from else
                     {"kind": "legacy_prefix", "max_doc_tokens": args.max_doc_tokens,
                      "skip_leading": skip}),
            "n_live": len(live), "n_dead": K - len(live), "live_min": args.live_min,
            "top_n": args.top_n, "spectrum_n": args.spectrum_n,
            "context_window": window,
            **({"doc_view": {**doc_meta, "n_hubs": sum(c["hub"] for c in out_clusters.values())}}
               if args.doc_view else {}),
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
        if args.doc_view:
            print(f"  docs: {r['n_docs']} distinct, {r['eff_docs']} effective, largest {r['top_doc_share']:.0%}"
                  f" | conf {r['conf_median']:.2f}{'  [HUB]' if r['hub'] else ''}")
            for td in r["top_docs"][:3]:
                print(f"    {td['n']:>4} × [{td['domain']}] {td['title']}")
        for item in r["top"][:5]:
            print(f"    d={item['dist']:.3f} [{item['domain']}] {item['context']}")


if __name__ == "__main__":
    main()
    # HF streaming spawns background parquet-prefetch threads that don't join
    # cleanly and crash the interpreter at finalization. All work is already
    # saved by here, so hard-exit to skip the broken teardown.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
