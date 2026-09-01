"""
Phase A — map where separable structure lives in raw residual space.

Answers, for RAW h-space (no AE), across layers x objects x label types:
  "is there separable manifold structure here, and how much?"

Two subcommands:

  extract   One LM pass over (a) the diverse corpus (token-level, aligned
            token ids / doc ids / domains) and (b) DBpedia-14 test docs
            (mean-pooled doc reps + class labels), capturing ALL target
            layers simultaneously via forward hooks (same convention as
            geoae.extract / SplicingHook: activation = output of block L).

  analyze   Per layer x object x label-set:
              fisher      tr(S_between)/tr(S_within)   — separation as a scalar
              knn_acc     10-NN label consistency (vs chance)
              silhouette  best silhouette over a coarse-K k-means curve
              twonn_id    TwoNN intrinsic dimension (Facco et al. 2017)
            Objects: diverse tokens, diverse mean-pooled docs, DBpedia docs.
            Labels:  token identity (top-N), function-vs-content, domain,
                     DBpedia class (incl. Hungarian-matched k-means accuracy).

Usage:
    python -m geoae.interp.manifold_map extract --out phase_a
    python -m geoae.interp.manifold_map analyze --dir phase_a
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geoae.lm_arch import decoder_layers

LAYERS = [16, 20, 24, 27]


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

class MultiLayerCapture:
    """Forward hooks on the decoder's layers[L], capturing output[0] per layer.

    Matches the geoae.extract / SplicingHook convention (residual AFTER block L,
    pre-final-norm) — do NOT swap for output_hidden_states, whose last entry is
    post-final-norm and would silently mismatch layer 27.
    """

    def __init__(self, lm, layers: list[int]):
        self.store: dict[int, torch.Tensor] = {}
        self.handles = []
        blocks = decoder_layers(lm)
        for L in layers:
            self.handles.append(blocks[L].register_forward_hook(self._make(L)))

    def _make(self, L):
        def fn(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            self.store[L] = hs.detach()
        return fn

    def pop(self) -> dict[int, torch.Tensor]:
        out, self.store = self.store, {}
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def _forward_batch(lm, capture, ids_list, pad_id, device):
    """Right-padded batch forward; returns {layer: [per-doc (T_i, D) fp16 cpu]}."""
    lens = [x.shape[1] for x in ids_list]
    T = max(lens)
    B = len(ids_list)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, : lens[i]] = x[0]
        mask[i, : lens[i]] = 1
    lm(input_ids=ids.to(device), attention_mask=mask.to(device))
    acts = capture.pop()
    return {L: [acts[L][i, : lens[i]].to(torch.float16).cpu() for i in range(B)]
            for L in acts}


def cmd_extract(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from geoae.extract import open_sources, stream_docs, DEFAULT_SOURCES

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    model_name = "meta-llama/Llama-3.2-3B"
    print(f"[map] Loading {model_name} (bf16) …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                              device_map={"": 0})
    lm.eval().requires_grad_(False)
    pad_id = tokenizer.eos_token_id
    capture = MultiLayerCapture(lm, LAYERS)

    # ---- (a) diverse corpus, token-level, aligned -------------------------
    print(f"[map] Diverse corpus: streaming ~{args.n_tokens:,} tokens "
          f"(skip_leading={args.skip_leading}) …")
    srcs = open_sources([dict(s) for s in DEFAULT_SOURCES])
    gen = stream_docs(srcs, args.n_tokens * 2, tokenizer, 10, 256)

    rows = {L: [] for L in LAYERS}
    tok_ids, doc_ids, domains = [], [], []
    batch, batch_dom = [], []
    n_saved, doc_i = 0, 0

    def flush():
        nonlocal n_saved, doc_i
        if not batch:
            return
        per_layer = _forward_batch(lm, capture, batch, pad_id, device)
        for i, ids in enumerate(batch):
            keep = ids.shape[1] - args.skip_leading
            if keep <= 0:
                continue
            for L in LAYERS:
                rows[L].append(per_layer[L][i][args.skip_leading:])
            tok_ids.append(ids[0, args.skip_leading:].numpy().astype(np.int32))
            doc_ids.append(np.full(keep, doc_i, dtype=np.int32))
            domains.append(batch_dom[i])
            n_saved += keep
            doc_i += 1
        batch.clear(); batch_dom.clear()

    for ids, domain in gen:
        batch.append(ids); batch_dom.append(domain)
        if len(batch) >= args.batch_docs:
            flush()
            if n_saved >= args.n_tokens:
                break
            if doc_i % 320 == 0:
                print(f"  … {n_saved:,} tokens / {doc_i} docs", flush=True)
    flush()

    print(f"[map] Saving diverse: {n_saved:,} tokens, {doc_i} docs")
    for L in LAYERS:
        np.save(out / f"diverse_acts_L{L}.npy",
                torch.cat(rows[L]).numpy())
        rows[L] = []
    np.save(out / "diverse_token_ids.npy", np.concatenate(tok_ids))
    np.save(out / "diverse_doc_ids.npy", np.concatenate(doc_ids))
    (out / "diverse_domains.json").write_text(json.dumps(domains))

    # ---- (b) DBpedia-14 docs, mean-pooled ---------------------------------
    from datasets import load_dataset
    print(f"[map] DBpedia-14: {args.dbpedia_per_class}/class …")
    ds = load_dataset("fancyzhx/dbpedia_14", split="test").shuffle(seed=0)
    per_class = {c: [] for c in range(14)}
    for ex in ds:
        c = int(ex["label"])
        if len(per_class[c]) < args.dbpedia_per_class:
            per_class[c].append(ex["content"].strip())
        if all(len(v) >= args.dbpedia_per_class for v in per_class.values()):
            break

    texts, labels = [], []
    for c in range(14):
        texts += per_class[c]
        labels += [c] * len(per_class[c])

    doc_means = {L: [] for L in LAYERS}
    for s in range(0, len(texts), args.batch_docs):
        chunk = texts[s : s + args.batch_docs]
        ids_list = [tokenizer(t, return_tensors="pt", truncation=True,
                              max_length=256)["input_ids"] for t in chunk]
        per_layer = _forward_batch(lm, capture, ids_list, pad_id, device)
        for L in LAYERS:
            for a in per_layer[L]:
                doc_means[L].append(a[1:].float().mean(0).to(torch.float16))  # skip BOS
        if s % (args.batch_docs * 20) == 0:
            print(f"  … {s}/{len(texts)} docs", flush=True)

    for L in LAYERS:
        np.save(out / f"dbpedia_docacts_L{L}.npy",
                torch.stack(doc_means[L]).numpy())
    np.save(out / "dbpedia_labels.npy", np.array(labels, dtype=np.int32))

    (out / "meta.json").write_text(json.dumps({
        "model": model_name, "layers": LAYERS, "n_tokens": n_saved,
        "n_docs": doc_i, "skip_leading": args.skip_leading,
        "dbpedia_docs": len(texts),
    }, indent=2))
    capture.remove()
    print(f"[map] Done → {out}/")


# ---------------------------------------------------------------------------
# analyze — metric helpers (GPU where it matters)
# ---------------------------------------------------------------------------

def zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sd


def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """tr(S_between)/tr(S_within), class-size weighted."""
    m = X.mean(0)
    sb = sw = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        mc = Xc.mean(0)
        sb += len(Xc) * float(((mc - m) ** 2).sum())
        sw += float(((Xc - mc) ** 2).sum())
    return sb / max(sw, 1e-12)


def knn_consistency(X: np.ndarray, y: np.ndarray, k: int = 10,
                    device="cuda", chunk: int = 2048) -> tuple[float, float]:
    """(10-NN label agreement, chance = sum p_c^2)."""
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y.astype(np.int64)).to(device)
    hits, total = 0, 0
    for s in range(0, len(Xt), chunk):
        d = torch.cdist(Xt[s : s + chunk], Xt)
        idx = d.topk(k + 1, largest=False).indices[:, 1:]      # drop self
        hits += (yt[idx] == yt[s : s + chunk, None]).sum().item()
        total += idx.numel()
    p = np.bincount(y) / len(y)
    return hits / total, float((p ** 2).sum())


def twonn_id(X: np.ndarray, device="cuda", chunk: int = 2048) -> float:
    """TwoNN intrinsic dimension (Facco et al. 2017): d = N / sum(log d2/d1)."""
    Xt = torch.from_numpy(X).to(device)
    mus = []
    for s in range(0, len(Xt), chunk):
        d = torch.cdist(Xt[s : s + chunk], Xt)
        v = d.topk(3, largest=False).values                    # self, d1, d2
        d1, d2 = v[:, 1], v[:, 2]
        ok = d1 > 1e-6
        mus.append((d2[ok] / d1[ok]).log())
    mu = torch.cat(mus)
    return float(len(mu) / mu.sum().item())


def gpu_kmeans_labels(X: np.ndarray, K: int, seed: int = 0, iters: int = 40,
                      device="cuda", chunk: int = 4096) -> np.ndarray:
    """Lloyd k-means on GPU (k-means++ init on a subsample); returns labels."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xt = torch.from_numpy(X).to(device)
    N = len(Xt)
    sub = Xt[torch.randperm(N, generator=g)[: min(20_000, N)].to(device)]
    C = sub[torch.randint(len(sub), (1,), generator=g).item()][None]
    for _ in range(K - 1):
        d2 = torch.cdist(sub, C).pow(2).min(1).values
        probs = (d2 / d2.sum()).cpu()
        C = torch.cat([C, sub[torch.multinomial(probs, 1, generator=g).item()][None]])
    for _ in range(iters):
        lab = torch.cat([torch.cdist(Xt[s : s + chunk], C).argmin(1)
                         for s in range(0, N, chunk)])
        newC = torch.zeros_like(C)
        cnt = torch.zeros(K, device=device)
        newC.index_add_(0, lab, Xt)
        cnt.index_add_(0, lab, torch.ones(N, device=device))
        dead = cnt == 0
        newC[~dead] /= cnt[~dead, None]
        if dead.any():  # respawn dead centroids on random points
            newC[dead] = Xt[torch.randint(0, N, (int(dead.sum()),), generator=g).to(device)]
        if torch.allclose(newC, C, atol=1e-5):
            C = newC
            break
        C = newC
    lab = torch.cat([torch.cdist(Xt[s : s + chunk], C).argmin(1)
                     for s in range(0, N, chunk)])
    return lab.cpu().numpy()


def silhouette_curve(X: np.ndarray, Ks: list[int], seed: int = 0) -> dict[int, float]:
    from sklearn.metrics import silhouette_score
    out = {}
    rng = np.random.RandomState(seed)
    for K in Ks:
        if K >= len(X):
            continue
        lab = gpu_kmeans_labels(X, K, seed=seed)
        idx = rng.choice(len(X), min(10_000, len(X)), replace=False)
        if len(np.unique(lab[idx])) < 2:
            out[K] = float("nan")
            continue
        out[K] = float(silhouette_score(X[idx], lab[idx]))
    return out


def hungarian_acc(X: np.ndarray, y: np.ndarray, K: int, seed: int = 0) -> tuple[float, float]:
    """k-means K → best bijection to classes: (accuracy, NMI)."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import normalized_mutual_info_score
    lab = gpu_kmeans_labels(X, K, seed=seed)
    C = np.zeros((K, len(np.unique(y))), dtype=np.int64)
    for p, t in zip(lab, y):
        C[p, t] += 1
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / len(y)), float(normalized_mutual_info_score(y, lab))


# ---------------------------------------------------------------------------
# analyze — label construction
# ---------------------------------------------------------------------------

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "by", "for", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "am", "do", "does", "did", "has", "have",
    "had", "will", "would", "can", "could", "shall", "should", "may", "might",
    "must", "not", "no", "nor", "so", "that", "this", "these", "those", "it",
    "its", "he", "she", "they", "we", "you", "i", "his", "her", "their",
    "our", "your", "my", "me", "him", "them", "us", "who", "whom", "which",
    "what", "when", "where", "why", "how", "there", "here", "than", "too",
    "very", "just", "also", "into", "over", "under", "between", "through",
    "during", "before", "after", "above", "below", "up", "down", "out",
    "about", "against", "again", "further", "once", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "s", "t", "don", "now", "while", "because",
}


def token_class_labels(token_ids: np.ndarray, tokenizer) -> np.ndarray:
    """0 = function/punct/digit token, 1 = content token."""
    uniq = np.unique(token_ids)
    is_content = {}
    for tid in uniq:
        s = tokenizer.decode([int(tid)]).strip().lower()
        alpha = any(ch.isalpha() for ch in s)
        is_content[int(tid)] = int(alpha and s not in FUNCTION_WORDS)
    return np.vectorize(is_content.get)(token_ids).astype(np.int64)


def top_token_labels(token_ids: np.ndarray, n_top: int = 100):
    """Labels 0..n_top-1 for the most frequent token ids; mask for the rest."""
    vals, counts = np.unique(token_ids, return_counts=True)
    top = vals[np.argsort(-counts)[:n_top]]
    lut = {int(t): i for i, t in enumerate(top)}
    lab = np.array([lut.get(int(t), -1) for t in token_ids], dtype=np.int64)
    return lab, lab >= 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def _metric_block(X, y, Ks, tag, results, do_curve=True):
    f = fisher_ratio(X, y)
    knn, chance = knn_consistency(X, y)
    row = {"fisher": round(f, 4), "knn": round(knn, 4), "knn_chance": round(chance, 4),
           "knn_lift": round(knn / max(chance, 1e-9), 2)}
    if do_curve:
        row["silhouette_by_K"] = {k: round(v, 4) for k, v in
                                  silhouette_curve(X, Ks).items()}
    results[tag] = row
    sil = row.get("silhouette_by_K", {})
    best = max(sil.values()) if sil else float("nan")
    print(f"    {tag:<28} fisher {f:8.4f} | knn {knn:.3f} (chance {chance:.3f}, "
          f"lift {row['knn_lift']:.1f}x) | best sil {best:.4f}")


def cmd_analyze(args):
    from transformers import AutoTokenizer
    d = Path(args.dir)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    rng = np.random.RandomState(0)

    token_ids = np.load(d / "diverse_token_ids.npy")
    doc_ids = np.load(d / "diverse_doc_ids.npy")
    domains = json.loads((d / "diverse_domains.json").read_text())
    dom_lut = {s: i for i, s in enumerate(sorted(set(domains)))}
    doc_dom = np.array([dom_lut[s] for s in domains], dtype=np.int64)

    n = len(token_ids)
    tok_sample = np.sort(rng.choice(n, min(args.n_token_sample, n), replace=False))
    fc_labels = token_class_labels(token_ids, tokenizer)
    top_lab, top_mask = top_token_labels(token_ids, n_top=100)

    dbp_labels = np.load(d / "dbpedia_labels.npy").astype(np.int64)

    results = {}
    for L in LAYERS:
        print(f"\n[map] ===== layer {L} =====")
        res_L = {}
        A = np.load(d / f"diverse_acts_L{L}.npy").astype(np.float32)

        # -- token object --------------------------------------------------
        X = zscore(A[tok_sample])
        y_dom = doc_dom[doc_ids[tok_sample]]
        y_fc = fc_labels[tok_sample]
        print(f"  tokens (n={len(X):,})  twonn_id …", end=" ", flush=True)
        tid = twonn_id(X[rng.choice(len(X), min(20_000, len(X)), replace=False)])
        res_L["token_twonn_id"] = round(tid, 1)
        print(f"{tid:.1f}")
        Ks = [8, 32, 128, 512]
        _metric_block(X, y_fc, Ks, "token/function-vs-content", res_L, do_curve=True)
        _metric_block(X, y_dom, Ks, "token/domain", res_L, do_curve=False)
        m = top_mask[tok_sample]
        _metric_block(X[m], top_lab[tok_sample][m], Ks, "token/token-identity", res_L,
                      do_curve=False)

        # -- doc object (mean-pooled diverse) -------------------------------
        n_docs = doc_ids.max() + 1
        sums = np.zeros((n_docs, A.shape[1]), dtype=np.float64)
        np.add.at(sums, doc_ids, A)
        cnt = np.bincount(doc_ids, minlength=n_docs)[:, None]
        Xd = zscore((sums / cnt).astype(np.float32))
        print(f"  docs (n={n_docs})  twonn_id …", end=" ", flush=True)
        did = twonn_id(Xd)
        res_L["doc_twonn_id"] = round(did, 1)
        print(f"{did:.1f}")
        _metric_block(Xd, doc_dom, [4, 8, 16, 64], "doc/domain", res_L)
        del A

        # -- DBpedia doc object ---------------------------------------------
        Xb = zscore(np.load(d / f"dbpedia_docacts_L{L}.npy").astype(np.float32))
        _metric_block(Xb, dbp_labels, [14], "dbpedia/class", res_L, do_curve=True)
        acc, nmi = hungarian_acc(Xb, dbp_labels, K=14)
        res_L["dbpedia_kmeans14"] = {"hungarian_acc": round(acc, 4), "nmi": round(nmi, 4)}
        print(f"    dbpedia k-means K=14         acc {acc:.3f} | nmi {nmi:.3f}")

        results[f"layer_{L}"] = res_L

    out = d / "manifold_map.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[map] Saved → {out}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--out", default="phase_a")
    e.add_argument("--n_tokens", type=int, default=500_000)
    e.add_argument("--dbpedia_per_class", type=int, default=500)
    e.add_argument("--batch_docs", type=int, default=16)
    e.add_argument("--skip_leading", type=int, default=4)

    a = sub.add_parser("analyze")
    a.add_argument("--dir", default="phase_a")
    a.add_argument("--n_token_sample", type=int, default=100_000)

    args = ap.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
