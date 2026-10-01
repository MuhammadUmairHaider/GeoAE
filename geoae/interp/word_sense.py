"""
Word-sense separation: do a codebook's clusters split the meanings ONE token carries?

The token-bypass AE encodes x - b[token], so its clusters should follow context, not
token identity; base (balanced k-means on the residual, same K) clusters tend to be
token-shaped. A word with two senses is the sharpest test: the token table holds the
word, so only the latent can hold which sense it is. Both spaces see the SAME
occurrences (a paired comparison); an optional token-mean control (the same k-means
fit on x - b[token], no encoder) says how much of any gap is the subtraction alone.

Data
  CoarseWSD-20 (Loureiro et al. 2021)  20 ambiguous nouns, 2-5 coarse senses each,
                                       Wikipedia sentences, LOWERCASED and pre-tokenised
                                       (fed as-is to every space).
  WiC (Pilehvar & Camacho-Collados 2019)  sentence pairs sharing a target word,
                                       label 1 = same sense (train + validation).

The model is causal: the representation at the target word sees only LEFT context.
Targets earlier than the dump's `skip_leading` (the AE never trained on them) are dropped.

Metrics (per space)
  CoarseWSD-20, within each word, macro-averaged over words:
    cnmi_sense         chance-corrected NMI(cluster, sense)   <- the headline number
    purity, top_cluster_share, n_clusters, eff_clusters
    knn_acc            leave-one-out 10-NN sense accuracy, cosine on the representation
  WiC:
    same_cluster_gap   P(same cluster | same sense) - P(same cluster | different sense)
    auroc_cos          ROC AUC of cosine(r1, r2) for "same sense"
    acc_threshold      validation accuracy, cosine threshold chosen on train
  Differences bypass - base with a paired bootstrap and BASIC intervals:
  est + quantile(boots - mean(boots)); the estimate is the full-sample value.

    python -m geoae.interp.word_sense --download_only --data_dir data/wsd
    python -m geoae.interp.word_sense --parse_only --data_dir data/wsd
    python -m geoae.interp.word_sense --checkpoint <bypass.pt> --baseline_kmeans <base.npz> \
        [--tokmean_kmeans <tokmean.npz>] --act_meta activations_sampled_10M/meta.json \
        --out results/delta/d6144/interp/word_sense.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

COARSEWSD_WORDS = ["apple", "arm", "bank", "bass", "bow", "chair", "club", "crane", "deck", "digit",
                   "hood", "java", "mole", "pitcher", "pound", "seal", "spring", "square", "trunk", "yard"]
COARSEWSD_URL = "https://raw.githubusercontent.com/danlou/bert-disambiguation/master/data/CoarseWSD-20"
COARSEWSD_FILES = ["classes_map.txt", "train.data.txt", "train.gold.txt", "test.data.txt", "test.gold.txt"]
COARSEWSD_EXTRA = ["LICENSE", "README.txt"]
WIC_REPO = "aps/super_glue"
# test is skipped: its labels are -1 (hidden for the SuperGLUE leaderboard)
WIC_FILES = {"train": "wic/train-00000-of-00001.parquet",
             "validation": "wic/validation-00000-of-00001.parquet"}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _fetch(url: str, dest: Path, retries: int = 4) -> int:
    """Download url -> dest unless dest already exists. Returns the file size in bytes."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "geoae-word-sense"})
            with urllib.request.urlopen(req, timeout=60) as r:
                blob = r.read()
            if not blob:
                raise OSError("empty response")
            tmp = dest.with_name(dest.name + ".part")     # never leave a truncated file under the real name
            tmp.write_bytes(blob)
            tmp.replace(dest)
            return len(blob)
        except Exception as e:                            # noqa: BLE001 -- retry on any network error
            err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not fetch {url}: {type(err).__name__}: {err}")


def download_data(data_dir: Path, datasets: list[str]) -> dict:
    """Fetch CoarseWSD-20 (GitHub raw) and WiC (HF dataset repo) into data_dir. Idempotent."""
    manifest: dict = {"downloaded": time.strftime("%Y-%m-%d"), "files": {}}
    if "coarsewsd20" in datasets:
        root = data_dir / "CoarseWSD-20"
        for name in COARSEWSD_EXTRA:
            n = _fetch(f"{COARSEWSD_URL}/{name}", root / name)
            manifest["files"][f"CoarseWSD-20/{name}"] = {"url": f"{COARSEWSD_URL}/{name}", "bytes": n}
        for w in COARSEWSD_WORDS:
            for name in COARSEWSD_FILES:
                url = f"{COARSEWSD_URL}/{w}/{name}"
                n = _fetch(url, root / w / name)
                manifest["files"][f"CoarseWSD-20/{w}/{name}"] = {"url": url, "bytes": n}
            print(f"[wsense] CoarseWSD-20/{w}: ok", flush=True)
    if "wic" in datasets:
        from huggingface_hub import hf_hub_download
        for split, rel in WIC_FILES.items():
            dest = data_dir / "wic" / Path(rel).name
            cached = hf_hub_download(WIC_REPO, rel, repo_type="dataset")
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, dest)             # self-contained copy, independent of HF_HOME
            parts = Path(cached).parts                    # .../snapshots/<commit>/wic/<file>
            rev = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
            manifest["files"][f"wic/{dest.name}"] = {
                "repo": WIC_REPO, "revision": rev, "bytes": dest.stat().st_size,
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
            print(f"[wsense] wic/{dest.name}: ok ({dest.stat().st_size:,} bytes, rev {rev})", flush=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------------------
# CoarseWSD-20 parsing
# ---------------------------------------------------------------------------

def read_lines(path: Path) -> list[str]:
    """Lines of a text file, split ONLY on "\\n". Universal-newline mode would also split on a
    stray "\\r" inside a sentence and silently shift every later gold label."""
    with open(path, encoding="utf-8", newline="\n") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def parse_coarsewsd_line(line: str) -> tuple[int, str]:
    """'<word_index>\\t<sentence>' -> (word_index, sentence)."""
    idx, sep, sentence = line.partition("\t")
    if not sep:
        raise ValueError(f"CoarseWSD line has no tab: {line[:60]!r}")
    return int(idx), sentence


def coarsewsd_span(sentence: str, index: int) -> tuple[int, int, str] | None:
    """Character span (start, end, surface) of whitespace-token `index` of `sentence`, or None
    when the index is out of range. The data is pre-tokenised on single spaces, so the start
    is the summed lengths of the preceding tokens plus one space each."""
    toks = sentence.split(" ")
    if not 0 <= index < len(toks):
        return None
    start = sum(len(t) for t in toks[:index]) + index
    return start, start + len(toks[index]), toks[index]


def load_coarsewsd(data_dir: Path, words: list[str] = COARSEWSD_WORDS) -> tuple[list[dict], dict]:
    """All train+test rows of the given words -> (rows, stats). Rows keep their split."""
    root = Path(data_dir) / "CoarseWSD-20"
    rows: list[dict] = []
    stats: dict = {}
    for w in words:
        names = {int(k): v for k, v in json.loads((root / w / "classes_map.txt").read_text()).items()}
        st = {"senses": names, "n_raw": 0, "out_of_range": 0, "form_mismatch": 0,
              "form_examples": [], "split_counts": Counter()}
        for split in ("train", "test"):
            data = read_lines(root / w / f"{split}.data.txt")
            gold = read_lines(root / w / f"{split}.gold.txt")
            if len(data) != len(gold):
                raise ValueError(f"{w}/{split}: {len(data)} data lines but {len(gold)} gold labels")
            for line_no, (ln, g) in enumerate(zip(data, gold)):
                idx, sentence = parse_coarsewsd_line(ln)
                sense = int(g)
                if sense not in names:
                    raise ValueError(f"{w}/{split} line {line_no}: gold {sense} not in classes_map")
                st["n_raw"] += 1
                sp = coarsewsd_span(sentence, idx)
                if sp is None:
                    st["out_of_range"] += 1
                    continue
                start, end, surface = sp
                if not surface.startswith(w):                 # index points at another word: drop
                    st["form_mismatch"] += 1
                    if len(st["form_examples"]) < 5:
                        st["form_examples"].append(surface)
                    continue
                rows.append(dict(dataset="coarsewsd20", word=w, sense=sense, sense_name=names[sense],
                                 split=split, line=line_no, text=sentence, start=start, end=end,
                                 surface=surface))
                st["split_counts"][split] += 1
        stats[w] = st
    return rows, stats


# ---------------------------------------------------------------------------
# WiC parsing
# ---------------------------------------------------------------------------

def load_wic(data_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """WiC train + validation -> (rows: one per sentence, pairs, stats). Each pair owns two
    consecutive rows; row["pair"] indexes `pairs`."""
    import pandas as pd
    rows, pairs = [], []
    stats = {"mismatch": 0, "examples": [], "splits": {}}
    for split, rel in WIC_FILES.items():
        df = pd.read_parquet(Path(data_dir) / "wic" / Path(rel).name)
        stats["splits"][split] = {"pairs": len(df), "same": int((df.label == 1).sum())}
        for r in df.itertuples(index=False):
            pi = len(pairs)
            pairs.append(dict(pair=pi, word=r.word, label=int(r.label), split=split))
            for side, (text, a, b) in enumerate(((r.sentence1, r.start1, r.end1),
                                                 (r.sentence2, r.start2, r.end2))):
                surface = text[int(a):int(b)]
                # WiC targets are inflected forms of `word` (run/ran, stand/stood): only an
                # empty span is a real error
                if not surface.strip():
                    stats["mismatch"] += 1
                    if len(stats["examples"]) < 5:
                        stats["examples"].append((r.word, surface))
                rows.append(dict(dataset="wic", word=r.word, pair=pi, side=side, split=split,
                                 label=int(r.label), text=text, start=int(a), end=int(b),
                                 surface=surface))
    return rows, pairs, stats


# ---------------------------------------------------------------------------
# Target token
# ---------------------------------------------------------------------------

def last_overlapping_token(offsets, start: int, end: int) -> int | None:
    """Index of the LAST token whose character range overlaps [start, end): the position
    that has read the whole target word. Zero-width offsets (BOS, specials) never count."""
    hit = None
    for i, (s, e) in enumerate(offsets):
        if e > s and s < end and e > start:
            hit = i
    return hit


DEFAULT_PREFIX = "Here is a sentence: "


def locate_targets(rows: list[dict], tokenizer, skip_leading: int, prefix: str = "") -> dict:
    """Tokenise prefix + text like the dump (default special tokens -> BOS first) and attach
    ids / target position / target token. Rows whose target is unlocatable or earlier than
    `skip_leading` get keep=False. Returns counts.

    The prefix exists because half of WiC's sentences put the target in the first four
    tokens, where the AE never trained (the attention-sink positions the dump skips). A
    short neutral frame moves every target past them; every space sees the same text."""
    n_none = n_early = 0
    # tokens before the sentence's first word (BOS + prefix); the prefix's trailing space
    # merges into the next word's token, so count the prefix without it
    n_pre = len(tokenizer(prefix.rstrip())["input_ids"]) if prefix.strip() else \
        len(tokenizer("")["input_ids"])
    for r in rows:
        enc = tokenizer(prefix + r["text"], return_offsets_mapping=True)
        t = last_overlapping_token(enc["offset_mapping"], r["start"] + len(prefix), r["end"] + len(prefix))
        r["ids"] = enc["input_ids"]
        r["pos"] = -1 if t is None else t
        r["tok"] = -1 if t is None else enc["input_ids"][t]
        r["ctx"] = -1 if t is None else t - n_pre          # sentence tokens of left context
        r["keep"] = t is not None and t >= skip_leading
        n_none += t is None
        n_early += (t is not None and t < skip_leading)
    return {"unlocatable": n_none, "early": n_early}


# ---------------------------------------------------------------------------
# Activations and spaces
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_targets(lm, layer: int, rows: list[dict], batch_size: int, device) -> torch.Tensor:
    """Layer-`layer` residual (float32, CPU) at each row's target position. Right padding:
    with a causal mask, positions before the padding are unaffected by it."""
    from geoae.hooks import SplicingHook
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    out = [None] * len(rows)
    captured: list[torch.Tensor] = []
    hook = SplicingHook(lm, layer)
    hook.activate(lambda hs: (captured.append(hs.detach()) or hs))
    try:
        for b0 in tqdm(range(0, len(order), batch_size), desc="[wsense] forward"):
            idx = order[b0:b0 + batch_size]
            T = max(len(rows[i]["ids"]) for i in idx)
            ids = torch.zeros(len(idx), T, dtype=torch.long)
            att = torch.zeros(len(idx), T, dtype=torch.long)
            for j, i in enumerate(idx):
                L = len(rows[i]["ids"])
                ids[j, :L] = torch.as_tensor(rows[i]["ids"])
                att[j, :L] = 1
            captured.clear()
            lm(input_ids=ids.to(device), attention_mask=att.to(device))
            hs = captured[0].float()
            for j, i in enumerate(idx):
                out[i] = hs[j, rows[i]["pos"]].cpu()
    finally:
        hook.deactivate()
    return torch.stack(out)


@torch.no_grad()
def check_batching(lm, layer: int, rows: list[dict], batch_size: int, device, n: int = 8) -> dict:
    """Batched vs one-at-a-time targets for n rows of different lengths (a padding check).
    bf16 rounding alone gives a relative error around 1e-2; a padding bug gives ~1."""
    sub = rows[:: max(1, len(rows) // n)][:n]
    a = capture_targets(lm, layer, sub, batch_size, device)
    b = torch.cat([capture_targets(lm, layer, [r], 1, device) for r in sub])
    rel = ((a - b).norm(dim=1) / b.norm(dim=1)).max()
    return {"max_abs": float((a - b).abs().max()), "max_rel_l2": float(rel), "n": len(sub)}


@torch.no_grad()
def nearest(q: torch.Tensor, centroids: torch.Tensor, metric: str, chunk: int = 4096) -> np.ndarray:
    from geoae.losses import pairwise_sq_dist
    out = []
    for i in range(0, len(q), chunk):
        out.append(pairwise_sq_dist(q[i:i + chunk], centroids, metric=metric).argmin(1).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def build_spaces(H: torch.Tensor, toks: torch.Tensor, args, device) -> dict:
    """{space: (representation float16 CPU, cluster ids, meta)} for base / bypass / tokmean."""
    from geoae.checkpoint import load_ae_checkpoint
    from geoae.interp.closest_tokens import load_baseline_kmeans
    from geoae.token_bias import TokenBiasLookup
    spaces = {}
    tok = toks.to(device)

    C, mean, std, K, _, _ = load_baseline_kmeans(Path(args.baseline_kmeans), device)
    x = (H.to(device) - mean) / std
    spaces["base"] = (x.half().cpu(), nearest(x, C, "euclidean"), {"K": K, "space": "residual"})

    ae, mean, std, _ = load_ae_checkpoint(args.checkpoint, device, allow_token_bias=True)
    x = (H.to(device) - mean) / std
    zs = [ae.encode(x[i:i + 4096], tok[i:i + 4096] if ae.has_token_bias else None)
          for i in range(0, len(x), 4096)]
    z = torch.cat(zs)
    spaces["bypass"] = (z.half().cpu(), nearest(z, ae.centroids, ae.metric),
                        {"K": ae.n_clusters, "space": "latent", "token_bias": bool(ae.has_token_bias)})
    del ae, z, zs

    if args.tokmean_kmeans:
        C, mean, std, K, _, _ = load_baseline_kmeans(Path(args.tokmean_kmeans), device, allow_erasure=True)
        d = np.load(args.tokmean_kmeans, allow_pickle=True)
        if "token_bias" not in d.files:
            raise SystemExit(f"{args.tokmean_kmeans} has no token_bias field: not a token-mean codebook")
        tb = TokenBiasLookup(str(d["token_bias"]), device)
        x = (H.to(device) - mean) / std - tb(tok)
        spaces["tokmean"] = (x.half().cpu(), nearest(x, C, "euclidean"),
                             {"K": K, "space": "residual - b[token]"})
    return spaces


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cnmi(labels, clusters, n_perm: int, rng) -> float:
    """(NMI - mean NMI over label permutations) / (1 - that mean): the project's
    chance-corrected NMI. Raw NMI rises with the number of clusters a word spreads over."""
    from sklearn.metrics import normalized_mutual_info_score as nmi
    labels, clusters = np.asarray(labels), np.asarray(clusters)
    raw = nmi(labels, clusters)
    null = float(np.mean([nmi(rng.permutation(labels), clusters) for _ in range(n_perm)]))
    return (raw - null) / (1 - null) if null < 1 else float("nan")


def partition_stats(labels, clusters) -> dict:
    labels, clusters = np.asarray(labels), np.asarray(clusters)
    _, cnt = np.unique(clusters, return_counts=True)
    p = cnt / cnt.sum()
    purity = sum(np.bincount(labels[clusters == c]).max() for c in np.unique(clusters)) / len(labels)
    return {"n_clusters": int(len(cnt)), "eff_clusters": float(np.exp(-(p * np.log(p)).sum())),
            "top_cluster_share": float(p.max()), "purity": float(purity)}


def knn_acc(R: torch.Tensor, labels, k: int = 10) -> float:
    """Leave-one-out k-NN majority vote with cosine similarity (ties -> nearest's label)."""
    labels = np.asarray(labels)
    if len(labels) <= k:
        return float("nan")
    Rn = torch.nn.functional.normalize(R.float(), dim=1)
    S = Rn @ Rn.T
    S.fill_diagonal_(-float("inf"))
    nn = S.topk(k, dim=1).indices.numpy()
    hit = 0
    for i, row in enumerate(nn):
        votes = Counter(labels[row])
        best = max(votes.values())
        winners = {l for l, v in votes.items() if v == best}
        pred = next(labels[j] for j in row if labels[j] in winners)
        hit += pred == labels[i]
    return hit / len(labels)


def basic_ci(est: float, boots) -> tuple[float, float]:
    """BASIC (centred) interval: est + quantile(boots - mean(boots))."""
    b = np.asarray([x for x in boots if np.isfinite(x)])
    if len(b) == 0:
        return float("nan"), float("nan")
    d = b - b.mean()
    return float(est + np.quantile(d, 0.025)), float(est + np.quantile(d, 0.975))


def comembership(c1, c2, label) -> dict:
    c1, c2, label = map(np.asarray, (c1, c2, label))
    same = c1 == c2
    p1 = float(same[label == 1].mean()) if (label == 1).any() else float("nan")
    p0 = float(same[label == 0].mean()) if (label == 0).any() else float("nan")
    return {"p_same_cluster_same_sense": p1, "p_same_cluster_diff_sense": p0, "gap": p1 - p0}


def auroc(score, label) -> float:
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label)
    return float(roc_auc_score(label, score)) if 0 < label.sum() < len(label) else float("nan")


def best_threshold(score, label) -> float:
    """Threshold on `score` maximising accuracy of (score >= thr) == label."""
    score, label = np.asarray(score), np.asarray(label)
    cand = np.unique(score)
    accs = [((score >= t) == label).mean() for t in cand]
    return float(cand[int(np.argmax(accs))])


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def analyse_coarsewsd(rows, spaces, args, rng) -> dict:
    idx_by_word = defaultdict(list)
    for i, r in enumerate(rows):
        idx_by_word[r["word"]].append(i)
    names = list(spaces)
    per_word, examples = {}, {}
    for w, idx in idx_by_word.items():
        idx = np.asarray(idx)
        y = np.asarray([rows[i]["sense"] for i in idx])
        if len(np.unique(y)) < 2:
            continue
        sense_names = {rows[i]["sense"]: rows[i]["sense_name"] for i in idx}
        per_word[w] = {"n": int(len(idx)), "senses": {sense_names[int(s)]: int((y == s).sum())
                                                     for s in np.unique(y)}}
        examples[w] = {}
        for s in names:
            R, c, _ = spaces[s]
            cw = c[idx]
            per_word[w][s] = {"cnmi_sense": cnmi(y, cw, args.n_perm, rng), **partition_stats(y, cw),
                              "knn_acc": knn_acc(R[torch.as_tensor(idx)], y)}
            top = Counter(cw.tolist()).most_common(3)
            examples[w][s] = [{"cluster": int(k), "n": int(n),
                               "by_sense": {sense_names[int(sv)]: int(((cw == k) & (y == sv)).sum())
                                            for sv in np.unique(y)}}
                              for k, n in top]
    metrics = ["cnmi_sense", "purity", "top_cluster_share", "eff_clusters", "knn_acc"]
    macro = {s: {m: float(np.nanmean([per_word[w][s][m] for w in per_word])) for m in metrics} for s in names}

    # paired bootstrap over occurrences within each word (kNN excluded: duplicated rows are
    # each other's nearest neighbours, which inflates kNN under resampling)
    boot_m = ["cnmi_sense", "purity", "top_cluster_share"]
    boots = {s: {m: [] for m in boot_m} for s in names}
    for _ in range(args.n_boot):
        acc = {s: {m: [] for m in boot_m} for s in names}
        for w in per_word:
            idx = np.asarray(idx_by_word[w])
            take = idx[rng.integers(0, len(idx), len(idx))]
            y = np.asarray([rows[i]["sense"] for i in take])
            if len(np.unique(y)) < 2:
                continue
            for s in names:
                cw = spaces[s][1][take]
                ps = partition_stats(y, cw)
                acc[s]["cnmi_sense"].append(cnmi(y, cw, args.n_perm_boot, rng))
                acc[s]["purity"].append(ps["purity"])
                acc[s]["top_cluster_share"].append(ps["top_cluster_share"])
        for s in names:
            for m in boot_m:
                boots[s][m].append(float(np.nanmean(acc[s][m])))
    diffs = {}
    for s in names:
        if s == "base":
            continue
        diffs[f"{s}-base"] = {}
        for m in boot_m:
            est = macro[s][m] - macro["base"][m]
            lo, hi = basic_ci(est, np.asarray(boots[s][m]) - np.asarray(boots["base"][m]))
            diffs[f"{s}-base"][m] = {"est": est, "lo": lo, "hi": hi}
        diffs[f"{s}-base"]["words_better_cnmi"] = int(sum(
            per_word[w][s]["cnmi_sense"] > per_word[w]["base"]["cnmi_sense"] for w in per_word))
        diffs[f"{s}-base"]["words_better_knn"] = int(sum(
            per_word[w][s]["knn_acc"] > per_word[w]["base"]["knn_acc"] for w in per_word))
        diffs[f"{s}-base"]["n_words"] = len(per_word)
    return {"per_word": per_word, "macro": macro, "diffs": diffs, "examples": examples}


def analyse_wic(rows, pairs, spaces, args, rng) -> dict:
    side = {(r["pair"], r["side"]): i for i, r in enumerate(rows)}
    keep = [p for p in pairs if (p["pair"], 0) in side and (p["pair"], 1) in side]
    i1 = np.asarray([side[(p["pair"], 0)] for p in keep])
    i2 = np.asarray([side[(p["pair"], 1)] for p in keep])
    y = np.asarray([p["label"] for p in keep])
    split = np.asarray([p["split"] for p in keep])
    minctx = np.minimum([rows[i]["ctx"] for i in i1], [rows[i]["ctx"] for i in i2])
    out = {"n_pairs": int(len(keep)), "per_space": {}, "by_position": {}, "diffs": {}}
    cos, clu = {}, {}
    for s, (R, c, _) in spaces.items():
        a = torch.nn.functional.normalize(R[torch.as_tensor(i1)].float(), dim=1)
        b = torch.nn.functional.normalize(R[torch.as_tensor(i2)].float(), dim=1)
        cos[s] = (a * b).sum(1).numpy()
        clu[s] = (c[i1], c[i2])
        tr, va = split == "train", split == "validation"
        thr = best_threshold(cos[s][tr], y[tr])
        out["per_space"][s] = {**comembership(*clu[s], y), "auroc_cos": auroc(cos[s], y),
                               "acc_threshold": float(((cos[s][va] >= thr) == y[va]).mean()),
                               "threshold": thr}
    # left context = sentence tokens before the target, the smaller of the pair's two
    bins = {"<4": minctx < 4, "4-11": (minctx >= 4) & (minctx < 12), ">=12": minctx >= 12}
    for bname, m in bins.items():
        out["by_position"][bname] = {"n": int(m.sum()), **{
            s: {"gap": comembership(clu[s][0][m], clu[s][1][m], y[m])["gap"],
                "auroc_cos": auroc(cos[s][m], y[m])} for s in spaces}}
    n = len(y)
    for s in spaces:
        if s == "base":
            continue
        bg, ba = [], []
        for _ in range(args.n_boot):
            t = rng.integers(0, n, n)
            g = lambda q: comembership(clu[q][0][t], clu[q][1][t], y[t])["gap"]
            bg.append(g(s) - g("base"))
            ba.append(auroc(cos[s][t], y[t]) - auroc(cos["base"][t], y[t]))
        d = {}
        for m, b in (("gap", bg), ("auroc_cos", ba)):
            est = out["per_space"][s][m] - out["per_space"]["base"][m]
            lo, hi = basic_ci(est, b)
            d[m] = {"est": est, "lo": lo, "hi": hi}
        out["diffs"][f"{s}-base"] = d
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_skip_leading(act_meta: str | None) -> int:
    if not act_meta:
        return 4
    p = Path(act_meta)
    p = p / "meta.json" if p.is_dir() else p
    return int(json.loads(p.read_text()).get("skip_leading", 4))


def cap_per_sense(rows: list[dict], cap: int, rng) -> list[dict]:
    if cap <= 0:
        return rows
    groups = defaultdict(list)
    for r in rows:
        groups[(r["word"], r["sense"])].append(r)
    out = []
    for key in sorted(groups):
        g = groups[key]
        if len(g) > cap:
            g = [g[i] for i in sorted(rng.choice(len(g), cap, replace=False))]
        out.extend(g)
    return out


def print_parse_report(cw_stats, wic_stats):
    if cw_stats:
        print(f"[wsense] CoarseWSD-20: {len(cw_stats)} words")
        for w, st in cw_stats.items():
            print(f"  {w:<8} raw {st['n_raw']:>5}  {dict(st['split_counts'])}  out_of_range "
                  f"{st['out_of_range']}  form_mismatch {st['form_mismatch']} {st['form_examples'][:3]}  "
                  f"senses {list(st['senses'].values())}")
    if wic_stats:
        print(f"[wsense] WiC: {wic_stats['splits']}  span mismatches {wic_stats['mismatch']} "
              f"{wic_stats['examples'][:3]}")


def fmt(d) -> str:
    return f"{d['est']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def print_tables(res: dict):
    if "coarsewsd20" in res:
        cw = res["coarsewsd20"]
        print("\n[wsense] CoarseWSD-20 (macro over words; within-word sense)")
        print(f"  {'metric':<20}{'base':>9}{'bypass':>9}   bypass - base [95% CI]")
        for m in ("cnmi_sense", "purity", "top_cluster_share", "eff_clusters", "knn_acc"):
            d = cw["diffs"]["bypass-base"].get(m)
            print(f"  {m:<20}{cw['macro']['base'][m]:>9.3f}{cw['macro']['bypass'][m]:>9.3f}   "
                  f"{fmt(d) if d else ''}")
        db = cw["diffs"]["bypass-base"]
        print(f"  words where bypass > base: cNMI {db['words_better_cnmi']}/{db['n_words']}, "
              f"kNN {db['words_better_knn']}/{db['n_words']}")
        if "tokmean-base" in cw["diffs"]:
            print(f"  control (tokmean, no encoder): cnmi_sense {cw['macro']['tokmean']['cnmi_sense']:.3f}"
                  f"  (tokmean - base {fmt(cw['diffs']['tokmean-base']['cnmi_sense'])})")
    if "wic" in res:
        wc = res["wic"]
        print(f"\n[wsense] WiC ({wc['n_pairs']} pairs)")
        print(f"  {'metric':<28}{'base':>9}{'bypass':>9}   bypass - base [95% CI]")
        for m in ("p_same_cluster_same_sense", "p_same_cluster_diff_sense", "gap", "auroc_cos", "acc_threshold"):
            d = wc["diffs"]["bypass-base"].get(m)
            print(f"  {m:<28}{wc['per_space']['base'][m]:>9.3f}{wc['per_space']['bypass'][m]:>9.3f}   "
                  f"{fmt(d) if d else ''}")
        for b, v in wc["by_position"].items():
            print(f"  left context {b:>4} sentence tokens (n={v['n']}): gap {v['base']['gap']:+.3f} -> "
                  f"{v['bypass']['gap']:+.3f}, AUROC {v['base']['auroc_cos']:.3f} -> {v['bypass']['auroc_cos']:.3f}")
        if "tokmean" in wc["per_space"]:
            t = wc["per_space"]["tokmean"]
            print(f"  control (tokmean, no encoder): gap {t['gap']:+.3f}, AUROC {t['auroc_cos']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="token-bypass AE checkpoint (.pt)")
    ap.add_argument("--baseline_kmeans", default=None, help="base: balanced k-means npz, same K")
    ap.add_argument("--tokmean_kmeans", default=None, help="optional control: k-means fit on x - b[token]")
    ap.add_argument("--datasets", default="coarsewsd20,wic")
    ap.add_argument("--data_dir", default="data/wsd")
    ap.add_argument("--act_meta", default=None, help="dump dir or meta.json, for skip_leading (default 4)")
    ap.add_argument("--max_per_sense", type=int, default=400, help="CoarseWSD rows per (word, sense); 0 = all")
    ap.add_argument("--max_sentences", type=int, default=0, help="smoke: cap total sentences (0 = all)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_boot", type=int, default=500)
    ap.add_argument("--n_perm", type=int, default=20, help="label permutations for cNMI point estimates")
    ap.add_argument("--n_perm_boot", type=int, default=3, help="permutations per bootstrap resample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save_rows", dest="save_rows", action="store_true", default=True)
    ap.add_argument("--no_save_rows", dest="save_rows", action="store_false")
    ap.add_argument("--check_batching", action="store_true", help="compare batched vs unbatched targets")
    ap.add_argument("--download_only", action="store_true")
    ap.add_argument("--parse_only", action="store_true", help="load + validate data; no model is loaded")
    ap.add_argument("--tokenizer", default=None, help="--parse_only: also count early/unlocatable targets")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX,
                    help="neutral text before every sentence ('' = none); see locate_targets")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    data_dir = Path(args.data_dir)

    if args.download_only:
        download_data(data_dir, datasets)
        return

    rng = np.random.default_rng(args.seed)
    cw_rows, cw_stats, wic_rows, wic_pairs, wic_stats = [], {}, [], [], {}
    if "coarsewsd20" in datasets:
        cw_rows, cw_stats = load_coarsewsd(data_dir)
        cw_rows = cap_per_sense(cw_rows, args.max_per_sense, rng)
    if "wic" in datasets:
        wic_rows, wic_pairs, wic_stats = load_wic(data_dir)

    skip = read_skip_leading(args.act_meta)
    if args.parse_only:
        print_parse_report(cw_stats, wic_stats)
        print(f"[wsense] after cap {args.max_per_sense}/sense: CoarseWSD rows {len(cw_rows)}, "
              f"WiC sentences {len(wic_rows)}")
        if args.tokenizer:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(args.tokenizer)
            for name, rr in (("coarsewsd20", cw_rows), ("wic", wic_rows)):
                if rr:
                    print(f"[wsense] {name}: {locate_targets(rr, tk, skip, args.prefix)} of {len(rr)} "
                          f"(skip_leading {skip})")
        return

    if not (args.checkpoint and args.baseline_kmeans and args.out):
        ap.error("--checkpoint, --baseline_kmeans and --out are required for a run")
    ckpt_cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    model_name = ckpt_cfg["extraction"]["model_name"]
    layer = int(ckpt_cfg["data"]["target_layer"])
    d = np.load(args.baseline_kmeans, allow_pickle=True)
    base_layer = int(d["layer"]) if "layer" in d.files else int(d["target_layer"])
    if base_layer != layer or ("model_name" in d.files and str(d["model_name"]) != model_name):
        raise SystemExit(f"base npz is {d.get('model_name', '?')} L{base_layer}, AE is {model_name} L{layer}")

    from transformers import AutoTokenizer
    from geoae.checkpoint import load_lm
    from geoae.seeding import seed_everything
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    loc = {}
    for name, rr in (("coarsewsd20", cw_rows), ("wic", wic_rows)):
        if rr:
            loc[name] = locate_targets(rr, tokenizer, skip, args.prefix)
            print(f"[wsense] {name}: {len(rr)} sentences, dropped {loc[name]} (skip_leading {skip})")
    cw_rows = [r for r in cw_rows if r["keep"]]
    wic_rows = [r for r in wic_rows if r["keep"]]
    if args.max_sentences:     # smoke: a random subset (a prefix would be one word / one split)
        n_cw = args.max_sentences // 2 if wic_rows else args.max_sentences
        if len(cw_rows) > n_cw:
            cw_rows = [cw_rows[i] for i in sorted(rng.choice(len(cw_rows), n_cw, replace=False))]
        keep_pairs = set()
        n_pairs = (args.max_sentences - len(cw_rows)) // 2
        if wic_rows and n_pairs > 0:
            pids = sorted({r["pair"] for r in wic_rows})
            keep_pairs = set(rng.choice(pids, min(n_pairs, len(pids)), replace=False).tolist())
        wic_rows = [r for r in wic_rows if r["pair"] in keep_pairs]
    rows = cw_rows + wic_rows

    print(f"[wsense] {model_name} L{layer} on {device}; {len(rows)} target occurrences")
    lm = load_lm(model_name, device_map="auto")
    batching_diff = check_batching(lm, layer, rows, args.batch_size, device) if args.check_batching else None
    if batching_diff is not None:
        print(f"[wsense] batched vs unbatched over {batching_diff['n']} targets: max |diff| "
              f"{batching_diff['max_abs']:.3e}, max relative L2 {batching_diff['max_rel_l2']:.3e}")
    H = capture_targets(lm, layer, rows, args.batch_size, device)
    del lm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    toks = torch.as_tensor([r["tok"] for r in rows], dtype=torch.long)
    spaces = build_spaces(H, toks, args, device)

    res = {"meta": {"checkpoint": args.checkpoint, "baseline_kmeans": args.baseline_kmeans,
                    "tokmean_kmeans": args.tokmean_kmeans, "model": model_name, "layer": layer,
                    "K": {s: v[2]["K"] for s, v in spaces.items()},
                    "spaces": {s: v[2] for s, v in spaces.items()},
                    "datasets": datasets, "skip_leading": skip, "prefix": args.prefix, "dropped": loc,
                    "max_per_sense": args.max_per_sense, "seed": args.seed, "n_boot": args.n_boot,
                    "batching_max_abs_diff": batching_diff,
                    "notes": ["CoarseWSD-20 text is lowercased and pre-tokenised; fed as-is",
                              "causal model: the target representation sees only left context",
                              "intervals are basic (centred) bootstrap intervals"]}}
    nc = len(cw_rows)
    sub = lambda a, s, e: {k: (v[0][s:e], v[1][s:e], v[2]) for k, v in a.items()}
    if cw_rows:
        res["meta"]["coarsewsd_stats"] = {w: {k: (dict(v) if isinstance(v, Counter) else v)
                                              for k, v in st.items()} for w, st in cw_stats.items()}
        res["coarsewsd20"] = analyse_coarsewsd(cw_rows, sub(spaces, 0, nc), args, rng)
    if wic_rows:
        res["meta"]["wic_stats"] = wic_stats
        res["wic"] = analyse_wic(wic_rows, wic_pairs, sub(spaces, nc, len(rows)), args, rng)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"[wsense] wrote {out}")
    if args.save_rows:
        npz = out.with_name(out.stem + "_rows.npz")
        np.savez_compressed(npz, dataset=np.asarray([r["dataset"] for r in rows]),
                            word=np.asarray([str(r["word"]) for r in rows]),
                            sense=np.asarray([r.get("sense", -1) for r in rows]),
                            pair=np.asarray([r.get("pair", -1) for r in rows]),
                            label=np.asarray([r.get("label", -1) for r in rows]),
                            split=np.asarray([r["split"] for r in rows]),
                            pos=np.asarray([r["pos"] for r in rows]),
                            ctx=np.asarray([r["ctx"] for r in rows]),
                            tok=toks.numpy(),
                            **{f"cluster_{s}": v[1] for s, v in spaces.items()})
        print(f"[wsense] wrote {npz}")
    print_tables(res)


if __name__ == "__main__":
    main()
