"""
ERASER Movie Reviews rationale benchmark (DeYoung et al. 2020): do token attributions read
from the base residual stream, the GeoAE latent, or a k-means codebook point at the words
humans marked as evidence?

Readouts (one soft score per ERASER token, positive = supports POS):
  h, h_tok, z     linear probe (doc-mean features, VAL-chosen C) applied per token:
                  s_t = w . ((r_t - mu) / sd);  mean_t s_t + b = the doc logit
                  h = normalised residual, h_tok = h - b[tok], z = ae.encode(h, tok)
  km_plain, kmt_plain, km_bal, kmt_bal, ae   cluster log-odds weights from TRAIN-doc counts
  random          uniform scores (n_random seeds)

Metrics (test split, per doc): plausibility = AUPRC / top-k token F1 against the human
rationale; faithfulness = AOPC comprehensiveness / sufficiency with the SAME Llama as
predictor (prompted zero-shot). Paired doc-bootstrap + Wilcoxon between readouts.

    python -u -m geoae.analysis.eraser_movies --data_dir data/eraser/movies \
        --ae <ckpt> --codebooks km_plain=<npz>,kmt_plain=<npz> --out out.json
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np

POS, NEG = 1, 0


# ---------------------------------------------------------------------------
# Data and alignment (pure python / numpy)
# ---------------------------------------------------------------------------

def load_doc(data_dir, docid):
    """Lines of whitespace tokens (empty lines dropped; flat token indices are unaffected)."""
    lines = [ln.split() for ln in Path(data_dir, "docs", docid).read_text().split("\n")]
    return [ln for ln in lines if ln]


def flat_tokens(lines):
    return [t for ln in lines for t in ln]


def build_text(lines, keep=None):
    """Lines joined by '\\n', tokens by ' '. `keep` = bool mask over flat tokens (None = all);
    order and line structure are preserved and lines that end up empty are dropped."""
    out, i = [], 0
    for ln in lines:
        toks = [t for j, t in enumerate(ln) if keep is None or keep[i + j]]
        i += len(ln)
        if toks:
            out.append(" ".join(toks))
    return "\n".join(out)


def text_with_spans(lines):
    """(text, spans) where spans[i] = (start_char, end_char) of flat ERASER token i."""
    parts, spans, pos = [], [], 0
    for li, ln in enumerate(lines):
        if li:
            pos += 1                       # the '\n'
        for ti, t in enumerate(ln):
            if ti:
                pos += 1                   # the ' '
            spans.append((pos, pos + len(t)))
            pos += len(t)
        parts.append(" ".join(ln))
    return "\n".join(parts), spans


def rationale_mask(evidences, n_tokens, docid=None):
    """Union of every [start_token, end_token) evidence span (end exclusive)."""
    m = np.zeros(n_tokens, dtype=bool)
    for group in evidences:
        for ev in group:
            if docid is not None and ev.get("docid", docid) != docid:
                continue
            m[ev["start_token"]:ev["end_token"]] = True
    return m


def check_evidence_convention(evidences, toks):
    """(n_checked, n_mismatch): " ".join(toks[start:end]) must equal the evidence text."""
    n = bad = 0
    for group in evidences:
        for ev in group:
            n += 1
            if " ".join(toks[ev["start_token"]:ev["end_token"]]) != ev["text"]:
                bad += 1
    return n, bad


def align_offsets(text, offsets, spans):
    """Per LM token: (lo, hi) range of ERASER tokens whose char span overlaps the LM token's
    span (leading/trailing whitespace stripped first); None if nothing overlaps (BOS, '\\n')."""
    starts = [s for s, _ in spans]
    ends = [e for _, e in spans]
    out = []
    for s, e in offsets:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s >= e:
            out.append(None)
            continue
        lo = bisect.bisect_right(ends, s)       # first ERASER token ending after s
        hi = bisect.bisect_left(starts, e)      # first ERASER token starting at/after e
        out.append((lo, hi) if hi > lo else None)
    return out


def alignment_pairs(align):
    """-> (kept_pos, pair_row, pair_tok): kept LM positions, and (row-in-kept, eraser token) pairs."""
    kept_pos, pr, pt = [], [], []
    for p, a in enumerate(align):
        if a is None:
            continue
        r = len(kept_pos)
        kept_pos.append(p)
        for t in range(a[0], a[1]):
            pr.append(r)
            pt.append(t)
    return (np.array(kept_pos, dtype=np.int64), np.array(pr, dtype=np.int64), np.array(pt, dtype=np.int64))


def aggregate_to_tokens(row_scores, pair_row, pair_tok, n_tokens):
    """ERASER-token score = mean over its LM subwords' row scores. Returns (scores, counts)."""
    cnt = np.bincount(pair_tok, minlength=n_tokens).astype(np.float64)
    s = np.bincount(pair_tok, weights=np.asarray(row_scores, dtype=np.float64)[pair_row], minlength=n_tokens)
    return s / np.maximum(cnt, 1), cnt


# ---------------------------------------------------------------------------
# Readout arithmetic
# ---------------------------------------------------------------------------

def cluster_weights(n_pos, n_neg):
    """w_k = log((n_pos+1)/(N_pos+K)) - log((n_neg+1)/(N_neg+K)); positive = POS-leaning."""
    n_pos, n_neg = np.asarray(n_pos, np.float64), np.asarray(n_neg, np.float64)
    K = len(n_pos)
    return np.log((n_pos + 1) / (n_pos.sum() + K)) - np.log((n_neg + 1) / (n_neg.sum() + K))


def probe_row_scores(R, w, mu, sd):
    """s_t = w . ((r_t - mu)/sd), bias excluded. mean_t s_t + b == the doc logit on mean(R)."""
    return ((np.asarray(R, np.float64) - mu) / sd) @ w


def rank_order(score, rng):
    """Indices from highest to lowest score; ties broken by a seeded random key (not position)."""
    return np.lexsort((rng.random(len(score)), -np.asarray(score, np.float64)))


def oriented_auprc(score, mask, sign):
    from sklearn.metrics import average_precision_score
    if mask.sum() == 0 or mask.all():
        return float("nan")
    return float(average_precision_score(mask, sign * np.asarray(score, np.float64)))


def topk_prf(score, mask, sign, k, rng):
    order = rank_order(sign * np.asarray(score, np.float64), rng)[:k]
    sel = np.zeros(len(mask), bool)
    sel[order] = True
    tp = float((sel & mask).sum())
    p = tp / max(sel.sum(), 1)
    r = tp / max(mask.sum(), 1)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def top_mask(score_toward_yhat, frac, rng):
    """Bool mask of the top ceil(frac * n) tokens by score (random tie-break)."""
    n = len(score_toward_yhat)
    k = min(n, max(1, math.ceil(frac * n)))
    m = np.zeros(n, bool)
    m[rank_order(score_toward_yhat, rng)[:k]] = True
    return m


def aopc(p_full, p_variants):
    """Comprehensiveness/sufficiency per bin: p_full - p_variant; AOPC = mean over bins."""
    per_bin = p_full - np.asarray(p_variants, np.float64)
    return per_bin, float(per_bin.mean())


def paired_stats(a, b, n_boot, rng):
    """a - b over docs (finite pairs): mean diff, doc-bootstrap 95% CI, Wilcoxon signed-rank p."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    if len(d) == 0:
        return dict(n=0, mean_diff=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"), wilcoxon_p=float("nan"))
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bm = d[idx].mean(1)
    try:
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    return dict(n=int(len(d)), mean_diff=float(d.mean()), ci_lo=float(np.percentile(bm, 2.5)),
                ci_hi=float(np.percentile(bm, 97.5)), wilcoxon_p=p)


# ---------------------------------------------------------------------------
# LM helpers
# ---------------------------------------------------------------------------

def prompt_for(text):
    return f"Review: {text}\nIs this movie review positive or negative?\nAnswer:"


def tokenise_docs(tok, texts, max_length):
    enc = tok(texts, return_offsets_mapping=True, add_special_tokens=True)
    for ids in enc["input_ids"]:
        if len(ids) > max_length:
            raise SystemExit(f"doc of {len(ids)} LM tokens exceeds --max_length {max_length} (no truncation allowed)")
    return enc["input_ids"], enc["offset_mapping"]


class PromptScorer:
    """p(POS) for prompts, batched by length, with an on-disk cache keyed by prompt hash."""

    def __init__(self, lm, tok, dev, cache_path, batch_size):
        import torch
        self.lm, self.tok, self.dev, self.bs = lm, tok, dev, batch_size
        self.path = Path(cache_path)
        self.cache = pickle.load(open(self.path, "rb")) if self.path.exists() else {}
        ids = []
        self.single_token = True
        for w in (" positive", " negative"):
            t = tok.encode(w, add_special_tokens=False)
            if len(t) != 1:
                self.single_token = False
                print(f"[eraser] WARNING {w!r} is {len(t)} tokens; using the first")
            ids.append(t[0])
        self.pos_id, self.neg_id = ids
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self.head = lm.get_output_embeddings()
        self._last = None

        def pre(mod, args):
            h = args[0]
            idx = self._last.to(h.device)
            return (h[torch.arange(h.shape[0], device=h.device), idx][:, None, :],) + tuple(args[1:])
        self.head.register_forward_pre_hook(pre)
        self.n_forward = 0

    @staticmethod
    def key(prompt):
        return hashlib.sha1(prompt.encode()).hexdigest()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.cache, open(self.path, "wb"))

    def p_pos(self, prompts):
        import torch
        keys = [self.key(p) for p in prompts]
        todo = sorted({k: p for k, p in zip(keys, prompts) if k not in self.cache}.items(), key=lambda kp: len(kp[1]))
        if todo:
            enc = self.tok([p for _, p in todo], add_special_tokens=True)["input_ids"]
            order = sorted(range(len(todo)), key=lambda i: len(enc[i]))
            for i in range(0, len(order), self.bs):
                sel = order[i:i + self.bs]
                seqs = [enc[j] for j in sel]
                T = max(map(len, seqs))
                ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
                mask = torch.zeros((len(seqs), T), dtype=torch.long)
                for b, s in enumerate(seqs):
                    ids[b, :len(s)] = torch.tensor(s)
                    mask[b, :len(s)] = 1
                self._last = torch.tensor([len(s) - 1 for s in seqs])
                with torch.no_grad():
                    lg = self.lm(input_ids=ids.to(self.dev), attention_mask=mask.to(self.dev), use_cache=False).logits
                two = lg[:, 0, [self.pos_id, self.neg_id]].float()
                p = torch.softmax(two, -1)[:, 0].cpu().numpy()
                for j, pj in zip(sel, p):
                    self.cache[todo[j][0]] = float(pj)
                self.n_forward += len(sel)
        return np.array([self.cache[k] for k in keys])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_split(data_dir, split, limit):
    rows = [json.loads(l) for l in open(Path(data_dir) / f"{split}.jsonl")]
    if limit:   # files are sorted by class: take a class-balanced subset (alternating)
        pos = [r for r in rows if r["classification"] == "POS"]
        neg = [r for r in rows if r["classification"] != "POS"]
        rows = [r for pair in zip(neg, pos) for r in pair][:limit]
    docs = []
    for r in rows:
        lines = load_doc(data_dir, r["annotation_id"])
        toks = flat_tokens(lines)
        text, spans = text_with_spans(lines)
        n_ev, n_bad = check_evidence_convention(r["evidences"], toks)
        docs.append(dict(id=r["annotation_id"], y=POS if r["classification"] == "POS" else NEG, lines=lines,
                         n=len(toks), text=text, spans=spans, mask=rationale_mask(r["evidences"], len(toks), r["annotation_id"]),
                         n_ev=n_ev, n_bad=n_bad))
    return docs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--ae", required=True)
    ap.add_argument("--codebooks", default="", help="name=npz,... (the AE codebook is added as 'ae')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--limit_test", type=int, default=0)
    ap.add_argument("--bins", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.2, 0.5])
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--min_lm_acc", type=float, default=0.8)
    ap.add_argument("--Cs", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1, 10])
    args = ap.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from geoae.checkpoint import load_lm
    from geoae.extraction.sampled import forward_batch
    from geoae.extraction.sampled import register_capture
    from geoae.evaluation.cohorts import ae_fingerprint
    from geoae.evaluation.cluster_steering import Arm

    t00 = time.time()
    timings = {}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # ---- data ---------------------------------------------------------------
    t0 = time.time()
    splits = {s: load_split(args.data_dir, s, lim) for s, lim in
              (("train", args.limit_train), ("val", args.limit_val), ("test", args.limit_test))}
    n_ev = sum(d["n_ev"] for v in splits.values() for d in v)
    n_bad = sum(d["n_bad"] for v in splits.values() for d in v)
    print("[eraser] docs " + ", ".join(f"{s}={len(v)}" for s, v in splits.items())
          + f"; evidence text mismatches {n_bad}/{n_ev}")
    if n_bad:
        print("[eraser] WARNING evidence index convention does not reproduce the evidence text")
    rbar = float(np.mean([d["mask"].mean() for d in splits["train"]]))
    print(f"[eraser] mean human rationale share on train r_bar = {rbar:.4f}; test = "
          f"{np.mean([d['mask'].mean() for d in splits['test']]):.4f}")
    timings["data"] = time.time() - t0

    # ---- model, arms --------------------------------------------------------
    t0 = time.time()
    lm = load_lm(args.model_name, device=dev)
    from geoae.checkpoint import load_tokenizer
    tok = load_tokenizer(args.model_name)
    assert tok.is_fast, "fast tokenizer needed for offset mapping"
    ae_arm = Arm("ae", args.ae, dev)
    cb_arms = [Arm(*spec.split("=", 1), dev) for spec in args.codebooks.split(",") if spec]
    arms = cb_arms + [ae_arm]
    mean, std = ae_arm.mean, ae_arm.std
    for a in arms:
        if not (torch.allclose(a.mean, mean) and torch.allclose(a.std, std)):
            raise SystemExit(f"arm {a.name} uses different norm stats than the AE")
    ae = ae_arm.ae
    tb_arm = next((a for a in cb_arms if a.tb is not None), None)
    tb_fn = tb_arm.tb if tb_arm is not None else (ae.token_bias if ae.has_token_bias else None)
    tb_src = f"{tb_arm.name} npz token_bias" if tb_arm is not None else ("AE table" if tb_fn else "none")
    print(f"[eraser] h_tok token-bias table: {tb_src}")
    timings["load"] = time.time() - t0

    scorer = PromptScorer(lm, tok, dev, cache / f"p_cache_{args.model_name.replace('/', '_')}.pkl", args.batch_size)
    meta = dict(args=vars(args), r_bar=rbar, evidence_mismatch=[n_bad, n_ev], token_bias_source=tb_src,
                ae_fingerprint=ae_fingerprint(ae), n_docs={s: len(v) for s, v in splits.items()},
                single_token_answers=scorer.single_token)

    # ---- LM zero-shot accuracy on test (stop early if the predictor is bad) ----
    t0 = time.time()
    test = splits["test"]
    p_full = scorer.p_pos([prompt_for(d["text"]) for d in test])
    yhat = (p_full > 0.5).astype(int)
    lm_acc = float(np.mean(yhat == np.array([d["y"] for d in test])))
    meta["lm_test_acc"] = lm_acc
    print(f"[eraser] LM zero-shot test accuracy {lm_acc:.3f} on {len(test)} docs")
    timings["lm_full"] = time.time() - t0
    if lm_acc < args.min_lm_acc:
        scorer.save()
        raise SystemExit(f"LM test accuracy {lm_acc:.3f} < {args.min_lm_acc}: STOP (not tuning further)")
    p_yhat_full = np.where(yhat == POS, p_full, 1 - p_full)

    # ---- pass 1: LM forward over prompt-free text ---------------------------
    t0 = time.time()
    key = hashlib.sha1(json.dumps([args.model_name, args.layer, str(args.ae), args.codebooks, meta["ae_fingerprint"],
                                   args.limit_train, args.limit_val, args.limit_test,
                                   args.max_length]).encode()).hexdigest()[:12]
    p1_path = cache / f"pass1_{key}.pkl"
    if p1_path.exists():
        print(f"[eraser] pass 1 loaded from {p1_path}")
        P1 = pickle.load(open(p1_path, "rb"))
    else:
        P1 = run_pass1(lm, tok, splits, arms, ae_arm, tb_fn, mean, std, args, dev, forward_batch, register_capture)
        pickle.dump(P1, open(p1_path, "wb"), protocol=4)
    timings["pass1"] = time.time() - t0
    meta["alignment"] = P1["align_stats"]
    meta["max_lm_len"] = P1["align_stats"]["max_lm_len"]
    print(f"[eraser] alignment: {P1['align_stats']}")
    if P1["align_stats"]["eraser_tokens_unaligned"]:
        print("[eraser] WARNING some ERASER tokens got no subword (they will score 0)")

    # ---- readouts -----------------------------------------------------------
    t0 = time.time()
    spaces = [s for s in ("h", "h_tok", "z") if s in P1["feat"]["train"]]
    ytr = np.array([d["y"] for d in splits["train"]])
    yva = np.array([d["y"] for d in splits["val"]])
    yte = np.array([d["y"] for d in splits["test"]])
    probes, meta["chosen_C"] = {}, {}
    acc = {}
    for sp in spaces:
        Ftr, Fva, Fte = (P1["feat"][s][sp] for s in ("train", "val", "test"))
        mu = Ftr.mean(0)
        sd = np.maximum(Ftr.std(0), 1e-6)
        n = lambda F: (F - mu) / sd
        best = None
        for C in args.Cs:
            clf = LogisticRegression(C=C, max_iter=5000).fit(n(Ftr), ytr)
            va = float((clf.predict(n(Fva)) == yva).mean())
            print(f"[eraser] probe {sp} C={C:g} val acc {va:.3f}")
            if best is None or va > best[0]:
                best = (va, C, clf)
        va, C, clf = best
        meta["chosen_C"][sp] = C
        probes[sp] = (clf.coef_[0].astype(np.float64), float(clf.intercept_[0]), mu, sd)
        acc[sp] = (va, float((clf.predict(n(Fte)) == yte).mean()))
    K_of = {a.name: a.C.shape[0] for a in arms}
    cw = {}
    for a in arms:
        npos = np.zeros(K_of[a.name])
        nneg = np.zeros(K_of[a.name])
        for d, asg in zip(splits["train"], P1["assign"]["train"][a.name]):
            np.add.at(npos if d["y"] == POS else nneg, asg, 1)
        cw[a.name] = cluster_weights(npos, nneg)
        va = np.mean([(np.mean(cw[a.name][x]) > 0) == (d["y"] == POS) for d, x in zip(splits["val"], P1["assign"]["val"][a.name])])
        te = np.mean([(np.mean(cw[a.name][x]) > 0) == (d["y"] == POS) for d, x in zip(splits["test"], P1["assign"]["test"][a.name])])
        acc[a.name] = (float(va), float(te))
    acc["random"] = (float("nan"), float("nan"))

    # per-test-doc ERASER-token scores for every readout
    names = spaces + [a.name for a in arms]
    scores = {nm: [] for nm in names}
    sign_own = {nm: [] for nm in names}
    R = P1["test_rows"]
    Wt = {sp: (torch.as_tensor(probes[sp][0], dtype=torch.float32, device=dev),
               torch.as_tensor(probes[sp][2], dtype=torch.float32, device=dev),
               torch.as_tensor(probes[sp][3], dtype=torch.float32, device=dev)) for sp in spaces}
    for i, d in enumerate(test):
        s0, s1 = R["offsets"][i], R["offsets"][i + 1]
        x = torch.from_numpy(R["x"][s0:s1]).to(dev).float()
        tk = torch.from_numpy(R["tok"][s0:s1]).to(dev)
        pr, pt = P1["pairs"]["test"][i]
        for sp in spaces:
            with torch.no_grad():
                r = x if sp == "h" else (x - tb_fn(tk) if sp == "h_tok" else ae_arm.coords(x, tk))
                w, mu_t, sd_t = Wt[sp]
                rs = (((r - mu_t) / sd_t) @ w).double().cpu().numpy()
            sc, _ = aggregate_to_tokens(rs, pr, pt, d["n"])
            scores[sp].append(sc)
            sign_own[sp].append(1.0 if rs.mean() + probes[sp][1] > 0 else -1.0)
        for a in arms:
            rs = cw[a.name][P1["assign"]["test"][a.name][i]]
            sc, _ = aggregate_to_tokens(rs, pr, pt, d["n"])
            scores[a.name].append(sc)
            sign_own[a.name].append(1.0 if rs.mean() > 0 else -1.0)
    rr = np.random.default_rng(args.seed + 1)
    rand_names = [f"random{j}" for j in range(args.n_random)]
    for rn in rand_names:
        scores[rn] = [rr.random(d["n"]) for d in test]
        sign_own[rn] = [1.0] * len(test)
    timings["readouts"] = time.time() - t0

    # ---- metrics ------------------------------------------------------------
    t0 = time.time()
    res = {}
    mrng = np.random.default_rng(args.seed + 2)
    for nm in scores:
        auprc, pp, rc, f1 = [], [], [], []
        for i, d in enumerate(test):
            auprc.append(oriented_auprc(scores[nm][i], d["mask"], sign_own[nm][i]))
            k = max(1, math.ceil(rbar * d["n"]))
            p_, r_, f_ = topk_prf(scores[nm][i], d["mask"], sign_own[nm][i], k, mrng)
            pp.append(p_); rc.append(r_); f1.append(f_)
        res[nm] = dict(auprc=auprc, prec=pp, rec=rc, f1=f1)
    timings["plausibility"] = time.time() - t0

    t0 = time.time()
    nb = len(args.bins)
    for nm in scores:
        comp = np.zeros((len(test), nb))
        suff = np.zeros((len(test), nb))
        prompts_c, prompts_s = [], []
        for i, d in enumerate(test):
            tw = scores[nm][i] * (1.0 if yhat[i] == POS else -1.0)
            for b in args.bins:
                m = top_mask(tw, b, mrng)
                prompts_c.append(prompt_for(build_text(d["lines"], ~m)))
                prompts_s.append(prompt_for(build_text(d["lines"], m)))
        pc = scorer.p_pos(prompts_c).reshape(len(test), nb)
        ps = scorer.p_pos(prompts_s).reshape(len(test), nb)
        for i in range(len(test)):
            fy = (lambda p: p) if yhat[i] == POS else (lambda p: 1 - p)
            comp[i] = p_yhat_full[i] - fy(pc[i])
            suff[i] = p_yhat_full[i] - fy(ps[i])
        res[nm]["comp_bins"] = comp.tolist()
        res[nm]["suff_bins"] = suff.tolist()
        res[nm]["aopc_comp"] = comp.mean(1).tolist()
        res[nm]["aopc_suff"] = suff.mean(1).tolist()
        scorer.save()
        print(f"[eraser] faithfulness done for {nm} ({scorer.n_forward} forwards so far)")
    timings["faithfulness"] = time.time() - t0

    # ---- aggregate: random mean over seeds, paired comparisons ---------------
    metrics = ("auprc", "aopc_comp", "aopc_suff")
    res["random"] = {k: np.mean([res[r][k] for r in rand_names], 0).tolist()
                     for k in ("auprc", "prec", "rec", "f1", "aopc_comp", "aopc_suff")}
    for r in rand_names:
        del res[r]
    pairs = [("z", "h"), ("z", "h_tok")] + [("ae", c) for c in ("km_plain", "kmt_plain", "km_bal", "kmt_bal")]
    pairs += [(nm, "random") for nm in names if nm in res]
    paired = {}
    for a, b in pairs:
        if a in res and b in res:
            paired[f"{a} vs {b}"] = {m: paired_stats(res[a][m], res[b][m], args.n_boot, rng) for m in metrics}
    timings["total"] = time.time() - t00
    meta["timings_s"] = timings
    meta["lm_forwards_prompted"] = scorer.n_forward
    summary = {}
    for nm in res:
        summary[nm] = dict(val_acc=acc[nm][0], test_acc=acc[nm][1],
                           **{k: float(np.nanmean(res[nm][k])) for k in ("auprc", "prec", "rec", "f1", "aopc_comp", "aopc_suff")})
    if torch.cuda.is_available():
        meta["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(dict(meta=meta, summary=summary, per_doc=res, paired=paired,
                   test_ids=[d["id"] for d in test], p_full=p_full.tolist(), yhat=yhat.tolist()),
              open(args.out, "w"))
    print(f"\n{'readout':10s} {'val':>6s} {'test':>6s} {'AUPRC':>7s} {'F1':>6s} {'comp+':>7s} {'suff-':>7s}")
    for nm, s in summary.items():
        print(f"{nm:10s} {s['val_acc']:6.3f} {s['test_acc']:6.3f} {s['auprc']:7.3f} {s['f1']:6.3f} "
              f"{s['aopc_comp']:7.3f} {s['aopc_suff']:7.3f}")
    print("\npaired (mean diff [95% CI], Wilcoxon p):")
    for k, v in paired.items():
        print(f"  {k:20s} " + "  ".join(f"{m}: {v[m]['mean_diff']:+.3f} [{v[m]['ci_lo']:+.3f},{v[m]['ci_hi']:+.3f}] p={v[m]['wilcoxon_p']:.3g}"
                                        for m in metrics))
    print(f"\n[eraser] timings {json.dumps({k: round(v, 1) for k, v in timings.items()})}")
    if torch.cuda.is_available():
        print(f"[eraser] peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def run_pass1(lm, tok, splits, arms, ae_arm, tb_fn, mean, std, args, dev, forward_batch, register_capture):
    import torch
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    captured, handles = register_capture(lm, [args.layer])
    feat = {s: {} for s in splits}
    assign = {s: {a.name: [None] * len(splits[s]) for a in arms} for s in splits}
    pairs = {s: [None] * len(splits[s]) for s in splits}
    tb_ae = ae_arm.ae.token_bias if ae_arm.ae.has_token_bias else None
    tb_diff = 0.0
    stats = dict(max_lm_len=0, eraser_tokens=0, eraser_tokens_unaligned=0, lm_tokens=0, lm_tokens_kept=0)
    test_x, test_tok = [], []
    for split, docs in splits.items():
        ids_all, off_all = tokenise_docs(tok, [d["text"] for d in docs], args.max_length)
        al = []
        for d, off in zip(docs, off_all):
            kp, pr, pt = alignment_pairs(align_offsets(d["text"], off, d["spans"]))
            al.append((kp, pr, pt))
            cov = np.bincount(pt, minlength=d["n"])
            stats["eraser_tokens"] += d["n"]
            stats["eraser_tokens_unaligned"] += int((cov == 0).sum())
            stats["lm_tokens"] += len(off)
            stats["lm_tokens_kept"] += len(kp)
        stats["max_lm_len"] = max(stats["max_lm_len"], max(map(len, ids_all)))
        F = {k: [None] * len(docs) for k in ("h", "h_tok", "z")}
        order = sorted(range(len(docs)), key=lambda i: len(ids_all[i]))
        rows_x = [None] * len(docs)
        rows_t = [None] * len(docs)
        for bi in range(0, len(order), args.batch_size):
            sel = order[bi:bi + args.batch_size]
            forward_batch(lm, [ids_all[i] for i in sel], pad_id, dev)
            cap = captured[args.layer]
            for b, i in enumerate(sel):
                kp = al[i][0]
                idx = torch.from_numpy(kp).to(dev)
                # the dump stores fp16: round-trip so rows match the codebooks' training distribution
                x = (cap[b, idx].half().float() - mean) / std
                tk = torch.tensor(ids_all[i], device=dev)[idx]
                F["h"][i] = x.mean(0).cpu().numpy()
                if tb_fn is not None:
                    F["h_tok"][i] = (x - tb_fn(tk)).mean(0).cpu().numpy()
                    if tb_ae is not None:
                        tb_diff = max(tb_diff, float((tb_fn(tk) - tb_ae(tk)).abs().max()))
                zs = []
                for c in range(0, len(x), 1024):
                    zs.append(ae_arm.coords(x[c:c + 1024], tk[c:c + 1024]))
                z = torch.cat(zs)
                F["z"][i] = z.mean(0).cpu().numpy()
                for a in arms:
                    if a.kind == "ae":
                        lab = torch.cat([torch.cdist(z[c:c + 1024], a.C).argmin(1) for c in range(0, len(z), 1024)])
                    else:
                        lab = torch.cat([a.assign(x[c:c + 1024], tk[c:c + 1024]) for c in range(0, len(x), 1024)])
                    assign[split][a.name][i] = lab.cpu().numpy().astype(np.int32)
                pairs[split][i] = (al[i][1], al[i][2])
                if split == "test":
                    rows_x[i] = x.half().cpu().numpy()
                    rows_t[i] = tk.cpu().numpy().astype(np.int64)
            if (bi // args.batch_size) % 25 == 0:
                print(f"[eraser] pass1 {split} {bi + len(sel)}/{len(docs)}")
        feat[split] = {k: np.stack(v) for k, v in F.items() if v[0] is not None}
        if split == "test":
            test_x = [rows_x[i] for i in range(len(docs))]
            test_tok = [rows_t[i] for i in range(len(docs))]
    for h in handles:
        h.remove()
    if tb_diff > 1e-3:
        raise SystemExit(f"token-bias table of the k-means npz differs from the AE's by {tb_diff:g}")
    stats["token_bias_max_abs_diff_vs_ae"] = tb_diff if tb_ae is not None and tb_fn is not tb_ae else None
    off = np.cumsum([0] + [len(x) for x in test_x])
    return dict(feat=feat, assign=assign, pairs=pairs, align_stats=stats,
                test_rows=dict(x=np.concatenate(test_x), tok=np.concatenate(test_tok), offsets=off))


if __name__ == "__main__":
    main()
