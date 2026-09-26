"""
Generation-level CONCEPT steering: several handles, one target.

cluster_steer_generate.py showed each codebook's OWN discovered clusters are usable
steering handles, but base and bypass steered DIFFERENT targets (their own content
clusters), so the comparison could not say which codebook is the better handle for a
GIVEN concept. This test picks named concepts (DBpedia-14 classes, bias_in_bios
professions) and builds several handles toward the SAME concept (--handles selects
which; default label,base,bypass), then steers with each at every generated position:

  label    supervised reference: v_c = mu_c - mu_d (class mean minus the class-balanced
           grand mean, in normalised residual space, fit on FIT documents), edited as
           x += alpha * R_c * v_c / ||v_c||  (a fixed direction, constant at every
           position). R_c matches the edit size to the base handle's typical gap (see
           below), so every handle moves by a comparable amount at a given alpha.
  label_z  the SAME supervised direction, but computed in the BYPASS AE's latent: fit
           z = ae.encode(x, tok) on the same FIT tokens, v_z = mu_c^z - mu_d^z, mapped
           to normalised residual space with the decoder WEIGHT only (u = ae.decoder(v_z)
           -- the decoder has no bias, and the token-bypass table only adds back onto an
           actual ae.decode() round trip, which a bare direction never needs -- see
           decode_direction()). Edited the same way: x += alpha * R_c * u / ||u||. Tests
           whether the AE's latent geometry, not just its clusters, carries a better
           supervised direction than the base residual does.
  base     balanced k-means codebook (no encoder): pick the codebook's own clusters that
           are class-c-informative on FIT tokens -- not a hub, with >= min_count class-c
           tokens from >= min_docs distinct documents -- ranked by CLASS-BALANCED PRECISION
           p_ck = f_ck / sum_c' f_c'k (f_ck = class-c token count / class c's total FIT
           tokens), keeping the top_k. Target mean m_c = their (held-out, general-reference)
           member means, weighted by class-c TOKEN COUNT n_ck at each selected cluster (NOT
           by p_ck -- precision only ranks which clusters are eligible). Edit
           x += alpha * (m_c - M[s(pos)])  (same form as cluster_steer_generate).
  bypass   the same construction, with the token-bypass AE's clusters.
  base_dir / bypass_dir
           the SAME cluster target mean m_c^cb (cb = base or bypass) used as a CONSTANT
           direction instead of the base/bypass handle's "move into cluster" translation:
           d_c^cb = m_c^cb - mean over the dataset's concepts that HAVE a cb handle of
           m_c'^cb (mirrors label's class-balanced contrast, but the group being averaged
           is "concepts with a cb handle", not "all kept classes"). Edited exactly like
           label: x += alpha * R_c * d / ||d|| (make_label_edit_fn/label_delta again -- no
           per-position assignment or token ids needed). Isolates whether base/bypass's
           advantage (if any) is the TARGET GEOMETRY the clusters point to, or specifically
           the per-position "snap into the nearest cluster, then translate" mechanism.

Concepts only SELECT which discovered clusters to use (via labelled data); the
target mean itself is the GENERAL reference rows' member mean, so the geometry
being steered toward is never fit on labelled data.

Coverage (HELD documents, no generation) reports each handle's precision/recall/
AUROC for identifying the concept, before any text is generated -- if a handle
cannot even describe the concept in its own partition, steering with it is moot.
For every DIRECTION handle (label, label_z, base_dir, bypass_dir) this is the AUROC
of held docs' mean x (or its projection) on the direction. Per concept, cos(label,
label_z), cos(base_dir, label), cos(bypass_dir, label) and cos(base_dir, bypass_dir)
are also recorded (all raw, pre-rescale directions in the SAME normalised residual
space), and per-dataset mean/min/max are printed for each.

Generation: unsteered continuations of 20 neutral prompts (PROMPTS, imported from
cluster_steer_generate), then for every (dataset, concept, handle, alpha) the same
prompts steered at EVERY position (prefill and each decoding step), greedy. Scored
by continuation perplexity under the UNEDITED model (fluency cost), distinct-bigram
share (repetition), and the mean edit norm actually applied. concept_steer_judge.py
does the (independent) LLM forced-choice scoring of whether the text moved on-topic.

    python -u -m geoae.interp.concept_steer_generate --base <km.npz> --bypass <ckpt> \
        --handles label,label_z,base,bypass --out eval_out/concept_steer_generate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from geoae.checkpoint import load_lm
from geoae.hooks import SplicingHook, TokenIdTap
from geoae.interp._shared import DATASET_CONFIGS
from geoae.interp.cluster_steer_generate import PROMPTS
from geoae.interp.cluster_steering import Arm

DB14_DESCRIPTIONS = {
    "Company": "companies and businesses",
    "EducationalInstitution": "schools and universities",
    "Artist": "artists and musicians (as people)",
    "Athlete": "athletes and sports",
    "OfficeHolder": "politicians and public officials",
    "MeanOfTransportation": "vehicles, ships and aircraft",
    "Building": "buildings and architecture",
    "NaturalPlace": "mountains, rivers and natural places",
    "Village": "villages and small towns",
    "Animal": "animals",
    "Plant": "plants",
    "Album": "music albums",
    "Film": "films and movies",
    "WrittenWork": "books and written works",
}


def concept_description(dataset: str, name: str) -> str:
    if dataset == "db14":
        return DB14_DESCRIPTIONS[name]
    if name == "dj":
        return "DJ"
    return name.replace("_", " ")


# Handles edited as a CONSTANT direction (x += alpha*R_c*d/||d||, via make_label_edit_fn) vs
# handles edited by moving a position into its own assigned cluster (make_cluster_edit_fn).
# base_dir/bypass_dir are the base/bypass CLUSTER target means used as directions, so they
# are direction handles even though their names look like the cluster handles they're derived
# from -- CLUSTER_HANDLES below is the "own-cluster-translation" pair they were built from.
DIRECTION_HANDLES = ("label", "label_z", "base_dir", "bypass_dir")
CLUSTER_HANDLES = ("base", "bypass")
HANDLE_DISPLAY_ORDER = ("label", "label_z", "base", "bypass", "base_dir", "bypass_dir")


# ---------------------------------------------------------------------------
# Pure, unit-testable helpers (numpy/torch in, plain values out)
# ---------------------------------------------------------------------------

def fit_class_cluster_stats(cls: np.ndarray, docid: np.ndarray, cluster: np.ndarray,
                             n_classes: int, K: int):
    """From per-token (class, doc id, cluster) arrays over FIT tokens of one codebook:
    n_ck (n_classes, K) token counts, n_docs_ck (n_classes, K) DISTINCT-document counts,
    p_ck (n_classes, K) class-balanced precision = f_ck / sum_c' f_c'k, f_ck = n_ck / N_c."""
    n_ck = np.zeros((n_classes, K), dtype=np.int64)
    n_docs_ck = np.zeros((n_classes, K), dtype=np.int64)
    for c in range(n_classes):
        m = cls == c
        if not m.any():
            continue
        k_c, d_c = cluster[m], docid[m]
        n_ck[c] = np.bincount(k_c, minlength=K)
        key = d_c.astype(np.int64) * K + k_c.astype(np.int64)
        uniq = np.unique(key)
        n_docs_ck[c] = np.bincount((uniq % K).astype(np.int64), minlength=K)
    N_c = np.maximum(np.bincount(cls, minlength=n_classes), 1).astype(np.float64)
    f_ck = n_ck / N_c[:, None]
    denom = f_ck.sum(0, keepdims=True)
    p_ck = np.divide(f_ck, denom, out=np.zeros_like(f_ck), where=denom > 0)
    return n_ck, n_docs_ck, p_ck


def select_handle_clusters(n_ck: np.ndarray, n_docs_ck: np.ndarray, p_ck: np.ndarray,
                            hub: np.ndarray, min_count: int, min_docs: int, top_k: int):
    """n_ck/n_docs_ck/p_ck/hub: (K,) arrays for ONE (concept, codebook). Eligible clusters
    are not a hub, have >= min_count class-c tokens from >= min_docs distinct documents.
    Returns (selected cluster ids, sorted by descending p_ck, weights ∝ n_ck summing to 1),
    or two empty arrays if nothing is eligible."""
    eligible = np.flatnonzero((~hub) & (n_ck >= min_count) & (n_docs_ck >= min_docs))
    if len(eligible) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    order = eligible[np.argsort(-p_ck[eligible], kind="stable")][:top_k]
    w = n_ck[order].astype(np.float64)
    w = w / w.sum()
    return order, w


def token_class_rate(is_in: np.ndarray, cls: np.ndarray, n_classes: int) -> np.ndarray:
    """(n_classes,) share of each class's tokens with is_in True."""
    tot = np.bincount(cls, minlength=n_classes).astype(np.float64)
    hit = np.bincount(cls[is_in], minlength=n_classes).astype(np.float64) if is_in.any() \
        else np.zeros(n_classes)
    return np.divide(hit, tot, out=np.zeros_like(hit), where=tot > 0)


def label_delta(v_c: torch.Tensor, R_c: float, alpha: float) -> torch.Tensor:
    """A direction handle's edit: alpha * R_c * v_c / ||v_c|| -- a fixed direction scaled to
    R_c's typical base-handle gap size, so every handle moves a comparable amount. Used by
    both `label` (v_c already in residual space) and `label_z` (v_c = decode_direction(...),
    already mapped into residual space before this is called)."""
    return alpha * R_c * v_c / v_c.norm()


def decode_direction(ae, v_z: torch.Tensor) -> torch.Tensor:
    """Map a latent DIRECTION (not an actual latent -- a difference of two class means) to
    normalised residual space using the decoder WEIGHT only: u = ae.decoder(v_z).

    geoae/model.py builds `self.decoder = nn.Linear(latent_dim, hidden_size, bias=False)`,
    so calling it directly is exactly W_dec @ v_z with no decoder bias to worry about. The
    token-bypass table (ae.token_bias) is added by ae.decode(z, tok) for an actual
    encode/decode ROUND TRIP; it is irrelevant here because it depends only on the current
    token, not on z, so it would cancel in any difference of two decode() calls at the same
    token anyway -- see tests/test_concept_steer.py for that equivalence. A bare direction
    therefore never needs token ids."""
    return ae.decoder(v_z)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """cos(a, b) between two raw (pre-rescale) direction vectors."""
    return float((a @ b) / (a.norm() * b.norm() + eps))


def contrast_direction(target_by_concept: dict) -> dict:
    """base_dir/bypass_dir's class-balanced contrast: for a codebook's per-concept target
    means {concept: m_c or None}, returns {concept: d_c} for every concept that HAS a target
    (None entries are dropped, not defaulted to 0), where
        d_c = m_c - mean(m_c' for every concept c' that also has a target)
    -- mirroring `label`'s v_c = mu_c - mu_d, except the group being averaged is "concepts
    with THIS handle", not "all kept classes". Works on anything supporting + and / (plain
    floats for tests, or torch tensors in the real pipeline); {} in -> {} out."""
    present = {c: t for c, t in target_by_concept.items() if t is not None}
    if not present:
        return {}
    mean = sum(present.values()) / len(present)
    return {c: t - mean for c, t in present.items()}


def doc_auroc(scores: np.ndarray, is_target: np.ndarray) -> float:
    scores, is_target = np.asarray(scores), np.asarray(is_target, dtype=bool)
    if is_target.sum() == 0 or (~is_target).sum() == 0:
        return float("nan")
    return float(roc_auc_score(is_target, scores))


def distinct2(row) -> float:
    bg = list(zip(row[:-1], row[1:]))
    return len(set(bg)) / max(len(bg), 1)


# ---------------------------------------------------------------------------
# Activation capture: every non-BOS, non-padding position of a raw document.
# Same capture point as neuronlens.capture_post_layer (post-layer, pre-final-norm).
# ---------------------------------------------------------------------------

@torch.no_grad()
def capture_docs(lm, tk, texts: list[str], layer: int, device, max_len: int, batch_size: int = 8):
    from geoae.lm_arch import decoder_layers
    layer_mod = decoder_layers(lm)[layer]
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s + batch_size]
        if not batch:
            continue
        enc = tk(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        captured = []

        def _hook(_m, _i, o):
            captured.append(o[0] if isinstance(o, tuple) else o)

        h = layer_mod.register_forward_hook(_hook)
        lm(**enc, use_cache=False)
        h.remove()
        hs = captured[0].float()
        am = enc["attention_mask"]
        B, T = am.shape
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        valid = am.bool() & (pos > 0)                      # exclude BOS (position 0) + padding
        b_idx, t_idx = valid.nonzero(as_tuple=True)
        if b_idx.numel() == 0:
            continue
        yield hs[b_idx, t_idx], enc["input_ids"][b_idx, t_idx], (b_idx + s)


def load_concept_docs(cfg: dict, n_fit: int, n_held: int, seed: int, smoke: bool):
    """Per-class TRAIN documents (text stripped), fit docs first, held-out after.
    Classes with fewer than n_fit + n_held docs are dropped (and logged). Under
    --smoke, only the first 2 classes (by config order) with enough docs are kept,
    to keep the capture pass fast."""
    from datasets import load_dataset
    ds = load_dataset(cfg["dataset_name"], split="train").shuffle(seed=seed)
    label_col = cfg.get("label_column", "label")
    text_col = cfg["text_column"]
    classes = cfg["classes"]
    need = n_fit + n_held
    by_class = {c: [] for c in range(len(classes))}
    remaining = set(range(len(classes)))
    for ex in ds:
        c = int(ex[label_col])
        if c in remaining:
            t = (ex[text_col] or "").strip()
            if t:
                by_class[c].append(t)
                if len(by_class[c]) >= need:
                    remaining.discard(c)
        if not remaining:
            break
    kept, dropped = [], []
    for c in range(len(classes)):
        (kept if len(by_class[c]) >= need else dropped).append(c)
    if dropped:
        print(f"[concept]   dropping classes with < {need} train docs: "
              + ", ".join(f"{classes[c]} ({len(by_class[c])})" for c in dropped))
    if smoke:
        kept = kept[:2]
    return by_class, kept


# ---------------------------------------------------------------------------
# Splice functions
# ---------------------------------------------------------------------------

def make_cluster_edit_fn(arm: Arm, M_cb: torch.Tensor, m_c: torch.Tensor, tap: TokenIdTap,
                          mean: torch.Tensor, std: torch.Tensor, alpha: float, log: list):
    def fn(hs):
        B, T, D = hs.shape
        x = (hs.reshape(B * T, D).float() - mean) / std
        tok = tap.ids_for(hs)
        s = arm.assign(x, tok)
        delta = alpha * (m_c[None, :] - M_cb[s])
        log.append(float(delta.norm(dim=1).mean()))
        x2 = x + delta
        return (x2 * std + mean).reshape(B, T, D).to(hs.dtype)
    return fn


def make_label_edit_fn(v_c: torch.Tensor, R_c: float, mean: torch.Tensor, std: torch.Tensor,
                        alpha: float, log: list):
    delta = label_delta(v_c, R_c, alpha)

    def fn(hs):
        B, T, D = hs.shape
        x = (hs.reshape(B * T, D).float() - mean) / std
        log.append(float(delta.norm()))
        x2 = x + delta[None, :]
        return (x2 * std + mean).reshape(B, T, D).to(hs.dtype)
    return fn


# ---------------------------------------------------------------------------
# Per-dataset concept construction
# ---------------------------------------------------------------------------

def run_dataset(ds_name: str, cfg: dict, lm, tk, arms: dict, mean, std, D: int, dev,
                 args, M: dict, hub: dict, lab_ref: dict, sanity_acc: dict):
    classes = cfg["classes"]
    n_classes = len(classes)
    by_class, kept = load_concept_docs(cfg, args.n_fit, args.n_held, args.seed, args.smoke)
    print(f"[concept] {ds_name}: {len(kept)}/{n_classes} classes with enough docs "
          f"({', '.join(classes[c] for c in kept)})")
    if not kept:
        return None

    need_z = "label_z" in args.handles              # only pay for the bypass encode pass if asked
    Lz = arms["bypass"].C.shape[1] if need_z else 0

    classsum_x = torch.zeros(n_classes, D, device=dev)
    classsum_z = torch.zeros(n_classes, Lz, device=dev) if need_z else None
    classcount = torch.zeros(n_classes, device=dev)
    fit_cls, fit_doc, fit_kb, fit_ky = [], [], [], []
    held_cls, held_doc, held_kb, held_ky = [], [], [], []
    held_doc_mean, held_doc_cls = [], []
    fit_doc_off = held_doc_off = 0

    tk.padding_side = "right"       # BOS is at position 0 only under right-padding
    for c in kept:
        texts_fit = by_class[c][: args.n_fit]
        texts_held = by_class[c][args.n_fit: args.n_fit + args.n_held]
        for h_raw, tok_ids, doc_local in capture_docs(lm, tk, texts_fit, args.layer, dev, args.max_doc_tokens):
            x = (h_raw - mean) / std
            classsum_x[c] += x.sum(0)
            classcount[c] += x.shape[0]
            sanity_acc["abs_sum"] += float(x.abs().sum())
            sanity_acc["n"] += x.shape[0]
            sanity_acc["sum_dim"] += x.sum(0)
            sanity_acc["sumsq_dim"] += (x * x).sum(0)
            lb, ly = arms["base"].assign(x, tok_ids), arms["bypass"].assign(x, tok_ids)
            if need_z:
                classsum_z[c] += arms["bypass"].coords(x, tok_ids).sum(0)
            n = x.shape[0]
            fit_cls.append(np.full(n, c, dtype=np.int64))
            fit_doc.append((doc_local + fit_doc_off).cpu().numpy())
            fit_kb.append(lb.cpu().numpy()); fit_ky.append(ly.cpu().numpy())
        fit_doc_off += len(texts_fit)

        n_held_c = len(texts_held)
        doc_sum = torch.zeros(max(n_held_c, 1), D, device=dev)
        doc_cnt = torch.zeros(max(n_held_c, 1), device=dev)
        for h_raw, tok_ids, doc_local in capture_docs(lm, tk, texts_held, args.layer, dev, args.max_doc_tokens):
            x = (h_raw - mean) / std
            lb, ly = arms["base"].assign(x, tok_ids), arms["bypass"].assign(x, tok_ids)
            n = x.shape[0]
            held_cls.append(np.full(n, c, dtype=np.int64))
            held_doc.append((doc_local + held_doc_off).cpu().numpy())
            held_kb.append(lb.cpu().numpy()); held_ky.append(ly.cpu().numpy())
            doc_sum.index_add_(0, doc_local, x)
            doc_cnt.index_add_(0, doc_local, torch.ones(n, device=dev))
        held_doc_mean.append((doc_sum[:n_held_c] / doc_cnt[:n_held_c].clamp_min(1)[:, None]).cpu().numpy())
        held_doc_cls.append(np.full(n_held_c, c, dtype=np.int64))
        held_doc_off += n_held_c
    tk.padding_side = "left"

    if not fit_cls:
        print(f"[concept] {ds_name}: no captured tokens, skipping")
        return None
    fit_cls, fit_doc = np.concatenate(fit_cls), np.concatenate(fit_doc)
    fit_kb, fit_ky = np.concatenate(fit_kb), np.concatenate(fit_ky)
    held_cls, held_doc = np.concatenate(held_cls), np.concatenate(held_doc)
    held_kb, held_ky = np.concatenate(held_kb), np.concatenate(held_ky)
    held_doc_mean = np.concatenate(held_doc_mean, axis=0)
    held_doc_cls = np.concatenate(held_doc_cls)

    mu = classsum_x / classcount.clamp_min(1)[:, None]              # (n_classes, D) FIT class means
    mu_d = mu[kept].mean(0)                                          # class-balanced grand mean
    v = mu - mu_d[None, :]
    if need_z:
        mu_z = classsum_z / classcount.clamp_min(1)[:, None]         # (n_classes, Lz) FIT class means, bypass latent
        mu_d_z = mu_z[kept].mean(0)
        v_z_all = mu_z - mu_d_z[None, :]

    K = {cb: arms[cb].C.shape[0] for cb in ("base", "bypass")}
    n_ck, n_docs_ck, p_ck = {}, {}, {}
    for cb, fit_k in (("base", fit_kb), ("bypass", fit_ky)):
        n_ck[cb], n_docs_ck[cb], p_ck[cb] = fit_class_cluster_stats(fit_cls, fit_doc, fit_k, n_classes, K[cb])

    need_dir = {"base_dir": "base_dir" in args.handles, "bypass_dir": "bypass_dir" in args.handles}

    handles, targets, fails = {}, {}, []
    for c in kept:
        cname = classes[c]
        handles[cname], targets[cname] = {}, {}
        for cb in ("base", "bypass"):
            hub_np = hub[cb].cpu().numpy()
            order, w = select_handle_clusters(n_ck[cb][c], n_docs_ck[cb][c], p_ck[cb][c], hub_np,
                                               args.min_count, args.min_docs, args.top_k)
            if len(order) == 0:
                handles[cname][cb], targets[cname][cb] = None, None
                fails.append({"dataset": ds_name, "concept": cname, "handle": cb, "reason": "no eligible cluster"})
                continue
            order_t = torch.as_tensor(order, device=dev, dtype=torch.long)
            w_t = torch.as_tensor(w, device=dev, dtype=torch.float32)
            m_c = (w_t[:, None] * M[cb][order_t]).sum(0)
            targets[cname][cb] = m_c
            handles[cname][cb] = {"clusters": order.tolist(), "weights": w.tolist(),
                                   "p_ck": p_ck[cb][c, order].tolist(), "n_ck": n_ck[cb][c, order].tolist(),
                                   "n_docs_ck": n_docs_ck[cb][c, order].tolist()}

    # base_dir/bypass_dir's contrast group is "concepts that have THIS cb's cluster handle"
    # -- computed once over every concept before the per-concept pass below, since a single
    # concept's direction needs the OTHERS' target means too.
    dir_contrasts = {}
    for cb, dirname in (("base", "base_dir"), ("bypass", "bypass_dir")):
        if need_dir[dirname]:
            dir_contrasts[cb] = contrast_direction({cname: targets[cname][cb] for cname in handles})

    for c in kept:
        cname = classes[c]
        if targets[cname]["base"] is not None:
            gaps = (targets[cname]["base"][None, :] - M["base"][lab_ref["base"]]).norm(dim=1)
            R_c = float(gaps.median())
            targets[cname]["label"] = (v[c].clone(), R_c)
            handles[cname]["label"] = {"R_c": R_c, "v_norm": float(v[c].norm())}
            if need_z:
                v_z = v_z_all[c]
                u = decode_direction(arms["bypass"].ae, v_z).detach()
                targets[cname]["label_z"] = (u.clone(), R_c)
                handles[cname]["label_z"] = {"R_c": R_c, "v_z_norm": float(v_z.norm()), "u_norm": float(u.norm())}
                handles[cname]["cos_label_label_z"] = cosine(v[c], u)
            for cb, dirname in (("base", "base_dir"), ("bypass", "bypass_dir")):
                if not need_dir[dirname]:
                    continue
                d = dir_contrasts[cb].get(cname)
                if d is None:
                    targets[cname][dirname] = None
                    handles[cname][dirname] = None
                    fails.append({"dataset": ds_name, "concept": cname, "handle": dirname,
                                  "reason": f"no {cb} handle"})
                    continue
                targets[cname][dirname] = (d.clone(), R_c)
                info = {"R_c": R_c, "raw_norm": float(d.norm()), "cos_label": cosine(d, v[c])}
                if need_z:
                    info["cos_label_z"] = cosine(d, u)
                handles[cname][dirname] = info
        else:
            targets[cname]["label"] = None
            handles[cname]["label"] = None
            fails.append({"dataset": ds_name, "concept": cname, "handle": "label", "reason": "no base handle for R_c"})
            if need_z:
                targets[cname]["label_z"] = None
                handles[cname]["label_z"] = None
                handles[cname]["cos_label_label_z"] = None
                fails.append({"dataset": ds_name, "concept": cname, "handle": "label_z",
                              "reason": "no base handle for R_c"})
            for dirname in ("base_dir", "bypass_dir"):
                if need_dir[dirname]:
                    targets[cname][dirname] = None
                    handles[cname][dirname] = None
                    fails.append({"dataset": ds_name, "concept": cname, "handle": dirname,
                                  "reason": "no base handle for R_c"})
        if need_dir["base_dir"] and need_dir["bypass_dir"]:
            db, bb = targets[cname].get("base_dir"), targets[cname].get("bypass_dir")
            handles[cname]["cos_base_dir_bypass_dir"] = cosine(db[0], bb[0]) if (db and bb) else None

    def _print_cos_summary(label: str, values: list):
        if values:
            print(f"[concept] {ds_name}: cos({label}) mean {np.mean(values):.3f} "
                  f"[min {np.min(values):.3f}, max {np.max(values):.3f}]  (n={len(values)})")

    if need_z:
        _print_cos_summary("label, label_z", [handles[c]["cos_label_label_z"] for c in handles
                                              if handles[c].get("cos_label_label_z") is not None])
    for dirname in ("base_dir", "bypass_dir"):
        if need_dir[dirname]:
            _print_cos_summary(f"{dirname}, label", [handles[c][dirname]["cos_label"] for c in handles
                                                      if handles[c].get(dirname) is not None])
    if need_dir["base_dir"] and need_dir["bypass_dir"]:
        _print_cos_summary("base_dir, bypass_dir", [handles[c]["cos_base_dir_bypass_dir"] for c in handles
                                                     if handles[c].get("cos_base_dir_bypass_dir") is not None])

    coverage = {}
    for cname in handles:
        c = classes.index(cname)
        coverage[cname] = {}
        for cb, held_k in (("base", held_kb), ("bypass", held_ky)):
            h = handles[cname][cb]
            if h is None:
                coverage[cname][cb] = None
                continue
            is_in = np.isin(held_k, h["clusters"])
            rate = token_class_rate(is_in, held_cls, n_classes)
            recall = float(rate[c])
            precision = float(rate[c] / rate.sum()) if rate.sum() > 0 else float("nan")
            doc_ntok = np.bincount(held_doc, minlength=held_doc_off)
            doc_in = np.bincount(held_doc[is_in], minlength=held_doc_off) if is_in.any() \
                else np.zeros(held_doc_off, dtype=np.int64)
            doc_share = np.divide(doc_in, doc_ntok, out=np.zeros(held_doc_off), where=doc_ntok > 0)
            coverage[cname][cb] = {"precision": precision, "recall": recall,
                                   "auroc": doc_auroc(doc_share, held_doc_cls == c)}
        for h in DIRECTION_HANDLES:
            if h not in targets[cname]:
                continue                                    # not requested (e.g. label_z, need_z False)
            lbl = targets[cname][h]
            if lbl is not None:
                v_c, _ = lbl
                scores = (torch.as_tensor(held_doc_mean, device=dev, dtype=torch.float32)
                          @ v_c / v_c.norm()).cpu().numpy()
                coverage[cname][h] = {"auroc": doc_auroc(scores, held_doc_cls == c)}
            else:
                coverage[cname][h] = None

    return {"kept": kept, "classes": classes, "handles": handles, "coverage": coverage,
            "fails": fails, "targets": targets}


def print_coverage_table(dataset_results: dict):
    print("\n[concept] coverage on HELD documents (precision / recall / AUROC; direction handles: AUROC only)")
    for ds_name, res in dataset_results.items():
        present = [h for h in HANDLE_DISPLAY_ORDER if any(h in hd for hd in res["coverage"].values())]
        print(f"  {ds_name}:")
        for cname, hd in res["coverage"].items():
            def fmt(d):
                if d is None:
                    return "n/a"
                if "recall" in d:
                    return f"p={d['precision']:.2f} r={d['recall']:.2f} auroc={d['auroc']:.2f}"
                return f"auroc={d['auroc']:.2f}"
            print(f"    {cname:26s} " + "  ".join(f"{h}[{fmt(hd.get(h))}]" for h in present))
        for hname in present:
            vals = [hd[hname]["auroc"] for hd in res["coverage"].values()
                    if hd.get(hname) is not None]
            if vals:
                print(f"    MEAN {hname:8s} auroc {np.mean(vals):.3f}  (n={len(vals)}/{len(res['coverage'])})")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base codebook: balanced k-means .npz on the plain residual")
    ap.add_argument("--bypass", required=True, help="token-bypass AE checkpoint")
    ap.add_argument("--datasets", default="db14,biasbios")
    ap.add_argument("--handles", default="label,base,bypass",
                    help="comma-separated: which handles to GENERATE text for (label, label_z, base, "
                         "bypass, base_dir, bypass_dir). label/base/bypass coverage is always computed; "
                         "label_z's extra bypass-encode pass over the FIT tokens, and base_dir/bypass_dir's "
                         "contrast-direction bookkeeping, only run when listed here.")
    ap.add_argument("--n_fit", type=int, default=150)
    ap.add_argument("--n_held", type=int, default=50)
    ap.add_argument("--max_doc_tokens", type=int, default=128)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--min_docs", type=int, default=5)
    ap.add_argument("--hub_x", type=float, default=3.0)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--activations_dir", default="activations_sampled_10M")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--n_eval", type=int, default=500_000)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.handles = [h.strip() for h in args.handles.split(",") if h.strip()]
    unknown = set(args.handles) - set(HANDLE_DISPLAY_ORDER)
    if unknown:
        raise SystemExit(f"--handles: unknown handle(s) {sorted(unknown)}")
    if args.smoke:
        args.n_fit, args.n_held, args.n_eval = 30, 10, 100_000
        args.alphas = [0.3, 1.0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    act = Path(args.activations_dir)
    meta = json.load(open(act / "meta.json"))
    tk = AutoTokenizer.from_pretrained(meta["model"])
    tk.pad_token, tk.padding_side = tk.eos_token, "left"
    lm = load_lm(meta["model"], device=dev)
    from geoae.lm_arch import decoder_layers
    if args.layer != len(decoder_layers(lm)) - 1:
        raise SystemExit("--layer must be the last block (edits act on the next-token choice)")

    arms = {"base": Arm("base", args.base, dev), "bypass": Arm("bypass", args.bypass, dev)}
    if arms["base"].kind != "km" or arms["bypass"].kind != "ae":
        raise SystemExit("--base must be a k-means .npz and --bypass an AE checkpoint")
    mean, std = arms["base"].mean, arms["base"].std
    if not (torch.allclose(arms["bypass"].mean, mean) and torch.allclose(arms["bypass"].std, std)):
        raise SystemExit("base and bypass use different norm stats")
    D = mean.shape[0]

    # ---- general reference stats: member means, hubs, per-row assignment ------------------
    X = np.load(str(act / f"layer_{args.layer}.npy"), mmap_mode="r")
    N = len(X)
    ev = np.arange(int(N * (1 - args.val_frac)), N)[: args.n_eval]
    doc = np.load(act / "rows_doc.npy")[ev]
    ref = ev[doc % 2 == 0]
    H = torch.from_numpy(np.array(X[ref])).to(dev)
    tok_ref = torch.from_numpy(np.load(act / "rows_tok.npy")[ref].astype(np.int64)).to(dev)
    print(f"[concept] {len(ref):,} reference rows (even docs of the held-out tail)")

    Xr = (H.float() - mean) / std
    ref_abs_mean, ref_std_mean = float(Xr.abs().mean()), float(Xr.std(dim=0).mean())
    print(f"[concept] reference scale: mean|x| {ref_abs_mean:.3f}  per-dim std {ref_std_mean:.3f}")
    del Xr

    M, hub, lab_ref = {}, {}, {}
    for cbname, arm in arms.items():
        K = arm.C.shape[0]
        lab = torch.empty(len(ref), dtype=torch.long, device=dev)
        Msum = torch.zeros(K, D, device=dev)
        for i in range(0, len(ref), args.chunk):
            x = (H[i:i + args.chunk].float() - mean) / std
            l = arm.assign(x, tok_ref[i:i + args.chunk])
            lab[i:i + len(l)] = l
            Msum.index_add_(0, l, x)
        cnt = torch.bincount(lab, minlength=K).float()
        Msum /= cnt.clamp_min(1)[:, None]
        hub_mask = cnt / cnt.sum() > args.hub_x / K
        M[cbname], hub[cbname], lab_ref[cbname] = Msum, hub_mask, lab
        print(f"[concept] {cbname}: K={K} hubs {int(hub_mask.sum())} "
              f"({float(cnt[hub_mask].sum() / cnt.sum()):.1%} of rows)")
    del H, tok_ref
    torch.cuda.empty_cache()

    # ---- labelled concept data ----------------------------------------------------------
    ds_names = args.datasets.split(",")
    sanity_acc = {"abs_sum": 0.0, "n": 0, "sum_dim": torch.zeros(D, device=dev),
                  "sumsq_dim": torch.zeros(D, device=dev)}
    dataset_results = {}
    for ds_name in ds_names:
        cfg = DATASET_CONFIGS[ds_name]
        res = run_dataset(ds_name, cfg, lm, tk, arms, mean, std, D, dev, args, M, hub, lab_ref, sanity_acc)
        if res is not None:
            dataset_results[ds_name] = res

    if sanity_acc["n"] > 0:
        cap_mean = sanity_acc["sum_dim"] / sanity_acc["n"]
        cap_var = (sanity_acc["sumsq_dim"] / sanity_acc["n"] - cap_mean * cap_mean).clamp_min(0)
        cap_abs_mean = sanity_acc["abs_sum"] / (sanity_acc["n"] * D)
        print(f"[concept] captured-row scale: mean|x| {cap_abs_mean:.3f}  per-dim std {float(cap_var.sqrt().mean()):.3f}"
              f"  (reference: {ref_abs_mean:.3f} / {ref_std_mean:.3f}; should be similar)")

    print_coverage_table(dataset_results)

    n_fail = sum(len(r["fails"]) for r in dataset_results.values())
    if n_fail:
        print(f"\n[concept] {n_fail} coverage failures (no eligible cluster / no base handle for R_c):")
        for r in dataset_results.values():
            for f in r["fails"]:
                print(f"    {f['dataset']:10s} {f['concept']:24s} {f['handle']:6s}: {f['reason']}")

    # ---- generation ----------------------------------------------------------------------
    n_prompts = 4 if args.smoke else 20
    prompts_used = PROMPTS[:n_prompts]
    hook, tap = SplicingHook(lm, args.layer), TokenIdTap(lm)
    tk.padding_side = "left"
    enc = tk(prompts_used, return_tensors="pt", padding=True).to(dev)
    L = enc["input_ids"].shape[1]

    def generate(fn=None):
        if fn is not None:
            hook.activate(fn)
        try:
            out = lm.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tk.pad_token_id)
        finally:
            if fn is not None:
                hook.deactivate()
        return out[:, L:]

    def ppl(cont):
        ids = torch.cat([enc["input_ids"], cont], 1)
        am = torch.cat([enc["attention_mask"], (cont != -1).long()], 1)
        logits = lm(input_ids=ids, attention_mask=am).logits[:, L - 1:-1].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), cont, reduction="none")
        return nll.mean(1).exp().cpu().numpy()

    base_cont = generate()
    base_ppl = ppl(base_cont)
    base_ids = base_cont.cpu().numpy()
    unsteered_text = [tk.decode(r, skip_special_tokens=True) for r in base_ids]
    unsteered_d2 = [distinct2(list(r)) for r in base_ids]
    print(f"\n[concept] unsteered: median ppl {np.median(base_ppl):.2f}; e.g. {unsteered_text[0][:70]!r}")

    gens = {}
    for ds_name, res in dataset_results.items():
        gens[ds_name] = {cname: {} for cname in res["handles"]}
        for hname in [h for h in HANDLE_DISPLAY_ORDER if h in args.handles]:
            avail = [c for c in res["handles"] if res["handles"][c].get(hname) is not None]
            if not avail:
                continue
            for cname in avail:
                gens[ds_name][cname][hname] = {"0.0": {
                    "text": unsteered_text, "ppl": base_ppl.tolist(),
                    "log_ppl_ratio": [0.0] * len(prompts_used), "distinct2": unsteered_d2, "edit_norm": 0.0}}
            for a in args.alphas:
                cell_lr, cell_d2, cell_edit = [], [], []
                for cname in avail:
                    log = []
                    if hname in DIRECTION_HANDLES:
                        v_c, R_c = res["targets"][cname][hname]
                        fn = make_label_edit_fn(v_c, R_c, mean, std, a, log)
                    else:
                        fn = make_cluster_edit_fn(arms[hname], M[hname], res["targets"][cname][hname],
                                                  tap, mean, std, a, log)
                    cont = generate(fn)
                    pp = ppl(cont)
                    ids = cont.cpu().numpy()
                    texts = [tk.decode(r, skip_special_tokens=True) for r in ids]
                    d2 = [distinct2(list(r)) for r in ids]
                    lr = (np.log(pp) - np.log(base_ppl)).tolist()
                    edit_norm = float(np.mean(log)) if log else 0.0
                    gens[ds_name][cname][hname][str(a)] = {"text": texts, "ppl": pp.tolist(),
                                                           "log_ppl_ratio": lr, "distinct2": d2,
                                                           "edit_norm": edit_norm}
                    cell_lr += lr; cell_d2 += d2; cell_edit.append(edit_norm)
                print(f"[concept] {hname:6s} {ds_name:9s} a{a:<4} ppl x{np.exp(np.mean(cell_lr)):.2f}  "
                      f"distinct2 {np.mean(cell_d2):.2f}  edit_norm {np.mean(cell_edit):.3f}  "
                      f"({len(avail)} concepts)", flush=True)
    tap.remove()

    res_out = {
        "meta": {**vars(args), "prompts": prompts_used,
                 "concept_descriptions": {ds: {n: concept_description(ds, n) for n in DATASET_CONFIGS[ds]["classes"]}
                                          for ds in ds_names}},
        "handles": {ds: r["handles"] for ds, r in dataset_results.items()},
        "coverage": {ds: r["coverage"] for ds, r in dataset_results.items()},
        "coverage_failures": [f for r in dataset_results.values() for f in r["fails"]],
        "generations": {"unsteered": {"text": unsteered_text, "ppl": base_ppl.tolist(), "distinct2": unsteered_d2},
                        "steered": gens},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res_out, indent=1, ensure_ascii=False))
    print(f"[concept] wrote {args.out}")


if __name__ == "__main__":
    main()
