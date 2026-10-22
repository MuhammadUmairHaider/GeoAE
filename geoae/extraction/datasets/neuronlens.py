"""External-labelled NeuronLens experiments on identical base/AE token rows.

Uses the variant already used for this project's shared-direction experiment:
d-prime saliency, top-p dimensions, +/-2 standard-deviation gate, complement
mean replacement, and reconstruction-preserving decoded AE differences.
Reports next-token predictive damage; these are not semantic-classifier scores.

Input JSONL: {doc_id, split:'fit'|'eval', text, spans:[{start,end,label}]}.
Input plan: {pairs:[[label,label],...], label_names:{...}}.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from geoae.extraction.datasets.neuronlens_summary import summarize
from geoae.checkpoint import load_ae_checkpoint, load_lm
from geoae.evaluation.interventions import dprime_saliency
from geoae.evaluation.interventions import fit_ranges
from geoae.evaluation.interventions import gate_edit
from geoae.evaluation.interventions import select_top
from geoae.lm_arch import decoder_layers, locate_lm_parts
from geoae.token_bias import TokenBiasLookup

CKPT = "checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc_sampled_tokbias/step_0014450.pt"
TBFILE = "e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz"
LM_NAME = "meta-llama/Llama-3.2-3B"
LAYER = 27
ARMS = ("base", "token_centred", "ae")


def log(*values):
    print(time.strftime("[%H:%M:%S]"), *values, flush=True)


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare_rows(path, plan, tok, args):
    """Select rows before looking at representations; anchor completed spans."""
    documents = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    document_keys = [(d["split"], d["doc_id"]) for d in documents]
    assert len(document_keys) == len(set(document_keys)), "duplicate document records"
    fit_ids = {d["doc_id"] for d in documents if d["split"] == "fit"}
    eval_ids = {d["doc_id"] for d in documents if d["split"] == "eval"}
    assert not fit_ids & eval_ids, "fit/eval documents overlap"
    labels = sorted({label for pair in plan["pairs"] for label in pair})
    pools, controls = defaultdict(list), defaultdict(list)
    failures = Counter()
    rng = np.random.default_rng(args.seed)
    for di, document in enumerate(documents):
        split, text = document["split"], document["text"]
        if split not in ("fit", "eval"):
            continue
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(ids) < 3:
            continue
        starts = np.asarray([s for s, _ in offsets])
        ends = np.asarray([e for _, e in offsets])
        occupied = np.zeros(len(ids), bool)
        anchors = {}
        for span in document["spans"]:
            start, end = int(span["start"]), int(span["end"])
            assert 0 <= start < end <= len(text), (document["doc_id"], span)
            overlap = (starts < end) & (ends > start)
            occupied |= overlap
            if span["label"] not in labels:
                continue
            nonwhite_end = end
            while nonwhite_end > start and text[nonwhite_end - 1].isspace():
                nonwhite_end -= 1
            indices = np.nonzero((starts <= nonwhite_end - 1) & (ends >= nonwhite_end))[0]
            if not len(indices):
                failures["no_endpoint_token"] += 1
                continue
            anchor = int(indices[-1])
            if int(ends[anchor]) > nonwhite_end:
                failures["endpoint_token_extends_span"] += 1
                continue
            # No right-context input. The next token is only the scoring target.
            if anchor >= len(ids) - 1:
                failures["no_next_token"] += 1
                continue
            if anchor < args.min_prefix:
                failures["short_prefix"] += 1
                continue
            anchors.setdefault(anchor, []).append(span)
        for anchor, spans in anchors.items():
            unique = {span["label"] for span in spans}
            if len(unique) != 1:
                failures["ambiguous_endpoint_labels"] += 1
                continue
            span = spans[0]
            pools[(split, span["label"])].append({
                "document": di, "anchor": anchor, "label": span["label"],
                "start": span["start"], "end": span["end"], "mention": text[span["start"]:span["end"]],
            })
        candidates = np.nonzero(~occupied & (np.arange(len(ids)) >= args.min_prefix)
                                & (np.arange(len(ids)) < len(ids) - 1))[0]
        if len(candidates):
            for anchor in rng.choice(candidates, min(2, len(candidates)), replace=False):
                controls[split].append({"document": di, "anchor": int(anchor), "label": "__random__",
                                        "start": int(starts[anchor]), "end": int(ends[anchor]),
                                        "mention": text[starts[anchor]:ends[anchor]]})
        document["ids"] = ids

    def sample(pool, maximum, cap):
        counts = Counter()
        selected = []
        for i in rng.permutation(len(pool)):
            row = pool[i]
            if counts[row["document"]] >= cap:
                continue
            selected.append(row)
            counts[row["document"]] += 1
            if len(selected) >= maximum:
                break
        return selected

    rows, support = [], {}
    for split in ("fit", "eval"):
        maximum = args.fit_per_label if split == "fit" else args.eval_per_label
        for label in labels:
            selected = sample(pools[(split, label)], maximum, args.cap_per_doc)
            support.setdefault(label, {})[split] = {"available": len(pools[(split, label)]),
                                                    "selected": len(selected),
                                                    "documents": len({r["document"] for r in selected})}
            rows.extend(selected)
        rows.extend(sample(controls[split], args.controls_per_split, 2))
    for row in rows:
        document = documents[row["document"]]
        anchor = row["anchor"]
        ids = document["ids"]
        row.update(doc_id=document["doc_id"], split=document["split"], tok=ids[anchor], nxt=ids[anchor + 1],
                   input_ids=ids[max(0, anchor + 1 - args.context):anchor + 1])
        row["prefix_text"] = tok.decode(row["input_ids"][-48:])
    log("selected span support:", json.dumps(support, sort_keys=True))
    return rows, {"support": support, "excluded": dict(failures),
                  "n_input_documents": len(documents), "n_selected_documents": len({r["doc_id"] for r in rows}),
                  "n_rows": len(rows), "n_random_rows": sum(r["label"] == "__random__" for r in rows)}


@torch.no_grad()
def capture_rows(rows, lm, norm, head, tok, args, dev):
    """Causal prefix windows, batched by length, fp32 saved activations."""
    layers = decoder_layers(lm)
    assert len(layers) == LAYER + 1, "head-only scoring requires final decoder layer"
    captured = []
    handle = layers[LAYER].register_forward_hook(
        lambda module, inp, out: captured.append(out[0] if isinstance(out, tuple) else out))
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
    h_all = np.empty((len(rows), head.weight.shape[1]), np.float32)
    sanity_differences = []
    for offset in range(0, len(order), args.batch_size):
        selected = order[offset:offset + args.batch_size]
        length = max(len(rows[i]["input_ids"]) for i in selected)
        ids = torch.full((len(selected), length), tok.eos_token_id, device=dev, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for bi, ri in enumerate(selected):
            seq = rows[ri]["input_ids"]
            ids[bi, -len(seq):] = torch.as_tensor(seq, device=dev)
            mask[bi, -len(seq):] = 1
        captured.clear()
        output = lm(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
        h = captured.pop()[:, -1].float()
        assert torch.isfinite(h).all()
        h_all[selected] = h.cpu().numpy()
        if offset == 0:
            target = torch.as_tensor([rows[i]["nxt"] for i in selected], device=dev)[:, None]
            reference = output.logits[:, -1].float().log_softmax(-1).gather(1, target)
            reconstructed = head(norm(h)).log_softmax(-1).gather(1, target)
            sanity_differences.extend((reference - reconstructed).abs().flatten().tolist())
            assert max(sanity_differences) < 0.5, "captured residual fails LM-head sanity"
        if offset == 0 or (offset // args.batch_size + 1) % 50 == 0:
            log("capture", min(offset + args.batch_size, len(rows)), "/", len(rows))
        del output, h
    handle.remove()
    return h_all, {"n_rows": len(sanity_differences), "max_abs_logprob_diff": max(sanity_differences),
                   "mean_abs_logprob_diff": float(np.mean(sanity_differences))}


def geometry(spaces, labels, splits, pair):
    """Fit common direction on fit only; evaluate range separation independently."""
    output = {}
    for space, array in spaces.items():
        fit = splits == "fit"
        t, s = labels == pair[0], labels == pair[1]
        complement = fit & ~t & ~s
        center = array[complement].mean(0)
        first, second = array[fit & t].mean(0) - center, array[fit & s].mean(0) - center
        first = first / max(float(first.norm()), 1e-12)
        second = second / max(float(second.norm()), 1e-12)
        direction = first + second
        direction = direction / max(float(direction.norm()), 1e-12)
        record = {"fit_cosine": float(first @ second)}
        for split in ("fit", "eval"):
            ti = torch.as_tensor(np.nonzero((splits == split) & t)[0], device=array.device)
            si = torch.as_tensor(np.nonzero((splits == split) & s)[0], device=array.device)
            a, b = (array[ti] - center) @ direction, (array[si] - center) @ direction
            record[split] = {"projected_dprime": float((a.mean() - b.mean()).abs() /
                                (0.5 * (a.var(correction=0) + b.var(correction=0)) + 1e-8).sqrt()),
                             "mean_t": float(a.mean()), "mean_s": float(b.mean()),
                             "sd_t": float(a.std(correction=0)), "sd_s": float(b.std(correction=0))}
        output[space] = record
    return output


@torch.no_grad()
def evaluate(rows, H, ae, tb, mean, std, norm, head, plan, args, dev):
    tok = torch.as_tensor([r["tok"] for r in rows], dtype=torch.long, device=dev)
    nxt = torch.as_tensor([r["nxt"] for r in rows], dtype=torch.long, device=dev)
    X = (H - mean) / std
    Z = torch.cat([ae.encode(X[i:i + 1024], tok[i:i + 1024]) for i in range(0, len(H), 1024)])
    HB = X - tb(tok)
    spaces = {"base": H, "token_centred": HB, "ae": Z}
    labels = np.asarray([r["label"] for r in rows])
    splits = np.asarray([r["split"] for r in rows])
    fit, ev = splits == "fit", splits == "eval"
    control = labels == "__random__"
    cases, skipped, geometries = [], [], []

    for pi, pair in enumerate(plan["pairs"]):
        counts = {label: {split: int(((labels == label) & (splits == split)).sum())
                           for split in ("fit", "eval")} for label in pair}
        if any(counts[l]["fit"] < args.min_fit or counts[l]["eval"] < args.min_eval for l in pair):
            skipped.append({"pair": pair, "reason": "insufficient labelled rows", "counts": counts})
            log("SKIP", pair, counts)
            continue
        geometries.append({"pair": pair, "spaces": geometry({"base_normalized": X,
                            "token_centred": HB, "ae": Z}, labels, splits, pair)})
        for T, S in (pair, pair[::-1]):
            start = time.time()
            ti = np.nonzero(ev & (labels == T))[0]
            si = np.nonzero(ev & (labels == S))[0]
            ci = np.nonzero(ev & control)[0]
            indices = torch.as_tensor(np.concatenate([ti, si, ci]), device=dev)
            groups = {"target": (0, len(ti)), "sibling": (len(ti), len(ti) + len(si)),
                      "random": (len(ti) + len(si), len(indices))}
            fit_target = fit & (labels == T)
            fit_complement = fit & (labels != T) & (labels != S)
            assert fit_complement.sum() > 0 and len(ci) > 0
            base_h, targets = H[indices], nxt[indices]
            # Cache clean distributions once per case; exactly same clean baseline for all arms.
            clean = torch.cat([head(norm(base_h[i:i + args.score_chunk])).log_softmax(-1)
                               for i in range(0, len(indices), args.score_chunk)])
            result = {}
            for arm in ARMS:
                source = spaces[arm]
                aT = source[torch.as_tensor(np.nonzero(fit_target)[0], device=dev)].cpu().numpy()
                aC = source[torch.as_tensor(np.nonzero(fit_complement)[0], device=dev)].cpu().numpy()
                acts = np.concatenate([aT, aC])
                sal = dprime_saliency(acts, np.r_[np.ones(len(aT), bool), np.zeros(len(aC), bool)])
                rep = aC.mean(0)
                result[arm] = []
                for p in args.grid:
                    salient = select_top(sal, p=p)
                    lo, hi, _, _ = fit_ranges(aT, salient, tao=2.0)
                    original = source[indices]
                    edited = gate_edit(lo, hi, rep, dev)(original)
                    changed = (edited != original).any(1)
                    edited_h = base_h.clone()
                    active = changed.nonzero(as_tuple=True)[0]
                    for offset in range(0, len(active), args.score_chunk):
                        selected = active[offset:offset + args.score_chunk]
                        if arm == "base":
                            edited_h[selected] = edited[selected]
                        elif arm == "token_centred":
                            edited_h[selected] += (edited[selected] - original[selected]) * std
                        else:
                            ids = tok[indices[selected]]
                            edited_h[selected] += (ae.decode(edited[selected], ids) -
                                                   ae.decode(original[selected], ids)) * std
                    assert torch.isfinite(edited_h).all()
                    dnll = torch.zeros(len(indices), device=dev)
                    kl = torch.zeros_like(dnll)
                    for offset in range(0, len(active), args.score_chunk):
                        selected = active[offset:offset + args.score_chunk]
                        lc = clean[selected]
                        le = head(norm(edited_h[selected])).log_softmax(-1)
                        y = targets[selected, None]
                        dnll[selected] = (lc.gather(1, y) - le.gather(1, y)).flatten()
                        kl[selected] = (lc.exp() * (lc - le)).sum(-1)
                    metrics = {g: {"dnll": float(dnll[start:end].mean()),
                                   "kl": float(kl[start:end].mean()),
                                   "frac_edited": float(changed[start:end].float().mean()),
                                   "n": end - start} for g, (start, end) in groups.items()}
                    result[arm].append({"p": p, "n_dimensions": int(salient.sum()), "metrics": metrics})
                del acts, aT, aC
            case = {"pair_id": pi, "T": T, "S": S, "counts": counts,
                    "n_fit_complement": int(fit_complement.sum()), "res": result}
            cases.append(case)
            standard = {a: next(r["metrics"] for r in result[a] if r["p"] == 0.3) for a in ARMS}
            log(f"{args.name} T={T} sibling={S} ({time.time() - start:.1f}s): " + " | ".join(
                f"{a} target/sibling/random=" + "/".join(f"{standard[a][g]['dnll']:.4f}"
                for g in ("target", "sibling", "random")) for a in ARMS))
            del clean
    summaries = {"all_three": summarize(cases, list(ARMS)),
                 "base_ae": summarize(cases, ["base", "ae"]),
                 "kl_all_three": summarize(cases, list(ARMS), metric="kl")}
    standard = {}
    for arm in ARMS:
        records = [next(r["metrics"] for r in c["res"][arm] if r["p"] == 0.3) for c in cases]
        standard[arm] = {g: float(np.mean([r[g]["dnll"] for r in records])) if records else None
                        for g in ("target", "sibling", "random")}
    return {"cases": cases, "skipped_pairs": skipped, "geometry": geometries,
            "summary": summaries, "standard_p_0.3": standard}


def write_report(output, path):
    primary = output["summary"]["all_three"]["matched"]
    lines = [f"# {output['meta']['name']}: NeuronLens", "",
             "Same labelled examples and clean baseline in all spaces. Lower collateral ΔNLL is better at matched target ΔNLL.",
             "", f"Complete matched pairs: {primary['n_complete_pairs']}; cases: {primary['n_complete_cases']}.",
             f"Mean matched target ΔNLL: {primary['L_mean']} nats (varies by case).", "",
             "| Space | Target ΔNLL | Sibling ΔNLL | Random ΔNLL |",
             "|---|---:|---:|---:|"]
    for arm, values in primary["arms"].items():
        values = [f"{values[g]:.5f}" if values[g] is not None else "unavailable"
                  for g in ("target", "sibling", "random")]
        lines.append(f"| {arm} | " + " | ".join(values) + " |")
    lines += ["", "## Fixed target loss reach", "",
              "| Target ΔNLL | Base cases | Token-centred cases | AE cases | Complete common pairs |",
              "|---|---:|---:|---:|---:|"]
    for level, values in output["summary"]["all_three"]["fixed"].items():
        reach = values["reach_by_arm"]
        lines.append(f"| {level} | {reach['base']} | {reach['token_centred']} | {reach['ae']} | {values['n_complete_pairs']} |")
    lines += ["", "These are descriptive results for the selected labels, with both orientations averaged per pair.",
              "Matched interpolation uses the first upward crossing in actual edit-strength order, without extrapolation.",
              "Method: d-prime saliency, ±2σ gates, complement-mean replacement; AE uses decoded differences added to the clean residual.",
              "Damage measures next-token prediction at the completed span; it does not demonstrate erasure of a semantic label.",
              "Random controls are non-annotated positions in the same dataset. Dataset-specific provenance and split limitations are in the input manifest.", ""]
    Path(path).write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--score-chunk", type=int, default=256)
    ap.add_argument("--fit-per-label", type=int, default=200)
    ap.add_argument("--eval-per-label", type=int, default=120)
    ap.add_argument("--controls-per-split", type=int, default=400)
    ap.add_argument("--cap-per-doc", type=int, default=4)
    ap.add_argument("--min-fit", type=int, default=40)
    ap.add_argument("--min-eval", type=int, default=30)
    ap.add_argument("--min-prefix", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", type=float, nargs="+", default=[0.003, 0.01, 0.03, 0.1, 0.3, 0.5])
    args = ap.parse_args()
    assert args.grid == sorted(set(args.grid)) and 0.3 in args.grid
    started = time.time()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    from geoae.checkpoint import load_tokenizer
    tokenizer = load_tokenizer(LM_NAME)
    plan = json.loads(Path(args.plan).read_text())
    settings = {k: getattr(args, k) for k in ("context", "fit_per_label", "eval_per_label",
                "controls_per_split", "cap_per_doc", "min_prefix", "seed")}
    identity = {"version": 2, "data_sha256": digest(args.data), "plan": plan, "model": LM_NAME,
                "layer": LAYER, "settings": settings, "dtype": "float32"}
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ae, mean, std, _ = load_ae_checkpoint(CKPT, dev, allow_token_bias=True)
    tb = TokenBiasLookup(TBFILE, dev)
    lm = load_lm(LM_NAME, device=dev)
    _, _, original_norm, original_head = locate_lm_parts(lm)
    norm, head = copy.deepcopy(original_norm).float(), copy.deepcopy(original_head).float()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            assert json.loads(str(saved["identity"])) == identity, "cache identity mismatch"
            H = torch.as_tensor(saved["H"], device=dev)
            meta = json.loads(str(saved["metadata"]))
        rows = [json.loads(line) for line in cache.with_suffix(".rows.jsonl").read_text().splitlines()]
        assert len(rows) == len(H) and digest(cache.with_suffix(".rows.jsonl")) == meta["rows_sha256"], "cache row alignment mismatch"
        log("reused cache", cache, len(H), "rows")
    else:
        rows, meta = prepare_rows(args.data, plan, tokenizer, args)
        h, sanity = capture_rows(rows, lm, norm, head, tokenizer, args, dev)
        meta["sanity"] = sanity
        for row in rows:
            row.pop("input_ids", None)
        cache.with_suffix(".rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        meta["rows_sha256"] = digest(cache.with_suffix(".rows.jsonl"))
        np.savez(cache, H=h, identity=np.asarray(json.dumps(identity, sort_keys=True)),
                 metadata=np.asarray(json.dumps(meta)))
        H = torch.as_tensor(h, device=dev)
        log("saved", cache, "sanity", sanity)
    # The final decoder output has no later attention blocks. Reuse its exact norm/head.
    del lm, original_norm, original_head
    torch.cuda.empty_cache()
    output = evaluate(rows, H, ae, tb, mean, std, norm, head, plan, args, dev)
    output["meta"] = {"name": args.name, "args": vars(args), "plan": plan, "cache_identity": identity,
                      "selection": meta, "seconds": time.time() - started,
                      "checkpoint": CKPT, "token_bias": TBFILE, "n_evaluated_cases": len(output["cases"]),
                      "interpretation": "next-token predictive damage; modified NeuronLens consistent with earlier comparison"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, allow_nan=False))
    write_report(output, out.with_suffix(".md"))
    log("DONE", args.name, f"{time.time() - started:.1f}s", "wrote", out)


if __name__ == "__main__":
    main()
