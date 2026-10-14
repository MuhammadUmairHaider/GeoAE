"""Atlas class corpus, stages `draw` and `prep` (no model run; CPU only, needs the Atlas concept table + the 300k cache).

  draw   freeze the three class draws (seeds 0,1,2) in <out>/draws.json BEFORE any model run (refuses to overwrite).
         --seeds 3,4,...: ADDITIONAL draws under the same rules, one frozen file per seed (<out>/s<seed>/draw.json);
         a seed that already has its file is left as it is, so the same line can be run again.
  prep   per seed: decode the eligible chunks, split by DOCUMENT into few-shot / fit / test, write
         <out>/s<seed>/{data.parquet, cfg.json, relatedness.json}. cfg.json is what set_dataset("atlas_s<seed>") loads.

    python eval_out/atlas_classes_prep.py draw
    python eval_out/atlas_classes_prep.py draw --seeds 3,4,5
    python eval_out/atlas_classes_prep.py prep --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoae.interp import atlas_classes as AC  # noqa: E402
from geoae.interp import atlas_siblings as AS  # noqa: E402

OUT_DEFAULT = "results/delta/d6144/interventions/atlas_classes"
N_CONCEPTS = 16790


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def load_atlas(atlas_dir):
    import pandas as pd
    conc = pd.read_parquet(Path(atlas_dir) / "fineweb-concept-atlas.parquet")
    assert (conc.concept_id.to_numpy() == np.arange(len(conc))).all()
    units = AC.build_units(conc.name.tolist(), conc.description.tolist(), conc.concept_id.to_numpy(),
                           conc.chunk_count.to_numpy(), (conc.concept_type == "content").to_numpy())
    return conc, units


def load_cache_labels(cache_dir):
    z = np.load(Path(cache_dir) / "chunks.npz", allow_pickle=False)
    return z, dict(ptr=z["label_ptr"], ids=z["label_ids"], n=len(z["doc_int_id"]), fingerprint=str(z["fingerprint"]))


def stage_draw(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "draws.json"
    R = json.loads(json.dumps(AC.RULES))
    extra = [int(s) for s in args.seeds.split(",")] if args.seeds else []
    if extra:
        clash = sorted(set(extra) & set(R["seeds"]))
        if clash:
            raise SystemExit(f"seeds {clash} are the original draws frozen in {f}; pick other seeds")
        todo = [s for s in extra if not (out / f"s{s}" / "draw.json").exists()]
        for s in sorted(set(extra) - set(todo)):
            log(f"seed {s}: {out / f's{s}' / 'draw.json'} exists, kept")
        if not todo:
            return
    elif f.exists() and not args.overwrite:
        raise SystemExit(f"{f} exists and is the frozen draw file (sha256 {json.load(open(f))['sha256'][:16]}); "
                         f"--overwrite to replace it")
    conc, units = load_atlas(args.atlas_dir)
    z, lab = load_cache_labels(args.cache)
    log(f"{len(units):,} content units; cache {lab['n']:,} chunks, fingerprint {lab['fingerprint'][:16]}")
    M = AS.labels_csr(lab["ptr"], lab["ids"], lab["n"], [u["ids"] for u in units], N_CONCEPTS)
    cnt = np.diff(M.indptr)
    names = [u["name"] for u in units]
    cand = [i for i, u in enumerate(units) if cnt[i] >= R["min_chunks"] and AC.name_ok(u["name"], R)]
    n_big = int((cnt >= R["min_chunks"]).sum())
    log(f"{n_big} units have >= {R['min_chunks']} chunks; {len(cand)} pass the name filter")
    draws = []
    for seed in (todo if extra else R["seeds"]):
        d = AC.draw_classes(seed, M, names, cand, R)
        sub = M[:, d["units"]]
        excl = d["stats"]["n_exclusive"]
        draws.append(dict(
            seed=seed, attempts=d["attempts"], n_rejected=d["n_rejected"], rejections=d["rejections"],
            max_pair_jaccard=round(d["stats"]["max_jaccard"], 5),
            classes=[dict(name=units[u]["name"], pid=units[u]["pid"], ids=units[u]["ids"], desc=units[u]["desc"],
                          cache_chunks=int(cnt[u]), n_exclusive=int(excl[k]))
                     for k, u in enumerate(d["units"])]))
        log(f"seed {seed}: {d['attempts']} attempts ({d['n_rejected']} rejected {d['rejections']}); "
            + ", ".join(c["name"] for c in draws[-1]["classes"]))
    universe = dict(n_candidate_units=len(cand), n_units_ge_min_chunks=n_big)
    cache = dict(dir=str(args.cache), n_chunks=lab["n"], fingerprint_sha256=lab["fingerprint"])
    if extra:       # one frozen file per additional seed, same format as draws.json with a single draw
        earlier = {}
        for g in [f] + sorted(out.glob("s*/draw.json")):
            if g.exists():
                for d in json.load(open(g))["draws"]:
                    earlier.update({c["name"]: d["seed"] for c in d["classes"]})
        for d in draws:
            rep = {c["name"]: earlier[c["name"]] for c in d["classes"] if c["name"] in earlier}
            log(f"seed {d['seed']}: {len(rep)} of {len(d['classes'])} classes also in an earlier draw {rep if rep else ''}")
            body = dict(rules=dict(R, seeds=[d["seed"]]), universe=universe, cache=cache, draws=[d])
            body["sha256"] = AC.canonical_sha256(body)
            g = out / f"s{d['seed']}" / "draw.json"
            g.parent.mkdir(parents=True, exist_ok=True)
            g.write_text(json.dumps(body, indent=1))
            log(f"wrote {g}  sha256 {body['sha256']}")
            earlier.update({c["name"]: d["seed"] for c in d["classes"]})
        return
    body = dict(rules=R, universe=universe, cache=cache, draws=draws)
    body["sha256"] = AC.canonical_sha256(body)
    f.write_text(json.dumps(body, indent=1))
    log(f"wrote {f}  sha256 {body['sha256']}")


def read_draws(out, seed=None):
    """The frozen draw file that holds `seed`: draws.json (seeds 0,1,2) or s<seed>/draw.json (an additional draw)."""
    f = Path(out) / "draws.json"
    if seed is not None and seed not in AC.RULES["seeds"]:
        f = Path(out) / f"s{seed}" / "draw.json"
    d = json.load(open(f))
    sha = d.pop("sha256")
    assert AC.canonical_sha256(d) == sha, f"{f} was modified after freezing"
    d["sha256"] = sha
    return d


def embed(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval()
    with torch.no_grad():
        enc = tk(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).float().numpy()


def relatedness(classes, atlas_dir):
    """MiniLM cosine of 'name: description' and the Atlas-wide co-occurrence (PMI with +1 smoothing, Jaccard) of the
    primary concept ids, read off the packed co-occurrence matrix of the full 95M-chunk Atlas."""
    names = [c["name"] for c in classes]
    E = embed([f"{c['name']}: {c['desc']}" for c in classes])
    cos = E @ E.T
    pk = np.load(Path(atlas_dir) / "fineweb-concept-cooccurrence-matrix/fineweb-atlas-cooccurrence-upper-uint32.npy", mmap_mode="r")
    K = len(classes)
    pid = np.array([c["pid"] for c in classes])
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            C[i, j] = float(pk[AS.packed_index(np.array([pid[i]]), np.array([pid[j]]), N_CONCEPTS)[0]])
    d = np.diag(C).copy()
    pmi = np.log((C + 1) * AS.N_CHUNKS_ATLAS / ((d[:, None] + 1) * (d[None, :] + 1)))
    jac = C / np.maximum(d[:, None] + d[None, :] - C, 1.0)
    return dict(names=names, minilm_cos=cos.round(5).tolist(), cooc_count=C.tolist(), cooc_pmi=pmi.round(5).tolist(),
                cooc_jaccard=jac.round(6).tolist(), embedder="sentence-transformers/all-MiniLM-L6-v2",
                embed_text="{name}: {description}", cooc_source="full-Atlas packed co-occurrence, primary concept ids")


def stage_prep(args):
    from transformers import AutoTokenizer
    out = Path(args.out_dir)
    D = read_draws(out, args.seed)
    R = D["rules"]
    dr = next(d for d in D["draws"] if d["seed"] == args.seed)
    classes = dr["classes"]
    K = len(classes)
    names = [c["name"] for c in classes]
    z, lab = load_cache_labels(args.cache)
    assert lab["fingerprint"] == D["cache"]["fingerprint_sha256"], "cache differs from the one the draws were made on"
    M = AS.labels_csr(lab["ptr"], lab["ids"], lab["n"], [c["ids"] for c in classes], N_CONCEPTS)
    ok, why, st = AC.check_draw(M, names, R)
    assert ok and st["n_exclusive"] == [c["n_exclusive"] for c in classes], f"draw does not reproduce: {why} {st}"
    doc = z["doc_int_id"].astype(np.int64)
    n_tok = z["n_tok"]
    split = AC.split_of_bucket(AC.doc_bucket(doc, args.seed, R["n_split_buckets"]), R)
    rng = np.random.default_rng([args.seed, 12345])
    tok_ptr, tok_flat = z["tok_ptr"], z["tok_flat"]
    tk = AutoTokenizer.from_pretrained(args.model_name)

    def text_of(i):
        ids = tok_flat[tok_ptr[i]:tok_ptr[i + 1]].astype(np.int64)
        if len(ids) and ids[0] == tk.bos_token_id:
            ids = ids[1:]
        return AC.clean_text(tk.decode(ids, skip_special_tokens=True))

    elig = [AC.exclusive_chunks(M, k) for k in range(K)]
    # few-shot exemplars: 4 classes at random, one chunk each from the few-shot document pool
    ex_classes = [int(x) for x in rng.choice(K, size=R["n_exemplars"], replace=False)]
    lo, hi = R["exemplar_tok_range"]
    exemplars, ex_chunks = [], set()
    for k in ex_classes:
        pool = elig[k][(split[elig[k]] == 0) & (n_tok[elig[k]] >= lo) & (n_tok[elig[k]] <= hi)]
        i = int(rng.choice(pool))
        exemplars.append((text_of(i), k))
        ex_chunks.add(i)
    log(f"seed {args.seed} exemplars (class: text): " + " | ".join(f"{names[k]}: {t[:70]}" for t, k in exemplars))
    head = AC.build_prompt_head(names, exemplars)
    rows, seen, n_dup = [], set(a for a, _ in exemplars), 0
    for k in range(K):
        for sp, nm in ((1, "fit"), (2, "test")):
            cand = rng.permutation(elig[k][split[elig[k]] == sp])
            got = 0
            for i in cand:
                t = text_of(int(i))
                if t in seen or len(t) < 20:
                    n_dup += 1
                    continue
                seen.add(t)
                rows.append(dict(text=t, label=k, split=nm, chunk_idx=int(i), doc=int(doc[i]), n_tok=int(n_tok[i])))
                got += 1
                if got >= R["cap_per_class_split"]:
                    break
            log(f"  {names[k]:<40} {nm:<4} eligible chunks {len(cand):>6} -> kept {got}")
    import pandas as pd
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sd = out / f"s{args.seed}"
    sd.mkdir(parents=True, exist_ok=True)
    df.to_parquet(sd / "data.parquet", index=False)
    # documents never straddle splits (by construction) -- verify, and verify exemplar docs are not in fit/test
    dsplit = df.groupby("doc").split.nunique()
    assert (dsplit == 1).all(), "a document appears in both fit and test"
    ex_docs = {int(doc[i]) for i in ex_chunks}
    assert not (set(df.doc) & ex_docs), "an exemplar document leaked into fit/test"
    max_new = max(len(tk(n, add_special_tokens=False)["input_ids"]) for n in names) + 3
    longest = df.sort_values("n_tok").text.tail(24).tolist()
    plen = max(len(tk(head.format(t))["input_ids"]) for t in longest)
    assert plen < 1000, f"prompt of {plen} tokens would hit the 1024 truncation"
    cfg = dict(classes=names, dataset_name=f"local:atlas_s{args.seed}", dataset_split="train", text_column="text",
               label_column="label", split_column="split", local_parquet="data.parquet", prompt_head=head,
               max_new=max_new, draws_sha256=D["sha256"], seed=args.seed,
               exemplars=[dict(text=t, cls=names[k]) for t, k in exemplars], max_prompt_tokens=plen,
               scoring_note="exact string match of the generated class name (up to '}'); first-token collisions are harmless")
    (sd / "cfg.json").write_text(json.dumps(cfg, indent=1))
    first = {}
    for n in names:
        first.setdefault(tk(n, add_special_tokens=False)["input_ids"][0], []).append(n)
    log(f"max_new={max_new}, longest prompt {plen} tokens, duplicate/short chunks skipped {n_dup}; "
        f"classes sharing a first token: {[v for v in first.values() if len(v) > 1]}")
    (sd / "relatedness.json").write_text(json.dumps(relatedness(classes, args.atlas_dir)))
    log(f"wrote {sd} ({len(df)} rows)")
    print("PROMPT HEAD:\n" + head.format("<DOC>"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["draw", "prep"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="", help="draw: comma list of ADDITIONAL seeds (not 0,1,2), one frozen file each")
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--atlas_dir", default="/scratch/dogar/geoae/atlas")
    ap.add_argument("--cache", default="/scratch/dogar/geoae/atlas_l27")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    {"draw": stage_draw, "prep": stage_prep}[args.stage](args)


if __name__ == "__main__":
    main()
