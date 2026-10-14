"""Step 1 of the Atlas sibling test: freeze the sibling pairs and the null pairs BEFORE any activation exists.

Everything here uses only the public FineWeb Atlas (guidelabs/fineweb-atlas): the concept table (names,
descriptions, LCC letter, chunk counts), the packed concept co-occurrence matrix, a public sentence embedder, and
the chunk LABELS of the first N_REF chunks of the deterministic stream (label ids only: no text is tokenised, no
language model is run). Pair membership therefore cannot depend on any model geometry.

Rules (all constants are in RULES below and are written into pairs.json together with the data they came from):

  universe  content concepts, case-insensitive duplicate names merged into one unit (primary = most frequent member),
            LCC letter != "None", present in >= MIN_STREAM chunks of the first N_REF ok chunks.
  Rule E    external semantics: embed "name: description" with all-MiniLM-L6-v2; a pair (A, B) is E when both lie in
            the same LCC letter, B is A's nearest neighbour among same-LCC units whose cosine is inside
            [TAU_LO, TAU_HI] (and A is B's: MUTUAL nearest neighbours), after the synonym/containment guard.
  Rule C    co-occurrence: cosine of PPMI co-occurrence profiles (all 16,790 concepts as contexts) >= TAU_C while the
            two concepts rarely share a chunk (overlap coefficient C_AB / min(C_AA, C_BB) <= MAX_OVERLAP): mutual
            nearest neighbours by that second-order similarity among same-LCC units. "Substitutes", not synonyms.
  guard     drop a pair when one name's tokens contain the other's, or when the chunk Jaccard of the pair in the
            Atlas co-occurrence is > MAX_JACCARD.
  data      keep a pair when BOTH members have >= MIN_EXCL exclusive chunks (A without B, B without A) among the
            first N_REF ok chunks (the same stream the extractor reads).
  nulls     per kept sibling pair one prevalence-matched random same-LCC pair and one random cross-LCC pair, drawn
            from the universe with the same guard and the same data filter, never from the sibling candidates.

Also written (pairs_extended.json): the one-way variant of both rules (A's nearest neighbour need not reciprocate),
~3x as many pairs, for a run that wants more statistical power. Nothing else differs.

    scripts/delta/py eval_out/atlas_siblings_pairs.py --atlas_dir /scratch/dogar/geoae/atlas \
        --out_dir results/delta/d6144/evals/atlas_siblings
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoae.interp import atlas_siblings as AS  # noqa: E402

RULES = dict(
    n_ref_chunks=300_000,
    min_stream=100,
    min_excl=100,
    embedder="sentence-transformers/all-MiniLM-L6-v2",
    embed_text="{name}: {description}",
    embed_max_len=256,
    E=dict(k=1, mutual_only=True, tau_lo=0.55, tau_hi=0.80),
    C=dict(k=1, mutual_only=True, tau_c=0.60, max_overlap=0.05, ppmi_min_count=20),
    guard=dict(max_jaccard=0.15, name_containment=True),
    same_lcc=True,
    exclude_lcc="None",
    null=dict(seed=0, tol=0.10),
)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def embed(texts, model_name, max_len, device):
    """Mean-pooled, L2-normalised sentence embeddings (the sentence-transformers recipe for MiniLM)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            enc = tk(texts[i:i + 256], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).float().cpu())
    return torch.cat(out).numpy(), str(getattr(mdl.config, "_commit_hash", None) or "unknown")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas_dir", default="/scratch/dogar/geoae/atlas",
                    help="holds fineweb-concept-atlas.parquet, the co-occurrence .npy and annotated shards (downloaded if missing)")
    ap.add_argument("--out_dir", default="results/delta/d6144/evals/atlas_siblings")
    ap.add_argument("--n_ref", type=int, default=RULES["n_ref_chunks"])
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing pairs.json (the frozen list); the old and new pair-list hashes are printed")
    args = ap.parse_args()
    frozen = Path(args.out_dir) / "pairs.json"
    if frozen.exists() and not args.overwrite:
        raise SystemExit(f"{frozen} already exists and is the frozen pair list; pass --overwrite to regenerate it")
    old_hash = json.load(open(frozen)).get("pair_list_sha256") if frozen.exists() else None
    R = json.loads(json.dumps(RULES))
    R["n_ref_chunks"] = args.n_ref
    atlas = Path(args.atlas_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    import torch

    # ---- concept table, units -------------------------------------------------------------------------------
    from huggingface_hub import hf_hub_download
    for f in ("fineweb-concept-atlas.parquet",
              "fineweb-concept-cooccurrence-matrix/fineweb-atlas-cooccurrence-upper-uint32.npy",
              "fineweb-concept-cooccurrence-matrix/fineweb-atlas-cooccurrence-upper-uint32.json"):
        if not (atlas / f).exists():
            hf_hub_download(AS.ATLAS_REPO, f, repo_type="dataset", local_dir=str(atlas))
    conc = pd.read_parquet(atlas / "fineweb-concept-atlas.parquet")
    n_concepts = len(conc)
    CID = conc.concept_id.to_numpy().astype(np.int64)
    assert (CID == np.arange(n_concepts)).all()
    col_marginal = conc.chunk_count.to_numpy().astype(np.float64)
    is_content = (conc.concept_type == "content").to_numpy()
    rows = np.flatnonzero(is_content)                              # positions into conc of content concepts
    names_c = [conc.name.iloc[i] for i in rows]
    counts_c = col_marginal[rows]
    groups = AS.merge_duplicate_names(names_c, counts_c)           # positions into `rows`
    units = []
    for g in groups:
        pm = AS.primary_member(g, counts_c)
        pr = int(rows[pm])
        units.append(dict(ids=sorted(int(CID[rows[m]]) for m in g), pid=int(CID[pr]), name=str(conc.name.iloc[pr]),
                          desc=str(conc.description.iloc[pr]), lcc=str(conc.taxonomy_lcc_path_primary.iloc[pr]),
                          count=int(sum(col_marginal[rows[m]] for m in g))))
    log(f"{len(rows):,} content concepts -> {len(units):,} units after merging {len(rows) - len(units)} duplicates")

    # ---- label stream: first N_REF ok chunks, LABELS ONLY ------------------------------------------------------
    t0 = time.time()
    docs, cids, lab = [], [], []
    for b in AS.iter_atlas_chunks(atlas, R["n_ref_chunks"], ["doc_int_id", "chunk_id", "content_ids"]):
        docs += b["doc_int_id"]
        cids += b["chunk_id"]
        lab += b["content_ids"]
    fp = AS.stream_fingerprint(docs, cids)
    ptr, flat = AS.ragged_to_csr(lab)
    log(f"streamed {len(lab):,} ok chunks in {time.time() - t0:.0f}s; fingerprint {fp[:16]}")
    M_all = AS.labels_csr(ptr, flat, len(lab), [u["ids"] for u in units], n_concepts)
    stream_cnt = np.diff(M_all.indptr)
    keep = [i for i, u in enumerate(units) if stream_cnt[i] >= R["min_stream"] and u["lcc"] != R["exclude_lcc"]]
    U = [units[i] for i in keep]
    n = len(U)
    log(f"universe: {n:,} units with >= {R['min_stream']} stream chunks and an LCC letter")
    # chunk x universe-unit incidence (columns re-indexed to the universe)
    M = AS.labels_csr(ptr, flat, len(lab), [u["ids"] for u in U], n_concepts)
    lcc = np.array([u["lcc"] for u in U])
    names = [u["name"] for u in U]
    count = np.array([u["count"] for u in U], dtype=np.float64)
    log_count = np.log(count)

    # ---- Rule E: embeddings ---------------------------------------------------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    texts = [R["embed_text"].format(name=u["name"], description=u["desc"]) for u in U]
    E, revision = embed(texts, R["embedder"], R["embed_max_len"], dev)
    cos = E @ E.T
    same = (lcc[:, None] == lcc[None, :])
    np.fill_diagonal(same, False)

    # ---- Rule C: PPMI profiles --------------------------------------------------------------------------------------
    packed = np.load(atlas / "fineweb-concept-cooccurrence-matrix/fineweb-atlas-cooccurrence-upper-uint32.npy", mmap_mode="r")
    prof = np.zeros((n, n_concepts))
    for k, u in enumerate(U):
        prof[k] = AS.packed_rows(packed, n_concepts, u["ids"]).sum(0)
    cij = np.zeros((n, n))
    for v, u in enumerate(U):
        cij[:, v] = prof[:, u["ids"]].sum(1)
    overlap, jac = AS.first_order_stats(cij, count)
    np.fill_diagonal(overlap, 0.0)
    np.fill_diagonal(jac, 0.0)
    for k, u in enumerate(U):
        prof[k, u["ids"]] = 0.0                                    # the unit's own columns are not co-occurrence
    P = AS.ppmi_profiles(prof, count, col_marginal, AS.N_CHUNKS_ATLAS, min_count=R["C"]["ppmi_min_count"])
    S2 = AS.cosine_matrix(P)
    del prof, P
    log("co-occurrence profiles done")

    guard = AS.synonym_guard(names, jac, R["guard"]["max_jaccard"])

    def rule_pairs(mutual_only):
        e = AS.top_k_pairs(cos, same & guard & (cos >= R["E"]["tau_lo"]) & (cos <= R["E"]["tau_hi"]), R["E"]["k"])
        c = AS.top_k_pairs(S2, same & guard & (overlap <= R["C"]["max_overlap"]) & (S2 >= R["C"]["tau_c"]), R["C"]["k"])
        keepm = (lambda m: m) if mutual_only else (lambda m: True)
        return ({(i, j): m for (i, j, m) in e if keepm(m)}, {(i, j): m for (i, j, m) in c if keepm(m)})

    E_main, C_main = rule_pairs(True)
    E_ext, C_ext = rule_pairs(False)
    candidates = set(E_main) | set(C_main) | set(E_ext) | set(C_ext)       # forbidden for the nulls

    def build(Epairs, Cpairs):
        allp = sorted(set(Epairs) | set(Cpairs))
        ex = AS.exclusive_counts(M, allp)
        recs, dropped = [], 0
        for (i, j), e in zip(allp, ex):
            if e[0] < R["min_excl"] or e[1] < R["min_excl"]:
                dropped += 1
                continue
            inE, inC = (i, j) in Epairs, (i, j) in Cpairs
            tag = "both" if inE and inC else ("E" if inE else "C")
            recs.append(dict(
                a=U[i]["pid"], b=U[j]["pid"], a_ids=U[i]["ids"], b_ids=U[j]["ids"], a_name=U[i]["name"], b_name=U[j]["name"],
                a_desc=U[i]["desc"], b_desc=U[j]["desc"], lcc=U[i]["lcc"], tag=tag,
                emb_cos=round(float(cos[i, j]), 4), ppmi_cos=round(float(S2[i, j]), 4),
                overlap=round(float(overlap[i, j]), 5), jaccard=round(float(jac[i, j]), 5),
                mutual_E=bool(Epairs.get((i, j), False)) if inE else None,
                mutual_C=bool(Cpairs.get((i, j), False)) if inC else None,
                atlas_count=[U[i]["count"], U[j]["count"]], stream_exclusive=[int(e[0]), int(e[1])],
                stream_shared=int(e[2]), _ij=(i, j)))
        return recs, dropped

    main_recs, drop_main = build(E_main, C_main)
    ext_recs, drop_ext = build(E_ext, C_ext)
    for r in main_recs:
        assert r["tag"] in ("E", "C", "both")
    tags = {t: sum(r["tag"] == t for r in main_recs) for t in ("E", "C", "both")}
    log(f"main set: {len(main_recs)} pairs survive the data filter ({drop_main} dropped) {tags}; "
        f"extended one-way set: {len(ext_recs)} ({drop_ext} dropped)")

    # ---- nulls ----------------------------------------------------------------------------------------------------
    def data_ok(x, y):
        ex = AS.exclusive_counts(M, [(x, y)])[0]
        return ex[0] >= R["min_excl"] and ex[1] >= R["min_excl"]

    def allowed(x, y):
        return bool(guard[x, y]) and data_ok(x, y)

    eligible = np.ones(n, dtype=bool)

    def nulls_for(recs, seed):
        sib = [r["_ij"] for r in recs]
        res = {}
        for mode in ("same", "cross"):
            rng = np.random.default_rng([R["null"]["seed"], seed, 0 if mode == "same" else 1])
            got = AS.sample_null_pairs(sib, log_count, lcc, eligible, candidates, mode, rng, R["null"]["tol"], allowed)
            lst = []
            for r, g in zip(recs, got):
                if g is None:
                    lst.append(None)
                    continue
                x, y, tol = g
                ex = AS.exclusive_counts(M, [(x, y)])[0]
                lst.append(dict(a=U[x]["pid"], b=U[y]["pid"], a_ids=U[x]["ids"], b_ids=U[y]["ids"], a_name=U[x]["name"],
                                b_name=U[y]["name"], lcc_a=U[x]["lcc"], lcc_b=U[y]["lcc"], matched_to=[r["a"], r["b"]],
                                match_tol=tol, atlas_count=[U[x]["count"], U[y]["count"]],
                                stream_exclusive=[int(ex[0]), int(ex[1])], stream_shared=int(ex[2])))
            res[mode] = lst
        return res

    nulls_main = nulls_for(main_recs, 0)
    nulls_ext = nulls_for(ext_recs, 1)
    for mode in ("same", "cross"):
        miss = sum(x is None for x in nulls_main[mode])
        if miss:
            log(f"WARNING: {miss} sibling pairs found no {mode}-LCC null")

    # ---- write ------------------------------------------------------------------------------------------------------
    def pct(M_, mask):
        iu = np.triu_indices(n, 1)
        v = M_[iu][mask[iu]]
        return {str(p): round(float(np.percentile(v, p)), 4) for p in (50, 90, 99, 99.9)}

    ctx = {"same_lcc_pair_emb_cos_percentiles": pct(cos, same), "same_lcc_pair_ppmi_cos_percentiles": pct(S2, same),
           "same_lcc_pair_overlap_percentiles": pct(overlap, same), "n_universe": n, "n_units": len(units)}

    def dump(recs, nulls, name, extended):
        pairs = [{k: v for k, v in r.items() if k != "_ij"} for r in recs]
        doc = {
            "meta": {"kind": "atlas sibling pairs (frozen before any activation was computed)", "extended_one_way": extended,
                     "rules": R, "context_distributions": ctx, "embedder_revision": revision,
                     "atlas_repo": AS.ATLAS_REPO, "stream": {"n_chunks": len(lab), "fingerprint_sha256": fp,
                                                             "shard_files": ["fineweb-atlas-annotated/shard_0000.parquet"]},
                     "concept_table_sha256": AS.file_sha256(atlas / "fineweb-concept-atlas.parquet"),
                     "cooccurrence_sha256": AS.file_sha256(atlas / "fineweb-concept-cooccurrence-matrix/fineweb-atlas-cooccurrence-upper-uint32.npy"),
                     "counts": {"pairs": len(pairs), "by_tag": {t: sum(p["tag"] == t for p in pairs) for t in ("E", "C", "both")},
                                "null_same": sum(x is not None for x in nulls["same"]),
                                "null_cross": sum(x is not None for x in nulls["cross"])},
                     "pair_list": "sha256 over sorted [min(a,b), max(a,b), tag] with a, b the primary concept ids"},
            "pair_list_sha256": AS.pair_list_sha256(pairs),
            "null_same_sha256": AS.pair_list_sha256([dict(x, tag="null") for x in nulls["same"] if x]),
            "null_cross_sha256": AS.pair_list_sha256([dict(x, tag="null") for x in nulls["cross"] if x]),
            "pairs": pairs, "null_same": nulls["same"], "null_cross": nulls["cross"],
        }
        path = out_dir / name
        path.write_text(json.dumps(doc, indent=1))
        (out_dir / (name + ".sha256")).write_text(f"{AS.file_sha256(path)}  {name}\n")
        log(f"wrote {path}  pair_list_sha256={doc['pair_list_sha256']}  file_sha256={AS.file_sha256(path)}")
        return doc

    doc = dump(main_recs, nulls_main, "pairs.json", False)
    if old_hash is not None:
        log(f"previous pair_list_sha256 {old_hash} -> new {doc['pair_list_sha256']}: "
            f"{'IDENTICAL' if old_hash == doc['pair_list_sha256'] else 'DIFFERENT'}")
    dump(ext_recs, nulls_ext, "pairs_extended.json", True)

    print("\n=== example pairs per rule ===")
    rng = np.random.default_rng(0)
    for t in ("E", "C", "both"):
        sub = [p for p in doc["pairs"] if p["tag"] == t]
        for p in [sub[i] for i in rng.choice(len(sub), min(5, len(sub)), replace=False)]:
            print(f"  [{t:>4}] ({p['lcc']}) {p['a_name']!r} ~ {p['b_name']!r}  emb_cos={p['emb_cos']:.2f} ppmi_cos={p['ppmi_cos']:.2f} "
                  f"overlap={p['overlap']:.3f} jaccard={p['jaccard']:.3f} exclusive={p['stream_exclusive']}")
    print("\n=== example null pairs ===")
    for mode in ("same", "cross"):
        for x in [x for x in doc[f"null_{mode}"] if x][:4]:
            print(f"  [{mode:>5}] {x['a_name']!r} ({x['lcc_a']}) ~ {x['b_name']!r} ({x['lcc_b']})  exclusive={x['stream_exclusive']} shared={x['stream_shared']}")
    print("\npair_list_sha256", doc["pair_list_sha256"])
    print(json.dumps(doc["meta"]["counts"]), json.dumps(ctx))


if __name__ == "__main__":
    main()
