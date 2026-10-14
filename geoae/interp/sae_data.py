"""Gemma Scope comparison data: native block outputs, document splits, frozen pairs."""
from __future__ import annotations

import copy
import hashlib
import json
import itertools
from pathlib import Path

import numpy as np
import torch

from geoae.checkpoint import load_ae_checkpoint
from geoae.extract_sampled import forward_batch, register_capture, sample_positions
from geoae.interp import atlas_siblings as AS
from geoae.interp.closest_tokens import sampled_sources, stream_sampled_docs
from geoae.lm_arch import decoder_layers, locate_lm_parts
from geoae.losses import pairwise_sq_dist

MODEL = "google/gemma-3-4b-pt"
LAYER = 33


def fingerprint(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def load_geoae(path, device):
    ae, mean, std, checkpoint = load_ae_checkpoint(path, device, allow_token_bias=True)
    cfg = checkpoint["config"]
    if cfg["extraction"]["model_name"] != MODEL or cfg["data"]["target_layer"] != LAYER:
        raise ValueError("comparison requires Gemma 3 4B PT layer 33 checkpoint")
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid GeoAE normalization")
    return ae, mean, std, checkpoint


@torch.inference_mode()
def discover_pairs(dump, ae, mean, std, *, n_rows=150000, min_n=50,
                   min_cos=.9, min_dprime=2., max_pairs=32, seed=0, batch=2048):
    """Discover on the dump's validation tail; score later on corpus-tail documents."""
    dump = Path(dump)
    h = np.load(dump / "layer_33.npy", mmap_mode="r")
    tok = np.load(dump / "rows_tok.npy", mmap_mode="r")
    start = max(int(len(h) * .95), len(h) - n_rows)
    k, d = ae.centroids.shape
    sums = torch.zeros(k, d, device=mean.device, dtype=torch.float64)
    counts = torch.zeros(k, device=mean.device, dtype=torch.float64)
    zs, labs = [], []
    for i in range(start, len(h), batch):
        raw = torch.tensor(np.array(h[i:i+batch]), device=mean.device, dtype=torch.float32)
        tids = torch.tensor(np.array(tok[i:i+batch]), device=mean.device, dtype=torch.long)
        z = ae.encode((raw - mean) / std, tids)
        lab = pairwise_sq_dist(z, ae.centroids, metric=ae.metric).argmin(1)
        sums.index_add_(0, lab, z.double())
        counts.index_add_(0, lab, torch.ones_like(lab, dtype=torch.float64))
        zs.append(z.cpu())
        labs.append(lab.cpu())
    origin = sums.sum(0) / counts.sum()
    mu = sums / counts.clamp_min(1)[:, None] - origin
    unit = torch.nn.functional.normalize(mu.float(), dim=1)
    cos = unit @ unit.T
    ok = counts >= min_n
    cos[~ok] = -2
    cos[:, ~ok] = -2
    cos.fill_diagonal_(-2)
    nearest = cos.argmax(1)
    cand = sorted({tuple(sorted((a, int(nearest[a])))) for a in ok.nonzero().flatten().tolist()
                   if cos[a, nearest[a]] >= min_cos})
    zall, lall = torch.cat(zs), torch.cat(labs)
    records = []
    for a, b in cand:
        u = torch.nn.functional.normalize(unit[a] + unit[b], dim=0).cpu()
        va, vb = zall[lall == a] @ u, zall[lall == b] @ u
        dp = float((va.mean() - vb.mean()).abs() / ((va.var(unbiased=False) + vb.var(unbiased=False))/2).sqrt().clamp_min(1e-8))
        if dp >= min_dprime:
            records.append(dict(a=a, b=b, cos=float(cos[a,b]), dprime=dp,
                                n_a=len(va), n_b=len(vb), geoae_direction_count=1))
    # Connected components prevent counting overlapping pairs as independent families.
    parents = {c:c for r in records for c in (r["a"],r["b"])}
    def find(c):
        while parents[c] != c:
            c = parents[c]
        return c
    for r in records:
        parents[find(r["a"])] = find(r["b"])
    for r in records:
        r["family"] = f"range_{find(r['a'])}"
    for family in {r["family"] for r in records}:
        family_records=[r for r in records if r["family"]==family]
        members=sorted({c for r in family_records for c in (r["a"],r["b"])})
        family_cos=float((unit[members]@unit[members].T).min())
        for r in family_records:
            r["family_min_cos"]=family_cos
            r["geoae_direction_count"]=1 if family_cos>=min_cos else None
    if max_pairs and len(records) > max_pairs:
        ix = np.sort(np.random.default_rng(seed).choice(len(records), max_pairs, replace=False))
        chosen = [records[i] for i in ix]
    else:
        chosen = records
    return {"meta": {"kind":"discovered GeoAE range pairs", "discovery_rows":len(zall),
                     "discovery_start":start, "min_n":min_n, "min_cos":min_cos,
                     "min_dprime":min_dprime, "seed":seed, "n_qualifying":len(records)},
            "pairs":chosen, "all_pairs":records}


class SoftcapHead(torch.nn.Module):
    def __init__(self, head, cap=None):
        super().__init__()
        self.head, self.cap = head, cap

    def forward(self, h):
        logits = self.head(h)
        return torch.tanh(logits / self.cap) * self.cap if self.cap else logits


def fp32_head(lm):
    _, _, norm, head = locate_lm_parts(lm)
    config = getattr(lm.config, "text_config", lm.config)
    return copy.deepcopy(norm).float().eval(), SoftcapHead(
        copy.deepcopy(head).float().eval(), getattr(config, "final_logit_softcapping", None))


@torch.inference_mode()
def check_head(lm, tokenizer, norm, head, device):
    """Verify hook and complete logits transformation against one real forward."""
    ids = tokenizer("The scientist measured the temperature of the water.", return_tensors="pt")["input_ids"].to(device)
    captured = []
    handle = decoder_layers(lm)[LAYER].register_forward_hook(
        lambda _m, _i, o: captured.append((o[0] if isinstance(o,tuple) else o).detach()))
    try:
        out = lm(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    raw = captured[0][0,4:-1].float()
    target = ids[0,5:]
    own = torch.log_softmax(head(norm(raw)), -1).gather(1,target[:,None])
    ref = torch.log_softmax(out.logits[0,4:-1].float(), -1).gather(1,target[:,None])
    diff = (own-ref).abs()
    report = {"max_abs_logprob_diff":float(diff.max()),"mean_abs_logprob_diff":float(diff.mean()),"n_rows":len(raw)}
    if report["max_abs_logprob_diff"] > .5:
        raise ValueError(f"raw block output does not reproduce LM logits: {report}")
    return report


def captured_documents(stream, lm, tokenizer, captured, device, *, max_docs, batch=16):
    """Batch forward passes while preserving source order and document IDs."""
    iterator=enumerate(itertools.islice(stream,max_docs))
    while True:
        sub=list(itertools.islice(iterator,batch))
        if not sub:
            return
        forward_batch(lm,[row[1][0][0].tolist() for row in sub],
                      tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,device)
        block=captured[LAYER]
        for bi,(di,(ids,domain,pos)) in enumerate(sub):
            yield di,ids,block[bi,:ids.shape[1]].float(),domain,pos
        captured.clear()


@torch.inference_mode()
def collect_range(out_dir, dump, pair_doc, ae, mean, std, lm, tokenizer, *, max_docs=6000,
                  rows_per_cluster=768, controls_per_doc=2, seed=0, corpus_dir=None, batch=16):
    """Collect corpus-tail rows; no document here appeared in the activation dump."""
    meta, sources = sampled_sources(Path(dump),Path(corpus_dir) if corpus_dir else None)
    clusters = sorted({c for p in pair_doc["pairs"] for c in (p["a"],p["b"])})
    if not clusters:
        raise ValueError("no qualifying Gemma range pairs; inspect discovery report")
    rng = np.random.default_rng(seed)
    counts = {(s,c):0 for s in (False,True) for c in clusters}
    parts = {k:[] for k in ("h","tok","nxt","doc","cluster","control","fit")}
    captured, handles = register_capture(lm, [LAYER])
    try:
        stream = stream_sampled_docs(sources,meta, max_docs*meta["positions_per_doc"],tokenizer,10,seed+7)
        for di,ids,raw,_domain,_sampled in captured_documents(stream,lm,tokenizer,captured,mean.device,
                                                              max_docs=max_docs,batch=batch):
            seq = ids[0].tolist()
            pos = sample_positions(len(seq)-1,meta["skip_leading"],meta["positions_per_doc"],rng)
            if not len(pos):
                continue
            p = torch.tensor(pos,device=mean.device)
            tids = ids[0,p.cpu()].to(mean.device)
            z = ae.encode((raw[p]-mean)/std,tids)
            lab = pairwise_sq_dist(z,ae.centroids,metric=ae.metric).argmin(1).cpu().numpy()
            split = bool(AS.split_half(np.asarray([di]), seed)[0])
            keep = []
            for c in clusters:
                ix = np.flatnonzero(lab==c)
                take = min(8,len(ix), rows_per_cluster-counts[split,c])
                if take > 0:
                    keep.extend(rng.choice(ix,take,replace=False))
                    counts[split,c] += take
            ctrl = rng.choice(len(pos),min(controls_per_doc,len(pos)),replace=False)
            keep = np.union1d(keep,ctrl).astype(np.int64)
            kp = p[torch.tensor(keep,device=mean.device)]
            parts["h"].append(raw[kp].cpu().numpy())
            parts["tok"].append(np.asarray(seq)[pos[keep]])
            parts["nxt"].append(np.asarray(seq)[pos[keep]+1])
            parts["doc"].append(np.full(len(keep),di))
            parts["cluster"].append(lab[keep])
            parts["control"].append(np.isin(keep,ctrl))
            parts["fit"].append(np.full(len(keep),split))
            if di % 200 == 0:
                print(f"range collection docs={di+1}, min/member/split={min(counts.values())}",flush=True)
            if min(counts.values()) >= rows_per_cluster:
                break
    finally:
        for handle in handles:
            handle.remove()
    write_rows(out_dir,parts,{"kind":"range","model":MODEL,"layer":LAYER,"pairs":pair_doc["pairs"],
                             "source_dump":str(Path(dump).resolve()),"seed":seed,"max_docs":max_docs,
                             "training_document_overlap":0})


@torch.inference_mode()
def collect_atlas(out_dir, atlas_dir, pair_doc, lm, tokenizer, *, n_chunks=300000,
                  max_pairs=32, rows_per_label=512, controls_per_split=1024,
                  seed=0, batch=16, max_len=256, positions=4):
    """Use frozen label-only Atlas pairs; exclusive chunks and document split."""
    pairs = list(pair_doc["pairs"])
    if max_pairs and len(pairs)>max_pairs:
        ix=np.sort(np.random.default_rng(seed).choice(len(pairs),max_pairs,replace=False))
        pairs=[pairs[i] for i in ix]
    units = sorted({c for p in pairs for side in ("a_ids","b_ids") for c in p[side]})
    selected = set(units)
    counts={(s,pi,side):0 for s in (False,True) for pi in range(len(pairs)) for side in ("a","b")}
    ctrl_counts={False:0,True:0}
    parts={k:[] for k in ("h","tok","nxt","doc","control","fit","labels")}
    captured,handles=register_capture(lm,[LAYER])
    chunk_index=0
    try:
        for blk in AS.iter_atlas_chunks(atlas_dir,n_chunks,["doc_int_id","chunk_id","chunk_text","content_ids","entity_ids","tone_ids","document_ids"]):
            todo=[]
            for j,text in enumerate(blk["chunk_text"]):
                labels=set(blk["content_ids"][j]+blk["entity_ids"][j]+blk["tone_ids"][j]+blk["document_ids"][j])
                doc=int(blk["doc_int_id"][j]); split=bool(AS.split_half(np.asarray([doc]),seed)[0])
                wanted=[]
                for pi,pair in enumerate(pairs):
                    has_a=bool(labels.intersection(pair["a_ids"]))
                    has_b=bool(labels.intersection(pair["b_ids"]))
                    for side,exclusive in (("a",has_a and not has_b),("b",has_b and not has_a)):
                        if exclusive and counts[split,pi,side]<rows_per_label:
                            wanted.append((pi,side))
                control=not bool(labels & selected) and ctrl_counts[split]<controls_per_split
                if not wanted and not control:
                    continue
                seq=tokenizer(text,truncation=True,max_length=max_len)["input_ids"]
                pos=sample_positions(len(seq)-1,4,positions,np.random.default_rng([seed,chunk_index+j]))
                if not len(pos):
                    continue
                for pi,side in wanted:
                    counts[split,pi,side]+=len(pos)
                if control:
                    ctrl_counts[split]+=len(pos)
                todo.append((seq,pos,doc,split,control,sorted(labels)))
            for s in range(0,len(todo),batch):
                sub=todo[s:s+batch]
                forward_batch(lm,[x[0] for x in sub],tokenizer.pad_token_id or tokenizer.eos_token_id,next(lm.parameters()).device)
                for b,(seq,pos,doc,split,control,labels) in enumerate(sub):
                    raw=captured[LAYER][b,torch.tensor(pos,device=captured[LAYER].device)].float().cpu().numpy()
                    parts["h"].append(raw)
                    parts["tok"].append(np.asarray(seq)[pos]); parts["nxt"].append(np.asarray(seq)[pos+1])
                    parts["doc"].append(np.full(len(pos),doc)); parts["fit"].append(np.full(len(pos),split))
                    parts["control"].append(np.full(len(pos),control)); parts["labels"].extend([labels]*len(pos))
                captured.clear()
            chunk_index+=len(blk["chunk_text"])
            print(f"Atlas chunks={chunk_index}, collected rows={sum(len(x) for x in parts['h'])}",flush=True)
    finally:
        for handle in handles:
            handle.remove()
    write_rows(out_dir,parts,{"kind":"atlas","model":MODEL,"layer":LAYER,"pairs":pairs,
                             "pairs_source_sha256":pair_doc.get("pair_list_sha256"),"seed":seed,
                             "n_chunks":chunk_index,"atlas_training_overlap":"not audited"})


def write_rows(out_dir, parts, meta):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    if not parts["h"]:
        raise ValueError("collected no rows")
    for name, arrays in parts.items():
        if name=="labels":
            ptr,ids=AS.ragged_to_csr(arrays)
            np.savez(out/"labels.npz",ptr=ptr,ids=ids)
        else:
            data=np.concatenate(arrays)
            if name=="h" and (data.dtype != np.float32 or not np.isfinite(data).all()):
                raise ValueError("residual rows must be finite float32 (Gemma fp16 overflows)")
            np.save(out/f"{name}.npy",data)
    meta["n_rows"]=sum(len(x) for x in parts["h"])
    meta["h_sha256"]=fingerprint(out/"h.npy")
    (out/"meta.json").write_text(json.dumps(meta,indent=2))


def load_rows(path, *, rows_per_label=128, control_rows=512, seed=0):
    path=Path(path); meta=json.loads((path/"meta.json").read_text())
    if meta["model"]!=MODEL or meta["layer"]!=LAYER:
        raise ValueError("cache model/hook mismatch")
    h=np.load(path/"h.npy",mmap_mode="r")
    if h.dtype != np.float32:
        raise ValueError("Gemma comparison requires float32 cached rows")
    if meta.get("h_sha256") != fingerprint(path/"h.npy"):
        raise ValueError("residual cache fingerprint mismatch")
    if h.ndim != 2 or h.shape[1] != 2560 or meta.get("n_rows") != len(h):
        raise ValueError("residual cache shape or row count mismatch")
    for start in range(0,len(h),8192):
        if not np.isfinite(h[start:start+8192]).all():
            raise ValueError("nonfinite residual cache")
    fit=np.load(path/"fit.npy").astype(bool)
    ctrl=np.load(path/"control.npy").astype(bool)
    doc=np.load(path/"doc.npy")
    for name,arr in (("fit",fit),("controls",ctrl),("doc",doc),
                     ("tok",np.load(path/"tok.npy")),("nxt",np.load(path/"nxt.npy"))):
        if arr.shape != (len(h),):
            raise ValueError(f"{name} cache row alignment mismatch")
    if np.intersect1d(doc[fit],doc[~fit]).size:
        raise ValueError("a document appears in both fit and evaluation rows")
    memberships={}; pairs=[]
    if meta["kind"]=="range":
        cluster=np.load(path/"cluster.npy")
        for p in meta["pairs"]:
            q=dict(p)
            for side in ("a","b"):
                label=f"cluster_{p[side]}"; q[side]=label
                memberships[label]=cluster==p[side]
            pairs.append(q)
    else:
        labels=np.load(path/"labels.npz")
        row_ids=np.repeat(np.arange(len(h)),np.diff(labels["ptr"]))
        parents={int(c):int(c) for p in meta["pairs"] for key in ("a_ids","b_ids") for c in p[key]}
        def find(c):
            while parents[c] != c:
                c=parents[c]
            return c
        for p in meta["pairs"]:
            ids=p["a_ids"]+p["b_ids"]
            for c in ids[1:]:
                parents[find(int(c))]=find(int(ids[0]))
        for pi,p in enumerate(meta["pairs"]):
            masks=[np.zeros(len(h),bool),np.zeros(len(h),bool)]
            for si,side in enumerate(("a","b")):
                masks[si][row_ids[np.isin(labels["ids"],p[f"{side}_ids"])]]=True
            q=dict(p,family=f"atlas_{find(int(p['a_ids'][0]))}")
            for si,side in enumerate(("a","b")):
                key=f"pair_{pi}:{side}"; q[side]=key
                memberships[key]=masks[si]&~masks[1-si]
            pairs.append(q)
    # Random controls are the same reference distribution for coding/removal,
    # excluding every selected alternative (Atlas already follows this rule).
    for mask in memberships.values():
        ctrl &= ~mask
    rng=np.random.default_rng(seed); keep=[]
    for mask in list(memberships.values())+[ctrl]:
        limit=control_rows if mask is ctrl else rows_per_label
        for split in (fit,~fit):
            ix=np.flatnonzero(mask&split)
            keep.extend(rng.choice(ix,min(len(ix),limit),replace=False))
    ix=np.unique(keep).astype(np.int64)
    return dict(h=np.array(h[ix]),tok=np.load(path/"tok.npy")[ix],nxt=np.load(path/"nxt.npy")[ix],
                doc=doc[ix],fit=fit[ix],controls=ctrl[ix],
                memberships={k:v[ix] for k,v in memberships.items()},pairs=pairs,meta=meta)
