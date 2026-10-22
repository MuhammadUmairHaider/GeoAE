"""Pretrained Gemma Scope 2 head-to-head against a Gemma 4B GeoAE.

python -m geoae.evaluation.sae_compare --stage all --device cuda
Preparation and analysis have independent reusable outputs. No SAE is trained.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from geoae.checkpoint import load_lm
import geoae.baselines.sae_data as D
from geoae.lm_arch import decoder_layers

CHECKPOINT="checkpoints/gemma3-4B/layer33/k2000_bnh_b32k_lam1_d5120_dpc_sampled_struct5_tokbias/best_val.pt"
REVISION="a0ffd6132a985bc84077a66d1a1033e10b604fa8"


def prepare_signature(args, kind, checkpoint_sha):
    signature={"checkpoint_sha256":checkpoint_sha,"kind":kind,"seed":args.seed,
               "max_pairs":args.max_pairs,"collect_rows_per_label":args.collect_rows_per_label}
    if kind=="range":
        signature.update(dump=str(Path(args.dump).resolve()),max_docs=args.max_docs,
                         discovery_rows=args.discovery_rows,discovery_min_n=args.discovery_min_n,
                         min_cos=args.min_cos,min_dprime=args.min_dprime,corpus_dir=args.corpus_dir,
                         extract_batch=args.extract_batch)
    else:
        signature.update(atlas_pairs_sha256=D.fingerprint(args.atlas_pairs),atlas_chunks=args.atlas_chunks,
                         atlas_positions=args.atlas_positions,extract_batch=args.extract_batch)
    return signature


def analysis_signature(args, kind, data_meta, checkpoint_sha, sae_path):
    names=("seed","rows_per_label","control_rows","feature_budgets","reconstruction_budgets",
           "strengths","target_damage","logit_chunk","min_rows","score_rows","bootstrap_samples",
           "sae_repo","sae_revision")
    result={"kind":kind,"checkpoint_sha256":checkpoint_sha,"h_sha256":data_meta["h_sha256"],
            "sae_path":sae_path,**{name:getattr(args,name) for name in names}}
    if getattr(args,"sae_range_control",False): result["sae_range_control"]=True
    return result


def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if isinstance(x,dict): return {str(k):convert(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [convert(v) for v in x]
        if isinstance(x,float) and not np.isfinite(x): return None
        if isinstance(x,np.generic): return convert(x.item())
        return x
    tmp=path.with_suffix(path.suffix+f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(convert(data),indent=2,allow_nan=False))
    tmp.replace(path)


@torch.inference_mode()
def prepare(args):
    ae,mean,std,ckpt=D.load_geoae(args.checkpoint,args.device)
    provenance={"checkpoint_sha256":D.fingerprint(args.checkpoint),"checkpoint":args.checkpoint,
                "epoch":ckpt.get("epoch"),"step":ckpt.get("step")}
    cache=Path(args.cache_dir); out=Path(args.out_dir)
    todo=[]
    for kind in args.kinds:
        path=cache/kind
        if (path/"meta.json").exists():
            meta=json.loads((path/"meta.json").read_text())
            if meta.get("checkpoint_sha256") != provenance["checkpoint_sha256"]:
                raise ValueError(f"{path} was collected with a different checkpoint; choose a fresh cache_dir")
            if meta.get("prepare_signature") != prepare_signature(args,kind,provenance["checkpoint_sha256"]):
                raise ValueError(f"{path} collection settings differ; choose a fresh cache_dir")
            print(f"reuse {kind} cached rows",flush=True)
        else:
            todo.append(kind)
    if not todo:
        return
    pair_doc=None
    if "range" in todo:
        pair_file=out/"range_pairs.json"
        if pair_file.exists():
            pair_doc=json.loads(pair_file.read_text())
            if pair_doc["meta"].get("checkpoint_sha256") != provenance["checkpoint_sha256"]:
                raise ValueError("range-pair discovery checkpoint mismatch")
            if pair_doc["meta"].get("prepare_signature") != prepare_signature(args,"range",provenance["checkpoint_sha256"]):
                raise ValueError("range-pair discovery settings differ; choose a fresh out_dir")
        else:
            print("discover Gemma range pairs from validation tail",flush=True)
            pair_doc=D.discover_pairs(args.dump,ae,mean,std,n_rows=args.discovery_rows,
                min_n=args.discovery_min_n,min_cos=args.min_cos,min_dprime=args.min_dprime,
                max_pairs=args.max_pairs,seed=args.seed)
            pair_doc["meta"].update(provenance)
            pair_doc["meta"]["prepare_signature"]=prepare_signature(args,"range",provenance["checkpoint_sha256"])
            write_json(pair_file,pair_doc)
        print(f"range pairs selected={len(pair_doc['pairs'])}, qualifying={pair_doc['meta']['n_qualifying']}",flush=True)
    from geoae.checkpoint import load_tokenizer
    tokenizer=load_tokenizer(D.MODEL)
    lm=load_lm(D.MODEL,device=args.device)
    if len(decoder_layers(lm))!=34:
        raise ValueError("expected Gemma 3 4B's 34 text decoder blocks")
    norm,head=D.fp32_head(lm)
    sanity=D.check_head(lm,tokenizer,norm,head,args.device)
    write_json(out/("head_sanity_"+"_".join(args.kinds)+".json"),sanity)
    print(f"head sanity {sanity}",flush=True)
    del norm,head
    torch.cuda.empty_cache() if str(args.device).startswith("cuda") else None
    for kind in todo:
        if kind=="range":
            D.collect_range(cache/kind,args.dump,pair_doc,ae,mean,std,lm,tokenizer,
                max_docs=args.max_docs,rows_per_cluster=args.collect_rows_per_label,seed=args.seed,
                corpus_dir=args.corpus_dir,batch=args.extract_batch)
        else:
            atlas_pairs=json.loads(Path(args.atlas_pairs).read_text())
            D.collect_atlas(cache/kind,args.atlas_dir,atlas_pairs,lm,tokenizer,n_chunks=args.atlas_chunks,
                max_pairs=args.max_pairs,rows_per_label=args.collect_rows_per_label,seed=args.seed,
                batch=args.extract_batch,positions=args.atlas_positions)
        meta_path=cache/kind/"meta.json"
        meta=json.loads(meta_path.read_text()); meta.update(provenance)
        meta["prepare_signature"]=prepare_signature(args,kind,provenance["checkpoint_sha256"])
        write_json(meta_path,meta)
    del lm,ae
    gc.collect()
    if str(args.device).startswith("cuda"): torch.cuda.empty_cache()


@torch.inference_mode()
def analyze(args):
    from geoae.baselines.gemma_scope import load_gemma_scope
    from geoae.baselines.sae_coding import analyze_coding
    from geoae.baselines.sae_coding import analyze_geoae_geometry
    from geoae.baselines.sae_coding import reconstruction_report
    from geoae.evaluation.sae_removal import run_removal
    ae,mean,std,ckpt=D.load_geoae(args.checkpoint,args.device)
    checkpoint_sha=D.fingerprint(args.checkpoint)
    out=Path(args.out_dir)
    norm=head=None
    if not args.skip_removal:
        lm=load_lm(D.MODEL,device=args.device)
        norm,head=D.fp32_head(lm)
        del lm
        gc.collect()
        if str(args.device).startswith("cuda"): torch.cuda.empty_cache()
    for kind in args.kinds:
        data=D.load_rows(Path(args.cache_dir)/kind,rows_per_label=args.rows_per_label,
                         control_rows=args.control_rows,seed=args.seed)
        if data["meta"].get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("cache/checkpoint mismatch")
        h=torch.tensor(data["h"],device=args.device,dtype=torch.float32)
        tok=torch.tensor(data["tok"],device=args.device,dtype=torch.long)
        z=torch.cat([ae.encode((h[s:s+args.encode_batch]-mean)/std,tok[s:s+args.encode_batch])
                     for s in range(0,len(h),args.encode_batch)])
        geo_recon=torch.cat([ae.decode(z[s:s+args.encode_batch],tok[s:s+args.encode_batch])*std+mean
                            for s in range(0,len(h),args.encode_batch)])
        geometry=analyze_geoae_geometry(z,data["memberships"],data["pairs"],data["fit"],~data["fit"],
                                       controls=data["controls"])
        write_json(out/kind/"geoae_geometry.json",{"analysis":geometry,"checkpoint_sha256":checkpoint_sha})
        print(f"{kind} analysis rows={len(h)}, pairs={len(data['pairs'])}",flush=True)
        for sae_path in args.sae_paths:
            dest=out/kind/sae_path.split("/")[-1]
            signature=analysis_signature(args,kind,data["meta"],checkpoint_sha,sae_path)
            if (dest/"complete.json").exists():
                complete=json.loads((dest/"complete.json").read_text())
                if complete.get("signature") != signature:
                    raise ValueError(f"{dest} analysis settings differ; choose a fresh out_dir")
                required=[dest/"coding.json",dest/"reconstruction.json"]
                if not args.skip_removal: required.append(dest/"removal.json")
                if all(p.exists() for p in required) and (args.skip_removal or complete.get("removal_ran")):
                    print(f"reuse completed {dest}",flush=True); continue
            start=time.time()
            sae=load_gemma_scope(repo_id=args.sae_repo,sae_path=sae_path,revision=args.sae_revision,
                                device=args.device)
            features=torch.empty((len(h),sae.decoder_directions.shape[0]),device=args.device,dtype=torch.float32)
            for s in range(0,len(h),args.encode_batch):
                features[s:s+args.encode_batch]=sae.encode(h[s:s+args.encode_batch])
            provenance={"sae":sae.metadata,"checkpoint_sha256":checkpoint_sha,
                "geoae_epoch":ckpt.get("epoch"),"rows":len(h),"data":data["meta"],
                "seed":args.seed,"model":D.MODEL,"layer":D.LAYER}
            coding=analyze_coding(features,data["memberships"],data["pairs"],sae.decoder_directions,
                data["fit"],~data["fit"],controls=data["controls"],budgets=tuple(args.feature_budgets))
            write_json(dest/"coding.json",{"provenance":provenance,"analysis":coding})
            # Reconstruction on a random held-out control sample, independent of target enrichment.
            rc=np.flatnonzero(data["controls"] & ~data["fit"])
            if len(rc):
                ix=torch.tensor(rc,device=args.device)
                reconstruction=reconstruction_report(h[ix],features[ix],sae,
                    budgets=tuple(args.reconstruction_budgets),geoae_reconstruction=geo_recon[ix],
                    geoae_features=z[ix],geoae_decoder_directions=ae.decoder.weight.T*std,
                    geoae_bias=ae.decode(torch.zeros_like(z[ix]),tok[ix])*std+mean,
                    geoae_parameter_count=sum(p.numel() for p in ae.parameters()))
                reconstruction["geoae_parameter_count"]=sum(p.numel() for p in ae.parameters())
                reconstruction["geoae_token_bias_values"]=int(ae.tb_table.numel()) if hasattr(ae,"tb_table") else 0
                reconstruction["sample"]="held-out random controls; whole-model and active-budget comparisons separate"
                write_json(dest/"reconstruction.json",{"provenance":provenance,"analysis":reconstruction})
            if not args.skip_removal:
                removal=run_removal(h,z,tok,torch.tensor(data["nxt"],device=args.device),
                    data["memberships"],data["pairs"],data["fit"],~data["fit"],data["controls"],
                    ae,mean,std,sae,features,norm,head,strengths=tuple(args.strengths),
                    feature_budgets=tuple(args.feature_budgets),target_damage=tuple(args.target_damage),
                    logit_chunk=args.logit_chunk,seed=args.seed,min_rows=args.min_rows,
                    bootstrap_samples=args.bootstrap_samples,max_rows_per_group=args.score_rows,
                    include_sae_range=args.sae_range_control)
                write_json(dest/"removal.json",{"provenance":provenance,"analysis":removal})
            write_json(dest/"complete.json",{"seconds":time.time()-start,"provenance":provenance,
                                             "signature":signature,
                                             "removal_ran":not args.skip_removal})
            print(f"completed {dest} in {time.time()-start:.1f}s",flush=True)
            del features,sae
            gc.collect()
            if str(args.device).startswith("cuda"): torch.cuda.empty_cache()
        del h,z,tok,geo_recon
    run_key="_".join(args.kinds)+"_"+"_".join(p.split("/")[-1] for p in args.sae_paths)
    write_json(out/f"run_complete_{run_key}.json",{"stage":"analysis","kinds":args.kinds,"sae_paths":args.sae_paths,
                                          "removal_ran":not args.skip_removal})


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage",choices=("prepare","analyze","all"),default="all")
    ap.add_argument("--kinds",nargs="+",choices=("range","atlas"),default=["range","atlas"])
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--checkpoint",default=CHECKPOINT)
    ap.add_argument("--dump",default="activations_gemma3_4b_sampled_struct5_10M")
    ap.add_argument("--corpus_dir",default="corpus/gemma3_4b_sampled_struct5_v1")
    ap.add_argument("--cache_dir",default="cache/gemma_scope_compare")
    ap.add_argument("--out_dir",default="results/delta/g4b_l33_d5120/evals/gemma_scope_compare")
    ap.add_argument("--atlas_dir",default="cache/atlas")
    ap.add_argument("--atlas_pairs",default="results/delta/d6144/evals/atlas_siblings/pairs.json")
    ap.add_argument("--sae_repo",default="google/gemma-scope-2-4b-pt")
    ap.add_argument("--sae_revision",default=REVISION)
    ap.add_argument("--sae_paths",nargs="+",default=["resid_post_all/layer_33_width_16k_l0_small",
                                                               "resid_post_all/layer_33_width_262k_l0_small"])
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--max_pairs",type=int,default=32)
    ap.add_argument("--max_docs",type=int,default=6000)
    ap.add_argument("--discovery_rows",type=int,default=150000)
    ap.add_argument("--discovery_min_n",type=int,default=50)
    ap.add_argument("--min_cos",type=float,default=.9)
    ap.add_argument("--min_dprime",type=float,default=2.)
    ap.add_argument("--atlas_chunks",type=int,default=300000)
    ap.add_argument("--atlas_positions",type=int,default=4)
    ap.add_argument("--extract_batch",type=int,default=16)
    ap.add_argument("--collect_rows_per_label",type=int,default=512)
    ap.add_argument("--rows_per_label",type=int,default=128)
    ap.add_argument("--control_rows",type=int,default=512)
    ap.add_argument("--encode_batch",type=int,default=256)
    ap.add_argument("--feature_budgets",nargs="+",type=int,default=[1,2,4,8])
    ap.add_argument("--reconstruction_budgets",nargs="+",type=int,default=[1,2,4,8,16,20,32,64])
    ap.add_argument("--strengths",nargs="+",type=float,default=[0,.25,.5,1,2,4])
    ap.add_argument("--target_damage",nargs="+",type=float,default=[.05,.1,.2])
    ap.add_argument("--logit_chunk",type=int,default=32)
    ap.add_argument("--min_rows",type=int,default=8)
    ap.add_argument("--score_rows",type=int,default=128)
    ap.add_argument("--bootstrap_samples",type=int,default=500)
    ap.add_argument("--skip_removal",action="store_true")
    ap.add_argument("--sae_range_control",action="store_true",
                    help="add a top-1 SAE scalar range gate; use a separate out_dir for this diagnostic")
    args=ap.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    Path(args.out_dir).mkdir(parents=True,exist_ok=True)
    if args.stage in ("prepare","all"): prepare(args)
    if args.stage in ("analyze","all"): analyze(args)


if __name__=="__main__":
    main()
