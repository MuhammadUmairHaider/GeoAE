"""Use the frozen head-to-head algorithms with a token-aware SAE adapter."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from geoae.checkpoint import load_lm
import geoae.baselines.sae_data as D
from geoae.baselines.sae_coding import analyze_coding
from geoae.baselines.sae_coding import analyze_geoae_geometry
from geoae.baselines.sae_coding import reconstruction_report
from geoae.evaluation.sae_matched import analyze_observed_matching
from geoae.evaluation.sae_removal import run_removal
from .common import GEOAE_CHECKPOINT, METHODS, ROOT, load_matched, model_name, write_json


class CenteredAffineView:
    """Decoder in raw units, with the identical frozen token term canceled."""
    def __init__(self, model):
        self.model = model
        self.decoder_directions = model.decoder_directions
        self.parameter_count = model.parameter_count

    def decode(self, features):
        return self.model.core.decode(features)*self.model.transform.norm_std+self.model.transform.norm_mean


def raw_variance_normalization(report, raw):
    """Keep pipeline SSEs, replace only the denominator with original raw SST."""
    original = report["normalization"]["raw_variance_per_component"]
    variance = float((raw-raw.mean(0)).double().square().mean())
    def visit(value):
        if isinstance(value, dict):
            if "raw_mse" in value and value["raw_mse"] is not None:
                value["normalized_mse"] = value["raw_mse"]/variance if variance > 0 else None
                value["raw_fve"] = 1-value["normalized_mse"] if variance > 0 else None
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(report)
    report["normalization"].update(raw_variance_per_component=variance,
        token_centered_variance_per_component=original,
        token_bias_handling="same frozen raw token bias canceled from target and both reconstructions")
    return report


@torch.inference_mode()
def next_token_nll(raw, nxt, norm, head, chunk=32):
    values = []
    for begin in range(0, len(raw), chunk):
        logits = head(norm(raw[begin:begin+chunk]))
        values.append(F.cross_entropy(logits.float(), nxt[begin:begin+chunk].long(), reduction="none"))
    return torch.cat(values)


def sanity_metrics(raw, reconstruction, features, nxt, norm, head):
    """Score raw residuals; callers must restore any token mean before splicing."""
    if reconstruction.shape != raw.shape or not torch.isfinite(reconstruction).all() or not torch.isfinite(features).all():
        raise ValueError("Nonfinite or wrong-shape SAE reconstruction/features")
    error = (raw-reconstruction).double().square().sum()
    variance = (raw-raw.mean(0)).double().square().sum()
    clean = next_token_nll(raw, nxt, norm, head)
    spliced = next_token_nll(reconstruction, nxt, norm, head)
    return {"rows":len(raw), "mean_l0":float((features != 0).sum(1).double().mean()),
            "raw_mse":float(error/raw.numel()), "raw_fve":float(1-error/variance) if variance > 0 else None,
            "clean_nll_nats":float(clean.mean()), "spliced_nll_nats":float(spliced.mean()),
            "spliced_delta_nll_nats":float((spliced-clean).mean()),
            "finite_gate_passed":bool(torch.isfinite(clean).all() and torch.isfinite(spliced).all()),
            "sample":"same held-out random controls", "intervention":"full reconstruction replacement",
            "reconstruction_space":"raw residual",
            "fve_denominator":"sum of squared raw residual deviations from the per-component sample mean"}


def checkpoint_identity(trained, method, width, k):
    """Reject checkpoints whose recipe differs from the requested evaluation arm."""
    recipe = trained["recipe"]
    expected = {"method":method, "width":width, "k":k}
    for key, value in expected.items():
        if recipe.get(key) != value:
            raise ValueError(f"SAE checkpoint recipe mismatch for {key}: {recipe.get(key)!r} != {value!r}")
    return {**expected, "name":model_name(method,width,k), "recipe":recipe,
            "token_bias_handling":"frozen token mean added back and normalization restored before FVE and next-token splice"}


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    root = Path(args.out); out = root/("smoke/evaluation" if args.smoke else "evaluation")
    ae, mean, std, ckpt = D.load_geoae(args.geoae_checkpoint, args.device)
    checkpoint_sha = D.fingerprint(args.geoae_checkpoint)
    lm = load_lm(D.MODEL, device=args.device)
    from geoae.checkpoint import load_tokenizer
    tokenizer = load_tokenizer(D.MODEL)
    norm, head = D.fp32_head(lm)
    head_sanity = D.check_head(lm, tokenizer, norm, head, args.device)
    del lm, tokenizer, ckpt; gc.collect(); torch.cuda.empty_cache()
    write_json(out/"head_sanity.json", head_sanity)
    for kind in ("range", "atlas"):
        data = D.load_rows(Path(args.cache_dir)/kind, rows_per_label=128, control_rows=512, seed=0)
        if data["meta"]["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Frozen cache and GeoAE checkpoint differ")
        h = torch.tensor(data["h"], dtype=torch.float32, device=args.device)
        tok = torch.tensor(data["tok"], dtype=torch.long, device=args.device)
        nxt = torch.tensor(data["nxt"], dtype=torch.long, device=args.device)
        z = torch.cat([ae.encode((h[s:s+256]-mean)/std, tok[s:s+256]) for s in range(0,len(h),256)])
        geo_reconstruction = torch.cat([ae.decode(z[s:s+256],tok[s:s+256])*std+mean for s in range(0,len(h),256)])
        rc = torch.as_tensor(np.flatnonzero(data["controls"] & ~data["fit"]), device=args.device)
        if not len(rc): raise ValueError("No held-out random controls")
        write_json(out/kind/"geoae_sanity.json",sanity_metrics(h[rc],geo_reconstruction[rc],z[rc],nxt[rc],norm,head))
        write_json(out/kind/"geoae_geometry.json",{"analysis":analyze_geoae_geometry(
            z,data["memberships"],data["pairs"],data["fit"],~data["fit"],controls=data["controls"])})
        for width in args.widths:
            for method in args.methods:
                name = model_name(method,width,args.k)
                path = root/("smoke" if args.smoke else "training")/name/("smoke.pt" if args.smoke else "last.pt")
                model, trained = load_matched(path,args.device)
                identity = checkpoint_identity(trained,method,width,args.k)
                if trained["provenance"]["geoae_checkpoint_sha256"] != checkpoint_sha:
                    raise ValueError("SAE trained against a different GeoAE transform")
                for ours, theirs in ((model.transform.tb_table,ae.tb_table),(model.transform.tb_index,ae.tb_index),
                                    (model.transform.norm_mean,mean),(model.transform.norm_std,std)):
                    torch.testing.assert_close(ours,theirs,rtol=0,atol=0)
                model.transform.assert_geoae_equivalence(ae,h[:64],tok[:64])
                dest = out/kind/name
                print(f"evaluate {kind} {name}",flush=True)
                features = torch.cat([model.encode(h[s:s+256],tok[s:s+256]) for s in range(0,len(h),256)])
                recon = model.decode(features[rc],tok[rc])
                # Reloaded smoke checkpoints must exactly reproduce their state in memory.
                sanity = sanity_metrics(h[rc],recon,features[rc],nxt[rc],norm,head)
                sanity.update(threshold=float(model.core.threshold), checkpoint_sha256=D.fingerprint(path),
                              learned_parameters=model.parameter_count,
                              frozen_token_table_values=model.transform.tb_table.numel(), **identity)
                if not sanity["finite_gate_passed"]: raise ValueError("Nonfinite spliced logits")
                write_json(dest/"sanity.json",sanity)
                directions = model.decoder_directions
                coding = analyze_coding(features,data["memberships"],data["pairs"],directions,
                    data["fit"],~data["fit"],controls=data["controls"],budgets=(1,2,4,8))
                write_json(dest/"coding.json",{"provenance":trained["provenance"],"analysis":coding})
                table = ae.token_bias(tok[rc])*std
                report = reconstruction_report(h[rc]-table,features[rc],CenteredAffineView(model),
                    budgets=tuple(sorted({1,2,4,8,16,20,32,64,args.k})),geoae_reconstruction=geo_reconstruction[rc]-table,
                    geoae_features=z[rc],geoae_decoder_directions=ae.decoder.weight.T*std,
                    geoae_bias=ae.decode(torch.zeros_like(z[rc]),tok[rc])*std+mean-table,
                    geoae_parameter_count=sum(p.numel() for p in ae.parameters()))
                report = raw_variance_normalization(report,h[rc])
                # Affine-coordinate cancellation must agree with direct raw reconstruction.
                np.testing.assert_allclose(report["full_sae"]["all"]["raw_mse"],sanity["raw_mse"],rtol=1e-5,atol=1e-5)
                write_json(dest/"reconstruction.json",{"analysis":report})
                pairs = data["pairs"][:1] if args.smoke else data["pairs"]
                removal = run_removal(h,z,tok,nxt,data["memberships"],pairs,data["fit"],~data["fit"],
                    data["controls"],ae,mean,std,model,features,norm,head,
                    strengths=(0,.25,.5,1,2,4),feature_budgets=(1,) if args.smoke else (1,2,4,8),
                    target_damage=(.05,) if args.smoke else (.05,.1,.2),logit_chunk=32,seed=0,min_rows=8,
                    bootstrap_samples=50 if args.smoke else 500,max_rows_per_group=8 if args.smoke else 128,
                    include_sae_range=True)
                write_json(dest/"removal.json",{"provenance":trained["provenance"],"analysis":removal})
                for arm in ("sae","sae_range"):
                    matched = analyze_observed_matching(removal,sae_arm=arm)
                    write_json(dest/f"observed_{arm}_matching.json",{"analysis":matched})
                write_json(dest/"complete.json",{"checkpoint":str(path),"checkpoint_sha256":sanity["checkpoint_sha256"],
                    "data":data["meta"],"head_sanity":head_sanity,"smoke":args.smoke, **identity})
                del model,trained,features,directions,recon; gc.collect(); torch.cuda.empty_cache()
        # Pretrained arms get the same reconstruction sanity gate, without rewriting original outputs.
        if args.scope_sanity or args.smoke:
            from geoae.baselines.gemma_scope import load_gemma_scope
            for width in ("16k","262k"):
                model = load_gemma_scope(sae_path=f"resid_post_all/layer_33_width_{width}_l0_small",
                    revision="a0ffd6132a985bc84077a66d1a1033e10b604fa8",device=args.device)
                f = torch.cat([model.encode(h[rc[s:s+32]]) for s in range(0,len(rc),32)])
                recon = torch.cat([model.decode(f[s:s+32]) for s in range(0,len(f),32)])
                write_json(out/kind/f"gemma_scope_{width}_sanity.json",sanity_metrics(h[rc],recon,f,nxt[rc],norm,head))
                del model,f,recon; gc.collect(); torch.cuda.empty_cache()
    if args.smoke:
        complete = {"widths":args.widths,"methods":args.methods,"k":args.k,
                    "head_sanity":head_sanity,"checkpoint_reload_and_pipeline_smokes_passed":True}
        if args.k == 20 and set(args.widths) == {5120,10240,20480} and set(args.methods) == set(METHODS):
            complete["all_six_checkpoint_reload_and_pipeline_smokes_passed"] = True
        write_json(root/"smoke/complete.json",complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default=str(ROOT)); parser.add_argument("--cache-dir",required=True)
    parser.add_argument("--geoae-checkpoint",default=GEOAE_CHECKPOINT)
    parser.add_argument("--widths",type=int,nargs="+",default=[5120])
    parser.add_argument("--k",type=int,default=20)
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--device",default="cuda"); parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--scope-sanity",action="store_true")
    run(parser.parse_args())


if __name__ == "__main__": main()
