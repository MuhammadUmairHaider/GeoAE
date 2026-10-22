"""Train reference BatchTopK and Matryoshka in lockstep on identical rows."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

import geoae.baselines.sae_data as D
from .common import (BATCH_SIZE, GEOAE_CHECKPOINT, METHODS, ROOT, TOKEN_VISITS,
                     BypassTransform, decoder_rows, make_trainer, model_name,
                     reference_provenance, write_json)


class RowStream:
    """Identical seeded permutation stream at every width; crosses epoch boundaries."""
    def __init__(self, n, seed=42, epoch=0, offset=0):
        self.n, self.seed, self.epoch, self.offset = n, seed, epoch, offset
        self.order = np.random.default_rng(seed+epoch).permutation(n)

    def take(self, count):
        pieces = []
        while count:
            size = min(count, self.n-self.offset)
            pieces.append(self.order[self.offset:self.offset+size]); self.offset += size; count -= size
            if self.offset == self.n:
                self.epoch += 1; self.offset = 0
                self.order = np.random.default_rng(self.seed+self.epoch).permutation(self.n)
        return np.concatenate(pieces)


def batch(raw, tokens, ids, transform, device):
    x = torch.from_numpy(np.array(raw[ids], dtype=np.float32))
    tok = torch.from_numpy(np.array(tokens[ids], dtype=np.int64))
    if str(device).startswith("cuda"):
        x, tok = x.pin_memory(), tok.pin_memory()
    x, tok = x.to(device, non_blocking=True), tok.to(device, non_blocking=True)
    return transform.preprocess(x, tok), x, tok


@torch.no_grad()
def validate(trainers, raw, tokens, start, transform, device, batch_size=2048):
    n = len(raw)-start
    totals = {name: {"sum_sq_error": 0., "raw_sum_sq_error": 0., "sum_l0": 0.} for name in trainers}
    total, square = torch.zeros(raw.shape[1], device=device, dtype=torch.float64), torch.zeros(raw.shape[1], device=device, dtype=torch.float64)
    raw_total, raw_square = torch.zeros_like(total), torch.zeros_like(square)
    for begin in range(start, len(raw), batch_size):
        x, h, tok = batch(raw, tokens, slice(begin, min(begin+batch_size, len(raw))), transform, device)
        total += x.double().sum(0); square += x.double().square().sum(0)
        raw_total += h.double().sum(0); raw_square += h.double().square().sum(0)
        for name, trainer in trainers.items():
            f = trainer.ae.encode(x, use_threshold=True)
            decoded = trainer.ae.decode(f)
            error = decoded-x
            totals[name]["sum_sq_error"] += float(error.double().square().sum())
            raw_error = transform.restore(decoded, tok)-h
            totals[name]["raw_sum_sq_error"] += float(raw_error.double().square().sum())
            totals[name]["sum_l0"] += float((f != 0).sum())
    variance_sum = float((square-total.square()/n).sum())
    raw_variance_sum = float((raw_square-raw_total.square()/n).sum())
    for name, result in totals.items():
        result.update(rows=n, mse=result["sum_sq_error"]/(n*raw.shape[1]),
                      fve=1-result["sum_sq_error"]/variance_sum if variance_sum > 0 else None,
                      mean_l0=result["sum_l0"]/n, threshold=float(trainers[name].ae.threshold),
                      current_k=int(trainers[name].ae.k))
        result.update(bypass_mse=result["mse"], bypass_fve=result["fve"],
                      raw_mse=result["raw_sum_sq_error"]/(n*raw.shape[1]),
                      raw_fve=1-result["raw_sum_sq_error"]/raw_variance_sum if raw_variance_sum > 0 else None,
                      raw_centered_sst=raw_variance_sum,
                      reconstruction_coordinates="raw residual; frozen token mean restored")
    return totals


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def save(path, trainer, transform, recipe, provenance, *, resume=False):
    payload = {"model_state": cpu_state(trainer.ae), "transform_state": cpu_state(transform),
               "recipe": recipe, "provenance": provenance}
    if resume:
        payload.update(optimizer_state=trainer.optimizer.state_dict(), scheduler_state=trainer.scheduler.state_dict(),
                       num_tokens_since_fired=trainer.num_tokens_since_fired.detach().cpu(),
                       rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    path = Path(path); temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def run(args):
    if not 0 < args.k < 2560 or any(width < 2560 for width in args.widths):
        raise ValueError("Require 0 < k < activation_dim=2560 and widths >= activation_dim")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    reference = reference_provenance()
    raw = np.load(Path(args.dump)/"layer_33.npy", mmap_mode="r")
    tokens = np.load(Path(args.dump)/"rows_tok.npy", mmap_mode="r")
    if len(tokens) != len(raw) or raw.shape[1] != 2560:
        raise ValueError("Dump shape/token mismatch")
    train_rows = int(len(raw)*.95)
    ae, mean, std, geo_checkpoint = D.load_geoae(args.geoae_checkpoint, args.device)
    if geo_checkpoint["config"]["data"]["val_frac"] != .05 or geo_checkpoint["config"]["train"]["seed"] != args.seed:
        raise ValueError("GeoAE split/seed mismatch")
    transform = BypassTransform.from_geoae(ae, mean, std)
    norm = np.load(Path(args.dump)/"norm_params_layer33.npz")
    np.testing.assert_array_equal(norm["mean"], mean.cpu().numpy()); np.testing.assert_array_equal(norm["std"], std.cpu().numpy())
    first = np.arange(min(args.batch_size, train_rows))
    x, h, tok = batch(raw, tokens, first, transform, args.device)
    equality = transform.assert_geoae_equivalence(ae, h, tok)
    # Independent check against the actual training dataset's NumPy normalization.
    from geoae.training.data import ActivationBuffer
    original_buffer = ActivationBuffer(args.dump,33,val_frac=.05,split="train",return_tokens=True)
    canonical = torch.stack([original_buffer[i][0] for i in range(min(32,len(first)))]).to(args.device)
    torch.testing.assert_close(x[:len(canonical)],canonical-ae.token_bias(tok[:len(canonical)]),rtol=0,atol=0)
    equality["activation_buffer_input_max_abs_error"] = 0.0
    del original_buffer,canonical
    base = {"reference": reference, "geoae_checkpoint_sha256": D.fingerprint(args.geoae_checkpoint),
            "geoae_epoch": geo_checkpoint.get("epoch"), "geoae_step": geo_checkpoint.get("step"),
            "geoae_parameter_count": sum(p.numel() for p in ae.parameters()),
            "dump": str(Path(args.dump).resolve()), "dump_shape": list(raw.shape),
            "train_rows": train_rows, "val_rows": len(raw)-train_rows,
            "preprocessing_equality": equality, "seed": args.seed, "job_id": os.environ.get("SLURM_JOB_ID"),
            "precision": "fp32; TF32 disabled; no autocast", "stream": "numpy PCG64 permutation(seed+epoch); contiguous 95/5 split; carry epoch tails",
            "analysis_plan_sha256": D.fingerprint(args.analysis_plan or Path(args.out)/"analysis_plan.md")}
    receipt = Path(args.dump)/"layer_33.npy.receipt.json"
    if receipt.exists(): base["staged_dump"] = json.loads(receipt.read_text())
    del ae, geo_checkpoint; gc.collect()
    specs = {model_name(method, width, args.k): (method, width)
             for width in args.widths for method in args.methods}
    trainers = {name: make_trainer(method, width, k=args.k, visits=args.visits,
                batch_size=args.batch_size, seed=args.seed, device=args.device)
                for name, (method, width) in specs.items()}
    root = Path(args.out)/("smoke" if args.smoke_steps else "training")
    root.mkdir(parents=True, exist_ok=True)
    steps = math.ceil(args.visits/args.batch_size)
    start_step = 0; seen = 0; stream = RowStream(train_rows, args.seed)
    validation_visits = 0
    best = {name: math.inf for name in trainers}
    if args.resume:
        positions = []
        for name, trainer in trainers.items():
            old = torch.load(root/name/"resume.pt", map_location=args.device, weights_only=False)
            method, width = specs[name]
            if any(old["recipe"].get(key) != value for key, value in
                   (("method", method), ("width", width), ("k", args.k), ("batch_size", args.batch_size))):
                raise ValueError("Resume SAE recipe mismatch")
            if old["recipe"]["visits"] != args.visits or old["provenance"]["analysis_plan_sha256"] != base["analysis_plan_sha256"]:
                raise ValueError("Resume budget/plan mismatch")
            trainer.ae.load_state_dict(old["model_state"]); trainer.optimizer.load_state_dict(old["optimizer_state"])
            trainer.scheduler.load_state_dict(old["scheduler_state"])
            trainer.num_tokens_since_fired.copy_(old["num_tokens_since_fired"])
            p = old["provenance"]; positions.append((p["completed_steps"],p["token_visits"],p["stream_epoch"],p["stream_offset"]))
            best[name] = p["best_validation_mse"]
        if len(set(positions)) != 1: raise ValueError("Lockstep resume checkpoints differ")
        start_step, seen, epoch, offset = positions[0]; stream = RowStream(train_rows, args.seed, epoch, offset)
        validation_visits = old["provenance"]["validation_row_visits"]
    run_key = '_'.join(map(str, args.widths)) + (f"_k{args.k}" if args.k != 20 else "")
    history = root/f"history_{run_key}.jsonl"
    start = time.monotonic(); times = []; peak_memory = 0
    stop = min(steps, start_step+args.smoke_steps) if args.smoke_steps else steps
    for step in range(start_step, stop):
        before_epoch = stream.epoch
        ids = stream.take(min(args.batch_size, args.visits-seen))
        x, _, _ = batch(raw, tokens, ids, transform, args.device)
        tic = time.monotonic()
        losses = {name: trainer.update(step, x) for name, trainer in trainers.items()}
        if not all(math.isfinite(loss) for loss in losses.values()): raise ValueError(f"Nonfinite loss: {losses}")
        seen += len(ids); times.append(time.monotonic()-tic)
        if step % args.log_every == 0 or step+1 == stop:
            progress = {"step": step+1, "token_visits": seen, "epoch": stream.epoch,
                        "losses": losses, "seconds": time.monotonic()-start,
                        "current_k": {n:int(t.ae.k) for n,t in trainers.items()},
                        "threshold": {n:float(t.ae.threshold) for n,t in trainers.items()},
                        "seconds_per_step_recent": sum(times[-100:])/len(times[-100:])}
            print(json.dumps(progress), flush=True)
            with history.open("a") as writer: writer.write(json.dumps(progress)+"\n")
            write_json(root/f"progress_{run_key}.json",progress)
        if args.smoke_steps:
            continue
        if stream.epoch != before_epoch or step+1 == stop:
            validation = validate(trainers, raw, tokens, train_rows, transform, args.device, args.batch_size)
            validation_visits += len(raw)-train_rows
            print(json.dumps({"validation_step":step+1,"validation":validation}), flush=True)
            for name, trainer in trainers.items():
                method, width = specs[name]
                recipe = {"method":method,"width":width,"activation_dim":2560,"k":args.k,
                          "visits":args.visits,"batch_size":args.batch_size,"steps":steps,
                          "reference_config":trainer.config,"lr_rationale":"explicit demo_config learning_rates=[5e-5]"}
                dest = root/name; dest.mkdir(exist_ok=True)
                report = validation[name]
                eligible = int(trainer.ae.k) == args.k and float(trainer.ae.threshold) >= 0
                improved = eligible and report["mse"] < best[name]
                if improved: best[name] = report["mse"]
                provenance = {**base,"completed_steps":step+1,"token_visits":seen,
                              "validation_row_visits":validation_visits,
                              "stream_epoch":stream.epoch,"stream_offset":stream.offset,
                              "validation":report,"best_validation_mse":best[name]}
                with (dest/"validation.jsonl").open("a") as writer:
                    writer.write(json.dumps({"step":step+1,"visits":seen,**report})+"\n")
                if improved: save(dest/"best_val.pt",trainer,transform,recipe,provenance)
                save(dest/"resume.pt",trainer,transform,recipe,provenance,resume=True)
                if step+1 == stop:
                    if seen != args.visits: raise AssertionError("Budget mismatch")
                    save(dest/"last.pt",trainer,transform,recipe,provenance)
                    write_json(dest/"complete.json",provenance)
    if args.smoke_steps:
        checks = {}
        for name, trainer in trainers.items():
            norms = decoder_rows(trainer.ae).norm(dim=1)
            if not torch.isfinite(norms).all(): raise ValueError("Nonfinite decoder")
            torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)
            # Exercise the real threshold, inference, and checkpoint round-trip branches.
            trainer.ae.k.fill_(args.k)
            f = trainer.ae.encode(x, use_threshold=False); trainer.update_threshold(f)
            inferred = trainer.ae.encode(x, use_threshold=True)
            report = {"last_loss":losses[name],"mean_l0":float((inferred!=0).sum(1).float().mean()),
                      "threshold":float(trainer.ae.threshold),"decoder_max_norm_error":float((norms.detach()-1).abs().max()),
                      "parameter_count":sum(p.numel() for p in trainer.ae.parameters())}
            if str(args.device).startswith("cuda"): peak_memory = max(peak_memory,torch.cuda.max_memory_allocated())
            checks[name] = report
            method, width = specs[name]
            recipe = {"method":method,"width":width,"activation_dim":2560,"k":args.k}
            dest = root/name; dest.mkdir(exist_ok=True)
            save(dest/"smoke.pt",trainer,transform,recipe,base)
        # Initial k is sparse on step zero, then starts at d_model; report both regimes.
        report = {"preprocessing":equality,"checks":checks,"seconds":time.monotonic()-start,
                  "step_seconds_excluding_first":sum(times[1:])/max(1,len(times)-1),
                  "peak_gpu_bytes":peak_memory,"full_schedule_steps":steps,"smoke_steps":args.smoke_steps}
        write_json(root/("smoke.json" if args.k == 20 else f"smoke_k{args.k}.json"),report)
        print(json.dumps(report),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump",required=True); parser.add_argument("--out",default=str(ROOT))
    parser.add_argument("--geoae-checkpoint",default=GEOAE_CHECKPOINT)
    parser.add_argument("--analysis-plan",help="Immutable plan snapshot; defaults to the output directory's plan")
    parser.add_argument("--widths",type=int,nargs="+",default=[5120])
    parser.add_argument("--k",type=int,default=20)
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--visits",type=int,default=TOKEN_VISITS); parser.add_argument("--batch-size",type=int,default=BATCH_SIZE)
    parser.add_argument("--seed",type=int,default=42); parser.add_argument("--device",default="cuda")
    parser.add_argument("--cpu-threads",type=int,default=4); parser.add_argument("--log-every",type=int,default=250)
    parser.add_argument("--smoke-steps",type=int,default=0); parser.add_argument("--resume",action="store_true")
    run(parser.parse_args())


if __name__ == "__main__": main()
