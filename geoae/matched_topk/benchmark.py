"""Throughput estimate only; these disposable updates never select checkpoints."""
from __future__ import annotations

import argparse
import time

import torch

from .common import METHODS, ROOT, make_trainer, model_name, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default=str(ROOT)); parser.add_argument("--steps",type=int,default=16)
    parser.add_argument("--widths",type=int,nargs="+",default=[5120,10240,20480])
    parser.add_argument("--k",type=int,default=20)
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    args = parser.parse_args()
    if args.steps <= 0: parser.error("--steps must be positive")
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Shape/precision matched random input; stage timing and validation overhead are reported separately.
    x = torch.randn(2048,2560,device="cuda",dtype=torch.float32)
    result = {}
    for width in args.widths:
        pair = {method:make_trainer(method,width,k=args.k,device="cuda") for method in args.methods}
        for t in pair.values():
            t.ae.k.fill_(args.k)
            t.num_tokens_since_fired.fill_(10_000_000)
        times = []
        for step in range(args.steps+3):
            start = time.monotonic()
            for trainer in pair.values(): trainer.update(3000+step,x)
            torch.cuda.synchronize()
            if step >= 3: times.append(time.monotonic()-start)
        result[str(width)] = {"pair_seconds_per_step":sum(times)/len(times),
                             "training_hours_lower_estimate":244141*sum(times)/len(times)/3600,
                             "dead_features":{m:t.dead_features for m,t in pair.items()},
                             "peak_gpu_bytes":torch.cuda.max_memory_allocated(),
                             "methods":args.methods, "k":args.k,
                             "model_names":[model_name(m,width,args.k) for m in args.methods],
                             "note":f"random input, k{args.k}, AuxK enabled; excludes staging, full validation and checkpoint I/O"}
        print(result[str(width)],flush=True)
        del pair; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    suffix = "" if args.k == 20 else f"_k{args.k}"
    write_json(f"{args.out}/smoke/throughput{suffix}.json",result)


if __name__ == "__main__": main()
