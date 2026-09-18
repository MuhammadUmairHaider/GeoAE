"""
Find the loss weights at which clustering starts COSTING reconstruction.

At the shipped weights the clustering terms barely participate: measured at init
on llama L27, recon 1.535 / cluster 0.451 / sep 0.366, so lambda_cluster 0.05 is
1.3% of the loss and lambda_sep 0.005 is 0.1%. Recon dominates and the geometry
terms ride along free. This sweeps them up until reconstruction visibly degrades.

WHY "UNTIL RECON DEGRADES" NEEDS A SECOND AXIS. cluster_loss rewards pulling
points toward centroids, which rewards CONTRACTING the whole latent space, and
sep_loss uses an adaptive sigma so it is scale-invariant and cannot resist that
at any weight. There is no norm layer in the model by default, so lambda_var is
the only counter-pressure. A contracting latent ALSO degrades FVE, and in the
FVE column it looks identical to a lambda that is genuinely working — so the
scoring pass reports |z| and effective rank alongside, and a drop in FVE only
counts if the geometry improved with it.

The `latent_norm: batch` arm tests whether an architectural scale constraint
(BatchNorm1d between the encoder's linear map and its activation) does the job
better than the lambda_var penalty.

Runs are short by construction: a --max_train_rows subset with the phase gates
pulled early, so clustering is active for most of each run.

Usage:
    python -m geoae.interp.lambda_sweep --base configs/base/llama3.2-3b_l27_k2000_sq3072_strong_mse.yaml
    python -m geoae.interp.lambda_sweep --score_only        # re-score finished runs
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

# (lambda_cluster, lambda_sep, latent_norm) — sep held at 0.6x cluster, the
# ratio of the current configs. Spans three orders of magnitude around 0.5/0.3.
# --grid lambda : how hard to push the clustering objective
# --grid sep    : whether a REFORMULATED separation term and/or an architectural
#                 scale constraint do anything, at the lambda already chosen.
#                 sep_mode "median" is scale-free and provably inert (a 240x
#                 lambda_sep range moves it by 0.01), so it is the control row.
SEP_GRID = [
    # (lambda_cluster, lambda_sep, latent_norm, sep_mode)
    (2.0, 0.3, "none",  "median"),   # control: current runs
    (2.0, 0.3, "none",  "intra"),
    (2.0, 0.3, "none",  "hinge"),
    (2.0, 0.3, "batch", "median"),   # BN alone (arm that never ran)
    (2.0, 0.3, "batch", "intra"),
    (2.0, 0.3, "batch", "hinge"),
]

# --grid cov : how hard to push DECORRELATION.
#
# cov is the only loss term that is not O(1): it sat at ~10 for the whole b32k
# L27 run while recon/clus/sep/var were all below 0.35. That is not a scaling
# artefact — covariance_loss already divides by L. Measured on the trained
# d6144 checkpoint over 32,768 real tokens:
#
#   cov on real z                                 9.84
#   cov, independent noise, std 1.0, same shape   0.19   <- achievable floor
#
# so the term is 52x above its floor and lambda_cov 0.01 contributes 0.098 of a
# 0.596 total loss. The latent is correlated: mean |r| 0.031 but p99 |r| 0.25
# and max |r| 0.995 — some dimension pairs are near-duplicates. Effective rank
# (squared-SV) is 546/6144 on z and 165/1999 on centroids, both ~9%.
#
# This grid holds cluster/sep/var at the b32k settings and moves ONLY lambda_cov,
# including a 0.0 control, so any change in rank or cov is attributable.
COV_GRID = [
    # (lambda_cluster, lambda_sep, latent_norm, sep_mode, lambda_cov)
    (1.0, 0.3, "batch", "hinge", 0.0),    # control: no decorrelation at all
    (1.0, 0.3, "batch", "hinge", 0.01),   # shipped
    (1.0, 0.3, "batch", "hinge", 0.1),
    (1.0, 0.3, "batch", "hinge", 0.5),
    (1.0, 0.3, "batch", "hinge", 2.0),
    (1.0, 0.3, "batch", "hinge", 8.0),
]

DEFAULT_GRID = [
    (0.05, 0.005, "none"),   # shipped weights, for reference
    (0.5,  0.3,   "none"),
    (2.0,  1.2,   "none"),
    (8.0,  4.8,   "none"),
    (32.0, 19.2,  "none"),
    (0.5,  0.3,   "batch"),
    (2.0,  1.2,   "batch"),
    (8.0,  4.8,   "batch"),
]

# var/cov appear only once geometry_start_epoch is reached, so they must be
# OPTIONAL here. A regex requiring sep to be followed directly by val_mse
# silently matches only the PRE-geometry epochs, which are recon-only and
# therefore identical across configs — that produced a convincing-looking
# "every lambda gives the same result" artefact on the first run.
STEP_RE = re.compile(
    r"step +(\d+) \| loss ([\d.]+) \| recon ([\d.]+) \| clus ([\d.]+) \| sep ([\d.]+) \|"
    r"(?: var ([\d.]+) \| cov ([\d.]+) \|)? val_mse ([\d.]+) \| fve ([\d.-]+) \| eff_K (\d+)")


def run_grid(args, grid):
    base = yaml.safe_load(open(args.base))
    for entry in grid:
        lc, ls, ln = entry[0], entry[1], entry[2]
        sm = entry[3] if len(entry) > 3 else "median"
        lcov = entry[4] if len(entry) > 4 else None
        tag = f"clu{lc}_sep{ls}_{ln}" + (f"_{sm}" if len(entry) > 3 else "")
        if lcov is not None:
            tag += f"_cov{lcov}"
        out = Path(args.out_dir) / tag
        if (out / "best_val.pt").exists() and not args.force:
            print(f"=== {tag}: already done, skipping (use --force to redo)")
            continue
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["loss"]["lambda_cluster"], cfg["loss"]["lambda_sep"] = lc, ls
        cfg["model"]["latent_norm"] = ln
        cfg["loss"]["sep_mode"] = sm
        if lcov is not None:
            cfg["loss"]["lambda_cov"] = lcov
        cfg["train"].update(
            n_epochs=args.epochs, recon_only_epochs=2, geometry_start_epoch=2,
            clustering_start_epoch=3, full_loss_start_epoch=3,
            checkpoints_dir=str(out), diag_every=100)
        cpath = Path(args.out_dir) / f"{tag}.yaml"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(cfg, open(cpath, "w"))
        log = Path(args.out_dir) / f"{tag}.log"
        print(f"\n=== {tag} ===", flush=True)
        # Stream to BOTH the log and the terminal. Redirecting only to the file
        # makes a healthy 13-minute run look like a hang.
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "geoae.train", "--config", str(cpath),
             "--max_train_rows", str(args.rows), "--no_wandb"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        with open(log, "w") as f:
            for line in proc.stdout:
                f.write(line)
                ls_ = line.rstrip()
                if "Epoch" in ls_ and "done" not in ls_:
                    print(f"    {ls_}", flush=True)
                elif m := STEP_RE.search(ls_):
                    if int(m.group(1)) % args.print_every == 0:
                        print(f"    {ls_}", flush=True)
                elif "Traceback" in ls_ or "Error" in ls_:
                    print(f"    {ls_}", flush=True)
        proc.wait()
        rows = STEP_RE.findall(log.read_text(errors="ignore").replace("\r", "\n"))
        if rows:
            _, _, rc, c, sp, var, cov, vm, fve, ek = rows[-1]
            print(f"  final: fve {fve}  val_mse {vm}  recon {rc}  clus {c}  sep {sp}  "
                  f"var {var or '-'}  cov {cov or '-'}  eff_K {ek}", flush=True)
        else:
            print(f"  no step lines parsed — see {log}", flush=True)


@torch.no_grad()
def score(args, grid):
    """Reconstruction cost vs the geometry it actually bought."""
    from geoae.checkpoint import load_ae_checkpoint
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mm = np.load(args.activations, mmap_mode="r")
    rng = np.random.RandomState(0)
    st = rng.choice(mm.shape[0] - 200, args.score_blocks, replace=False)
    X = np.concatenate([np.asarray(mm[s:s + 200]) for s in sorted(st)]).astype(np.float32)
    if not np.isfinite(X).all():
        raise SystemExit("non-finite activations — check the dump's dtype")
    Xt = torch.from_numpy(X).to(dev)

    print(f"\n{'config':<24}{'FVE':>8}{'|z|':>8}{'erank':>7}{'liveK':>7}"
          f"{'top10%':>8}{'inter/intra':>12}")
    for entry in grid:
        lc, ls, ln = entry[0], entry[1], entry[2]
        sm = entry[3] if len(entry) > 3 else "median"
        lcov = entry[4] if len(entry) > 4 else None
        tag = f"clu{lc}_sep{ls}_{ln}" + (f"_{sm}" if len(entry) > 3 else "")
        if lcov is not None:
            tag += f"_cov{lcov}"
        # best_val.pt selects on val_mse, which is monotonically best BEFORE
        # clustering engages, so on a phased run it captures a pre-clustering
        # model whose centroids were never initialised (1 live cluster). Always
        # score the LAST step checkpoint instead.
        steps = sorted((Path(args.out_dir) / tag).glob("step_*.pt"))
        p = steps[-1] if steps else Path(args.out_dir) / tag / "best_val.pt"
        if not p.exists():
            continue
        ae, m, s, _ = load_ae_checkpoint(p, dev)
        ae.eval()
        xn = (Xt - m) / s
        z = torch.cat([ae.encoder(xn[i:i + 8192]) for i in range(0, len(xn), 8192)])
        xh = torch.cat([ae.decoder(z[i:i + 8192]) for i in range(0, len(z), 8192)])
        fve = float(1 - ((xn - xh) ** 2).sum() / ((xn - xn.mean(0)) ** 2).sum())
        C = ae.centroids
        lab = torch.cat([torch.cdist(z[i:i + 8192], C).argmin(1) for i in range(0, len(z), 8192)])
        cnt = Counter(lab.tolist())
        top10 = sum(v for _, v in cnt.most_common(10)) / len(lab)
        zc = (z[:20000] - z[:20000].mean(0)).float()
        sv = torch.linalg.svdvals(zc)
        pv = sv / sv.sum()
        erank = float(torch.exp(-(pv * pv.log()).sum()))
        used = torch.tensor(sorted(cnt), device=dev)
        dd = torch.cdist(C[used], C[used])
        dd.fill_diagonal_(float("inf"))
        intra = float(np.mean([float((z[lab == k] - C[k]).norm(dim=1).mean())
                               for k in list(cnt)[:200]]))
        print(f"{tag:<24}{fve:>8.4f}{float(z.norm(dim=1).mean()):>8.1f}{erank:>7.0f}"
              f"{len(cnt):>7}{top10:>8.1%}{float(dd.min()) / max(intra, 1e-9):>12.3f}")
    print("\nRead FVE together with |z| and erank: a contracting latent loses FVE too,")
    print("and buys nothing. A useful lambda trades FVE for a HIGHER inter/intra ratio.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="configs/base/llama3.2-3b_l27_k2000_sq3072_strong_mse.yaml")
    ap.add_argument("--activations", default="activations_diverse_10M/layer_27.npy")
    ap.add_argument("--out_dir", default="sweeps/lambda_l27")
    ap.add_argument("--rows", type=int, default=1_500_000, help="--max_train_rows per run")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--score_blocks", type=int, default=600, help="200-row blocks for scoring")
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--force", action="store_true", help="Redo configs that already finished")
    ap.add_argument("--grid", default="lambda", choices=["lambda", "sep", "cov"],
                    help="lambda: sweep clustering strength. sep: sweep sep_mode x latent_norm.")
    ap.add_argument("--print_every", type=int, default=500,
                    help="Echo a step line to the terminal every N steps.")
    args = ap.parse_args()

    grid = {"lambda": DEFAULT_GRID, "sep": SEP_GRID, "cov": COV_GRID}[args.grid]
    if not args.score_only:
        run_grid(args, grid)
    score(args, grid)


if __name__ == "__main__":
    main()
