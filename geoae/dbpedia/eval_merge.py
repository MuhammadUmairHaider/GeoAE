"""
Over-cluster -> merge eval for K>14 DBpedia runs.

Names each of the K learned clusters by its dominant TRAIN class, maps test
points through that naming to 14 labels, and reports accuracy/purity/NMI/ARI +
class coverage. This is the correct metric when K>n_classes (Hungarian
bijection wastes the extra clusters and understates quality).

Usage:
  python -m geoae.dbpedia.eval_merge \
    --checkpoint e2e/checkpoints/llama3.2-3B/layer27/last_gelu_k20/best_val.pt \
    --pooling last
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from geoae.checkpoint import load_ae_checkpoint

CLASSES = ["Company","EducationalInstitution","Artist","Athlete","OfficeHolder",
           "MeanOfTransportation","Building","NaturalPlace","Village","Animal",
           "Plant","Album","Film","WrittenWork"]
NC = len(CLASSES)


@torch.no_grad()
def _assign(ae, acts, mean, std, device, bs=2048):
    """Nearest-centroid cluster id per row (dist2.argmin — NOT Sinkhorn Q)."""
    out_ids = np.empty(len(acts), dtype=np.int64)
    for s in range(0, len(acts), bs):
        x = torch.from_numpy(((acts[s:s+bs] - mean) / std).astype(np.float32)).to(device)
        out_ids[s:s+bs] = ae(x).dist2.argmin(dim=1).cpu().numpy()
    return out_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="llama3.2-3B")
    ap.add_argument("--pooling", default="last", choices=["last", "mean"])
    ap.add_argument("--mode", default="unprompted")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, _, _, ckpt = load_ae_checkpoint(Path(args.checkpoint), device)
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]
    layer = ckpt["config"]["data"]["target_layer"]
    K = ckpt["config"]["model"]["n_clusters"]

    base = Path("dbpedia/activations") / args.model / args.pooling / args.mode
    Xtr = np.load(base / f"layer_{layer}.npy")
    ytr = np.load(base / "labels_train.npy").astype(np.int64)
    Xte = np.load(base / f"layer_{layer}_test.npy")
    yte = np.load(base / "labels_test.npy").astype(np.int64)

    ltr = _assign(ae, Xtr, mean, std, device)
    lte = _assign(ae, Xte, mean, std, device)

    # Name each cluster by its dominant TRAIN class
    dom = np.full(K, -1)
    for k in range(K):
        c = ytr[ltr == k]
        if len(c):
            dom[k] = np.bincount(c, minlength=NC).argmax()

    merged = dom[lte]                       # test point -> named class (-1 if empty cluster)
    valid = merged >= 0
    acc = (merged[valid] == yte[valid]).mean()
    # purity: fraction of test in the majority class of its cluster
    pur = sum(np.bincount(yte[lte == k], minlength=NC).max() for k in np.unique(lte)) / len(yte)

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    nmi = normalized_mutual_info_score(yte, lte, average_method="arithmetic")
    ari = adjusted_rand_score(yte, lte)

    covered = sorted(set(dom[dom >= 0].tolist()))
    starved = [CLASSES[i] for i in range(NC) if i not in covered]
    from collections import Counter
    cnt = Counter(dom[dom >= 0].tolist())
    over = {CLASSES[c]: v for c, v in cnt.items() if v >= 2}

    print(f"\n  Over-cluster -> merge eval  (K={K} -> {NC} classes)")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  {'merged accuracy':<22} {acc*100:6.2f}%")
    print(f"  {'purity':<22} {pur*100:6.2f}%")
    print(f"  {'NMI':<22} {nmi:6.4f}")
    print(f"  {'ARI':<22} {ari:6.4f}")
    print(f"  {'coverage':<22} {len(covered)}/{NC}")
    print(f"  starved classes: {starved if starved else 'none'}")
    print(f"  over-covered (>=2 clusters): {over if over else 'none'}")


if __name__ == "__main__":
    main()
