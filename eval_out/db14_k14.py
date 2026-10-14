"""DB14 fair test: k-means K=14 on latents, Hungarian-matched accuracy + NMI."""
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from geoae.checkpoint import load_ae_checkpoint

D = "dbpedia/activations/llama3.2-3B/last/unprompted"
dev = "cuda"
Htr = np.load(f"{D}/layer_27.npy", mmap_mode="r")
ytr = np.load(f"{D}/labels_train.npy")
n = min(40000, len(Htr))
rng = np.random.RandomState(0); idx = np.sort(rng.choice(len(Htr), n, replace=False))
H = np.asarray(Htr[idx]).astype(np.float32); y = ytr[idx]
print(f"pool {H.shape}  classes {len(np.unique(y))}")

def hung_acc(y, p):
    K = max(p.max(), y.max()) + 1
    M = np.zeros((K, K), dtype=np.int64)
    for a, b in zip(p, y): M[a, b] += 1
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(y)

def score(name, Z):
    km = KMeans(14, n_init=10, random_state=0).fit(Z)
    print(f"{name:16s} acc {hung_acc(y, km.labels_):.4f}   NMI {nmi(y, km.labels_):.4f}")

# raw residual, z-scored (the control that has beaten every general AE so far)
score("raw zscore", (H - H.mean(0)) / (H.std(0) + 1e-6))

arms = {
 "kmeanspp":     "checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144/best_val.pt",
 "seeded_atlas": "checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_seeded_atlas/best_val.pt",
 "dpc":          "checkpoints/llama3.2-3B/layer27/k2000_bnh_b32k_lam1_d6144_dpc/best_val.pt",
}
for name, p in arms.items():
    ae, _, _, d = load_ae_checkpoint(p, dev); ae.eval()
    m = torch.as_tensor(d["norm_mean"], device=dev).float()
    s = torch.as_tensor(d["norm_std"], device=dev).float()
    zs = []
    with torch.no_grad():
        for i in range(0, len(H), 8192):
            h = torch.from_numpy(H[i:i+8192]).to(dev)
            zs.append(ae.encoder((h - m) / s).cpu().numpy())
    score(name, np.concatenate(zs))
