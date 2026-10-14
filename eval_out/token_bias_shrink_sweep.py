"""Held-out check of the token-bypass shrinkage constant (read-only; nothing is written to the repo).

For a bypass table built with b_t = n/(n+30) * mean_t, recover the unshrunk train
means and score, on the held-out val tail, the variance removed by
b_t(k) = n/(n+k) * mean_t for a grid of k. Per val token t the squared error is
quadratic in the factor s:  A_t - 2 s B_t + s^2 C_t,  so one pass gives every k.
"""
import json
import sys
import numpy as np

REPO = "/projects/bbyl/mhaider/GeoAE"
RUNS = {
    "llama": ("e2e/checkpoints/general/llama3.2-3B/layer27/token_bias_sampled.npz", "activations_sampled_10M", 27),
    "gemma": ("e2e/checkpoints/general/gemma3-4B/layer33/token_bias_sampled_struct5.npz",
              "activations_gemma3_4b_sampled_struct5_10M", 33),
}
KS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 100, 300]
BINS = [10, 20, 30, 50, 100, 300, 1000, 3000, 10 ** 9]


def run(name):
    tab_path, act, layer = RUNS[name]
    d = np.load(f"{REPO}/{tab_path}", allow_pickle=True)
    ids, n = d["token_ids"].astype(np.int64), d["counts"].astype(np.float64)
    k0 = float(d["shrink_k"])
    means = d["table"].astype(np.float32) * ((n + k0) / n)[:, None].astype(np.float32)   # unshrunk train means
    mnorm2 = (means.astype(np.float64) ** 2).sum(1)
    V = int(d["vocab_size"])
    index = np.full(V, -1, np.int64)
    index[ids] = np.arange(len(ids))
    mu, sd = d["norm_mean"].astype(np.float32), d["norm_std"].astype(np.float32)
    X = np.load(f"{REPO}/{act}/layer_{layer}.npy", mmap_mode="r")
    tok = np.load(f"{REPO}/{act}/rows_tok.npy", mmap_mode="r")
    N = len(X)
    val_start = int(N * 0.95)
    out = {}
    for label, n_eval in (("first200k", 200_000), ("all_val", N - val_start)):
        M = len(ids)
        B = np.zeros(M)
        nval = np.zeros(M)
        A_tok = np.zeros(M)
        sum_z = np.zeros(X.shape[1])
        tot_sq = 0.0
        rows = 0
        for s in range(val_start, val_start + n_eval, 20_000):
            e = min(s + 20_000, val_start + n_eval)
            z = (np.asarray(X[s:e]).astype(np.float32) - mu) / sd
            idx = index[np.asarray(tok[s:e]).astype(np.int64)]
            cov = idx >= 0
            sq = np.einsum("ij,ij->i", z, z, dtype=np.float64)
            tot_sq += sq.sum()
            sum_z += z.sum(0, dtype=np.float64)
            rows += len(z)
            dot = np.einsum("ij,ij->i", z[cov], means[idx[cov]], dtype=np.float64)
            B += np.bincount(idx[cov], weights=dot, minlength=M)
            A_tok += np.bincount(idx[cov], weights=sq[cov], minlength=M)
            nval += np.bincount(idx[cov], minlength=M)
        C = nval * mnorm2
        tot = tot_sq - (sum_z ** 2).sum() / rows          # same denominator as geoae.token_bias
        res = {"rows": rows, "val_coverage": float(nval.sum() / rows), "by_k": {}, "bins": []}
        for k in KS:
            sfac = n / (n + k)
            sse = tot_sq - 2 * (sfac * B).sum() + (sfac ** 2 * C).sum()
            res["by_k"][k] = float(1 - sse / tot)
        fine = np.concatenate([np.arange(0, 20, 0.25), np.arange(20, 200, 1.0)])
        vr = [1 - (tot_sq - 2 * (n / (n + k) * B).sum() + ((n / (n + k)) ** 2 * C).sum()) / tot for k in fine]
        res["best_k"], res["best_var_removed"] = float(fine[int(np.argmax(vr))]), float(max(vr))
        for a, b in zip(BINS[:-1], BINS[1:]):
            m = (n >= a) & (n < b)
            if nval[m].sum() == 0:
                continue
            sstar = B[m].sum() / C[m].sum()                      # best single factor for this bin
            fb = [(-2 * (n[m] / (n[m] + k) * B[m]).sum() + ((n[m] / (n[m] + k)) ** 2 * C[m]).sum()) for k in fine]
            gain = lambda k: float((2 * (n[m] / (n[m] + k) * B[m]).sum() - ((n[m] / (n[m] + k)) ** 2 * C[m]).sum())
                                   / A_tok[m].sum())
            res["bins"].append({"bin": f"{a}-{b - 1}" if b < 10 ** 9 else f"{a}+", "types": int(m.sum()),
                                "val_rows": int(nval[m].sum()), "val_row_share": float(nval[m].sum() / rows),
                                "best_factor": float(sstar), "best_k": float(fine[int(np.argmin(fb))]),
                                "factor_at_k30_rowmean": float((n[m] / (n[m] + 30) * nval[m]).sum() / nval[m].sum()),
                                "sqnorm_removed_k0": gain(0), "sqnorm_removed_k30": gain(30),
                                "sqnorm_removed_best": gain(float(fine[int(np.argmin(fb))]))})
        out[label] = res
    return out


if __name__ == "__main__":
    allres = {nm: run(nm) for nm in sys.argv[1:]}
    print(json.dumps(allres, indent=1))
