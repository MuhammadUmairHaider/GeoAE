"""CoarseWSD-20 sense alignment for every Gemma arm: base, token-removed base, GeoAE, with paired bootstrap CIs."""

from geoae.paths import resolve_input_path
import json
from collections import defaultdict

import numpy as np

from geoae.analysis.word_sense import basic_ci
from geoae.analysis.word_sense import cnmi

ARMS = ["g1b_l13_d1152", "g1b_l13_d2304", "g1b_l13_d4608", "g1b_l25_d1152", "g1b_l25_d2304", "g1b_l25_d4608"]
for arm in ARMS:
    d = np.load(resolve_input_path(f"results/delta/{arm}/interp/word_sense_rows.npz"), allow_pickle=True)
    m = d["dataset"] == "coarsewsd20"
    word, sense = d["word"][m], d["sense"][m]
    C = {s: d["cluster_" + s][m] for s in ("base", "bypass", "tokmean")}
    rng = np.random.default_rng(0)
    idx = defaultdict(list)
    for i, w in enumerate(word):
        idx[w].append(i)
    words = [w for w, ix in idx.items() if len(np.unique(sense[ix])) >= 2]

    def macro(take, n_perm):
        out = {s: [] for s in C}
        for w in words:
            ix = take(np.asarray(idx[w]))
            y = sense[ix]
            if len(np.unique(y)) < 2:
                continue
            for s in C:
                out[s].append(cnmi(y, C[s][ix], n_perm, rng))
        return {s: float(np.mean(v)) for s, v in out.items()}

    est = macro(lambda ix: ix, 20)
    boots = []
    for _ in range(200):
        mb = macro(lambda ix: ix[rng.integers(0, len(ix), len(ix))], 3)
        boots.append((mb["bypass"] - mb["base"], mb["bypass"] - mb["tokmean"]))
    boots = np.array(boots)
    e1, e2 = est["bypass"] - est["base"], est["bypass"] - est["tokmean"]
    l1, h1 = basic_ci(e1, boots[:, 0])
    l2, h2 = basic_ci(e2, boots[:, 1])
    wic = json.load(open(resolve_input_path(f"results/delta/{arm}/interp/word_sense.json")))["wic"]["per_space"]
    print(f"{arm}: base {est['base']:.3f} tokrem {est['tokmean']:.3f} geoae {est['bypass']:.3f} | "
          f"geoae-base {e1:+.3f} [{l1:+.3f},{h1:+.3f}] | geoae-tokrem {e2:+.3f} [{l2:+.3f},{h2:+.3f}] | "
          f"WiC AUROC base {wic['base']['auroc_cos']:.3f} tokrem {wic['tokmean']['auroc_cos']:.3f} "
          f"geoae {wic['bypass']['auroc_cos']:.3f}", flush=True)
