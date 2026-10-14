"""Old recipe vs the two 12 Oct recipes (no balancing + staged + 100 epochs; no balancing + no staging + 50 epochs), per model,
from the result files (missing files = dash).

    python3 eval_out/ne100_compare.py
Geometry: cq_std.json (1M rows, balanced k-means baseline); single-token: token_structure.json trivial_cur_80;
MMLU / perplexity with the reconstruction spliced in. Old Gemma no-bypass was trained on the train10M dump, the new one
on the bypass dump, so that row is not a like-for-like pair.
"""
import json
from pathlib import Path

R = Path("results/delta")
Q = Path("results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35")
J = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
g = lambda d, *k: (None if d is None else g(d.get(k[0]), *k[1:]) if k else d) if isinstance(d, dict) or not k else None
f = lambda v, fmt, mul=1.0: "—" if v is None else format(mul * v, fmt)

ROWS = [  # label, old (cq, key, token, key, mmlu, ppl), new folder, key
    ("Llama bypass", (R/"d6144/evals/cq_std.json", "bypass", R/"d6144/evals/token_structure.json", "bypass", R/"d6144/evals/mmlu.json", R/"d6144/evals/splice_ppl.json"), R/"d6144_ne100/evals", "bypass"),
    ("Llama no bypass", (R/"d6144_nobypass/evals/cq.json", None, R/"d6144_nobypass/evals/token_structure.json", None, R/"d6144_nobypass/evals/mmlu.json", R/"d6144/evals/splice_ppl_nobypass.json"), R/"d6144_nobypass_ne100/evals", "nobypass"),
    ("Gemma bypass", (R/"g4b_l33_d5120/evals/cq_std.json", "bypass", R/"g4b_l33_d5120/evals/token_structure.json", "bypass", R/"g4b_l33_d5120/evals/mmlu.json", R/"g4b_l33_d5120/evals/splice_cost/splice_ppl_ae.json"), R/"g4b_l33_d5120_ne100/evals", "bypass"),
    ("Gemma no bypass", (R/"g4b_l33_nobypass/evals/cq.json", None, R/"g4b_l33_nobypass/evals/token_structure.json", None, R/"g4b_l33_nobypass/evals/mmlu.json", None), R/"g4b_l33_nobypass_ne100/evals", "nobypass"),
    ("Qwen bypass", (Q/"bypass/evals/cq_std.json", "bypass", Q/"bypass/evals/token_structure.json", "bypass", Q/"bypass/evals/mmlu.json", R/"qwen25_l35_d4096/evals/splice_ppl.json"), R/"qwen25_l35_d4096_ne100/evals", "bypass"),
    ("Qwen no bypass", (Q/"plain/evals/cq_std.json", None, Q/"plain/evals/token_structure.json", None, Q/"plain/evals/mmlu.json", None), R/"qwen25_l35_nobypass_ne100/evals", "nobypass"),
]


def pick(d, key):
    if d is None:
        return None
    if key and key in d:
        return d[key]
    ks = [k for k in d if not k.startswith(("base", "plain", "meta"))]
    return d[ks[0]] if ks else None


def line(cq, cqk, ts, tsk, mm, pp):
    c, b = pick(J(cq), cqk), g(J(cq), "base")
    t = pick(J(ts), tsk) if ts else None
    m, p = (J(mm) if mm else None), (J(pp) if pp else None)
    ch, chb = g(c, "calinski_harabasz"), g(b, "calinski_harabasz")
    return (f"{f(ch, '.0f')} ({f(None if not (ch and chb) else ch / chb, '.1f')}×) | {f(g(c, 'davies_bouldin'), '.2f')} | {f(g(c, 'silhouette'), '.3f')} "
            f"| {f(g(c, 'effective_k'), '.0f')} | {f(g(t, 'trivial_cur_80'), '.1f', 100)} | {f(g(m, 'meta', 'recon_acc'), '.1f', 100)} (base {f(g(m, 'meta', 'base_acc'), '.1f', 100)}) "
            f"| {f(g(p, 'spliced', 'ppl'), '.2f')} (base {f(g(p, 'clean', 'ppl'), '.2f')})")


print("| Model | Recipe | CH (× k-means) | Davies–Bouldin | Silhouette | Clusters in use | Single-token (%) | MMLU (%) | Perplexity |")
print("|---|---|---|---|---|---|---|---|---|")
for label, old, new, key in ROWS:
    print(f"| {label} | old (balanced, staged, 50 ep) | {line(*old)} |")
    for tag, name in (("_ne100", "no balancing, staged, 100 ep"), ("_ns50", "no balancing, no staging, 50 ep")):
        d = Path(str(new).replace("_ne100", tag))
        if (d / "cq_std.json").exists() or (d / "mmlu.json").exists():
            print(f"| | {name} | {line(d/'cq_std.json', key, d/'token_structure.json', key, d/'mmlu.json', d/'splice_ppl.json')} |")
