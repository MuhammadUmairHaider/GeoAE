"""Paper Table 1 (geometry and faithfulness), straight from the result JSONs, as LaTeX.

Columns: Llama-3.2-3B (Base, GeoAE) | Gemma 3 4B (Base, SAE, GeoAE) | Qwen2.5-3B (Base, GeoAE).
Geometry block (1M activations sampled from the dump; each space in its own coordinates, Euclidean distances):
  Base  = balanced k-means on the standardized residual stream        <arm>/evals/cq_std.json  "base"
  GeoAE = the token-bypass latent with its trained prototypes          <arm>/evals/cq_std.json  "bypass"
  SAE   = Gemma Scope 2 16k codes, plain k-means on the full codes     g4b_l33_d5120/evals/sae_suite/cq_gs16k_plain.json "sae_plain"
  Within-cluster variance share = W / (W + B). cq_std.json stores it; the SAE file does not, so it is derived from the
  SAE's Calinski-Harabasz index with CH = (B / (K - 1)) / (W / (N - K)), i.e. W / (W + B) = 1 / (1 + CH (K - 1) / (N - K)).
  The same identity is checked against the stored value for every Base / GeoAE cell (must agree to 1e-3).
Faithfulness block (the reconstruction spliced in at the layer output):
  MMLU      2,000 test questions, zero-shot      <arm>/evals/mmlu.json; SAE: g4b_l33_d5120/evals/sae_suite/mmlu_gs16k.json
            (the SAE run has its own unmodified-model pass, 55.3 vs 55.2: one question)
  WikiText-2 perplexity, `spliced` condition     Llama d6144/evals/splice_ppl.json; Gemma evals/splice_cost/splice_ppl_{ae,gs16k}.json;
                                                 Qwen qwen25_l35_d4096/evals/splice_ppl.json

    python eval_out/paper_geometry_table.py
"""
import json
from pathlib import Path

R = Path("results/delta")
Q = Path("results/qwen25_3b_l35_2x_k2000_20261008/qwen2.5-3B/layer35/bypass/evals")
G = R / "g4b_l33_d5120/evals"
J = lambda p: json.loads(Path(p).read_text())
K, N = 2000, 1_000_000

# GEOAE_NE100=1: the 12 Oct no-balancing / 100-epoch models (results/delta/<arm>_ne100/evals/); base and SAE columns unchanged
import os
SFX = os.environ.get("GEOAE_SFX") or ("_ne100" if os.environ.get("GEOAE_NE100") == "1" else "")     # GEOAE_SFX=_ns100: the no-staging models
NE = bool(SFX)
NEW = {"llama": R / f"d6144{SFX}/evals", "gemma": R / f"g4b_l33_d5120{SFX}/evals", "qwen": R / f"qwen25_l35_d4096{SFX}/evals"}
NEW = {m: d for m, d in NEW.items() if all((d / f).exists() for f in ("cq_std.json", "mmlu.json", "splice_ppl.json"))}   # finished models only
cq = ({m: J(d / "cq_std.json") for m, d in NEW.items()} if NE else
      {"llama": J(R / "d6144/evals/cq_std.json"), "gemma": J(G / "cq_std.json"), "qwen": J(Q / "cq_std.json")})
sae = J(G / "sae_suite/cq_gs16k_plain.json")
assert sae["meta"]["n_sample"] == N and sae["sae_plain"]["n_empty_clusters"] == 0
within = lambda ch: 1.0 / (1.0 + ch * (K - 1) / (N - K))
for m, d in cq.items():
    for s in ("base", "bypass"):
        assert abs(within(d[s]["calinski_harabasz"]) - d[s]["within_var_share"]) < 1e-3, (m, s)
cols = [("llama", "base"), ("llama", "bypass"), ("gemma", "base"), ("gemma", "sae"), ("gemma", "bypass"), ("qwen", "base"), ("qwen", "bypass")]
if NE:
    cols = [c for c in cols if c[0] in cq]
    print("% columns:", ", ".join(f"{m} {s}" for m, s in cols))
def geo(m, s, key):
    d = sae["sae_plain"] if s == "sae" else cq[m][s]
    return within(d["calinski_harabasz"]) * 100 if key == "within" else d[key]

mm = ({m: J(d / "mmlu.json")["meta"] for m, d in NEW.items()} if NE else
      {"llama": J(R / "d6144/evals/mmlu.json")["meta"], "gemma": J(G / "mmlu.json")["meta"], "qwen": J(Q / "mmlu.json")["meta"]})
mm_sae = J(G / "sae_suite/mmlu_gs16k.json")["meta"]
pp = ({m: J(d / "splice_ppl.json") for m, d in NEW.items()} if NE else
      {"llama": J(R / "d6144/evals/splice_ppl.json"), "gemma": J(G / "splice_cost/splice_ppl_ae.json"),
       "qwen": J(R / "qwen25_l35_d4096/evals/splice_ppl.json")})
pp_sae = J(G / "splice_cost/splice_ppl_gs16k.json")
assert "gemma" not in pp or abs(pp_sae["clean"]["ppl"] - pp["gemma"]["clean"]["ppl"]) < (2e-2 if NE else 1e-6)   # NE: the SAE run is an older, separate run
def faith(m, s, key):
    if key == "mmlu":
        return 100 * (mm_sae["recon_acc"] if s == "sae" else mm[m]["base_acc"] if s == "base" else mm[m]["recon_acc"])
    return (pp_sae if s == "sae" else pp[m])["clean" if s == "base" else "spliced"]["ppl"]

ROWS = [("Calinski--Harabasz $\\uparrow$", "calinski_harabasz", "{:.1f}", max, geo),
        ("Davies--Bouldin $\\downarrow$", "davies_bouldin", "{:.2f}", min, geo),
        ("Silhouette $\\uparrow$", "silhouette", "{:.3f}", max, geo),
        ("Within-cluster variance (\\%) $\\downarrow$", "within", "{:.1f}", min, geo),
        None,
        ("MMLU accuracy (\\%) $\\uparrow$", "mmlu", "{:.1f}", max, faith),
        ("WikiText-2 perplexity $\\downarrow$", "ppl", "{:.2f}", min, faith)]
def fmt(v, f, bold):
    t = f.format(v).replace("-", "{-}")
    return "$\\mathbf{" + t + "}$" if bold else "$" + t + "$"
for row in ROWS:
    if row is None:
        print("\\addlinespace[2pt]\n\\multicolumn{8}{@{}l}{\\emph{Faithfulness of the spliced model}} \\\\")
        continue
    name, key, f, best, fn = row
    cells = []
    for m in ("llama", "gemma", "qwen"):
        mine = [(mm_, s) for mm_, s in cols if mm_ == m]
        vals = {s: fn(m, s, key) for _, s in mine}
        cand = {s: v for s, v in vals.items() if not (fn is faith and s == "base")}      # the unmodified model is the reference
        b = best(cand.values()) if len(cand) > 1 else None
        cells += [fmt(vals[s], f, b is not None and vals[s] == b) for _, s in mine]
    print("\\quad " + name + " & " + " & ".join(cells) + " \\\\")
print("% MMLU retained: " + ", ".join(f"{m} {100 * mm[m]['recon_acc'] / mm[m]['base_acc']:.1f}%" for m in mm)
      + f"; SAE {100 * mm_sae['recon_acc'] / mm_sae['base_acc']:.1f}% (its own base {100 * mm_sae['base_acc']:.1f})")
print("% perplexity ratio: " + ", ".join(f"{m} {pp[m]['spliced']['ppl_ratio']:.3f}" for m in pp) + f"; SAE {pp_sae['spliced']['ppl_ratio']:.3f}")
print("% CH ratio GeoAE / Base: " + ", ".join(f"{m} {cq[m]['bypass']['calinski_harabasz'] / cq[m]['base']['calinski_harabasz']:.2f}" for m in cq))
print("% prototypes in use (GeoAE effective_k of 2000): " + ", ".join(f"{m} {cq[m]['bypass']['effective_k']}" for m in cq))
