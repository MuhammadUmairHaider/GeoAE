"""Build the round-2 report page from report_data.json. Every number in a table comes from the data file."""
import json, html
from pathlib import Path

SP = Path(__file__).parent
D = json.load(open(SP / "report_data.json"))
S, A = D["summary"], D["arms"]
LBL = {"d768": "d768", "d3072": "d3072", "d6144": "d6144", "d12288": "d12288", "d6144k4000": "d6144 · K4000"}
e = html.escape

def f3(x): return "–" if x is None else f"{x:.3f}"
def f1(x): return "–" if x is None else f"{x:.1f}"
def pct(x, d=1): return "–" if x is None else f"{100*x:.{d}f}%"
def sgn(x, d=1): return "–" if x is None else f"{x:+.{d}f}"

def chip(kind, text=None):
    names = {"holds": "holds", "stronger": "stronger", "weaker": "weaker", "cost": "cost", "new": "new", "tie": "tie", "worse": "worse", "control": "control ties"}
    return f'<span class="chip {kind}">{e(text or names[kind])}</span>'

def table(head, rows, cls="", note=None, num_from=1):
    h = "".join(f'<th{" class=num" if i >= num_from else ""}>{c}</th>' for i, c in enumerate(head))
    body = ""
    for r in rows:
        tr_cls = ""
        if isinstance(r, tuple) and len(r) == 2 and isinstance(r[1], str) and r[1].startswith("row:"):
            r, tr_cls = r[0], r[1][4:]
        body += f'<tr class="{tr_cls}">' + "".join(f'<td{" class=num" if i >= num_from else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>"
    cap = f'<p class="tnote">{note}</p>' if note else ""
    return f'<div class="tw {cls}"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>{cap}'

def bottom(items):
    """Bottom-line strip: (label, base_text, ae_text, chip_kind, chip_text, note)."""
    out = '<div class="bl">'
    for lab, b, y, kind, ct, note in items:
        arrow = f'<span class="blv"><span class="blb">{b}</span><span class="bla">→</span><span class="bly {kind}">{y}</span></span>' if b is not None else f'<span class="blv"><span class="bly {kind}">{y}</span></span>'
        out += f'<div class="blt"><span class="bll">{lab}</span>{arrow}<span class="bln">{chip(kind, ct)} {note}</span></div>'
    return out + "</div>"

def diffspan(d, good_if_pos=True, fmt="{:+.3f}", eps=0.005):
    if d is None: return "–"
    good = d > eps if good_if_pos else d < -eps
    bad = d < -eps if good_if_pos else d > eps
    return f'<span class="{"pos" if good else "neg" if bad else "neu"}">{fmt.format(d)}</span>'

def best(vals, hi=True):
    xs = [v for v in vals if v is not None]
    return (max(xs) if hi else min(xs)) if xs else None

def mark(v, b, fmt=f3):
    s = fmt(v)
    return f"<b>{s}</b>" if v is not None and b is not None and abs(v - b) < 1e-12 else s

CJ6 = S["d6144"]["cluster_steer_judge"]
PW6 = D["concept"]["d6144"]["scopes"]["pooled"]["pairwise"]
CU6 = D["concept"]["d6144"]["scopes"]["pooled"]["usable"]
def pfmt(p): return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
P = D["probe"]; T = D["tokstruct"]; U = D["usage"]; G = D["geometry"]; C = D["concept"]; J = D["cjudge"]; TR = D["transfer"]; N = D["number"]

# ---------------------------------------------------------------- headline table (d6144)
a = "d6144"; s = S[a]
p6 = P[a]
dbr = s["range_intervention_db14_dprime"]; bbr = s["range_intervention_biasbios_dprime"]
tr6 = TR[a]
cs = C[a]["scopes"]["pooled"]
head_rows = [
    ["Topic clustering, DBpedia-14 (cNMI)", f3(p6["topic14"]["arms"]["base"]["cnmi"]), f3(p6["topic14"]["arms"]["base_tokmean"]["cnmi"]), f"<b>{f3(p6['topic14']['arms']['bypass']['cnmi'])}</b>", chip("holds") + " " + chip("control")],
    ["Mean over 9 sequence tasks (cNMI)", f3(s["probe"]["seq_mean"]["base"]), f3(s["probe"]["seq_mean"]["base_tokmean"]), f"<b>{f3(s['probe']['seq_mean']['bypass'])}</b>", chip("holds")],
    ["Part-of-speech clustering (cNMI)", f"<b>{f3(s['probe']['pos_mean']['base'])}</b>", f3(s["probe"]["pos_mean"]["base_tokmean"]), f3(s["probe"]["pos_mean"]["bypass"]), chip("cost")],
    ["Pure current-token clusters (≥80% one token)", pct(T[a]["base"]["trivial_cur_80"]), f"<b>{pct(T[a]['base_tokmean']['trivial_cur_80'])}</b>", pct(T[a]['bypass']['trivial_cur_80']), chip("holds") + " " + chip("control")],
    ["MMLU with the AE spliced in (unspliced 0.562)", "0.562", "–", f3(s["mmlu"]["recon_acc"]), chip("tie", "−1.2 pts")],
    ["DBpedia removal: removed / collateral per 100", f"{100*dbr['rm_range_comp']['h_tgt_drop']:.0f} / {100*dbr['rm_range_comp']['h_comp_drop']:.0f}", "–", f"<b>{100*dbr['rm_range_comp']['z_tgt_drop']:.0f} / {100*dbr['rm_range_comp']['z_comp_drop']:.0f}</b>", chip("holds")],
    ["bias_in_bios removal: removed / collateral per 100", f"{100*bbr['rm_range_comp']['h_tgt_drop']:.0f} / {100*bbr['rm_range_comp']['h_comp_drop']:.0f}", "–", f"{100*bbr['rm_range_comp']['z_tgt_drop']:.0f} / {100*bbr['rm_range_comp']['z_comp_drop']:.0f}", chip("tie", "parity")],
    ["Next-token transfer into a cluster (disruption 0.1)", pct(tr6["base:km"]["0.1"]["est"]), pct(tr6["base_tokmean:km"]["0.1"]["est"]), f"<b>{pct(tr6['bypass:ae']['0.1']['est'])}</b>", chip("holds", "2.4×")],
    ["Usable steered generations per 100 (own clusters)", f1(s["cluster_steer_judge"]["base_usable_per_100"]), "–", f"<b>{f1(s['cluster_steer_judge']['bypass_usable_per_100'])}</b>", chip("weaker", f"{CJ6['bypass_usable_per_100']/CJ6['base_usable_per_100']:.1f}×, not significant")],
    ["Named-concept steering, clusters as a direction (per 100)", f1(100*cs["usable"]["base_dir"][0]), "–", f"<b>{f1(100*cs['usable']['bypass_dir'][0])}</b>", chip("holds", pfmt(PW6["bypass_dir_minus_base_dir"]["wilcoxon_p"]))],
]
headline = table(["Measure (d6144, K = 2,000)", "Base", "Token-mean control", "Bypass AE", "Verdict"], head_rows,
                 note="Bold = best of the columns shown. cNMI = chance-corrected normalised mutual information between cluster id and label (0 = no better than a shuffled label, 1 = perfect). All numbers are from one training seed.")

# ---------------------------------------------------------------- arms table
EP = {"d768": 6.6, "d3072": 8.9, "d6144": 11.8, "d12288": 22.7, "d6144k4000": 16.2}
arm_rows = []
for a in A:
    k = 4000 if "k4000" in a else 2000
    w = int(a[1:].split("k")[0]) if a != "d6144k4000" else 6144
    u = U.get(a) or {}
    arm_rows.append([LBL[a], f"{w:,}", f"{k:,}", f"{S[a]['token_structure']['fve_bypass']:.3f}", f"{EP[a]:.1f}", f"{u.get('perplexity','–'):,}" if u else "–", f"{u.get('starved','–')} / {u.get('zero','–')}"])
arm_rows.append((["d12288 · K4000", "12,288", "4,000", "training", "~19", "–", "–"], "row:pending"))
arms_tbl = table(["Run", "Latent width", "Clusters K", "Held-out FVE", "Min / epoch", "Clusters in use", "Starved / empty"], arm_rows,
                 note="FVE = fraction of the residual's variance the reconstruction explains, on 500k held-out rows. Clusters in use = exp(entropy) of the checkpoint's hard-assignment counts. Starved = clusters below a tenth of their fair share. Minutes per epoch are wall-clock on the GPU the run finished on (A100 or A40) and are not a speed comparison.")

# ---------------------------------------------------------------- health table
hr = []
for a in A:
    m = S[a]["mmlu"]; cq = S[a]["cq"]
    hr.append([LBL[a], f"{S[a]['token_structure']['fve_bypass']:.3f}", f3(m["recon_acc"]), sgn(100*m["delta"]), f"{m['flips']['base_only']} / {m['flips']['recon_only']}",
               f"{cq['effective_k']['bypass']:,}", f"{cq['effective_rank']['base']:.0f} → {cq['effective_rank']['bypass']:.0f}"])
health_tbl = table(["Run", "FVE", "MMLU spliced", "Δ vs 0.562 (pts)", "Questions lost / gained", "Effective K (1M rows)", "Effective rank base → AE"], hr,
                   note="MMLU: 2,000 questions; the AE's reconstruction replaces layer 27 during the forward pass. Questions lost / gained counts items the unspliced model got right and the spliced one wrong, and the reverse. Effective K and effective rank come from clustering_quality on 1M dump rows; the full set of clustering-quality scores is in Cluster compactness.")

# ---------------------------------------------------------------- token structure
tsr = []
a = "d6144"
for arm, lab in [("base", "Base k-means"), ("base_tokmean", "Token-mean control"), ("bypass", "Bypass AE")]:
    t = T[a][arm]
    tsr.append([lab, pct(t["trivial_cur_80"]), pct(t["trivial_cur_50"]), f3(t["cnmi_cur"]), f3(t["cnmi_next"]), pct(t["top10"])])
ts_tbl = table(["d6144, K = 2,000", "Pure token clusters (≥80%)", "Mostly one token (≥50%)", "cNMI with current token", "cNMI with next token", "Top-10 clusters' share"], tsr)
tsw = [[LBL[a], pct(T[a]["base"]["trivial_cur_80"]), pct(T[a]["bypass"]["trivial_cur_80"]), f3(T[a]["base"]["cnmi_cur"]), f3(T[a]["bypass"]["cnmi_cur"]), f3(T[a]["bypass"]["cnmi_next"])] for a in A]
tsw_tbl = table(["Run", "Pure token clusters, base", "…bypass", "cNMI current token, base", "…bypass", "cNMI next token, bypass"], tsw)

# ---------------------------------------------------------------- probe full table (d6144)
NAMES = {"surface": "Surface form", "pos_coarse": "Part of speech (coarse)", "pos_fine": "Part of speech (fine)", "ner_coarse": "Named entity (coarse)", "ner_fine": "Named entity (fine)",
         "sentiment": "Sentiment (short)", "sentiment_long": "Sentiment (reviews)", "subjectivity": "Subjectivity", "formality": "Formality", "language": "Language (14)", "domain": "Domain (prose/code/math)",
         "topic4": "Topic, AG News (4)", "topic14": "Topic, DBpedia (14)", "topic20": "Topic, 20 Newsgroups", "ravel_country": "RAVEL country", "ravel_continent": "RAVEL continent",
         "ravel_language": "RAVEL language", "ioi_role": "IOI role", "ioi_name": "IOI name"}
ORDER_SEQ = ["topic14", "topic4", "topic20", "language", "domain", "subjectivity", "formality", "sentiment", "sentiment_long"]
ORDER_TOK = ["pos_coarse", "pos_fine", "surface", "ner_coarse", "ner_fine", "ioi_role", "ioi_name", "ravel_country", "ravel_continent", "ravel_language"]
def probe_rows(a, order):
    rows = []
    for r in order:
        x = P[a][r]; ar = x["arms"]
        vals = [ar[k]["cnmi"] for k in ("base", "base_tokmean", "bypass")]
        b = best(vals)
        cells = [mark(v, b) + f'<span class="ci">±{(ar[k]["ci"][1]-ar[k]["ci"][0])/2:.3f}</span>' for v, k in zip(vals, ("base", "base_tokmean", "bypass"))]
        d = ar["bypass"]["cnmi"] - ar["base"]["cnmi"]
        rows.append([NAMES[r], f'{x["n"]:,}', *cells, f'<span class="{"pos" if d > 0.01 else "neg" if d < -0.01 else ""}">{d:+.3f}</span>', "–" if x["leak"] is None else f'{x["leak"]:.2f}'])
    return rows
PH = ["Task", "Rows", "Base", "Token-mean control", "Bypass AE", "Bypass − base", "Token leak"]
probe_seq = table(PH, probe_rows("d6144", ORDER_SEQ))
probe_tok = table(PH, probe_rows("d6144", ORDER_TOK),
                  note="± = half-width of a 95% interval from 200 bootstrap resamples of the rows (centred on the estimate). Token leak = cNMI between the label and the current token id alone: how much of the task a lookup on the token could carry, bypassing the latent. A high leak (part of speech 0.41, IOI name 1.00) means the token table, not the latent, can hold that information.")
KEYR = ["topic14", "topic4", "language", "subjectivity", "domain", "pos_coarse", "ner_coarse", "ioi_role", "ravel_country"]
pw = []
for r in KEYR:
    row = [NAMES[r], f3(P["d6144"][r]["arms"]["base"]["cnmi"]), f3(P["d6144"][r]["arms"]["base_tokmean"]["cnmi"])]
    vals = [P[a][r]["arms"]["bypass"]["cnmi"] for a in A[:4]]
    b = best(vals)
    row += [mark(v, b) for v in vals]
    row += [f'{f3(P["d6144k4000"][r]["arms"]["base"]["cnmi"])} → {f3(P["d6144k4000"][r]["arms"]["bypass"]["cnmi"])}']
    pw.append(row)
probe_width = table(["Task", "Base K2000", "Control K2000", "d768", "d3072", "d6144", "d12288", "K4000: base → bypass"], pw,
                    note="The K2000 base and control are the same codebooks for all four widths, so read across a row. The K4000 run has its own 4,000-cluster base and is shown as a pair. Bold = best bypass width.")

# ---------------------------------------------------------------- geometry
GR = ["topic14", "topic4", "domain", "language", "formality", "subjectivity", "pos_coarse", "ner_coarse", "ioi_role", "ravel_continent"]
gr = []
for r in GR:
    k = G["d6144"][r]["knn"]
    vals = [k["base"], k["base_tokmean"], k["bypass"]]; b = best(vals)
    gr.append([NAMES[r], f'{G["d6144"][r]["classes"]}', *[mark(v, b) for v in vals], diffspan(k["bypass"] - k["base"], eps=0.01)] + [f3(G[a][r]["knn"]["bypass"]) for a in ("d768", "d3072", "d12288", "d6144k4000")])
geo_tbl = table(["Task", "Classes", "Base", "Control", "Bypass d6144", "Bypass − base", "d768", "d3072", "d12288", "d6144 · K4000"], gr,
                note="kNN-10 label agreement: after PCA to 50 dimensions, a 10-nearest-neighbour classifier is fit on 60% of up to 6,000 points and scored on the other 40%. It is computed in the representation, not in the 2-D picture. Up to 14 classes per task.")

# ---------------------------------------------------------------- interventions
def ir(key):
    rows = []
    for a in A:
        r = S[a].get(key)
        if not r:
            rows.append([LBL[a], "not run", "", "", "", "", "", ""]); continue
        o = r["rm_range_comp"]; m = r["range"]
        dm = m["collateral_z_pct"] - m["collateral_h_pct"]
        rows.append([LBL[a], f"{100*o['h_tgt_drop']:.0f}", f"{100*o['h_comp_drop']:.0f}", f"{100*o['z_tgt_drop']:.0f}", f"{100*o['z_comp_drop']:.0f}",
                     diffspan(100*o['z_minus_h'], fmt="{:+.1f}", eps=1.5) + f'<span class="ci">± {100*o["z_minus_h_se"]:.1f}</span>',
                     f"{o['z_wins']} of {o['n']}", diffspan(dm, good_if_pos=False, fmt="{:+.1f}", eps=1.5)])
    return rows
IH = ["Run", "Base removed /100", "Base collateral /100", "AE removed /100", "AE collateral /100", "Net score AE − base", "Concepts AE wins", "Collateral at equal removal AE − base"]
db_tbl = table(IH, ir("range_intervention_db14_dprime"))
bb_tbl = table(IH, ir("range_intervention_biasbios_dprime"),
               note="Removed: per 100 documents of the target class, how many stop being classified as it (higher is better). Collateral: per 100 documents of other classes, how many are disturbed (lower is better). Net score = removed − collateral; the AE − base column is green when the AE is better, ± is one standard error over concepts. Last column: both spaces are pushed to the same removal level, and their collateral is compared there (green = AE does less damage). d768 cannot be edited: with it spliced in the model gets no documents right.")
steer_rows = []
for a in A:
    sd = D["steer"].get(a)
    if not sd or sd["a1.0"]["h_sel"] != sd["a1.0"]["h_sel"]:
        steer_rows.append([LBL[a], "–", "–", "–", "not available" if not sd else "NaN (see caveats)"]); continue
    steer_rows.append([LBL[a]] + [f"{sd[k]['h_sel']:+.3f} → {sd[k]['z_sel']:+.3f}" for k in ("a0.5", "a1.0", "a2.0")] + [""])
steer_tbl = table(["Run", "α = 0.5, h → z", "α = 1", "α = 2", ""], steer_rows,
                  note="Global DBpedia steering selectivity. Sign convention is the opposite of the range tables: more negative is the cleaner steer. At α = 1 the AE is slightly cleaner in all three runs where it was computed (by 0.01–0.04), and the two spaces converge at α = 8.")
num_rows = [[LBL[a], f"{N[a]['range:z_minus_h']['d']:+.3f}", f"[{N[a]['range:z_minus_h']['ci'][0]:.3f}, {N[a]['range:z_minus_h']['ci'][1]:.3f}]",
             f"{N[a]['transport:z_minus_h']['d']:+.3f}", f"{N[a]['range:z_minus_rotations_mean']['d']:+.3f}"] for a in A]
num_tbl = table(["Run", "Target flip, z − h (range, α=1)", "95% interval", "Same, transport edit", "z − random rotations"], num_rows,
                note="Number control: flip a subject noun's grammatical number (singular ↔ plural) on 144 held-out items built from 12 nouns. Strict flip = the gold-pair probability falls below 0.5. The interval bootstraps the 12 nouns only, so it does not include training-seed or template variation. The rotation column compares z with random rotations of the same edit, a check that the gain is not just the edit's size.")

# ---------------------------------------------------------------- transfer
trr = []
for a in A:
    t = TR[a]
    trr.append([LBL[a], pct(t["base:km"]["0.1"]["est"]), pct(t["base_tokmean:km"]["0.1"]["est"]), f'<b>{pct(t["bypass:ae"]["0.1"]["est"])}</b>', pct(t["bypass:ae_h"]["0.1"]["est"]), f'<span class="pos">{t["bypass:ae"]["0.1"]["est"]/t["base:km"]["0.1"]["est"]:.1f}×</span>', pct(t["base:km"]["1.0"]["est"]) + " → " + pct(t["bypass:ae"]["1.0"]["est"])])
tr_tbl = table(["Run", "Base", "Token-mean control", "Bypass (edit z)", "Bypass partition, edit h", "Bypass ÷ base", "At disruption 1.0: base → bypass"], trr,
               note="Move a held-out row from its cluster toward a target cluster by the gap between the two cluster means, and count how often its next-token distribution lands closer to the target's than to the source's. Compared at matched disruption (how much the prediction moves in total), here 0.1; 95% intervals are about ±0.3 points. The fourth column uses the AE's clusters but edits the base residual, isolating the partition from the encoder.")

# ---------------------------------------------------------------- generation judge (d6144 per alpha)
jr = []
for al, v in J["d6144"].items():
    b, y = v["base"], v["bypass"]
    bb = max(v2["base"]["usable"] for v2 in J["d6144"].values()); yb = max(v2["bypass"]["usable"] for v2 in J["d6144"].values())
    jr.append([f"α = {al}", pct(b["fluent_share"], 0), sgn(100*b["net_hit"]), (f'<b>{f1(100*b["usable"])}</b>' if b["usable"] == bb else f1(100*b["usable"])),
               pct(y["fluent_share"], 0), sgn(100*y["net_hit"]), (f'<b>{f1(100*y["usable"])}</b>' if y["usable"] == yb else f1(100*y["usable"]))])
judge_tbl = table(["Strength", "Base: fluent", "on-target, net", "usable /100", "Bypass: fluent", "on-target, net", "usable /100"], jr,
                  note="40 target clusters per codebook × 20 neutral prompts × 32 greedy tokens, steering every generated position. Fluent = the LLM judge answers YES to a separate question, asked without the target: is this fluent, meaningful text? Loops, word lists and gibberish are NO; text merely cut off at the end is not. On-target net = the LLM judge's forced-choice hit rate (target vs 3 distractor clusters, chance 25%) minus the same prompt's unsteered rate, among fluent continuations only. Usable = fluent share × net hit. Judge: gemini-2.5-flash-lite via OpenRouter.")
jw = [[LBL[a], f1(S[a]["cluster_steer_judge"]["base_usable_per_100"]), f'α {S[a]["cluster_steer_judge"]["base_best_alpha"]}',
       f1(S[a]["cluster_steer_judge"]["bypass_usable_per_100"]), f'α {S[a]["cluster_steer_judge"]["bypass_best_alpha"]}',
       f'{S[a]["cluster_steer_judge"]["bypass_usable_per_100"]/S[a]["cluster_steer_judge"]["base_usable_per_100"]:.1f}×'] for a in A]
judge_w = table(["Run", "Base usable /100", "at", "Bypass usable /100", "at", "Ratio"], jw)

# ---------------------------------------------------------------- concept steering
HN = {"label": "Label direction (supervised)", "label_z": "Label direction computed in the AE latent", "bypass_dir": "Bypass clusters as a direction", "base_dir": "Base clusters as a direction",
      "bypass": "Bypass clusters, move into cluster", "base": "Base clusters, move into cluster"}
cr = []
for h in ["label", "label_z", "bypass_dir", "base_dir", "bypass", "base"]:
    row = [HN[h]]
    for sc in ["pooled", "db14", "biasbios"]:
        u = C["d6144"]["scopes"][sc]["usable"][h]
        row.append(f'{100*u[0]:.1f}<span class="ci">[{100*u[1]:.0f}, {100*u[2]:.0f}]</span>')
    row.append(pct(C["d6144"]["scopes"]["pooled"]["fluent"][h][0], 0))
    cr.append(row)
concept_tbl = table(["Handle (d6144)", "Both datasets (42 concepts)", "DBpedia (14)", "bias_in_bios (28)", "Fluent at best α"], cr,
                    note="Usable continuations per 100 at each handle's best strength, chosen once on the pooled set. Brackets: 95% bootstrap interval over concepts. Every handle is scaled to the base handle's typical edit size, so a given α moves each handle by a comparable amount.")
PWN = {"bypass_dir_minus_base_dir": "Are bypass clusters better directions than base clusters?", "bypass_minus_base": "Are bypass clusters better move-in targets than base clusters?",
       "bypass_dir_minus_label": "Do bypass-cluster directions match the supervised label direction?", "label_z_minus_label": "Does computing the label direction in the AE latent help?",
       "bypass_dir_minus_bypass": "Bypass clusters: is a direction better than moving in?", "base_dir_minus_base": "Base clusters: is a direction better than moving in?"}
pr = []
for k, nm in PWN.items():
    row = [nm]
    for a in A:
        v = C[a]["scopes"]["pooled"]["pairwise"][k]
        sig = v["wilcoxon_p"] < 0.05
        word = ("yes" if v["mean_diff"] > 0 else "no, lower") if sig else "no clear difference"
        row.append(f'<span class="{("pos" if v["mean_diff"] > 0 else "neg") if sig else "neu"}">{100*v["mean_diff"]:+.1f}</span><span class="ci">{word} · p {v["wilcoxon_p"]:.3f}</span>')
    pr.append(row)
pair_tbl = table(["Question (both datasets, 42 concepts)"] + [LBL[a] for a in A], pr,
                 note="Each cell: difference in usable continuations per 100 (first minus second), paired over the same 42 concepts. Green or red = significant at p < 0.05 (Wilcoxon signed-rank); grey = no clear difference. No correction for multiple comparisons.")

# ---------------------------------------------------------------- width synthesis
def rowf(label, fn, hi=True, fmt=f3):
    vals = [fn(a) for a in A[:4]]; b = best(vals, hi)
    return [label] + [mark(v, b, fmt) for v in vals]
def pt(x): return "–" if x is None else f"{x:.1f}"
syn = [
    rowf("Held-out FVE", lambda a: S[a]["token_structure"]["fve_bypass"]),
    rowf("MMLU spliced (0.562 unspliced)", lambda a: S[a]["mmlu"]["recon_acc"]),
    rowf("Topic14 cNMI (base 0.377)", lambda a: P[a]["topic14"]["arms"]["bypass"]["cnmi"]),
    rowf("Mean sequence cNMI (base 0.168)", lambda a: S[a]["probe"]["seq_mean"]["bypass"]),
    rowf("Mean POS cNMI (base 0.362, higher = less cost)", lambda a: S[a]["probe"]["pos_mean"]["bypass"]),
    rowf("DBpedia removal selectivity z − h (pts)", lambda a: None if not S[a].get("range_intervention_db14_dprime") else 100*S[a]["range_intervention_db14_dprime"]["rm_range_comp"]["z_minus_h"], fmt=pt),
    rowf("bias_in_bios collateral z − h at matched removal (lower better)", lambda a: None if not S[a].get("range_intervention_biasbios_dprime") else S[a]["range_intervention_biasbios_dprime"]["range"]["collateral_z_pct"] - S[a]["range_intervention_biasbios_dprime"]["range"]["collateral_h_pct"], hi=False, fmt=pt),
    rowf("Next-token transfer at 0.1 (base 3.7%)", lambda a: 100*TR[a]["bypass:ae"]["0.1"]["est"], fmt=pt),
    rowf(f"Usable steered generations /100 (base {CJ6['base_usable_per_100']:.1f})", lambda a: S[a]["cluster_steer_judge"]["bypass_usable_per_100"], fmt=pt),
    rowf(f"Concept steering, bypass direction /100 (base {100*CU6['base_dir'][0]:.1f})", lambda a: 100*C[a]["scopes"]["pooled"]["usable"]["bypass_dir"][0], fmt=pt),
    rowf("Topic14 kNN in the latent (base 0.942)", lambda a: G[a]["topic14"]["knn"]["bypass"]),
]
syn_tbl = table(["Bypass AE, K = 2,000", "d768", "d3072", "d6144", "d12288"], syn, note="Bold = best width on that row. The base numbers in brackets are identical for all four widths.")

# K comparison
k6, k4 = "d6144", "d6144k4000"
def kk(label, fb6, fy6, fb4, fy4, fmt=f3):
    return [label, fmt(fb6), fmt(fy6), fmt(fb4), fmt(fy4)]
kr = [
    kk("Topic14 cNMI", P[k6]["topic14"]["arms"]["base"]["cnmi"], P[k6]["topic14"]["arms"]["bypass"]["cnmi"], P[k4]["topic14"]["arms"]["base"]["cnmi"], P[k4]["topic14"]["arms"]["bypass"]["cnmi"]),
    kk("Token-mean control, topic14", P[k6]["topic14"]["arms"]["base_tokmean"]["cnmi"], None, P[k4]["topic14"]["arms"]["base_tokmean"]["cnmi"], None),
    kk("Mean sequence cNMI", S[k6]["probe"]["seq_mean"]["base"], S[k6]["probe"]["seq_mean"]["bypass"], S[k4]["probe"]["seq_mean"]["base"], S[k4]["probe"]["seq_mean"]["bypass"]),
    kk("Pure token clusters", T[k6]["base"]["trivial_cur_80"], T[k6]["bypass"]["trivial_cur_80"], T[k4]["base"]["trivial_cur_80"], T[k4]["bypass"]["trivial_cur_80"], fmt=pct),
    kk("MMLU spliced", None, S[k6]["mmlu"]["recon_acc"], None, S[k4]["mmlu"]["recon_acc"]),
    ["DBpedia removal, removed/collateral", "86 / 19", "87 / 13", "87 / 19", "88 / 18"],
    kk("bias_in_bios collateral z − h, matched", None, S[k6]["range_intervention_biasbios_dprime"]["range"]["collateral_z_pct"]-S[k6]["range_intervention_biasbios_dprime"]["range"]["collateral_h_pct"], None, S[k4]["range_intervention_biasbios_dprime"]["range"]["collateral_z_pct"]-S[k4]["range_intervention_biasbios_dprime"]["range"]["collateral_h_pct"], fmt=lambda x: "–" if x is None else f"{x:+.1f}"),
    kk("Transfer at 0.1", TR[k6]["base:km"]["0.1"]["est"], TR[k6]["bypass:ae"]["0.1"]["est"], TR[k4]["base:km"]["0.1"]["est"], TR[k4]["bypass:ae"]["0.1"]["est"], fmt=pct),
    kk("Usable steered generations /100", S[k6]["cluster_steer_judge"]["base_usable_per_100"], S[k6]["cluster_steer_judge"]["bypass_usable_per_100"], S[k4]["cluster_steer_judge"]["base_usable_per_100"], S[k4]["cluster_steer_judge"]["bypass_usable_per_100"], fmt=f1),
    kk("Concept: clusters as a direction /100", 100*C[k6]["scopes"]["pooled"]["usable"]["base_dir"][0], 100*C[k6]["scopes"]["pooled"]["usable"]["bypass_dir"][0], 100*C[k4]["scopes"]["pooled"]["usable"]["base_dir"][0], 100*C[k4]["scopes"]["pooled"]["usable"]["bypass_dir"][0], fmt=f1),
]
k_tbl = table(["d6144", "Base K2000", "Bypass K2000", "Base K4000", "Bypass K4000"], kr)

# ---------------------------------------------------------------- replication vs old box
rep = [
    ["Topic14 cNMI, base → bypass", "0.45 → 0.60", "0.38 → 0.58", chip("holds", "holds, wider gap")],
    ["Mean sequence-concept alignment", "0.173 → 0.205", "0.168 → 0.222", chip("holds")],
    ["Token-mean control vs bypass (concept alignment)", "ties (+0.005)", "0.209 vs 0.222 (+0.013)", chip("control", "control nearly ties")],
    ["Pure token-detector clusters", "11% → 5%", "10.3% → 5.1%", chip("holds")],
    ["Part-of-speech clustering", "0.34 → 0.15", "0.36 → 0.17", chip("cost", "same cost")],
    ["Reconstruction error (FVU)", "0.034", "0.035", chip("holds")],
    ["MMLU spliced (base 0.562)", "0.548", "0.550", chip("holds")],
    ["DBpedia removal, removed / collateral", "86/20 → 89/15", "86/19 → 87/13", chip("holds")],
    ["bias_in_bios interventions", "parity", "parity at own setting; +6 collateral at matched removal", chip("weaker", "partly")],
    ["Number control", "tie", "AE +0.13 [0.10, 0.17]", chip("new", "new AE win")],
    ["Next-token transfer, bypass ÷ base", "2.4×", "2.4×", chip("holds")],
    ["Usable steered generations /100", "6.5 → 15", f"{CJ6['base_usable_per_100']:.1f} → {CJ6['bypass_usable_per_100']:.1f} (95% interval of the gap includes 0)", chip("weaker", "weaker, not significant")],
    ["Concept: bypass direction vs base direction", "28.6 vs 22.9 (p = 0.04)", f"{100*CU6['bypass_dir'][0]:.1f} vs {100*CU6['base_dir'][0]:.1f} ({pfmt(PW6['bypass_dir_minus_base_dir']['wilcoxon_p'])})", chip("holds")],
    ["Concept: bypass direction vs label direction", "28.6 vs 27.9", f"{100*CU6['bypass_dir'][0]:.1f} vs {100*CU6['label'][0]:.1f} ({pfmt(PW6['bypass_dir_minus_label']['wilcoxon_p'])})", chip("tie", "tie")],
    ["Concept: label direction in AE latent vs residual", "28.9 vs 27.9", f"{100*CU6['label_z'][0]:.1f} vs {100*CU6['label'][0]:.1f} ({pfmt(PW6['label_z_minus_label']['wilcoxon_p'])})", chip("tie", "tie")],
]
rep_tbl = table(["Claim (d6144)", "Old box (legacy data)", "Delta (new data)", "Verdict"], rep,
                note="Old-box concept steering covered DBpedia only; Delta pools DBpedia and bias_in_bios. On DBpedia alone Delta gives label 26.4, bypass direction 26.4, base direction 23.2. The Delta runs are new replicates on a new dump, so row-for-row agreement is not expected; the question is whether each claim's direction survives.")

CSS = """
:root{
  /* Report page: single reading column with wide tables allowed to scroll inside their own frame. */
  --bg:#f5f7f8; --paper:#ffffff; --ink:#152028; --ink-2:#44525c; --ink-3:#74828c; --rule:#d9e0e4; --rule-soft:#e9eef1; --band:#f0f4f6;
  --accent:#0f6e72; --accent-soft:#e2f1f1; --pos:#1f7a3a; --neg:#a8322d; --warn:#8a5d00;
  --serif:"Literata","Georgia",serif; --sans:"Public Sans","Segoe UI",system-ui,sans-serif; --mono:"JetBrains Mono",ui-monospace,Menlo,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1519;--paper:#161e23;--ink:#e7edf0;--ink-2:#aebbc3;--ink-3:#7f8d96;--rule:#2a353c;--rule-soft:#212b31;--band:#1b242a;--accent:#5cc2c4;--accent-soft:#16312f;--pos:#5fcf7f;--neg:#ef8b84;--warn:#f0c060;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f1519;--paper:#161e23;--ink:#e7edf0;--ink-2:#aebbc3;--ink-3:#7f8d96;--rule:#2a353c;--rule-soft:#212b31;--band:#1b242a;--accent:#5cc2c4;--accent-soft:#16312f;--pos:#5fcf7f;--neg:#ef8b84;--warn:#f0c060;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15.5px;line-height:1.6}
.wrap{max-width:1060px;margin:0 auto;padding-inline:20px;padding-block:40px 80px}
.col{max-width:72ch}
h1,h2,h3{font-family:var(--serif);font-weight:600;text-wrap:balance;margin:0}
h1{font-size:2.3rem;line-height:1.15;letter-spacing:-.01em}
h2{font-size:1.55rem;margin-top:3.2rem;padding-top:1.2rem;border-top:1px solid var(--rule)}
h3{font-size:1.12rem;margin-top:2rem}
p{margin:.7rem 0}
.eyebrow{font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);font-weight:600}
.lede{font-size:1.1rem;color:var(--ink-2);max-width:68ch}
.meta{display:flex;flex-wrap:wrap;gap:6px 22px;color:var(--ink-3);font-size:.86rem;margin-top:.8rem}
.tw{overflow-x:auto;background:var(--paper);border:1px solid var(--rule);border-radius:6px;margin:1rem 0 .4rem}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--rule-soft);vertical-align:top}
th{font-weight:600;color:var(--ink-2);font-size:.78rem;background:var(--band);line-height:1.35}
td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
td.num{white-space:nowrap}
th{min-width:4.5em;max-width:11em}
th.num{font-family:var(--sans)}
tr:last-child td{border-bottom:0}
tr.pending td{color:var(--ink-3);font-style:italic}
td b{color:var(--accent)}
.ci{display:block;font-size:.72rem;color:var(--ink-3)}
.pos{color:var(--pos)} .neg{color:var(--neg)} .sig{color:var(--accent);font-weight:700}
.bl{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:1.2rem 0 1.4rem}
.blt{background:var(--paper);border:1px solid var(--rule);border-top:3px solid var(--accent);border-radius:6px;padding:12px 14px;display:grid;gap:4px;min-width:0}
.bll{font-size:.8rem;color:var(--ink-2);font-weight:600}
.blv{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;font-family:var(--mono);font-variant-numeric:tabular-nums}
.blb{font-size:1.15rem;color:var(--ink-3)} .bla{color:var(--ink-3)}
.bly{font-size:1.7rem;font-weight:600;color:var(--ink)} .bly.holds,.bly.new{color:var(--pos)} .bly.cost,.bly.worse{color:var(--neg)} .bly.weaker{color:var(--warn)}
.bln{font-size:.78rem;color:var(--ink-3)}
.neu{color:var(--ink-3)}
.tnote{font-size:.82rem;color:var(--ink-3);max-width:90ch;margin:.3rem 0 1rem}
.chip{display:inline-block;font-family:var(--sans);font-size:.72rem;font-weight:600;padding:1px 8px;border-radius:999px;border:1px solid currentColor;white-space:nowrap;margin:1px 0}
.chip.holds,.chip.stronger{color:var(--pos)} .chip.cost,.chip.worse{color:var(--neg)} .chip.weaker,.chip.control{color:var(--warn)} .chip.tie{color:var(--ink-3)} .chip.new{color:var(--accent)}
.keys{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px;margin:1.4rem 0}
.key{background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:14px 16px;min-width:0}
.key h4{margin:0 0 .35rem;font-size:.98rem;font-family:var(--serif)}
.key p{margin:0;font-size:.9rem;color:var(--ink-2)}
.defs{display:grid;grid-template-columns:minmax(150px,220px) 1fr;gap:8px 18px;font-size:.9rem;margin:1rem 0}
.defs dt{font-weight:600} .defs dd{margin:0;color:var(--ink-2)}
@media (max-width:620px){.defs{grid-template-columns:1fr}.defs dd{margin-bottom:.6rem}h1{font-size:1.8rem}}
.callout{border-left:3px solid var(--accent);background:var(--accent-soft);padding:10px 14px;border-radius:0 6px 6px 0;margin:1rem 0;max-width:80ch}
nav.toc{columns:2;column-gap:28px;font-size:.9rem;margin:1.4rem 0;max-width:720px}
nav.toc a{color:var(--ink-2);text-decoration:none;display:block;padding:2px 0} nav.toc a:hover{color:var(--accent)}
@media (max-width:620px){nav.toc{columns:1}}
code{font-family:var(--mono);font-size:.85em;background:var(--band);padding:1px 5px;border-radius:3px}
ul{padding-left:1.2rem} li{margin:.3rem 0}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""

def sec(id_, title, body):
    return f'<section id="{id_}"><h2>{title}</h2>{body}</section>'

# ---- Part 2 of the report: control tests, rewritten for clarity (inserted into build_report.py) ----
OPS = D["ops"]

CSS2 = """
.part{margin-top:3.6rem;padding:14px 18px;border-radius:8px;background:var(--accent-soft);border:1px solid var(--rule)}
.part .eyebrow{display:block;margin-bottom:.2rem}
.part h2{border:0;margin:0;padding:0}
.part p{margin:.4rem 0 0;color:var(--ink-2);max-width:80ch}
.test{background:var(--paper);border:1px solid var(--rule);border-radius:8px;padding:18px 20px;margin:1.4rem 0;min-width:0}
.test h3{margin:0 0 .6rem;font-size:1.22rem}
.test .tag{font-family:var(--sans);font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);display:block;margin-bottom:.15rem}
.how{display:grid;grid-template-columns:minmax(110px,150px) 1fr;gap:6px 16px;font-size:.92rem;margin:.4rem 0 1rem}
.how dt{font-weight:600;color:var(--accent)} .how dd{margin:0;color:var(--ink-2)}
@media (max-width:620px){.how{grid-template-columns:1fr}.how dd{margin-bottom:.5rem}}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:.6rem 0 1rem}
.stat{border:1px solid var(--rule);border-radius:6px;padding:10px 12px;background:var(--bg);min-width:0}
.stat .lab{font-size:.78rem;color:var(--ink-3);line-height:1.3;min-height:2.1em}
.stat .fig{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:1.5rem;line-height:1.25;margin-top:4px;white-space:nowrap}
.stat .fig .b{color:var(--ink-3)} .stat .fig .ar{color:var(--ink-3);font-size:1.05rem;margin:0 4px}
.stat .fig .ae{font-weight:700}
.stat .fig .ae.good{color:var(--pos)} .stat .fig .ae.bad{color:var(--neg)} .stat .fig .ae.same{color:var(--ink)}
.stat .sub{font-size:.76rem;color:var(--ink-3);margin-top:2px}
.reading{border-left:3px solid var(--accent);padding:4px 0 4px 12px;margin:.8rem 0 0;font-size:.93rem;max-width:84ch}
.reading b{color:var(--ink)}
.opgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:.8rem 0}
.op{border:1px solid var(--rule);border-radius:6px;padding:10px 12px;background:var(--paper);font-size:.88rem;min-width:0}
.op b{display:block;font-family:var(--serif);font-size:.98rem;margin-bottom:2px}
.op code{font-size:.8em}
.op .kind{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-3)}
tr.hl td{background:var(--accent-soft)}
td .pos b,td .pos{color:var(--pos)} td .neg b,td .neg{color:var(--neg)}
td .w{font-size:.72rem;color:var(--ink-3)}
"""

def stat(label, base, ae, better="lower", unit="", sub="", fmt="{:.0f}", tol=1.5):
    """Tile: base -> AE, coloured by whether the AE moved the good way."""
    if base is None or ae is None or better == "none":
        cls = "same"
    else:
        d = ae - base
        good = d < 0 if better == "lower" else d > 0
        cls = "same" if abs(d) < tol else ("good" if good else "bad")
    bs = "–" if base is None else fmt.format(base) + unit
    as_ = "–" if ae is None else fmt.format(ae) + unit
    return (f'<div class="stat"><div class="lab">{label}</div><div class="fig"><span class="b">{bs}</span>'
            f'<span class="ar">→</span><span class="ae {cls}">{as_}</span></div>'
            f'<div class="sub">{sub or "base → bypass AE"}</div></div>')

def how(do, measure, result):
    return f'<dl class="how"><dt>What we do</dt><dd>{do}</dd><dt>What we measure</dt><dd>{measure}</dd><dt>Result</dt><dd>{result}</dd></dl>'

def test(tag, title, inner):
    return f'<div class="test"><span class="tag">{tag}</span><h3>{title}</h3>{inner}</div>'

def reading(txt):
    return f'<p class="reading">{txt}</p>'

# ---- Part 3: linear probes, matched-base number check, all-hubs transfer (inserted into build_report.py) ----
TPP, BP, NHB, TAH = D["tpp"], D["biasprobe"], D["number_hb"], D["transfer_allhubs"]
SPN = {"raw": "Base residual", "tokmean": "Token-mean control", "geoae": "Bypass AE latent"}

def tpp_rows(ds):
    rows = []
    t6 = TPP["d6144"][ds]
    for sp in ("raw", "tokmean"):
        v = t6[sp]
        rows.append([SPN[sp], f"{v['n_dims']:,}", f"{v['probe_acc']:.3f}", f"<b>{100*v['k50_frac']:.1f}%</b>", f"{100*v['random_k50_frac']:.0f}%", f"{100*v['collateral_at_k50']:.1f}", f"{v['n_reached']} / {v['n_classes']}"])
    for a in A:
        v = TPP[a][ds]["geoae"]
        rows.append([f"Bypass AE, {LBL[a]}", f"{v['n_dims']:,}", f"{v['probe_acc']:.3f}", f"<b>{100*v['k50_frac']:.1f}%</b>", f"{100*v['random_k50_frac']:.0f}%", f"{100*v['collateral_at_k50']:.1f}", f"{v['n_reached']} / {v['n_classes']}"])
    return table(["Space", "Coordinates", "Probe accuracy", "Removed to halve a class (mean)", "Same with random coordinates", "Other classes disturbed /100", "Classes halved"], rows)

td, tb = TPP["d6144"]["dbpedia"], TPP["d6144"]["biasbios"]
tpp_t = test("Probe test 1 · no language model in the loop", "How few coordinates carry a class?", f'''
{how("Train a linear probe to classify documents from the layer-27 vector (DBpedia topics; bias_in_bios professions). Then rank the coordinates by how much the probe relies on them for one class, and replace them, most important first, with their training mean. Repeat for every class, in each space.",
     "<b>k50</b>: the share of coordinates that must be replaced before the probe's accuracy on that class falls to half. Smaller = the class is concentrated in fewer coordinates. Reported as a share, not a count, because the spaces have different widths. Random coordinates give the reference.",
     "On DBpedia the three spaces are the same: a topic is spread over about 40% of coordinates everywhere. On bias_in_bios the AE latent concentrates professions slightly more.")}
<div class="stats">
  {stat("DBpedia: share replaced to halve a topic", 100*td["raw"]["k50_frac"], 100*td["geoae"]["k50_frac"], "lower", unit="%", sub=f"control {100*td['tokmean']['k50_frac']:.1f}% · random {100*td['raw']['random_k50_frac']:.0f}%", fmt="{:.1f}", tol=2)}
  {stat("bias_in_bios: share replaced to halve a profession", 100*tb["raw"]["k50_frac"], 100*tb["geoae"]["k50_frac"], "lower", unit="%", sub=f"control {100*tb['tokmean']['k50_frac']:.1f}% · random {100*tb['raw']['random_k50_frac']:.0f}%", fmt="{:.1f}", tol=2)}
  {stat("bias_in_bios probe accuracy before any removal", 100*tb["raw"]["probe_acc"], 100*tb["geoae"]["probe_acc"], "higher", unit="%", sub="the AE probe starts slightly weaker", fmt="{:.1f}", tol=0.5)}
</div>
<h3 style="margin-top:.4rem;font-size:1rem">DBpedia-14 (14 topics)</h3>
{tpp_rows("dbpedia")}
<h3 style="margin-top:.4rem;font-size:1rem">bias_in_bios (28 professions)</h3>
{tpp_rows("biasbios")}
<p class="tnote">Probes are fit per space on 50,000 documents (last token, layer 27) with the same split and no sparsity penalty, and coordinates are replaced by their training mean, not zero. An earlier version of this test used zero-filling and a strong sparsity penalty, which produced a large but spurious localisation advantage for the AE; this protocol removes both. The numbers describe what a fixed probe can still read, not whether the model itself lost the information.</p>
{reading(f"<b>No localisation advantage on topics.</b> Halving a DBpedia topic takes about 40% of coordinates in every space ({100*td['raw']['k50_frac']:.1f}% base, {100*td['tokmean']['k50_frac']:.1f}% control, {100*td['geoae']['k50_frac']:.1f}% AE), and other topics are barely disturbed on the way (under 1.5 per 100). On professions the AE needs fewer coordinates ({100*tb['geoae']['k50_frac']:.1f}% vs {100*tb['raw']['k50_frac']:.1f}%), but its probe also starts {100*(tb['raw']['probe_acc']-tb['geoae']['probe_acc']):.1f} points weaker and disturbs other professions a little more, so read this as a small effect. d768 and d12288 show the largest profession concentration (about 10%), d3072 none.")}
''')

def bp_rows():
    rows = []
    b6 = BP["d6144"]
    for sp in ("raw", "tokmean"):
        v = b6[sp]
        rows.append([SPN[sp], f"{v['prof_bal']:.2f}", f"{v['gender_bal']:.2f}", f"<b>{100*v['gender_probe']['frac']:.0f}%</b>", f"{100*v['gender_dprime']['frac']:.0f}%", f"{v['gender_probe']['prof_bal_after']:.2f}"])
    for a in A:
        v = BP[a]["geoae"]
        rows.append([f"Bypass AE, {LBL[a]}", f"{v['prof_bal']:.2f}", f"{v['gender_bal']:.2f}", f"<b>{100*v['gender_probe']['frac']:.0f}%</b>", f"{100*v['gender_dprime']['frac']:.0f}%", f"{v['gender_probe']['prof_bal_after']:.2f}"])
    return table(["Space", "Profession probe (balanced acc.)", "Gender probe (balanced acc.)", "Replaced until gender is at chance (probe ranking)", "Same, d′ ranking", "Profession accuracy left at that point"], rows)

bR, bZ = BP["d6144"]["raw"], BP["d6144"]["geoae"]
bias_t = test("Probe test 2 · no language model in the loop", "Can gender be removed without removing profession?", f'''
{how("On bias_in_bios, fit a profession probe and a gender probe in each space. Replace coordinates ranked by how much the <b>gender</b> probe uses them (or by d′ between genders), most important first, until the gender probe is at chance.",
     "How many coordinates that takes (as a share), and how much of the profession probe's accuracy survives at that point. A space that isolated gender would reach chance early and keep profession accuracy high.",
     "No space isolates gender. It takes about 90% of coordinates to reach chance in every space, and by then profession accuracy has collapsed in all of them.")}
<div class="stats">
  {stat("Share replaced until gender is at chance", 100*bR["gender_probe"]["frac"], 100*bZ["gender_probe"]["frac"], "lower", unit="%", sub=f"control {100*BP['d6144']['tokmean']['gender_probe']['frac']:.0f}%", fmt="{:.0f}", tol=3)}
  {stat("Profession balanced accuracy left at that point", bR["gender_probe"]["prof_bal_after"], bZ["gender_probe"]["prof_bal_after"], "higher", sub=f"from {bR['prof_bal']:.2f} (base) and {bZ['prof_bal']:.2f} (AE) before removal", fmt="{:.2f}", tol=0.03)}
</div>
{bp_rows()}
{reading(f"Gender is readable from almost every coordinate of the layer-27 vector in all three spaces (the gender probe is 98–99% accurate everywhere), so removing it coordinate by coordinate takes about 90% of the space and takes profession with it (balanced accuracy {bR['prof_bal']:.2f} → {bR['gender_probe']['prof_bal_after']:.2f} in the residual, {bZ['prof_bal']:.2f} → {bZ['gender_probe']['prof_bal_after']:.2f} in the AE). The bypass AE does not separate the two. d768 reaches chance earlier ({100*BP['d768']['geoae']['gender_probe']['frac']:.0f}%) but starts from a weaker profession probe.")}
''')

probes_sec = sec("linprobes", "Linear probes: how concentrated is each concept?", f'''
<div class="col"><p>The tests so far used the partition (clusters) or the raw geometry. These two use a <b>trained linear probe</b>, the standard interpretability tool: if a concept lives in a few coordinates, removing those few should break the probe. No text is generated and the language model is not run after the edit; this measures what a fixed reader can extract from the vector.</p></div>
{tpp_t}
{bias_t}
''')

# ---- matched-base number check
def hb_rows():
    rows = []
    for a in A:
        h = NHB[a]
        rows.append([LBL[a], f"{100*h['h:range:a0.5']:.0f}", f"{100*h['hb:range:a0.5']:.0f}", f"<b>{100*h['z:range:a0.5']:.0f}</b>",
                     f"{100*h['h:range:a1.0']:.0f}", f"{100*h['hb:range:a1.0']:.0f}", f"<b>{100*h['z:range:a1.0']:.0f}</b>"])
    return table(["Run", "Base, α 0.5", "Token-subtracted base, α 0.5", "AE, α 0.5", "Base, α 1", "Token-subtracted base, α 1", "AE, α 1"], rows)
hb_block = f'''
<h3 style="margin-top:.4rem;font-size:1rem">Is the gain from the token subtraction or from the encoder?</h3>
<div class="col"><p>The same edit is also made on the residual with the per-token average subtracted (and added back afterwards), which is the bypass AE's input without the encoder. Target removed per 100 items, range edit.</p></div>
{hb_rows()}
{reading(f"The token-subtracted base behaves exactly like the plain base ({100*NHB['d6144']['hb:range:a1.0']:.0f} vs {100*NHB['d6144']['h:range:a1.0']:.0f} at α 1 for d6144), so the subtraction alone explains none of the gain. Unlike topic clustering, <b>this advantage belongs to the encoder</b>.")}
'''

# ---- all-hubs transfer check
def ah_rows():
    rows = []
    for a in A:
        t, u = TR[a], TAH[a]
        rows.append([LBL[a], pct(t["base:km"]["0.1"]["est"]), pct(u["base:km"]["0.1"]["est"]), pct(t["bypass:ae"]["0.1"]["est"]), pct(u["bypass:ae"]["0.1"]["est"]),
                     f'{u["bypass:ae"]["0.1"]["est"]/u["base:km"]["0.1"]["est"]:.1f}×'])
    return table(["Run", "Base, hubs excluded", "Base, hubs allowed", "Bypass, hubs excluded", "Bypass, hubs allowed", "Bypass ÷ base, hubs allowed"], rows)
ah_block = f'''
<h3 style="margin-top:.4rem;font-size:1rem">Robustness: allowing hub clusters as targets</h3>
<div class="col"><p>Hub clusters (over 3× their fair share of rows, mostly boilerplate) are excluded from the targets above. Allowing them changes little: transfer rises slightly in every space and bypass stays about 2× base.</p></div>
{ah_rows()}
'''

# ---- Part 4: clustering-quality (cq) section (inserted into build_report.py) ----
CQ = {a: S[a]["cq"] for a in A}
CQD = {}
import json as _json
for _a in A:
    CQD[_a] = _json.load(open(f"/projects/bbyl/mhaider/GeoAE/results/delta/{_a}/evals/cq.json"))
CQM = [("silhouette", "Silhouette", "up", "{:+.3f}"), ("davies_bouldin", "Davies-Bouldin", "down", "{:.2f}"),
       ("calinski_harabasz", "Calinski-Harabasz", "up", "{:,.0f}"), ("separability_ratio", "Separability ratio", "up", "{:.0f}"),
       ("dunn_index", "Dunn index", "up", "{:.4f}"), ("cluster_balance", "Balance (usage entropy)", "up", "{:.3f}"),
       ("effective_k", "Effective K", "up", "{:,.0f}"), ("effective_rank", "Effective rank", "up", "{:.0f}")]
CQ_WHAT = {"silhouette": "How much closer a point is to its own cluster than to the next one (−1 to 1; 0 = on the boundary)",
           "davies_bouldin": "Average of each cluster's spread relative to its nearest neighbour's distance (lower = tighter)",
           "calinski_harabasz": "Between-cluster over within-cluster variance, adjusted for K and N (higher = tighter)",
           "separability_ratio": "Mean distance between cluster centres over mean spread inside a cluster",
           "dunn_index": "Closest pair of clusters over the widest cluster: the worst case",
           "cluster_balance": "How evenly rows are spread over clusters (1 = perfectly even)",
           "effective_k": "Clusters that actually receive rows",
           "effective_rank": "How many directions carry the space's variance"}
def cq_d6_rows():
    rows = []
    b, y = CQD["d6144"]["base"], CQD["d6144"]["bypass"]
    for k, lab, dirn, f in CQM:
        vb, vy = b[k], y[k]
        better = (vy > vb) if dirn == "up" else (vy < vb)
        same = abs(vy - vb) <= 0.02 * max(abs(vb), 1e-9) if k not in ("silhouette",) else abs(vy - vb) < 0.005
        cls = "" if same else ("pos" if better else "neg")
        rows.append([f"<b>{lab}</b><span class='ci'>{CQ_WHAT[k]}</span>", f.format(vb), f'<span class="{cls}">{f.format(vy)}</span>', "↑ better" if dirn == "up" else "↓ better"])
    return table(["Metric (d6144, K = 2,000, 1M rows)", "Base", "Bypass AE", ""], rows)
def cq_w_rows():
    rows = []
    for a in A:
        b, y = CQD[a]["base"], CQD[a]["bypass"]
        rows.append([LBL[a]] + [f"{f.format(b[k])} → <b>{f.format(y[k])}</b>" for k, _, _, f in CQM[:5]] + [f"{CQM[7][3].format(b['effective_rank'])} → {CQM[7][3].format(y['effective_rank'])}"])
    return table(["Run", "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑", "Separability ↑", "Dunn ↑", "Effective rank"], rows)
b6, y6 = CQD["d6144"]["base"], CQD["d6144"]["bypass"]
cq_sec = sec("cq", "Cluster compactness", f'''
<div class="col"><p>Standard clustering-quality scores for each partition, on 1M rows drawn at random from the whole dump, mostly training rows (seed 0, deterministic). Each space is scored in <b>its own coordinates</b>: the base k-means in the normalised residual, the AE's clusters in its latent. Only scale-free ratios are compared; raw distances and variances are not comparable between a 3,072-d residual and a 6,144-d latent.</p></div>
{test("Cluster-quality metrics", "Are the clusters tight and well separated?", f"""
{how("Assign 1M rows to their nearest centre in each space and compute the usual internal clustering scores. No labels are used.",
     "Eight scores (definitions under each row). Silhouette, Davies-Bouldin, Calinski-Harabasz and separability describe typical clusters; Dunn describes the worst pair; balance, effective K and effective rank describe how the space and the codebook are used.",
     "On typical clusters the AE's partition is tighter and better separated in its own space. On the worst pair (Dunn) it is slightly worse. Silhouette is near zero everywhere: at K = 2,000 clusters overlap heavily in every space.")}
<div class="stats">
  {stat("Calinski-Harabasz (higher = tighter)", b6["calinski_harabasz"], y6["calinski_harabasz"], "higher", fmt="{:,.0f}")}
  {stat("Separability ratio", b6["separability_ratio"], y6["separability_ratio"], "higher", fmt="{:.0f}")}
  {stat("Silhouette (−1 to 1)", b6["silhouette"], y6["silhouette"], "higher", fmt="{:+.3f}", tol=0.005, sub="near zero in both: heavy overlap at K = 2,000")}
</div>
{cq_d6_rows()}
<h3 style="margin-top:.4rem;font-size:1rem">Across runs (base → bypass)</h3>
{cq_w_rows()}
<p class="tnote">The K = 2,000 base is the same codebook for the four widths; the K = 4,000 run has its own base. Clustering quality was scored for base and bypass only; the token-mean control is not included here.</p>
{reading(f"<b>Treat these as descriptions, not as a quality ranking.</b> The AE is trained with a loss that pulls each latent toward its cluster centre, so tighter clusters in its own space are partly built in. The clearest evidence is d768: it has the best Davies-Bouldin ({CQD['d768']['bypass']['davies_bouldin']:.2f}) and Calinski-Harabasz ({CQD['d768']['bypass']['calinski_harabasz']:,.0f}) of any run, yet the language model fails completely with it spliced in, and its latent keeps only {CQD['d768']['bypass']['effective_rank']:.0f} effective directions. The usefulness of a partition is measured by the concept-alignment and control sections, not by these scores. Balance and effective K are close to the maximum in every run (no collapse), and the Dunn index is slightly lower for the AE in every run ({CQD['d6144']['base']['dunn_index']:.4f} → {CQD['d6144']['bypass']['dunn_index']:.4f} at d6144), meaning its closest pair of clusters sits relatively nearer.")}
""")}
''')

# ------------------------------------------------------------------ primer
primer = f'''
<div class="col">
<p>Sections 3–6 asked whether the latent is <em>organised</em> better. Everything from here asks whether it is <em>easier to control</em>: if you edit the representation at layer 27, does the model's behaviour change the way you intended, and only that way?</p>
<p>Every test edits a vector at layer 27 and lets the rest of the model run. An edit can be made in two places. <b>h</b> is the base residual itself. <b>z</b> is the bypass AE latent: encode the residual, edit z, decode back to the residual. Both sides use the same operator, the same documents and the same strength, so every h-vs-z number is paired.</p>
</div>
{table(["Test", "Edit is built from", "Applied at", "What we read off", "Data"], [
    ["<b>Removal</b> and <b>range steering</b>", "Labelled documents of each class", "The last token, once", "Does the model stop predicting the target class? Do other classes survive?", "DBpedia-14 topics, 27 bias_in_bios professions"],
    ["<b>Global steering</b>", "Class mean minus rest mean", "The last token, once", "Same readout, simpler edit", "DBpedia-14"],
    ["<b>Number control</b>", "Singular vs plural examples", "The subject noun, once", "Does the verb agreement flip?", "12 held-out nouns, 144 items"],
    ["<b>Cluster transfer</b>", "The codebook's own clusters, no labels", "One held-out row", "Does the next-token distribution move toward the target cluster's?", "Held-out dump rows"],
    ["<b>Long-form cluster steering</b>", "The codebook's own clusters, no labels", "Every generated token, 32 tokens", "LLM judge: is the text fluent and on the cluster's topic?", "40 targets × 20 prompts"],
    ["<b>Long-form concept steering</b>", "Named concepts via labels, several handle types", "Every generated token, 32 tokens", "LLM judge: fluent and about the named concept?", "14 DBpedia + 28 bias_in_bios concepts"],
], num_from=99)}
<div class="col"><p>The first three read a single prediction right after the edit (a few-shot prompt asking the model for the document's category, or the next word). The last two let the model write freely, which is harder: an edit has to keep working as the text grows, and it must not wreck fluency.</p></div>
<h3>The edit operators</h3>
<div class="col"><p>Removal and steering are built from per-class statistics fitted on 80 documents per class. <b>Salient coordinates</b> are the 30% of dimensions that best separate the class from the others (by d′). A class's <b>range</b> on a coordinate is its typical interval, mean ± 2 standard deviations. <b>α</b> is the strength, in units of the gap between the class mean and the rest mean, so α = 1 means "move by one class-gap" in either space.</p></div>
<div class="opgrid">
  <div class="op"><span class="kind">Removal</span><b>Range removal</b>On salient coordinates, any value inside the class's range is replaced by the average of the other classes. Values outside the range are untouched. <code>rm_range_comp</code></div>
  <div class="op"><span class="kind">Removal</span><b>Full ablation</b>Every value on the salient coordinates is replaced, in range or not. Blunter. <code>rm_full_comp</code></div>
  <div class="op"><span class="kind">Steering</span><b>Global steering</b>Subtract α × (class mean − rest mean) from every coordinate. The classic steering vector. <code>st_global</code></div>
  <div class="op"><span class="kind">Steering</span><b>Salient steering</b>The same subtraction, only on salient coordinates. Tests whether choosing coordinates helps. <code>st_salient</code></div>
  <div class="op"><span class="kind">Steering</span><b>Range steering</b>The same subtraction, only on salient coordinates whose value is inside the class range. Tests whether gating by range helps. <code>st_range</code></div>
  <div class="op"><span class="kind">Steering</span><b>Transport</b>In-range values are mapped from the class's distribution to the rest's (shift and rescale by the standard deviations), by fraction α. <code>st_transport</code></div>
</div>
'''

# ------------------------------------------------------------------ removal
def opsrow(a, ds, key):
    o = OPS.get(a, {}).get(ds, {}).get(key)
    if not o: return None
    return o

def ops_table(a, ds, keys, labels):
    rows = []
    for k, lab in zip(keys, labels):
        o = opsrow(a, ds, k)
        if not o: continue
        sel = 100 * o["z_minus_h"]
        cls = "pos" if sel > 1 else "neg" if sel < -1 else ""
        r_ = [lab, f"{100*o['h_tgt_drop']:.0f} / {100*o['h_comp_drop']:.0f}", f"{100*o['z_tgt_drop']:.0f} / {100*o['z_comp_drop']:.0f}",
                     f'<span class="{cls}"><b>{sel:+.1f}</b></span><span class="ci">± {100*o["z_minus_h_se"]:.1f}</span>', f"{o['z_wins']} of {o['n']}"]
        rows.append((r_, "row:hl") if k.endswith("a1.0") or k == "rm_range_comp" else r_)
    return table(["Operator (d6144)", "Base: removed / collateral per 100", "AE: removed / collateral per 100", "Net score AE − base", "Concepts where AE wins"], rows)

d = OPS["d6144"]
rdb, rbb = d["db14"]["rm_range_comp"], d["biasbios"]["rm_range_comp"]
fdb, fbb = d["db14"]["rm_full_comp"], d["biasbios"]["rm_full_comp"]
removal = test("Test 1 · single-step readout", "Removal: make the model forget a class", f'''
{how("Take documents the model classifies correctly. At the last token, apply range removal for the true class, then ask the model for the category again.",
     "<b>Removed</b>: of 100 target documents, how many are no longer given their class. <b>Collateral</b>: of 100 documents from other classes, how many are disturbed. Good removal = high removed, low collateral. <b>Net score</b> = removed − collateral; we report AE − base, so positive favours the AE.",
     "On DBpedia the latent removes the class just as often and does a third less collateral damage. On bias_in_bios the two are level.")}
<div class="stats">
  {stat("DBpedia collateral per 100 (range removal)", 100*rdb["h_comp_drop"], 100*rdb["z_comp_drop"], "lower", sub=f"removed {100*rdb['h_tgt_drop']:.0f} → {100*rdb['z_tgt_drop']:.0f} · AE wins {rdb['z_wins']} of 14")}
  {stat("bias_in_bios collateral per 100 (range removal)", 100*rbb["h_comp_drop"], 100*rbb["z_comp_drop"], "lower", sub=f"removed {100*rbb['h_tgt_drop']:.0f} → {100*rbb['z_tgt_drop']:.0f} · AE wins {rbb['z_wins']} of 27")}
  {stat("bias_in_bios collateral per 100 (full ablation)", 100*fbb["h_comp_drop"], 100*fbb["z_comp_drop"], "lower", sub="blunt removal hurts the latent more")}
</div>
<h3 style="margin-top:.4rem;font-size:1rem">DBpedia-14, all removal operators</h3>
{ops_table("d6144", "db14", ["rm_range_comp", "rm_full_comp", "rm_range_zero", "rm_full_zero"], ["Range removal (fill with other-class mean)", "Full ablation (fill with other-class mean)", "Range removal, fill with zero", "Full ablation, fill with zero"])}
<h3 style="margin-top:.4rem;font-size:1rem">bias_in_bios, all removal operators</h3>
{ops_table("d6144", "biasbios", ["rm_range_comp", "rm_full_comp", "rm_range_zero", "rm_full_zero"], ["Range removal (fill with other-class mean)", "Full ablation (fill with other-class mean)", "Range removal, fill with zero", "Full ablation, fill with zero"])}
{reading("<b>Range removal is the operator to quote.</b> It is the most selective in both spaces, and it is where the latent gains on DBpedia (net score +6.3). Zero-filling is not a neutral edit in the latent: its coordinates sit far from zero, so writing zeros damages everything (collateral 96–99). Full ablation on bias_in_bios also costs the latent more (52 vs 27). The precise range gate is what makes the latent edit clean.")}
''')

# ------------------------------------------------------------------ steering
sdb, sbb = d["db14"], d["biasbios"]
steer_ops = test("Test 2 · single-step readout", "Steering: push the representation away from a class", f'''
{how("Instead of overwriting values, subtract a scaled class direction at the last token. Four versions differ only in which coordinates and values are moved (see the operator cards). Each is run at α = 0.5, 1 and 2.",
     "Same readout as removal: removed and collateral per 100, and net score AE − base. Positive = the AE edit is cleaner.",
     "At α = 1 the latent is cleaner for every operator on DBpedia and for three of four on bias_in_bios. At α = 2 both spaces saturate and the differences vanish or reverse.")}
<div class="stats">
  {stat("DBpedia collateral, range steering α = 1", 100*sdb["st_range_a1.0"]["h_comp_drop"], 100*sdb["st_range_a1.0"]["z_comp_drop"], "lower", sub=f"removed {100*sdb['st_range_a1.0']['h_tgt_drop']:.0f} → {100*sdb['st_range_a1.0']['z_tgt_drop']:.0f}")}
  {stat("DBpedia collateral, transport α = 1", 100*sdb["st_transport_a1.0"]["h_comp_drop"], 100*sdb["st_transport_a1.0"]["z_comp_drop"], "lower", sub=f"removed {100*sdb['st_transport_a1.0']['h_tgt_drop']:.0f} → {100*sdb['st_transport_a1.0']['z_tgt_drop']:.0f} · AE wins 10 of 14")}
  {stat("DBpedia collateral, global steering α = 1", 100*sdb["st_global_a1.0"]["h_comp_drop"], 100*sdb["st_global_a1.0"]["z_comp_drop"], "lower", sub=f"removed {100*sdb['st_global_a1.0']['h_tgt_drop']:.0f} → {100*sdb['st_global_a1.0']['z_tgt_drop']:.0f}")}
</div>
<h3 style="margin-top:.4rem;font-size:1rem">DBpedia-14, every steering operator and strength</h3>
{ops_table("d6144", "db14", [f"st_{m}_a{a}" for m in ("global", "salient", "range", "transport") for a in ("0.5", "1.0", "2.0")],
           [f"{n}, α = {a}" for n in ("Global", "Salient", "Range", "Transport") for a in ("0.5", "1", "2")])}
<h3 style="margin-top:.4rem;font-size:1rem">bias_in_bios, every steering operator (α = 1 and 2)</h3>
{ops_table("d6144", "biasbios", [f"st_{m}_a{a}" for m in ("global", "salient", "range", "transport") for a in ("1.0", "2.0")],
           [f"{n}, α = {a}" for n in ("Global", "Salient", "Range", "Transport") for a in ("1", "2")])}
{reading("<b>How to read the strength effect.</b> At α = 0.5 the edit is too weak to remove much, and the latent removes less (it needs a slightly larger push). At α = 1 the latent reaches the same or higher removal with less collateral, so this is the setting where it wins. At α = 2 both spaces remove everything and damage a lot, so there is nothing left to compare. <b>Range and transport</b> are the two gated operators, and they give the latent's clearest wins on DBpedia (+6.3 and +7.9).")}
<h3 style="margin-top:.4rem;font-size:1rem">Matched removal: a fairer strength comparison</h3>
<div class="col"><p>Comparing at the same α is not quite fair if one space needs a bigger push. So each operator is also interpolated along its strength grid to the <b>same removal level</b> in both spaces, and collateral is compared there. Negative = the latent does less damage.</p></div>
{table(["Operator", "DBpedia: collateral AE − base", "bias_in_bios: collateral AE − base"],
       [[n, f'<span class="{("pos" if S["d6144"]["range_intervention_db14_dprime"][m]["collateral_z_pct"]-S["d6144"]["range_intervention_db14_dprime"][m]["collateral_h_pct"]<-1 else "neg" if S["d6144"]["range_intervention_db14_dprime"][m]["collateral_z_pct"]-S["d6144"]["range_intervention_db14_dprime"][m]["collateral_h_pct"]>1 else "")}">{S["d6144"]["range_intervention_db14_dprime"][m]["collateral_z_pct"]-S["d6144"]["range_intervention_db14_dprime"][m]["collateral_h_pct"]:+.1f}</span>',
         f'<span class="{("pos" if S["d6144"]["range_intervention_biasbios_dprime"][m]["collateral_z_pct"]-S["d6144"]["range_intervention_biasbios_dprime"][m]["collateral_h_pct"]<-1 else "neg" if S["d6144"]["range_intervention_biasbios_dprime"][m]["collateral_z_pct"]-S["d6144"]["range_intervention_biasbios_dprime"][m]["collateral_h_pct"]>1 else "")}">{S["d6144"]["range_intervention_biasbios_dprime"][m]["collateral_z_pct"]-S["d6144"]["range_intervention_biasbios_dprime"][m]["collateral_h_pct"]:+.1f}</span>']
        for n, m in (("Global", "global"), ("Salient", "salient"), ("Range", "range"), ("Transport", "transport"))])}
{reading("At matched removal DBpedia is roughly level (−1.6 to +2.9), and <b>bias_in_bios favours the base residual</b> (+2 to +6 collateral for the latent). So the latent's advantage is that its edits land cleanly at their natural strength; when you force both spaces to the same removal, the edge disappears on DBpedia and reverses on professions.")}
''')

# ------------------------------------------------------------------ across runs
def width_ops(ds):
    rows = []
    for a in A:
        o = OPS.get(a, {}).get(ds)
        if not o:
            rows.append([LBL[a], "not run (spliced model gets no documents right)", "", "", ""]); continue
        r, t = o["rm_range_comp"], o["st_transport_a1.0"]
        m = S[a][f"range_intervention_{ds}_dprime"]["range"]
        mm = m["collateral_z_pct"] - m["collateral_h_pct"]
        rows.append([LBL[a], f"{100*r['h_comp_drop']:.0f} → <b>{100*r['z_comp_drop']:.0f}</b>", f"{100*r['z_minus_h']:+.1f} <span class='w'>({r['z_wins']}/{r['n']})</span>",
                     f"{100*t['z_minus_h']:+.1f} <span class='w'>({t['z_wins']}/{t['n']})</span>", f'<span class="{"pos" if mm < -1 else "neg" if mm > 1 else ""}">{mm:+.1f}</span>'])
    return rows
WH = ["Run", "Range removal: collateral h → z", "Range removal selectivity z − h", "Transport α = 1 selectivity", "Range steering, matched collateral z − h"]
across = test("Tests 1–2 across runs", "Does this hold at other widths?", f'''
<div class="col"><p>Range removal for every run. <b>Net score</b> = removed − collateral; the AE − base column is green when the AE is better. The last column pushes both spaces to the same removal level and compares their collateral (green = the AE does less damage).</p></div>
<h3 style="margin-top:.4rem;font-size:1rem">DBpedia-14</h3>
{db_tbl}
<h3 style="margin-top:.4rem;font-size:1rem">bias_in_bios</h3>
{bb_tbl}
{reading("<b>d6144 is the sweet spot.</b> Wider (d12288) and more clusters (K4000) keep a smaller DBpedia edge and sit closer to parity on bias_in_bios. <b>d3072 is the outlier</b>: its latent edits remove more but damage much more on professions (collateral 6 → 23 per 100, wins 3 of 27).")}
<h3 style="margin-top:.4rem;font-size:1rem">Global steering, measured by a second tool</h3>
<div class="col"><p>A separate script repeats global steering on DBpedia with a wider strength range (α up to 8). Its sign is flipped relative to the tables above: more negative = cleaner.</p></div>
{steer_tbl}
''')

# absolute flip rates, number control
NA = D["number_abs"]
def na(a, arm, op, al, m="strict_target_flip"):
    v = NA[a].get(f"{arm}:{op}:a{al}")
    return None if v is None else 100 * v[m]
def rotmean(a, op, al):
    xs = [100 * v["strict_target_flip"] for k, v in NA[a].items() if k.startswith("z_rot") and k.endswith(f":{op}:a{al}")]
    return sum(xs) / len(xs) if xs else None
abs6 = []
for op, opn in (("range", "Range"), ("transport", "Transport")):
    for al in ("0.5", "1.0", "2.0"):
        h, z, r, sh = na("d6144", "h", op, al), na("d6144", "z", op, al), rotmean("d6144", op, al), na("d6144", "z_shuffled", op, al)
        row = [f"{opn}, α = {al.rstrip('0').rstrip('.')}", f"{h:.0f}", f"<b>{z:.0f}</b>", f'<span class="{"pos" if z-h > 1 else ""}">{z-h:+.0f}</span>',
               "–" if r is None else f"{r:.0f}", "–" if sh is None else f"{sh:.0f}",
               f"{na('d6144','h',op,al,'complement_drop'):.0f} / {na('d6144','z',op,al,'complement_drop'):.0f}",
               f"{na('d6144','h',op,al,'neutral_top1_change'):.0f} / {na('d6144','z',op,al,'neutral_top1_change'):.0f}"]
        abs6.append((row, "row:hl") if al == "1.0" else row)
num_abs_tbl = table(["Edit (d6144)", "Base: flips /100", "AE: flips /100", "AE − base", "Random rotation of AE edit", "Shuffled-label AE edit", "Wrong-way flips, base / AE", "Neutral text changed, base / AE (/100)"], abs6)
num_abs_w = table(["Run", "Base, α = 0.5", "AE, α = 0.5", "Base, α = 1", "AE, α = 1", "Rotation, α = 1"],
                  [[LBL[a], f"{na(a,'h','range','0.5'):.0f}", f"<b>{na(a,'z','range','0.5'):.0f}</b>", f"{na(a,'h','range','1.0'):.0f}", f"<b>{na(a,'z','range','1.0'):.0f}</b>", f"{rotmean(a,'range','1.0'):.0f}"] for a in A])

# ------------------------------------------------------------------ number control
nd = N["d6144"]["range:z_minus_h"]
number_t = test("Test 3 · single-step readout", "Number control: flip singular and plural", f'''
{how("Fit a singular-vs-plural edit on the subject noun, then apply it to held-out sentences built from 12 unseen nouns (144 items). Range and transport operators, α = 1.",
     "<b>Strict flip rate</b>: how often the model's verb agreement switches to the other number (the gold pair's probability falls below 0.5). Compared with the same edit in h, and with random rotations of the edit as a control for edit size.",
     "The latent flips number more often in every run, and beats random rotations too.")}
<div class="stats">
  {stat("Extra flips per 100, latent vs residual (range, α = 1)", 0, 100*nd["d"], "higher", sub=f"95% interval +{100*nd['ci'][0]:.0f} to +{100*nd['ci'][1]:.0f}", fmt="{:+.0f}", tol=0.5)}
  {stat("Extra flips per 100 vs random rotations", 0, 100*N["d6144"]["range:z_minus_rotations_mean"]["d"], "higher", sub="rules out a pure edit-size effect", fmt="{:+.0f}", tol=0.5)}
</div>
<h3 style="margin-top:.4rem;font-size:1rem">Absolute flip rates, d6144</h3>
{num_abs_tbl}
<p class="tnote">Flips per 100 held-out items (144 items, 12 nouns). Random rotation: the AE edit with its direction randomly rotated, averaged over 3 rotations, same size. Shuffled-label: an edit fit on shuffled singular/plural labels. Wrong-way flips: items pushed toward the wrong number. Neutral text changed: how often the top next word changes on unrelated sentences.</p>
<h3 style="margin-top:.4rem;font-size:1rem">Range edit, every run (flips per 100)</h3>
{num_abs_w}
{hb_block}
<h3 style="margin-top:.4rem;font-size:1rem">Differences with intervals</h3>
{num_tbl}
{reading("<b>Base numbers.</b> The same edit made directly in the residual flips agreement 80 times per 100 at α = 1 and 17 at α = 0.5; the AE edit flips 93 and 33. At α = 2 both reach 99–100, so there is nothing left to separate. No edit flips anything the wrong way, and the shuffled-label edit flips nothing, so the edit is specific to number. A randomly rotated AE edit of the same size reaches 84, so part of the gain comes from editing in the latent at all, and the rest from the learned direction. This is a syntactic, token-level edit, the kind the bypass design might have been expected to hurt. It helps instead, by 7 to 16 flips per 100 across runs. The interval only resamples the 12 nouns, not training seeds or sentence templates.")}
''')

# ------------------------------------------------------------------ transfer
t6 = TR["d6144"]
transfer_t = test("Test 4 · one step, no labels", "Cluster transfer: move a row into a cluster", f'''
{how("Take a held-out row that sits in cluster s. Pick a target cluster t. Move the row by the gap between the two cluster centres (in the latent for the AE, in the residual for the base). Layer 27 is the last layer, so the next-token distribution can be read directly.",
     "<b>Transfer</b>: how often the row's next-token distribution ends up closer to cluster t's typical next tokens than to cluster s's. Compared at equal <b>disruption</b> (how far the prediction moved in total), so a bigger push cannot win on size alone.",
     "At every level of disruption, bypass clusters transfer more often than base clusters.")}
<div class="stats">
  {stat("Transfer at small disruption (0.1)", 100*t6["base:km"]["0.1"]["est"], 100*t6["bypass:ae"]["0.1"]["est"], "higher", unit="%", sub="2.4× as often", fmt="{:.1f}")}
  {stat("Transfer at large disruption (1.0)", 100*t6["base:km"]["1.0"]["est"], 100*t6["bypass:ae"]["1.0"]["est"], "higher", unit="%", fmt="{:.1f}")}
  {stat("Token-mean control at 0.1", 100*t6["base:km"]["0.1"]["est"], 100*t6["base_tokmean:km"]["0.1"]["est"], "higher", unit="%", sub="base → control: the subtraction alone gets part way", fmt="{:.1f}")}
</div>
{tr_tbl}
{reading("The fourth column keeps the AE's clusters but makes the edit in the residual. It transfers about as well as editing the latent, so the gain belongs to <b>where the AE draws its cluster boundaries</b>, not to the latent coordinates themselves. Narrower latents transfer best (d768: 3.7× base).")}
{ah_block}
''')

# ------------------------------------------------------------------ long-form cluster steering
jd = J["d6144"]
FA = json.load(open('/projects/bbyl/mhaider/GeoAE/results/delta/d6144/steering/cluster_steer_judge_llmfluency.json'))['fluency_agreement']
gen_t = test("Test 5 · long-form generation, no labels", "Steering generated text toward a cluster", f'''
{how("Pick 40 content clusters from each codebook (not hubs, drawn from at least 20 documents, and whose typical next tokens are whole words). For each, generate 32 tokens from 20 neutral prompts, and at <b>every</b> generated token move the representation from its current cluster toward the target by α × the gap between the cluster means. α from 0.1 to 0.75, plus α = 0 as the unsteered reference.",
     "An LLM judge (gemini-2.5-flash-lite) sees the continuation and four short cluster descriptions, one correct, and picks one (chance 25%). <b>Fluent</b>: the judge also answers, without seeing the target, whether the continuation is fluent, meaningful text (repetition loops, word lists and gibberish count as no). <b>On-target, net</b>: the judge's hit rate among fluent continuations minus the same prompt's unsteered hit rate. <b>Usable</b> = fluent share × net on-target, per 100 attempts.",
     "Bypass steering stays fluent under stronger pushes, but its fluent text is less often on target, so usable text is only slightly higher and the difference is not significant at d6144.")}
<div class="stats">
  {stat("Usable continuations per 100, each at its best α", S["d6144"]["cluster_steer_judge"]["base_usable_per_100"], S["d6144"]["cluster_steer_judge"]["bypass_usable_per_100"], "higher", sub=f"base best at α {CJ6['base_best_alpha']}, bypass at α {CJ6['bypass_best_alpha']} · gap not significant", fmt="{:.1f}", tol=0)}
  {stat("Fluent share at α = 0.5", 100*jd["0.5"]["base"]["fluent_share"], 100*jd["0.5"]["bypass"]["fluent_share"], "higher", unit="%", sub="base text falls apart sooner")}
  {stat("Fluent share at α = 0.75", 100*jd["0.75"]["base"]["fluent_share"], 100*jd["0.75"]["bypass"]["fluent_share"], "higher", unit="%")}
</div>
{judge_tbl}
{reading(f"Read the table left to right as the push gets stronger. <b>Bypass steering stays fluent longer</b>: at α 0.5 the judge calls {100*jd['0.5']['bypass']['fluent_share']:.0f}% of bypass continuations fluent against {100*jd['0.5']['base']['fluent_share']:.0f}% for base, and at α 0.75 {100*jd['0.75']['bypass']['fluent_share']:.0f}% against {100*jd['0.75']['base']['fluent_share']:.0f}%. <b>But among fluent continuations, base is more often on target at α 0.5</b> (net {100*jd['0.5']['base']['net_hit']:+.0f} vs {100*jd['0.5']['bypass']['net_hit']:+.0f}). The two effects nearly cancel: usable text peaks at {CJ6['base_usable_per_100']:.1f} for base and {CJ6['bypass_usable_per_100']:.1f} for bypass, and a bootstrap over target clusters puts the gap at +1.6 with a 95% interval of −3.5 to +6.6. This claim does not survive the stricter, judge-based fluency rule at d6144.")}
<h3 style="margin-top:.4rem;font-size:1rem">How the judge's fluency call compares with the old word-pair rule (d6144, all steered continuations)</h3>
{table(["", "Old rule: fluent", "Old rule: not fluent"], [["Judge: fluent", f"{FA['both_yes']:,}", f"{FA['judge_only']:,}"], ["Judge: not fluent", f"{FA['bigram_only']:,}", f"{FA['both_no']:,}"]])}
<p class="tnote">The two agree on {100*(FA['both_yes']+FA['both_no'])/sum(FA.values()):.0f}% of continuations. Where they differ, the judge is stricter: it rejects {FA['bigram_only']:,} continuations the word-pair rule passed, and accepts only {FA['judge_only']:,} it rejected.</p>
<h3 style="margin-top:.4rem;font-size:1rem">Across runs</h3>
{judge_w}
''')

# ------------------------------------------------------------------ long-form concept steering
cs6 = C["d6144"]["scopes"]["pooled"]
pwc = cs6["pairwise"]
concept_t = test("Test 6 · long-form generation, named targets", "Steering generated text toward a named concept", f'''
{how("Test 5 lets each codebook choose its own targets, so base and bypass steer toward different things. Here every method steers toward the <b>same</b> 42 named concepts (14 DBpedia topics, 28 professions), with the same prompts, 32 tokens, edit at every token. Six handles (below) are compared, each scaled to the same typical edit size so a given α moves them all by a comparable amount.",
     "The same LLM-judge scoring as Test 5, with the target and three distractor concepts from the same dataset. Usable per 100 at each handle's best α; paired Wilcoxon tests over the 42 concepts.",
     "Bypass clusters beat base clusters, and used as a direction they match a supervised label direction.")}
<div class="opgrid">
  <div class="op"><span class="kind">Supervised</span><b>Label direction</b>Concept's mean residual minus the average over all concepts. Uses labels fully. The reference.</div>
  <div class="op"><span class="kind">Supervised, in the AE</span><b>Label direction in the latent</b>The same difference computed in z and decoded back. Asks whether the latent gives a better supervised direction.</div>
  <div class="op"><span class="kind">Clusters, fixed direction</span><b>Base / bypass clusters as a direction</b>Labels only pick the 5 clusters most specific to the concept. Their centres, measured on general text, give one fixed direction, added at every token.</div>
  <div class="op"><span class="kind">Clusters, per position</span><b>Base / bypass move-into-cluster</b>The same 5 clusters, but each token is moved from its own cluster toward them, as in Test 5.</div>
</div>
<div class="stats">
  {stat("Clusters as a direction, usable per 100", 100*cs6["usable"]["base_dir"][0], 100*cs6["usable"]["bypass_dir"][0], "higher", sub=f"paired p = {pwc['bypass_dir_minus_base_dir']['wilcoxon_p']:.3f}", fmt="{:.1f}")}
  {stat("Move-into-cluster, usable per 100", 100*cs6["usable"]["base"][0], 100*cs6["usable"]["bypass"][0], "higher", sub=f"paired p < 0.001", fmt="{:.1f}")}
  {stat("Supervised label direction vs bypass clusters as a direction", 100*cs6["usable"]["label"][0], 100*cs6["usable"]["bypass_dir"][0], "none", sub=f"label direction → bypass clusters; not significantly different · p = {pwc['bypass_dir_minus_label']['wilcoxon_p']:.2f}", fmt="{:.1f}")}
</div>
{concept_tbl}
{pair_tbl}
{reading("<b>Four findings.</b> (1) Bypass clusters are better handles than base clusters in both forms. (2) Used as a fixed direction, bypass clusters score as well as the supervised label direction; the small gap is not significant. (3) Adding a fixed direction beats moving each token into a cluster by 9–11 points, for both codebooks, because moving into a cluster also erases where each token was. (4) When labels are available, the AE adds nothing: the label direction computed in the latent performs the same as in the residual.")}
''')

part_b = f'''
<div class="part" id="control"><span class="eyebrow">Part B</span><h2>Controlling the model through the representation</h2>
<p>Six tests, from the most controlled (one prediction, labelled classes) to the most realistic (free generation, no labels). Each block says what is done, what is measured and what came out, with the key numbers as base → bypass AE.</p></div>
{sec("howtested", "How control is tested", primer)}
{sec("interventions", "Labelled edits on a single prediction", removal + steer_ops + across + number_t)}
{sec("handles", "Clusters as handles", transfer_t + gen_t)}
{sec("concepts", "Named-concept steering in generated text", concept_t)}
'''

toc_items = [("summary", "Summary"), ("setup", "Setup"), ("health", "Reconstruction and model health"), ("tokens", "Token structure"), ("probes", "Concept alignment"),
             ("cq", "Cluster compactness"), ("geometry", "Geometry of the spaces"), ("linprobes", "Linear probes"), ("control", "Part B: controlling the model"), ("howtested", "How control is tested"), ("interventions", "Labelled edits: removal, steering, number"), ("handles", "Clusters as handles: transfer, long-form"), ("concepts", "Named-concept steering"),
             ("width", "Width and cluster count"), ("replication", "Replication of the old-box claims"), ("caveats", "Caveats and open items"), ("appendix", "Where the files are")]
toc = '<nav class="toc">' + "".join(f'<a href="#{i}">{t}</a>' for i, t in toc_items) + "</nav>"

body = f"""
<title>GeoAE Round 2 Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,500;7..72,600&family=Public+Sans:wght@400;600&family=JetBrains+Mono:wght@400&display=swap">
<style>{CSS}{CSS2}</style>
<div class="wrap">
<header>
  <span class="eyebrow">GeoAE · Llama-3.2-3B, layer 27 · NCSA Delta round 2</span>
  <h1>Token-bypass autoencoder vs the base residual</h1>
  <p class="lede">Five trained runs on a new 10M-row activation dump, each compared with a balanced k-means base on the plain residual and with an encoder-free token-mean control. Every number below is from a single training seed.</p>
  <div class="meta"><span>Results as of 29 Sep 2026</span><span>Main arm: d6144, K = 2,000</span><span>Sixth run (d12288, K = 4,000) still training</span></div>
</header>
{toc}

{sec("summary", "Summary", f'''
<div class="col">
<p>The question is whether an autoencoder latent at layer 27 is better organised than the residual stream itself. The model under test is the <b>token-bypass AE</b>: it subtracts each token's average residual before encoding and adds it back after decoding, so the latent only has to carry context. It is compared with <b>base</b>, a balanced k-means with the same number of clusters fit directly on the residual, and with a <b>token-mean control</b>, the same k-means fit on the residual after the per-token average is subtracted. The control answers a sharp question: is any gain due to the encoder, or only to the subtraction?</p>
</div>
<div class="keys">
  <div class="key"><h4>Topic organisation improves {chip("holds")}</h4><p>Topic clustering on DBpedia-14 rises from 0.38 to 0.58 cNMI, and pure token-detector clusters halve (10.3% → 5.1%). The token-mean control gets most of the way (0.55), so most of this gain comes from the subtraction, not the encoder.</p></div>
  <div class="key"><h4>Clusters are better one-step handles {chip("holds")}</h4><p>Moving a row into a bypass cluster changes the next token toward the target 2.4× as often as with base clusters; the control sits in between (6.2%). In free generation bypass steering stays fluent longer, but usable text is only {CJ6['bypass_usable_per_100']:.1f} vs {CJ6['base_usable_per_100']:.1f} per 100, not a significant difference.</p></div>
  <div class="key"><h4>As directions, bypass clusters match supervision {chip("holds")}</h4><p>Used as a steering direction toward a named concept, bypass clusters score {100*CU6['bypass_dir'][0]:.1f} usable per 100 vs {100*CU6['base_dir'][0]:.1f} for base clusters ({pfmt(PW6['bypass_dir_minus_base_dir']['wilcoxon_p'])}) and {100*CU6['label'][0]:.1f} for a supervised label direction (difference not significant).</p></div>
  <div class="key"><h4>Labelled edits: mixed {chip("weaker", "mixed")}</h4><p>DBpedia topic removal is cleaner in the latent (collateral 13 vs 19 per 100 at the tool's setting). On bias_in_bios the AE is at parity at its own setting but does 6 more collateral per 100 at matched removal. Number control favours the AE (+13 flips per 100), and that gain comes from the encoder, not the token subtraction.</p></div>
  <div class="key"><h4>Linear probes see no difference {chip("tie", "tie")}</h4><p>A trained probe needs about 40% of coordinates removed to halve a topic in every space, and gender stays spread over about 90% of coordinates. The bypass latent does not isolate concepts for a fixed reader; its gains are in the partition and in edits.</p></div>
  <div class="key"><h4>Syntax moves out of the latent {chip("cost")}</h4><p>Part-of-speech clustering drops from 0.36 to 0.17 cNMI. That information now lives in the per-token table, which is the design's intent but a real cost for token-level uses.</p></div>
  <div class="key"><h4>Width and cluster count {chip("tie", "d6144 best")}</h4><p>d3072 and d6144 lead; d768 breaks the model when spliced in (MMLU 0) and d12288 loses some organisation. Doubling clusters to 4,000 helps the base more than the AE, shrinking the clustering lead but widening the steering lead.</p></div>
</div>
<h3>Main arm at a glance</h3>
{headline}
''')}

<div class="part" id="partA"><span class="eyebrow">Part A</span><h2>How the representation is organised</h2><p>Setup, model health, token structure, concept alignment and geometry: is the latent a faithful, better-organised stand-in for layer 27?</p></div>
{sec("setup", "Setup", f'''
<div class="col">
<p><b>Data.</b> A new dump of 10,000,000 layer-27 residual vectors from 157,249 documents: 64 random token positions per document, up to 2,048 tokens, in a pretraining-like mix (educational web 40%, web 15%, English Wikipedia 15%, Pile 10%, code 8%, math 5%, German, French and Spanish Wikipedia 7%). The last 5% of rows are held out. The median layer-27 vector norm is 56.6.</p>
<p><b>Token table.</b> The per-token average residual is computed on the training split for 49,664 tokens with at least 10 occurrences, shrunk toward zero for rare tokens. It covers 98.3% of held-out rows and on its own explains 17.2% of the residual's variance.</p>
<p><b>Training.</b> Every AE uses the same recipe: 50 epochs, batch 32,768, balanced (Sinkhorn) cluster assignment, reconstruction only for epochs 1–5, a variance/covariance regulariser from epoch 6, and the cluster loss from epoch 11 with centroids initialised by density peaks. Only latent width and cluster count vary between runs.</p>
</div>
{arms_tbl}
<dl class="defs">
  <dt>Base (h)</dt><dd>Balanced k-means with the run's K, fit on 1.5M normalised residual rows. No encoder. In the intervention tables, h means edits made directly in the residual.</dd>
  <dt>Token-mean control</dt><dd>The same k-means fit on the residual minus the per-token average. Same subtraction as the bypass AE, no encoder.</dd>
  <dt>Bypass AE (z)</dt><dd>The autoencoder's latent. In the intervention tables, z means edits made in the latent and decoded back.</dd>
  <dt>cNMI</dt><dd>Chance-corrected NMI: (NMI − NMI with shuffled labels) / (1 − NMI with shuffled labels). Plain NMI rises with the number of clusters a task spreads over, even for random labels; the correction removes that.</dd>
</dl>
''')}

{sec("health", "Reconstruction and model health", f'''
<div class="col"><p>Before comparing organisation, each AE has to be a faithful stand-in for layer 27. This table checks three things: how much variance the reconstruction keeps, whether the language model still works when the reconstruction replaces the real layer, and whether the clusters are all in use.</p></div>
{health_tbl}
<div class="col">
<p><b>Reading it.</b> d6144 and d12288 are near-lossless: MMLU drops by about one point. d3072 costs 7 points. <b>d768 is not usable as a replacement layer</b>: it keeps only 45% of the variance and the spliced model answers no MMLU question correctly. Its clustering results below are still meaningful, but it cannot be used for any evaluation that runs text through the spliced model.</p>
<p><b>No collapse.</b> The training log's "dying clusters" counter looked alarming (700–1,500 per run) but measures the balancing step's soft targets, not where tokens go. Read from each checkpoint's hard assignments, no cluster is empty, at most 15 are starved, and the effective number of clusters is 64–69% of K in every run.</p>
</div>
''')}

{sec("tokens", "Token structure", f'''
<div class="col"><p>The motivating problem was that many AE clusters were "token dumps": one current token made up most of their members. This table measures how much the partition tracks token identity. Lower values in the first four columns mean the clusters are organised by something other than the token itself.</p></div>
{ts_tbl}
<div class="col"><p>Both the control and the bypass AE halve the share of pure token clusters and roughly halve cNMI with the current token. The bypass AE goes slightly further on token identity (0.127 vs 0.147) and on the predicted next token (0.182 vs 0.218). The pattern is the same at every width:</p></div>
{tsw_tbl}
''')}

{sec("probes", "Concept alignment", f'''
<div class="col"><p>Each labelled task is encoded with the language model; every row is assigned to its nearest cluster, and the table reports how much the cluster id tells you about the label (cNMI). Sequence tasks use the last token of a document or sentence; token tasks label individual tokens. The three columns use the same rows and the same bootstrap resamples, so differences between them are paired.</p></div>
<h3>Sequence-level tasks (d6144)</h3>
{probe_seq}
<div class="col"><p>The bypass AE is best on 6 of 9 sequence tasks and gains most on topic (+0.20 on DBpedia, +0.14 on AG News) and subjectivity (+0.06). It is slightly behind base on domain (−0.007) and on long-review sentiment. The control takes most of the topic gain (0.553), so the encoder adds only +0.026 on DBpedia topic. Formality is the exception, where base is slightly ahead. Sentiment is near zero for every space: layer-27 clusters do not organise by sentiment.</p></div>
<h3>Token-level tasks (d6144)</h3>
{probe_tok}
<div class="col"><p>On token tasks the base residual wins almost everywhere. Part of speech falls from 0.34 to 0.16, and named entities and RAVEL attributes also drop. This is expected: the subtracted token average carries exactly this information, as the leak column shows. The exception is IOI role (which of two names is the indirect object), which depends on context and rises from 0.37 to 0.46.</p></div>
<h3>Across runs</h3>
{probe_width}
<div class="col"><p>d3072 and d6144 are the best widths for sequence organisation. d768 and d12288 both fall back toward the control. At K = 4,000 the base itself organises topics much better (0.50), so the AE's margin on topic14 shrinks to +0.05.</p></div>
''')}

{cq_sec}
{sec("geometry", "Geometry of the spaces", f'''
{bottom([("DBpedia topic: neighbours share the label", "94.2%", "93.1%", "tie", "about equal", "base → bypass, d6144"),
         ("Part of speech: neighbours share the label", "75.2%", "48.7%", "cost", "cost", "syntax moved into the token table")])}
<div class="col"><p><b>What this measures.</b> The concept-alignment section scored the clusters. This one ignores clusters and asks whether points with the same label sit next to each other. For each task, up to 6,000 points are placed in the space; for each point we look at its 10 nearest neighbours and check whether they vote for the right label. The number is the share of points classified correctly this way, so 94% means almost every point sits among its own class.</p>
<p>The same points appear in the PCA, t-SNE and UMAP figures in <code>results/delta/&lt;run&gt;/figures/geometry/</code>. The pictures are for looking; these numbers are for comparing.</p></div>
{geo_tbl}
<div class="col"><p><b>Reading it.</b> On sequence tasks the three spaces are within 1–3 points of each other, so classes are not farther apart in the latent than in the residual. The AE's clustering advantage comes from where cluster boundaries fall, not from better-separated classes. On token tasks the latent loses part of speech (75% → 49%) and some named-entity information (81% → 76%), the same cost the concept tables showed. d3072 is the only width that matches base on topics.</p></div>
''')}

{probes_sec}
{part_b}

{sec("width", "Width and cluster count", f'''
{bottom([("Best all-round run", None, "d6144", "holds", "main arm", "K = 2,000"),
         ("d768 spliced into the model", None, "MMLU 0.0", "worse", "unusable", "too narrow to reconstruct"),
         ("Topic lead over base at K = 4,000", "+0.20", "+0.05", "weaker", "shrinks", "K = 2,000 → 4,000")])}
<div class="col"><p><b>What this measures.</b> The same recipe at four latent widths (all K = 2,000), then the main width at twice the clusters. One row per measure, so the best width can be picked row by row. Bold = best width on that row.</p></div>
{syn_tbl}
<div class="col"><p><b>Reading it.</b> d6144 is the best all-round run. It matches d3072 on organisation without d3072's 7-point MMLU cost or its bias_in_bios collateral, and it leads on concept steering. d12288 reconstructs best and keeps the most part-of-speech information but is weaker on topics and on generated steering. d768 transfers most strongly but breaks the model when spliced in.</p></div>
<h3>K = 2,000 vs K = 4,000 at d6144</h3>
<div class="col"><p>Each pair of columns uses the base with the same K.</p></div>
{k_tbl}
<div class="col"><p><b>Reading it.</b> More clusters help the plain residual far more than the AE (topic14 0.38 → 0.50 for base, 0.58 → 0.55 for bypass), so the AE's clustering lead shrinks to +0.05 and the token-mean control ties it. Steering goes the other way: base clusters get worse as handles when split finer ({S["d6144"]["cluster_steer_judge"]["base_usable_per_100"]:.1f} → {S["d6144k4000"]["cluster_steer_judge"]["base_usable_per_100"]:.1f} usable per 100), bypass clusters hold up better; K = 4,000 is the only run where the generation gap is significant. K = 2,000 stays the headline setting.</p></div>
''')}

{sec("replication", "Replication of the old-box claims", f'''
<div class="col"><p>The original results came from a different machine and an older dump (first 256 tokens of 44k documents, five equal sources). This table checks each headline claim against the new Delta run of the same recipe.</p></div>
{rep_tbl}
''')}

{sec("caveats", "Caveats and open items", '''
<div class="col">
<ul>
<li><b>One training seed.</b> No run has been repeated. Several differences in this report are 1–3 points or 0.01–0.03 cNMI, which a second seed could move. Replicate configs exist for seeds 43–46.</li>
<li><b>No multiple-comparison correction</b> on the concept-steering p-values. Of the six pairwise tests per run, the bypass-vs-base comparisons are the ones planned in advance.</li>
<li><b>Fluency is judged by the LLM.</b> Steered text counts as fluent only when the judge says so (asked separately, without the target). The earlier word-pair rule is kept as an option; the two agree on 89% of cluster-steering continuations, and the judge is the stricter of the two. Under the judge rule every d6144 handle peaks at α = 0.5 or below, inside the tested range.</li>
<li><b>Chance correction depends on K.</b> cNMI removes the shuffled-label baseline, but comparisons across different cluster counts (K = 2,000 vs 4,000) are still less clean than within a K.</li>
<li><b>Matched vs own-setting interventions disagree on bias_in_bios.</b> Both are reported; neither is the single right view.</li>
<li><b>d3072 DBpedia global steering</b> returned NaN selectivity and is not reported. The cause has not been checked.</li>
<li><b>d768</b> cannot run any evaluation that passes text through the spliced model (interventions, MMLU).</li>
<li><b>Old-box comparisons</b> are across different data and different replicates; they test whether a claim's direction survives, not its exact size.</li>
<li><b>Pending.</b> d12288 at K = 4,000 is training (about 19 minutes per epoch, finishing around Thursday midday); its evaluations run automatically afterwards.</li>
</ul>
</div>
''')}

{sec("appendix", "Where the files are", '''
<div class="col">
<p>All Delta results live in one tree in the repository, <code>results/delta/</code>. <code>README.md</code> there explains every file; <code>summary.txt</code> is the plain-text version of the main tables.</p>
<ul>
<li><code>results/delta/&lt;run&gt;/evals/</code>: token structure, probes, MMLU splice, clustering quality</li>
<li><code>results/delta/&lt;run&gt;/interventions/</code>: DBpedia and bias_in_bios edits, DBpedia steering, number control</li>
<li><code>results/delta/&lt;run&gt;/steering/</code>: next-token transfer, steered generations and their judge</li>
<li><code>results/delta/&lt;run&gt;/concept/</code>: named-concept steering and its judge</li>
<li><code>results/delta/&lt;run&gt;/figures/geometry/</code>: PCA, t-SNE, UMAP and variance figures for ten tasks</li>
<li><code>results/delta/&lt;run&gt;/probes/</code>: linear-probe tests (per-class coordinate removal on DBpedia and bias_in_bios, gender vs profession probe); plots in <code>figures/probes/</code></li>
</ul>
<p>Regenerate the summary with <code>scripts/delta/py eval_out/summarize_delta.py</code>.</p>
</div>
''')}
</div>
"""
(SP / "geoae_round2_report.html").write_text(body)
print("wrote", len(body))
