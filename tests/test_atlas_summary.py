"""atlas_summary: usage weighting, token-detector / nameable / conceptual rules, judge-run matching,
missing-field robustness and the bootstrap interval, on tiny synthetic judge + closest_tokens files.
CPU only, no LM, no API."""
import json

import numpy as np
import pytest

from geoae.interp import atlas_summary as A

TOTAL = 10_000          # candidate tokens in the fake closest_tokens runs


# ------------------------------------------------------------------ fixtures
def ct_cluster(n, tokens=None, hub=False, doc_view=True, n_ctx=4):
    spread = [["a", 5], ["b", 4], ["c", 3]]                   # a diffuse token mix: never a token detector
    c = {"n_assigned": n, "usage_pct": round(100 * n / TOTAL, 3), "dist_min": 0.1, "dist_p50": 0.5,
         "monosemanticity": 0.1, "domains": {"web": n},
         "token_distribution": [list(t) for t in tokens] if tokens is not None else spread,
         "top": [{"dist": 0.1 * i, "token": "x", "context": f"ctx{i} «x» tail", "domain": "web"}
                 for i in range(n_ctx)],
         "spectrum": []}
    if doc_view:
        c.update(n_docs=7, eff_docs=5.5, top_doc_share=0.2, hub=hub)
    return c


def judge_rec(cid, n, level="semantics", sem=5, intr=1.0, surf=1, gen=3, det=0.9, nul=0.4, expl="a concept"):
    return {"cluster": cid, "n_assigned": n, "usage_pct": round(100 * n / TOTAL, 3), "explanation": expl,
            "level": level, "mono_llm": 4, "surface_coh": surf, "semantic_coh": sem, "generality": gen,
            "coverage": 0.9, "intruder_trials": 3, "intruder_acc": intr, "intruder_chance": 1 / 6,
            "detect_acc": det, "null_acc": nul}


def make_space(rows, mode, extra_live=(), doc_view=True):
    """rows: [(n_assigned, judge overrides, closest_tokens overrides)] -> (ct dict, judge records).
    extra_live: usage of live clusters that were never judged."""
    clusters, recs = {}, []
    for cid, (n, j, c) in enumerate(rows, start=1):
        clusters[str(cid)] = ct_cluster(n, doc_view=doc_view, **c)
        recs.append(judge_rec(cid, n, **j))
    for k, n in enumerate(extra_live, start=len(rows) + 1):
        clusters[str(k)] = ct_cluster(n, doc_view=doc_view)
    meta = {"mode": mode, "n_tokens": TOTAL, "n_docs": 50, "n_clusters": len(clusters) + 3, "n_live": len(clusters),
            "target_layer": 27, "model_name": "m", "data": {"seed": 42, "doc_offsets": {"web": 10}, "context_len": 64}}
    return {"meta": meta, "clusters": clusters}, recs


def write_case(tmp_path, base_rows, byp_rows, swap=False, relative=False, base_kw=None, byp_kw=None):
    """-> (judge json path, base ct path, bypass ct path). The judge file lists one run per space with
    `source` = the ct path as a user would have typed it."""
    ct_b, jb = make_space(base_rows, "raw_kmeans_baseline", **(base_kw or {}))
    ct_y, jy = make_space(byp_rows, "learned_ae", **(byp_kw or {}))
    pb = tmp_path / "base_stem" / "interp" / "closest_tokens.json"
    py = tmp_path / "d6144" / "interp" / "closest_tokens.json"
    for p, ct in ((pb, ct_b), (py, ct_y)):
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(ct))
    src = (lambda p: str(p.relative_to(tmp_path))) if relative else str
    runs = [{"source": src(pb), "provider": "openrouter", "model": "judge-x", "n_clusters": len(jb), "clusters": jb},
            {"source": src(py), "provider": "openrouter", "model": "judge-x", "n_clusters": len(jy), "clusters": jy}]
    if swap:
        runs.reverse()
    jp = tmp_path / "judge.json"
    jp.write_text(json.dumps(runs))
    return jp, pb, py


def run_atlas(tmp_path, jp, pb, py, *extra, n_boot="200"):
    out = tmp_path / "out"
    A.main(["--judge", str(jp), "--ct_base", str(pb), "--ct_bypass", str(py), "--out_dir", str(out),
            "--n_boot", n_boot, *extra])
    load = lambda name: json.loads((out / name).read_text())
    return load("atlas_summary.json"), load("atlas_base.json"), load("atlas_bypass.json")


def by_cluster(atlas):
    return {r["cluster"]: r for r in atlas}


def approx(x, tol=1e-5):                  # the summary JSON is rounded to 6 decimals
    return pytest.approx(x, abs=tol)


GOOD = ({}, {})                       # intruder 1.0, semantic 5, level semantics, not a token detector


# ------------------------------------------------------------------ weighting
def test_mass_weighting_vs_count(tmp_path):
    base = [(100, {}, {}), (300, {"intr": 0.0}, {}), (600, {"intr": 0.0, "sem": 2}, {})]
    byp = [(500, {}, {}), (500, {"intr": 1 / 3}, {})]
    jp, pb, py = write_case(tmp_path, base, byp, base_kw={"extra_live": [500]})
    s, _, _ = run_atlas(tmp_path, jp, pb, py)
    assert s["base"]["mass"]["nameable"] == approx(0.1)              # 100 of 1000 judged tokens
    assert s["base"]["count"]["nameable"] == approx(1 / 3)           # 1 of 3 judged clusters
    assert s["base"]["mass"]["semantic_coh"] == approx((500 + 1500 + 1200) / 1000)
    assert s["base"]["count"]["semantic_coh"] == approx(4.0)
    assert s["bypass"]["mass"]["nameable"] == approx(0.5)
    assert s["diff"]["mass"]["nameable"]["diff"] == approx(0.4)
    # coverage: 3 of 4 live clusters judged, holding 1000 of 1500 live tokens
    cov = s["base"]["coverage"]
    assert (cov["n_judged"], cov["n_live"]) == (3, 4)
    assert cov["judged_usage_mass"] == approx(1000 / 1500)
    assert s["bypass"]["coverage"]["judged_usage_mass"] == approx(1.0)
    assert sum(s["base"]["level_mass"].values()) == approx(1.0)
    assert set(s["base"]["level_mass"]) == set(A.CONCEPT_LEVELS) | {"unparsed"}


# ------------------------------------------------------------------ token detector
def test_token_share_uses_the_closest_core_not_the_listed_counts():
    tail = [(f"t{i}", 2) for i in range(1, 15)]
    assert A.token_stats({"token_distribution": [["the", 160], ["a", 40]]}, 1000) == ("the", 0.8)
    # 100 of the 200 closest tokens, the rest scattered: 0.5 - not 100 / (sum of the 15 listed) = 0.78
    assert A.token_stats({"token_distribution": [["t0", 100]] + [list(t) for t in tail]}, 1000) == ("t0", 0.5)
    # a cluster smaller than the core: the denominator is its own size
    assert A.token_stats({"token_distribution": [["x", 80], ["y", 20]]}, 100) == ("x", 0.8)
    assert A.token_stats({"token_distribution": []}, 100) == (None, None)
    assert A.token_stats({}, 100) == (None, None)
    assert A.token_stats({"token_distribution": [["x", None], "junk"]}, 100) == (None, None)


def test_token_detector_threshold_is_inclusive(tmp_path):
    long_tail = [("t0", 100)] + [(f"t{i}", 2) for i in range(1, 15)]
    base = [(1000, {}, {"tokens": [("the", 160), ("a", 20), ("b", 10)]}),     # 0.800 -> detector
            (1000, {}, {"tokens": [("the", 159), ("a", 41)]}),                # 0.795
            (100, {}, {"tokens": [("x", 80), ("y", 20)]}),                    # core is 100 -> 0.800
            (1000, {}, {"tokens": long_tail}),                                # 0.5
            (1000, {}, {"tokens": []})]                                       # no token mix at all
    jp, pb, py = write_case(tmp_path, base, [(100, *GOOD)])
    s, atlas, _ = run_atlas(tmp_path, jp, pb, py)
    r = by_cluster(atlas)
    assert [r[i]["token_detector"] for i in (1, 2, 3, 4, 5)] == [True, False, True, False, False]
    assert (r[1]["top_token"], r[1]["top_token_share"]) == ("the", 0.8)
    assert r[4]["top_token_share"] == 0.5 and r[5]["top_token"] is None and r[5]["top_token_share"] is None
    assert s["base"]["mass"]["token_detector"] == approx(1100 / 4100)
    assert s["base"]["count"]["token_detector"] == approx(2 / 5)


# ------------------------------------------------------------------ nameable / conceptual
def test_nameable_and_conceptual_rules(tmp_path):
    det = {"tokens": [("the", 190), ("a", 10)]}                  # 0.95 of the core
    cases = [                                                    # (judge, closest_tokens) -> (nameable, conceptual)
        (({}, {}), (True, True)),
        (({"intr": 2 / 3, "sem": 4}, {}), (True, True)),         # both thresholds are inclusive
        (({"intr": 1 / 3}, {}), (False, False)),
        (({"sem": 3}, {}), (False, False)),
        (({"level": "syntax"}, {}), (True, False)),
        (({"level": "topic"}, det), (True, False)),              # nameable, but as a string
        (({"level": "discourse"}, {}), (True, True)),
        (({"level": "entity"}, {}), (True, True)),
        (({"level": "surface"}, {}), (True, False)),
        (({"level": "morphology"}, {}), (True, False)),
        (({"level": "mixed"}, {}), (True, False)),
        (({"level": " Semantics "}, {}), (True, True)),          # case / whitespace normalised
        (({"level": None}, {}), (True, False)),                  # nameable, but no concept level
        (({"level": "lexical"}, {}), (True, False)),             # not in the judge's taxonomy
        (({"intr": 0.0, "level": "topic"}, {}), (False, False)),
    ]
    base = [(1000, j, c) for (j, c), _ in cases]
    jp, pb, py = write_case(tmp_path, base, [(100, *GOOD)])
    s, atlas, _ = run_atlas(tmp_path, jp, pb, py)
    r = by_cluster(atlas)
    for i, (_, (want_n, want_c)) in enumerate(cases, start=1):
        assert (r[i]["nameable"], r[i]["conceptual"]) == (want_n, want_c), (i, cases[i - 1])
    assert r[12]["level"] == "semantics" and r[13]["level"] == "unparsed" and r[14]["level"] == "unparsed"
    n_nameable = sum(w for _, (w, _) in cases)
    n_conceptual = sum(c for _, (_, c) in cases)
    assert s["base"]["count"]["nameable"] == approx(n_nameable / len(cases))
    assert s["base"]["count"]["conceptual"] == approx(n_conceptual / len(cases))
    assert s["base"]["unparsed_level_values"] == {"None": 1, "'lexical'": 1}


# ------------------------------------------------------------------ which run is which
def test_source_matching_ignores_run_order_and_path_spelling(tmp_path, monkeypatch):
    base, byp = [(100, {"intr": 0.0}, {})], [(100, {}, {})]       # base nameable 0, bypass nameable 1
    jp, pb, py = write_case(tmp_path, base, byp, swap=True, relative=True)
    monkeypatch.chdir(tmp_path)                                    # `source` is relative to the cwd it ran in
    s, _, _ = run_atlas(tmp_path, jp, pb, py)
    assert s["base"]["mass"]["nameable"] == 0.0 and s["bypass"]["mass"]["nameable"] == 1.0
    assert s["meta"]["inputs"]["judge_source_base"] == "base_stem/interp/closest_tokens.json"


def test_source_matching_falls_back_to_the_path_tail():
    runs = [{"source": "/old/root/d6144/interp/closest_tokens.json", "clusters": [], "tag": "y"},
            {"source": "/old/root/base_stem/interp/closest_tokens.json", "clusters": [], "tag": "b"}]
    b, y = A.match_runs(runs, "/new/place/base_stem/interp/closest_tokens.json",
                        "/new/place/d6144/interp/closest_tokens.json")
    assert (b["tag"], y["tag"]) == ("b", "y")
    with pytest.raises(SystemExit):                                # nothing in common
        A.match_runs(runs, "/x/other/closest_tokens.json", "/new/place/d6144/interp/closest_tokens.json")
    with pytest.raises(SystemExit):                                # both spaces claim the same run
        A.match_runs(runs[:1], "/a/d6144/interp/closest_tokens.json", "/b/d6144/interp/closest_tokens.json")


def test_swapped_ct_arguments_are_refused(tmp_path):
    jp, pb, py = write_case(tmp_path, [(100, *GOOD)], [(100, *GOOD)])
    with pytest.raises(SystemExit, match="swapped"):
        run_atlas(tmp_path, jp, py, pb)                            # --ct_base is the AE file
    s, _, _ = run_atlas(tmp_path, jp, py, pb, "--allow_mode_mismatch")
    assert s["meta"]["mode_check"]["ok"] is False


# ------------------------------------------------------------------ missing / None fields
def test_missing_and_none_fields(tmp_path, capsys):
    base = [(200, {"level": None, "sem": None, "surf": None, "gen": None, "intr": None, "det": None,
                   "nul": None, "expl": ""}, {}),                           # the judge answer did not parse
            (300, {"sem": 7, "intr": "0.5"}, {}),                           # out of range / not a number
            (500, {}, {})]
    jp, pb, py = write_case(tmp_path, base, [(100, *GOOD)], base_kw={"doc_view": False})
    runs = json.loads(jp.read_text())
    recs = runs[0]["clusters"]
    for k in ("intruder_trials", "intruder_acc", "intruder_chance"):        # early-return records lack these keys
        recs[0].pop(k)
    recs += [judge_rec(99, 100),                                            # id absent from the closest_tokens file
             judge_rec(3, 500),                                             # duplicate of cluster 3
             None,                                                          # not a record
             {**judge_rec(1, 100), "cluster": "x"}]                         # no usable id
    jp.write_text(json.dumps(runs))
    s, atlas, _ = run_atlas(tmp_path, jp, pb, py)
    b = s["base"]
    assert (b["coverage"]["n_judge_records"], b["coverage"]["n_judged"]) == (7, 3)
    assert b["dropped"]["n_dropped"] == 4
    assert b["dropped"]["reasons"] == {"cluster not in the closest_tokens file": 1, "duplicate cluster id": 1,
                                       "not a record": 1, "no cluster id": 1}
    r = by_cluster(atlas)
    assert r[1]["level"] == "unparsed" and r[1]["semantic_coh"] is None and r[1]["explanation"] == ""
    assert r[2]["semantic_coh"] is None and r[2]["intruder_acc"] is None and not r[2]["nameable"]
    assert b["level_mass"]["unparsed"] == approx(0.2)
    assert b["mass"]["nameable"] == approx(0.5)                      # only cluster 3 is nameable
    assert b["mass"]["semantic_coh"] == approx(5.0)                  # means skip the clusters lacking it
    assert b["n_present"]["semantic_coh"] == 1 and b["mass_present"]["semantic_coh"] == approx(0.5)
    assert b["n_present"]["detect_minus_null"] == 2
    assert b["mass"]["detect_minus_null"] == approx(0.5)
    assert b["mass"]["hub"] is None and b["count"]["hub"] is None           # no doc_view in this closest_tokens file
    assert s["diff"]["mass"]["hub"]["diff"] is None
    out = capsys.readouterr().out
    assert "dropped 4 of 7 judge records" in out and "n/a" in out


def test_unusable_space_aborts(tmp_path):
    jp, pb, py = write_case(tmp_path, [(100, *GOOD)], [(100, *GOOD)])
    runs = json.loads(jp.read_text())
    runs[0]["clusters"] = [judge_rec(42, 100)]                              # nothing that can be placed
    jp.write_text(json.dumps(runs))
    with pytest.raises(SystemExit, match="no usable judged clusters in the base"):
        run_atlas(tmp_path, jp, pb, py)


def test_inconsistent_inputs_are_flagged(tmp_path, capsys):
    jp, pb, py = write_case(tmp_path, [(1000, *GOOD), (1000, *GOOD)], [(100, *GOOD)])
    ct = json.loads(pb.read_text())
    ct["meta"]["data"]["seed"] = 7                                          # not the text the bypass run read
    pb.write_text(json.dumps(ct))
    runs = json.loads(jp.read_text())
    runs[0]["clusters"][0]["n_assigned"] = 100                              # judged from a different closest_tokens run
    jp.write_text(json.dumps(runs))
    s, _, _ = run_atlas(tmp_path, jp, pb, py)
    assert s["base"]["n_assigned_mismatch"] == 1 and s["bypass"]["n_assigned_mismatch"] == 0
    assert s["meta"]["text_differs"] == ["data.seed"]
    out = capsys.readouterr().out
    assert "n_assigned differs between the judge file and the closest_tokens file" in out
    assert "did not read identical held-out text" in out


def test_signed_never_prints_negative_zero():
    assert A._signed(-0.04, 1) == "+0.0" and A._signed(-0.06, 1) == "-0.1" and A._signed(2.5, 1) == "+2.5"
    e = {"diff": -0.0004, "lo": -0.01, "hi": 0.01}
    assert A._dif(e, "share", 1) == "+0.0 pp [-1.0, +1.0]"
    assert A._dif({"diff": None}, "share", 1) == "n/a"


# ------------------------------------------------------------------ bootstrap interval
def test_centred_ci_is_centred_on_the_estimate_not_on_the_bootstrap_mean():
    boots = np.linspace(4, 6, 2001)                                         # mean 5, symmetric
    lo, hi = A.centred_ci(5.1, boots)                                       # est != mean(boots) on purpose
    assert lo == approx(5.1 - 0.95, tol=1e-3) and hi == approx(5.1 + 0.95, tol=1e-3)
    assert lo < 5.1 < hi
    assert A.centred_ci(float("nan"), boots) == (None, None)
    assert A.centred_ci(1.0, [1.0]) == (None, None)
    assert A.centred_ci(1.0, [float("nan")] * 5) == (None, None)


def _half(n_clusters, n_nameable, n=100):
    return [(n, {} if i < n_nameable else {"intr": 0.0}, {}) for i in range(n_clusters)]


def test_interval_contains_the_estimate(tmp_path):
    jp, pb, py = write_case(tmp_path, _half(100, 30), _half(100, 50))
    s, _, _ = run_atlas(tmp_path, jp, pb, py, n_boot="500")
    e = s["diff"]["mass"]["nameable"]
    assert e["diff"] == approx(0.2)                                  # the full-sample value
    assert e["lo"] < e["diff"] < e["hi"]
    assert 0.15 < e["hi"] - e["lo"] < 0.45                                  # ~ +-1.96 * 0.068
    assert s["diff"]["count"]["nameable"]["diff"] == approx(0.2)


def test_identical_spaces_give_a_zero_difference_that_the_interval_contains(tmp_path):
    jp, pb, py = write_case(tmp_path, _half(60, 20), _half(60, 20))
    s, _, _ = run_atlas(tmp_path, jp, pb, py)
    for sect in ("mass", "count"):
        for k in ("nameable", "conceptual", "token_detector", "semantic_coh", "intruder_acc"):
            e = s["diff"][sect][k]
            assert e["diff"] == approx(0.0, tol=1e-12) and e["lo"] <= 0.0 <= e["hi"], (sect, k)


def test_seeded_and_n_boot_zero(tmp_path):
    jp, pb, py = write_case(tmp_path, _half(30, 10), _half(30, 15))
    s1, _, _ = run_atlas(tmp_path, jp, pb, py)
    s2, _, _ = run_atlas(tmp_path, jp, pb, py)
    assert s1["diff"] == s2["diff"]                                         # same seed, same draws
    s3, _, _ = run_atlas(tmp_path, jp, pb, py, "--seed", "1")
    assert s3["diff"]["mass"]["nameable"]["diff"] == s1["diff"]["mass"]["nameable"]["diff"]   # est never moves
    assert s3["diff"]["mass"]["nameable"]["lo"] != s1["diff"]["mass"]["nameable"]["lo"]
    s0, _, _ = run_atlas(tmp_path, jp, pb, py, n_boot="0")
    e = s0["diff"]["mass"]["nameable"]
    assert e["diff"] == approx(s1["diff"]["mass"]["nameable"]["diff"]) and e["lo"] is None and e["hi"] is None


# ------------------------------------------------------------------ outputs
def test_atlas_files(tmp_path, capsys):
    base = [(100, {"expl": "small"}, {}), (900, {"expl": "big"}, {"n_ctx": 5}), (400, {"expl": "mid"}, {"n_ctx": 2})]
    jp, pb, py = write_case(tmp_path, base, [(100, *GOOD)])
    s, atlas_b, atlas_y = run_atlas(tmp_path, jp, pb, py)
    assert [r["explanation"] for r in atlas_b] == ["big", "mid", "small"]   # biggest first
    assert len(atlas_y) == 1
    keys = ["cluster", "usage_pct", "level", "explanation", "semantic_coh", "surface_coh", "generality",
            "intruder_acc", "detect_acc", "null_acc", "top_token", "top_token_share", "token_detector",
            "n_docs", "eff_docs", "hub", "examples"]
    assert list(atlas_b[0])[:len(keys)] == keys and "n_assigned" not in atlas_b[0]
    assert atlas_b[0]["examples"] == ["ctx0 «x» tail", "ctx1 «x» tail", "ctx2 «x» tail"]   # first 3 of 5
    assert len(atlas_b[1]["examples"]) == 2                                 # fewer if the cluster has fewer
    assert (atlas_b[0]["n_docs"], atlas_b[0]["eff_docs"], atlas_b[0]["hub"]) == (7, 5.5, False)
    assert atlas_b[0]["usage_pct"] == 9.0
    th = s["meta"]["thresholds"]
    assert (th["token_detector_top_share"], th["semantic_coh_min"]) == (0.8, 4)
    assert th["intruder_acc_min"] == approx(2 / 3)
    assert s["meta"]["judge"]["model"] == "judge-x" and s["meta"]["bootstrap"]["n_boot"] == 200
    assert s["meta"]["intruder_chance_pass_rate"] == {"n_trials": 3, "chance": approx(1 / 6, tol=1e-3),
                                                      "pass_rate": approx(0.0741)}
    out = capsys.readouterr().out
    assert "bypass - base [95% CI]" in out and "level_mass" in out and "judged usage mass" in out
