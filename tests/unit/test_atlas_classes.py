"""Draw / exclusivity / rejection logic, splits, prompt template and the collateral-matrix aggregation of the Atlas class
corpus (geoae.extraction.atlas_classes, the lazy atlas_s<N> dataset configs in _shared)."""
import json

import numpy as np
import pytest
import scipy.sparse as sp

import geoae.evaluation.cohorts as shared
import geoae.extraction.atlas_classes as AC

RULES = dict(AC.RULES, K=3, min_eligible=10, max_jaccard=0.02, max_draw_attempts=500)


def membership(blocks, n_chunks, overlaps=()):
    """chunk x unit 0/1 csc. blocks[u] = chunk range of unit u; overlaps = [(u, v, chunks)] extra shared chunks."""
    rows, cols = [], []
    for u, rng_ in enumerate(blocks):
        for c in rng_:
            rows.append(c); cols.append(u)
    for u, v, chunks in overlaps:
        for c in chunks:
            rows += [c, c]; cols += [u, v]
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n_chunks, len(blocks)))
    M.data[:] = 1
    return M.tocsc()


def test_exclusive_and_jaccard():
    # chunks 60, 61 are carried by units 0 and 1 only (n_chunks grows) -> not exclusive to either
    M = membership([range(0, 20), range(20, 40), range(40, 60)], 62, overlaps=[(0, 1, range(60, 62))])
    assert len(AC.exclusive_chunks(M, 0)) == 20 and len(AC.exclusive_chunks(M, 2)) == 20
    J = AC.pair_jaccard(M)
    assert J[0, 1] == pytest.approx(2 / 42) and J[0, 2] == 0 and J[0, 0] == 1


def test_check_draw_rejections():
    names = ["alpha", "bravo", "charlie"]
    ok_M = membership([range(0, 20), range(20, 40), range(40, 60)], 60)
    ok, why, st = AC.check_draw(ok_M, names, RULES)
    assert ok and why == [] and st["n_exclusive"] == [20, 20, 20]
    # overlap of 5 chunks: Jaccard 5/35 > 0.02
    M = membership([range(0, 20), range(20, 40), range(40, 60)], 60, overlaps=[(0, 1, range(0, 5))])
    ok, why, _ = AC.check_draw(M, names, RULES)
    assert not ok and "jaccard" in why
    # case-insensitive name containment, either direction, token-wise
    ok, why, _ = AC.check_draw(ok_M, ["Hiking", "mountain HIKING trips", "charlie"], RULES)
    assert not ok and why == ["name_containment"]
    ok, why, _ = AC.check_draw(ok_M, ["Art and design", "Graphic design", "charlie"], RULES)
    assert ok                                       # shares a word, not a contiguous name
    # a class with too few exclusive chunks (a shared chunk is not exclusive): 12 shared -> 8 exclusive < 10
    M = membership([range(0, 20), range(20, 40), range(40, 60)], 60, overlaps=[(0, 2, range(0, 12))])
    ok, why, st = AC.check_draw(M, names, dict(RULES, max_jaccard=1.0))
    assert not ok and why == ["min_eligible"] and st["n_exclusive"][0] == 8


def test_draw_classes_deterministic_rejects_and_valid():
    # 8 units: 0..5 clean disjoint blocks of 30; units 6,7 are near-duplicates of unit 0 (jaccard ~ 1)
    blocks = [range(30 * u, 30 * u + 30) for u in range(6)] + [range(0, 30), range(0, 30)]
    M = membership(blocks, 180)
    names = [f"concept{u}x" for u in range(8)]      # distinct, non-contained names
    a = AC.draw_classes(7, M, names, range(8), RULES)
    b = AC.draw_classes(7, M, names, range(8), RULES)
    assert a == b                                   # same seed stream -> same draw, same rejection count
    assert len(set(a["units"])) == 3 and a["attempts"] == a["n_rejected"] + 1
    assert len({0, 6, 7} & set(a["units"])) <= 1    # never two near-duplicates
    seen_rej = sum(AC.draw_classes(s, M, names, range(8), RULES)["n_rejected"] for s in range(30))
    assert seen_rej > 0                             # the rejection path is actually exercised
    with pytest.raises(RuntimeError):
        AC.draw_classes(0, M, names, [0, 6, 7], dict(RULES, max_draw_attempts=20))


def test_name_ok_and_build_units_merge():
    assert AC.name_ok("Hiking") and not AC.name_ok('He said "x"') and not AC.name_ok("a}b") and not AC.name_ok("x" * 41)
    u = AC.build_units(["Hiking", "hiking", "Cooking", "Tone X"], ["d1", "d2", "d3", "d4"], np.arange(4),
                       [10, 30, 5, 99], [True, True, True, False])
    assert [x["name"] for x in u] == ["hiking", "Cooking"] and u[0]["ids"] == [0, 1] and u[0]["count"] == 40


def test_doc_bucket_is_document_level_and_splits_roughly_as_specified():
    docs = np.arange(20000)
    b = AC.doc_bucket(docs, 0)
    assert (AC.doc_bucket(docs, 0) == b).all() and (AC.doc_bucket(docs, 1) != b).any()
    sp_ = AC.split_of_bucket(b)
    fr = np.bincount(sp_) / len(sp_)
    assert fr[0] == pytest.approx(0.10, abs=0.02) and fr[1] == pytest.approx(0.40, abs=0.02) and fr[2] == pytest.approx(0.50, abs=0.02)
    # chunks of one document: same doc id -> same bucket
    d2 = np.array([5, 5, 5, 9, 9])
    bb = AC.doc_bucket(d2, 3)
    assert bb[0] == bb[1] == bb[2] and bb[3] == bb[4]


def test_prompt_head_formats_with_braces_in_exemplars():
    names = ["Hiking", "Cooking techniques"]
    head = AC.build_prompt_head(names, [("Use {x} and } then\n go", 1), ("Walk up the hill", 0)])
    p = head.format("DOC TEXT {with} braces")
    assert p.endswith('{"DOC TEXT {with} braces":')
    assert "{ Use {x} and } then go:Cooking techniques}" in p and "{ Walk up the hill:Hiking}" in p
    assert p.startswith("Choose from one of these categories: Hiking, Cooking techniques. Be careful")


def test_comp_by_class_and_drop_matrix():
    labels = [1, 1, 2, 2, 2, 3]
    base = [1, 1, 1, 1, 0, 1]
    now = [1, 0, 1, 0, 0, 1]
    r = AC.comp_by_class(labels, base, now)
    assert r["1"] == dict(n=2, base_acc=1.0, acc=0.5, drop=0.5)
    assert r["2"]["n"] == 3 and r["2"]["drop"] == pytest.approx(round(2 / 3 - 1 / 3, 4))
    assert r["3"]["drop"] == 0.0
    concepts = {"0": {"h_op": {"comp_by_class": r}}, "1": {"h_op": {}}}
    D = AC.drop_matrix(concepts, "h_op", 4)
    assert D[0, 1] == 0.5 and D[0, 3] == 0.0 and np.isnan(D[0, 0]) and np.isnan(D[1]).all()


def test_set_dataset_atlas_lazy_config_and_db14_unchanged(tmp_path, monkeypatch):
    sd = tmp_path / "s1"
    sd.mkdir()
    (sd / "cfg.json").write_text(json.dumps(dict(classes=["A", "B"], dataset_name="local:atlas_s1", dataset_split="train",
                                                 text_column="text", label_column="label", split_column="split",
                                                 local_parquet="data.parquet", prompt_head='x {{"{}":', max_new=9)))
    monkeypatch.setenv("GEOAE_ATLAS_CLASSES_DIR", str(tmp_path))
    monkeypatch.setattr(shared, "DATASET_CONFIGS", dict(shared.DATASET_CONFIGS))
    cfg = shared.set_dataset("atlas_s1")
    assert shared.CLASSES == ["A", "B"] and shared.MAX_NEW == 9 and cfg["local_parquet"] == str(sd / "data.parquet")
    assert shared.build_prompt("T") == 'x {"T":'
    with pytest.raises(FileNotFoundError):
        shared.set_dataset("atlas_s2")
    cfg = shared.set_dataset("db14")
    assert shared.MAX_NEW == 6 and len(shared.CLASSES) == 14 and "local_parquet" not in cfg and "split_column" not in cfg
    shared.set_dataset("biasbios")
    assert shared.MAX_NEW == 6 and len(shared.CLASSES) == 28


def _analysis():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("pacia", Path(__file__).resolve().parents[2] / "experiments/paper/analysis/paper_atlas_class_interventions.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_collateral_structure_statistics_on_synthetic_matrices():
    A = _analysis()
    K = 8
    rng = np.random.default_rng(0)
    cos = rng.uniform(0, 1, (K, K)); cos = (cos + cos.T) / 2
    rel = dict(minilm_cos=cos.tolist(), cooc_pmi=(-cos).tolist())
    concepts = {}
    for c in range(K):
        oth = [o for o in range(K) if o != c]
        # h: collateral proportional to relatedness; z: all collateral on ONE other class (concentrated, unrelated order)
        h = {str(o): dict(n=10, base_acc=1.0, acc=1 - cos[c, o], drop=float(cos[c, o])) for o in oth}
        z = {str(o): dict(n=10, base_acc=1.0, acc=1.0, drop=0.0) for o in oth}
        z[str(oth[0])]["drop"] = 0.9
        concepts[str(c)] = {"h_op": dict(comp_by_class=h), "z_op": {}, "z_op2": {}}
        concepts[str(c)]["z_op"] = dict(comp_by_class=z)
    # key naming used by the tool is h_<key>/z_<key>
    for c in concepts.values():
        c["h_k"], c["z_k"] = c.pop("h_op"), c.pop("z_op"); c.pop("z_op2")
    draws = [dict(seed=0, res=dict(concepts=concepts), rel=rel)]
    out = A.collateral_structure(draws, ["k"])["k"]
    assert out["minilm_cos"]["rho_h"]["mean"] == pytest.approx(1.0)        # h collateral ranks exactly like relatedness
    assert out["cooc_pmi"]["rho_h"]["mean"] == pytest.approx(-1.0)
    ts = out["top_class_share"]
    assert ts["z"]["mean"] == pytest.approx(1.0) and ts["h"]["mean"] < 0.4 and ts["uniform_share"] == pytest.approx(1 / 7)
    assert A.spearman([1, 2, 3, 4], [1, 1, 1, 1]) != A.spearman([1, 2, 3, 4], [1, 2, 3, 4])   # constant -> nan
