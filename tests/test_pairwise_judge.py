import random

from geoae.interp import pairwise_judge as P


def test_score_for_second_maps_both_orders():
    # order 0: second space is B; order 1: second space is A
    assert P.score_for_second("B", second_is_a=False) == 1.0
    assert P.score_for_second("A", second_is_a=False) == 0.0
    assert P.score_for_second("A", second_is_a=True) == 1.0
    assert P.score_for_second("tie", second_is_a=True) == 0.5
    assert P.score_for_second("neither", second_is_a=False) == 0.5
    assert P.score_for_second(None, second_is_a=False) is None


def test_parse_answer_validates_fields():
    txt = ('noise {"A": {"label": "x", "coherence": 7, "abstraction": "5", "level": "Topic"}, '
           '"B": {"label": "y", "coherence": 11, "abstraction": 2, "level": "nonsense"}, '
           '"clearer_concept": "A", "more_semantic": "maybe"}')
    a = P.parse_answer(txt)
    assert a["clearer_concept"] == "A" and a["more_semantic"] is None
    assert a["A"]["coherence"] == 7 and a["A"]["abstraction"] == 5 and a["A"]["level"] == "topic"
    assert a["B"]["coherence"] is None and a["B"]["level"] is None
    assert P.parse_answer("no json here") is None


def test_fake_cluster_draws_from_distinct_clusters():
    space = {k: (100, [f"c{k}-{i}" for i in range(8)]) for k in range(20)}
    ex = P.fake_cluster(space, 8, random.Random(0))
    assert len({e.split("-")[0] for e in ex}) == 8


def _rec(kind, first, second, pair, order, answer):
    second_is_a = order == 1
    return {"kind": kind, "first": first, "second": second, "pair": pair, "order": order,
            "second_is_a": second_is_a, "space_a": second if second_is_a else first,
            "space_b": first if second_is_a else second,
            "answer": {"clearer_concept": answer, "more_semantic": answer,
                       "A": {"label": "", "coherence": 6.0, "abstraction": 4.0, "level": "syntax"},
                       "B": {"label": "", "coherence": 8.0, "abstraction": 7.0, "level": "topic"}}}


def test_summarise_win_rate_and_position_consistency():
    recs = []
    for i in range(10):                     # y wins every pair in both orders: B in order 0, A in order 1
        recs += [_rec("main", "x", "y", i, 0, "B"), _rec("main", "x", "y", i, 1, "A")]
    s = P.summarise(recs, ["x", "y"], n_boot=50, seed=0)
    w = s["pairings"]["x vs y"]["clearer_concept"]
    assert w["win"] == 1.0 and w["n"] == 10
    assert s["position"]["order_consistent"] == 1.0
    assert s["position"]["first_slot_win"] == 0.5
    # a judge that always says "A" has no preference once both orders are averaged
    recs = []
    for i in range(10):
        recs += [_rec("main", "x", "y", i, 0, "A"), _rec("main", "x", "y", i, 1, "A")]
    s = P.summarise(recs, ["x", "y"], n_boot=50, seed=0)
    assert s["pairings"]["x vs y"]["clearer_concept"]["win"] == 0.5
    assert s["position"]["first_slot_win"] == 1.0 and s["position"]["order_consistent"] == 0.0


def test_export_import_round_trip(tmp_path):
    import argparse
    import json
    spaces = {s: {k: (100, [f"{s}{k}-{i}" for i in range(8)]) for k in range(30)} for s in ("x", "y")}
    args = argparse.Namespace(seed=0, pairs=6, control_pairs=2, weight="usage", n_examples=8)
    calls = P.build_jobs(spaces, args)
    P.export_batches(calls, tmp_path / "b", tmp_path / "key.json", batch_size=5, seed=0)
    key = json.loads((tmp_path / "key.json").read_text())
    assert len(key) == len(calls) and "ex_a" not in key[0]
    batches = sorted((tmp_path / "b").glob("batch_*.md"))
    for f in batches:                       # no batch holds both orders of one pair
        ids = [line.split()[2] for line in f.read_text().splitlines() if line.startswith("=== ITEM")]
        orders = {k["id"]: (k["kind"], k["first"], k["pair"], k["order"]) for k in key}
        assert len({orders[i][3] for i in ids}) == 1
    # every item answered "B" -> the second space wins whenever it sits in B (order 0), loses in order 1
    lines = [json.dumps({"id": k["id"], "A": {}, "B": {}, "clearer_concept": "B", "more_semantic": "tie"})
             for k in key]
    (tmp_path / "answers_00.jsonl").write_text("\n".join(lines))
    recs = P.import_answers(tmp_path / "key.json", tmp_path)
    assert all(r["answer"] is not None for r in recs)
    s = P.summarise(recs, ["x", "y"], n_boot=20, seed=0)
    assert s["pairings"]["x vs y"]["clearer_concept"]["win"] == 0.5
