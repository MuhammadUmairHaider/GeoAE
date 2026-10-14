"""Independent numerical examples for intervention-curve reporting (stdlib only)."""
import unittest

from eval_out.dataset_neuronlens_summary import first_crossing, summarize


def measurement(target, sibling, random):
    return {group: {"dnll": value} for group, value in
            zip(("target", "sibling", "random"), (target, sibling, random))}


class MatchingChecks(unittest.TestCase):
    def test_first_strength_crossing_with_nonmonotone_target(self):
        points = [(0.1, measurement(0.4, 0.2, 0.1)),
                  (0.3, measurement(0.1, 0.8, 0.9)),
                  (0.5, measurement(0.8, 0.3, 0.2))]
        result = first_crossing(points, 0.2)
        self.assertEqual(result["segment"], [0.0, 0.1])
        self.assertAlmostEqual(result["sibling"], 0.1)
        self.assertAlmostEqual(result["random"], 0.05)

    def test_negative_initial_effect_preserves_strength_adjacency(self):
        result = first_crossing([(0.1, measurement(-0.2, 0.2, 0.1)),
                                 (0.3, measurement(0.2, 0.4, 0.3))], 0.1)
        self.assertAlmostEqual(result["sibling"], 0.35)
        self.assertAlmostEqual(result["strength"], 0.25)

    def test_no_extrapolation(self):
        self.assertIsNone(first_crossing([(0.1, measurement(0.2, 1.0, 1.0))], 0.3))

    def test_complete_pairs_only_and_reachability(self):
        cases = []
        for pair, target, maximum in [(0, "a", 1.0), (0, "b", 1.0), (1, "c", 1.0), (1, "d", -1.0)]:
            cases.append({"pair_id": pair, "T": target, "S": "sibling", "res": {
                arm: [{"p": 0.3, "metrics": measurement(maximum, 0.4, 0.2)}]
                for arm in ("base", "ae")}})
        result = summarize(cases, ["base", "ae"])
        self.assertEqual(result["matched"]["n_complete_pairs"], 1)
        self.assertEqual(result["matched"]["n_complete_cases"], 2)
        self.assertEqual(result["matched"]["n_incomplete_cases"], 1)
        self.assertEqual(len(result["skipped"]), 1)
        self.assertEqual(result["fixed"]["0.5"]["reach_by_arm"], {"base": 3, "ae": 3})
        self.assertAlmostEqual(result["matched"]["arms"]["base"]["target"], 0.5)
        self.assertAlmostEqual(result["matched"]["arms"]["base"]["sibling"], 0.2)


if __name__ == "__main__":
    unittest.main()
