import unittest

import numpy as np

from deco_humanlm.rewards import DIMENSIONS, aggregate_rewards


def score_map(*rows):
    matrix = np.asarray(rows, dtype=float)
    return {name: matrix[:, index] for index, name in enumerate(DIMENSIONS)}


class RewardAggregationTests(unittest.TestCase):
    def test_deco_is_equal_weight_mean(self):
        scores = score_map([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        np.testing.assert_allclose(aggregate_rewards(scores, "deco"), [0.5])

    def test_pdn_balances_different_scales(self):
        scores = score_map(
            [0.4, 0.2, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.4, 0.5, 0.5, 0.5, 0.5],
            [0.6, 0.6, 0.5, 0.5, 0.5, 0.5],
        )
        np.testing.assert_allclose(
            aggregate_rewards(scores, "pdn"),
            [0.0, 2.0, 4.0],
            atol=2e-5,
        )

    def test_pdn_rejects_single_rollout(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            aggregate_rewards(score_map([0.5] * 6), "pdn")

    def test_rejects_out_of_range_scores(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            aggregate_rewards(score_map([1.1] * 6), "deco")


if __name__ == "__main__":
    unittest.main()

