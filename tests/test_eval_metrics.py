from itertools import combinations
import unittest

from evaluator import pass_at_k, pass_hat_k


class ReliabilityMetricTests(unittest.TestCase):
    def test_metrics_match_exhaustive_sampling_without_replacement(self):
        for n in range(1, 9):
            for correct in range(n + 1):
                outcomes = [True] * correct + [False] * (n - correct)
                for k in range(1, n + 1):
                    samples = list(combinations(outcomes, k))
                    with self.subTest(n=n, correct=correct, k=k):
                        self.assertAlmostEqual(pass_at_k(n, correct, k), sum(any(s) for s in samples) / len(samples))
                        self.assertAlmostEqual(pass_hat_k(n, correct, k), sum(all(s) for s in samples) / len(samples))

    def test_more_attempts_raise_capability_but_lower_reliability(self):
        self.assertAlmostEqual(pass_at_k(10, 6, 1), 0.6)
        self.assertAlmostEqual(pass_hat_k(10, 6, 1), 0.6)
        self.assertAlmostEqual(pass_at_k(10, 6, 3), 29 / 30)
        self.assertAlmostEqual(pass_hat_k(10, 6, 3), 1 / 6)

    def test_invalid_counts_and_unsupported_k_are_rejected(self):
        for args in ((0, 0, 1), (3, 4, 1), (3, -1, 1), (3, 1, 0), (3, 1, 4),
                     (-1, 0, 1), (3.0, 1, 1), (3, True, 1), (3, 1, False), ("3", 1, 1)):
            for metric in (pass_at_k, pass_hat_k):
                with self.subTest(metric=metric.__name__, args=args), self.assertRaises(ValueError):
                    metric(*args)
