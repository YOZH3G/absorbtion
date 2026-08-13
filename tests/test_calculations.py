import unittest

from calculations import calculate_xna, calculate_xog, combine_fractions


class CalculationTests(unittest.TestCase):
    def setUp(self):
        self.gg = 1000
        self.xg = 0.5
        self.gog = 625
        self.ga = 468000
        self.gna = 7800
        self.xa = 0.5

    def test_zero_disturbance_preserves_initial_values(self):
        self.assertAlmostEqual(calculate_xog(self.gg, self.xg, self.gog), 0.8)
        self.assertAlmostEqual(calculate_xna(self.ga, self.gna, self.xa), 30)

    def test_single_disturbance_is_a_fractional_increase(self):
        self.assertAlmostEqual(calculate_xog(self.gg, self.xg, self.gog, k_xg=0.1), 0.88)
        self.assertAlmostEqual(calculate_xna(self.ga, self.gna, self.xa, k_ga=0.1), 33)

    def test_two_disturbances_are_compounded(self):
        self.assertAlmostEqual(combine_fractions(0.1, 0.1), 0.21)
        self.assertAlmostEqual(
            calculate_xog(self.gg, self.xg, self.gog, k_xg=0.1, k_gg=0.1),
            0.968,
        )
        self.assertAlmostEqual(
            calculate_xna(self.ga, self.gna, self.xa, k_xa=0.1, k_ga=0.1),
            36.3,
        )


if __name__ == "__main__":
    unittest.main()
