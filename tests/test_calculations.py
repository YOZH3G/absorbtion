import unittest

import numpy as np

from calculations import (
    IMPULSE,
    RAMP,
    RECTANGLE,
    STEP,
    calculate_xna,
    calculate_xog,
    combine_fractions,
    disturbance_profile,
    first_order_response,
    transition_metrics,
)


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

    def test_negative_disturbances_reduce_the_result(self):
        self.assertAlmostEqual(combine_fractions(-0.1, -0.2), -0.28)
        self.assertAlmostEqual(
            calculate_xog(self.gg, self.xg, self.gog, k_xg=-0.1, k_gg=-0.2),
            0.576,
        )


class DynamicModelTests(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0.0, 20.0, 201)

    def test_step_profile_starts_at_requested_time(self):
        profile = disturbance_profile(self.time, STEP, start_time=5.0, duration=2.0)
        self.assertTrue(np.all(profile[self.time < 5.0] == 0.0))
        self.assertTrue(np.all(profile[self.time >= 5.0] == 1.0))

    def test_impulse_is_a_bounded_temporary_half_sine(self):
        profile = disturbance_profile(self.time, IMPULSE, start_time=5.0, duration=4.0)
        self.assertTrue(np.all(profile[(self.time < 5.0) | (self.time > 9.0)] == 0.0))
        self.assertAlmostEqual(profile[np.argmin(np.abs(self.time - 7.0))], 1.0)

    def test_rectangle_returns_to_zero_after_duration(self):
        profile = disturbance_profile(self.time, RECTANGLE, start_time=5.0, duration=4.0)
        self.assertTrue(np.all(profile[(self.time >= 5.0) & (self.time < 9.0)] == 1.0))
        self.assertTrue(np.all(profile[self.time >= 9.0] == 0.0))

    def test_ramp_reaches_and_keeps_full_amplitude(self):
        profile = disturbance_profile(self.time, RAMP, start_time=5.0, duration=4.0)
        self.assertAlmostEqual(profile[np.argmin(np.abs(self.time - 7.0))], 0.5)
        self.assertTrue(np.all(profile[self.time >= 9.0] == 1.0))

    def test_first_order_step_matches_analytical_solution(self):
        time = np.linspace(0.0, 10.0, 101)
        profile = disturbance_profile(time, STEP, start_time=2.0, duration=1.0)
        response = first_order_response(
            time,
            baseline=10.0,
            target=10.0 + 10.0 * profile,
            time_constant=2.0,
        )
        expected = 10.0 + 10.0 * (1.0 - np.exp(-(time - 2.0) / 2.0))
        expected[time < 2.0] = 10.0
        np.testing.assert_allclose(response, expected, rtol=0.0, atol=1e-12)

    def test_dead_time_delays_the_response(self):
        profile = disturbance_profile(self.time, STEP, start_time=2.0, duration=1.0)
        response = first_order_response(
            self.time,
            baseline=10.0,
            target=10.0 + 10.0 * profile,
            time_constant=2.0,
            delay=3.0,
        )
        self.assertTrue(np.all(response[self.time < 5.0] == 10.0))
        self.assertGreater(response[self.time > 5.0][0], 10.0)

    def test_transition_metrics_for_settled_step(self):
        time = np.linspace(0.0, 20.0, 201)
        target = np.where(time >= 2.0, 20.0, 10.0)
        response = first_order_response(time, 10.0, target, time_constant=2.0)

        metrics = transition_metrics(time, response, target, baseline=10.0)

        self.assertEqual(metrics["initial_value"], 10.0)
        self.assertEqual(metrics["steady_state"], 20.0)
        self.assertAlmostEqual(metrics["maximum_deviation"], 10.0, places=2)
        self.assertGreater(metrics["relative_deviation"], 99.9)
        self.assertGreater(metrics["settling_time"], 7.0)
        self.assertLess(metrics["settling_time"], 8.2)
        self.assertEqual(metrics["static_error"], -10.0)

    def test_transition_metrics_report_unsettled_response(self):
        time = np.linspace(0.0, 3.0, 31)
        target = np.where(time >= 2.0, 20.0, 10.0)
        response = first_order_response(time, 10.0, target, time_constant=10.0)

        metrics = transition_metrics(time, response, target, baseline=10.0)

        self.assertIsNone(metrics["settling_time"])

    def test_transition_metrics_for_temporary_disturbance_return_to_baseline(self):
        time = np.linspace(0.0, 30.0, 301)
        profile = disturbance_profile(time, RECTANGLE, start_time=2.0, duration=3.0)
        target = 10.0 + 5.0 * profile
        response = first_order_response(time, 10.0, target, time_constant=2.0)

        metrics = transition_metrics(time, response, target, baseline=10.0)

        self.assertEqual(metrics["steady_state"], 10.0)
        self.assertGreater(metrics["maximum_deviation"], 0.0)
        self.assertIsNotNone(metrics["settling_time"])
        self.assertEqual(metrics["static_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
