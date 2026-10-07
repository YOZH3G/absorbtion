import unittest

import numpy as np

from app.calculations import (
    CONTROLLER_TYPES,
    IMPULSE,
    RAMP,
    RECTANGLE,
    STEP,
    combine_fractions,
    disturbance_profile,
    first_order_response,
    transition_metrics,
    tune_controller_parameters,
)


class CalculationTests(unittest.TestCase):
    def test_combined_relative_change(self):
        self.assertAlmostEqual(combine_fractions(0.1, 0.1), 0.21)
        self.assertAlmostEqual(combine_fractions(-0.1, -0.2), -0.28)


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

    def test_settling_time_is_absolute_position_on_time_axis(self):
        time = np.linspace(0.0, 30.0, 301)
        target = np.where(time >= 10.0, 20.0, 10.0)
        response = first_order_response(time, 10.0, target, time_constant=2.0)

        metrics = transition_metrics(time, response, target, baseline=10.0)

        self.assertGreater(metrics["settling_time"], 15.0)
        self.assertLess(metrics["settling_time"], 16.2)
        self.assertGreater(metrics["settling_time"] - 10.0, 5.0)
        self.assertLess(metrics["settling_time"] - 10.0, 6.2)

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


class TuningTests(unittest.TestCase):
    def test_tuning_rejects_invalid_object_parameters(self):
        for time_constant, delay in ((0, 0), (10, -1), (float("nan"), 2)):
            with self.assertRaises(ValueError):
                tune_controller_parameters("PI", time_constant, delay)

    def test_all_controller_types_use_their_components(self):
        for kind in CONTROLLER_TYPES:
            tuning = tune_controller_parameters(kind, 10, 2)
            self.assertEqual(tuning["integral_time"] is not None, "I" in kind)
            self.assertEqual(tuning["derivative_time"] is not None, "D" in kind)


if __name__ == "__main__":
    unittest.main()
