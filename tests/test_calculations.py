import unittest

import numpy as np

from calculations import (
    CONTROLLER_TYPES,
    IMPULSE,
    RAMP,
    RECTANGLE,
    STEP,
    calculate_xna,
    calculate_xog,
    combine_fractions,
    controller_response,
    controller_steady_state,
    disturbance_profile,
    first_order_response,
    pi_control_response,
    transition_metrics,
    tune_controller_parameters,
    tune_pi_parameters,
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


class PIControllerTests(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0.0, 100.0, 1001)
        self.baseline = 10.0

    def simulate(self, target, **overrides):
        parameters = {
            "time_constant": 5.0,
            "proportional_gain": 2.0,
            "integral_time": 10.0,
            "control_limit": 20.0,
            "setpoint": self.baseline,
        }
        parameters.update(overrides)
        return pi_control_response(
            self.time,
            self.baseline,
            target,
            **parameters,
        )

    def test_zero_disturbance_keeps_setpoint(self):
        target = np.full_like(self.time, self.baseline)

        response, error, control = self.simulate(target)

        np.testing.assert_allclose(response, self.baseline)
        np.testing.assert_allclose(error, 0.0)
        np.testing.assert_allclose(control, 0.0)

    def test_pi_controller_reduces_steady_disturbance_error(self):
        target = np.where(self.time >= 5.0, 15.0, self.baseline)
        open_response = first_order_response(
            self.time,
            self.baseline,
            target,
            time_constant=5.0,
        )

        controlled, error, control = self.simulate(target)

        self.assertLess(abs(error[-1]), abs(self.baseline - open_response[-1]))
        self.assertAlmostEqual(controlled[-1], self.baseline, delta=0.005)
        self.assertLess(control[-1], 0.0)

    def test_control_signal_respects_limit(self):
        target = np.where(self.time >= 5.0, 50.0, self.baseline)

        _response, _error, control = self.simulate(target, control_limit=1.5)

        self.assertLessEqual(np.max(np.abs(control)), 1.5)
        self.assertTrue(np.any(np.isclose(np.abs(control), 1.5)))

    def test_zero_gain_matches_uncontrolled_object(self):
        target = np.where(self.time >= 5.0, 15.0, self.baseline)
        open_response = first_order_response(
            self.time,
            self.baseline,
            target,
            time_constant=5.0,
        )

        controlled, _error, control = self.simulate(target, proportional_gain=0.0)

        np.testing.assert_allclose(controlled, open_response)
        np.testing.assert_allclose(control, 0.0)

    def test_object_delay_postpones_controlled_response(self):
        target = np.where(self.time >= 5.0, 15.0, self.baseline)

        response, _error, _control = self.simulate(target, delay=3.0)

        self.assertTrue(np.all(response[self.time <= 8.0] == self.baseline))
        self.assertGreater(response[self.time > 8.0][0], self.baseline)

    def test_rejects_invalid_controller_parameters(self):
        target = np.full_like(self.time, self.baseline)

        with self.assertRaises(ValueError):
            self.simulate(target, proportional_gain=-1.0)
        with self.assertRaises(ValueError):
            self.simulate(target, integral_time=0.0)
        with self.assertRaises(ValueError):
            self.simulate(target, control_limit=0.0)
        with self.assertRaises(ValueError):
            self.simulate(target, delay=-1.0)

    def test_imc_tuning_without_delay(self):
        tuning = tune_pi_parameters(time_constant=10.0, delay=0.0)

        self.assertEqual(tuning["closed_loop_time"], 5.0)
        self.assertEqual(tuning["proportional_gain"], 2.0)
        self.assertEqual(tuning["integral_time"], 10.0)

    def test_imc_tuning_reduces_gain_when_delay_grows(self):
        without_delay = tune_pi_parameters(time_constant=10.0, delay=0.0)
        with_delay = tune_pi_parameters(time_constant=10.0, delay=8.0)

        self.assertLess(with_delay["proportional_gain"], without_delay["proportional_gain"])
        self.assertEqual(with_delay["integral_time"], 10.0)

    def test_imc_tuned_controller_reduces_final_error(self):
        target = np.where(self.time >= 5.0, 15.0, self.baseline)
        open_response = first_order_response(
            self.time,
            self.baseline,
            target,
            time_constant=5.0,
            delay=2.0,
        )
        tuning = tune_pi_parameters(time_constant=5.0, delay=2.0)

        _controlled, error, _control = self.simulate(
            target,
            time_constant=5.0,
            delay=2.0,
            proportional_gain=tuning["proportional_gain"],
            integral_time=tuning["integral_time"],
        )

        self.assertLess(abs(error[-1]), abs(self.baseline - open_response[-1]))

    def test_imc_tuning_rejects_invalid_object_parameters(self):
        with self.assertRaises(ValueError):
            tune_pi_parameters(time_constant=0.0, delay=0.0)
        with self.assertRaises(ValueError):
            tune_pi_parameters(time_constant=10.0, delay=-1.0)


class ControllerTypeTests(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(0.0, 100.0, 1001)
        self.baseline = 10.0
        self.target = np.where(self.time >= 5.0, 15.0, self.baseline)

    def simulate(self, controller_type, **overrides):
        parameters = {
            "controller_gain": 2.0,
            "integral_time": 10.0,
            "derivative_time": 1.0,
            "control_limit": 20.0,
            "setpoint": self.baseline,
            "delay": 0.0,
        }
        parameters.update(overrides)
        return controller_response(
            self.time,
            self.baseline,
            self.target,
            time_constant=5.0,
            controller_type=controller_type,
            **parameters,
        )

    def test_all_requested_controller_types_produce_finite_signals(self):
        self.assertEqual(CONTROLLER_TYPES, ("P", "PI", "PID", "PD"))

        for controller_type in CONTROLLER_TYPES:
            with self.subTest(controller_type=controller_type):
                response, error, control = self.simulate(controller_type)
                self.assertTrue(np.all(np.isfinite(response)))
                self.assertTrue(np.all(np.isfinite(error)))
                self.assertTrue(np.all(np.isfinite(control)))
                self.assertLessEqual(np.max(np.abs(control)), 20.0)

    def test_integral_component_reduces_final_error(self):
        _p_response, p_error, _p_control = self.simulate("P")
        _pi_response, pi_error, _pi_control = self.simulate("PI")

        self.assertLess(abs(pi_error[-1]), abs(p_error[-1]))

    def test_rejects_unknown_controller_type(self):
        with self.assertRaises(ValueError):
            self.simulate("UNKNOWN")

    def test_pid_matches_nonuniform_delay_reference(self):
        time = np.array([0.0, 0.2, 0.7, 1.4, 2.5, 4.0, 6.0, 9.0])
        target = np.where(time >= 0.7, 1.1, 0.8)

        response, error, control = controller_response(
            time,
            0.8,
            target,
            2.3,
            "PID",
            1.7,
            3.1,
            0.4,
            0.35,
            0.8,
            0.65,
        )

        np.testing.assert_allclose(response, [
            0.8, 0.8, 0.8, 0.8, 0.9140418580530887, 0.9513181886296848,
            0.8809031708104486, 0.8542955308366388,
        ])
        np.testing.assert_allclose(error, [
            0.0, 0.0, 0.0, 0.0, -0.11404185805308864, -0.1513181886296847,
            -0.08090317081044851, -0.05429553083663874,
        ])
        np.testing.assert_allclose(control, [
            0.0, 0.0, 0.0, 0.0, -0.26436976185034183, -0.2741395238651876,
            -0.2466930492009278, -0.21937010224326123,
        ])

    def test_steady_state_depends_on_controller_components(self):
        p_steady = controller_steady_state(15.0, 10.0, "P", 2.0, 20.0)
        pi_steady = controller_steady_state(15.0, 10.0, "PI", 2.0, 20.0)
        pd_steady = controller_steady_state(15.0, 10.0, "PD", 2.0, 20.0)

        self.assertAlmostEqual(p_steady, 35.0 / 3.0)
        self.assertEqual(pi_steady, 10.0)
        self.assertAlmostEqual(pd_steady, p_steady)

    def test_integral_controller_respects_steady_control_limit(self):
        steady = controller_steady_state(15.0, 10.0, "PI", 2.0, 1.0)

        self.assertEqual(steady, 14.0)

    def test_autotuning_returns_only_parameters_used_by_each_type(self):
        expected_terms = {
            "P": (False, False),
            "PI": (True, False),
            "PID": (True, True),
            "PD": (False, True),
        }

        for controller_type, (uses_integral, uses_derivative) in expected_terms.items():
            with self.subTest(controller_type=controller_type):
                tuning = tune_controller_parameters(controller_type, 10.0, 3.0)
                self.assertEqual(tuning["integral_time"] is not None, uses_integral)
                self.assertEqual(tuning["derivative_time"] is not None, uses_derivative)

    def test_pid_autotuning_uses_faster_target_and_delay_derivative(self):
        pi_tuning = tune_controller_parameters("PI", 10.0, 3.0)
        pid_tuning = tune_controller_parameters("PID", 10.0, 3.0)

        self.assertLess(pid_tuning["closed_loop_time"], pi_tuning["closed_loop_time"])
        self.assertEqual(pid_tuning["derivative_time"], 1.0)
        self.assertGreater(pid_tuning["proportional_gain"], pi_tuning["proportional_gain"])

    def test_autotuned_controllers_reduce_final_error(self):
        open_response = first_order_response(
            self.time,
            self.baseline,
            self.target,
            time_constant=5.0,
            delay=2.0,
        )
        open_error = abs(self.baseline - open_response[-1])

        for controller_type in CONTROLLER_TYPES:
            with self.subTest(controller_type=controller_type):
                tuning = tune_controller_parameters(controller_type, 5.0, 2.0)
                _response, error, _control = self.simulate(
                    controller_type,
                    delay=2.0,
                    controller_gain=tuning["proportional_gain"],
                    integral_time=tuning["integral_time"] or 1.0,
                    derivative_time=tuning["derivative_time"] or 0.0,
                )
                self.assertLess(abs(error[-1]), open_error)


if __name__ == "__main__":
    unittest.main()
