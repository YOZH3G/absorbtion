import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from app.experiment_control import compare_pi, validate_step
from app.identification import identify_step, validate_step_quality


def reference(gain=.4):
    time = np.linspace(0, 80, 401)
    signal = np.where(time < 10, .3, .5)
    output = .2 + gain * .2 * -np.expm1(-np.maximum(time - 12, 0) / 8)
    return identify_step(time, signal, output)


class ExperimentControlTests(unittest.TestCase):
    def test_long_baseline_cannot_hide_absent_response(self):
        time = np.r_[np.linspace(0, 9999, 101), np.linspace(10000, 10080, 401)]
        signal = np.where(time < 10000, .3, .5)
        record = validate_step(time, signal, np.full_like(time, .2),
                               dict(gain=.4, time_constant=8, delay=2))
        self.assertLess(record["normalized_rmse"], .1)
        self.assertGreater(record["quality"]["response_normalized_rmse"], .8)
        self.assertFalse(record["quality"]["error_acceptable"])
        self.assertIn("Ошибка отклика", record["warning"])

    def test_clustered_samples_cannot_hide_transition_gap(self):
        time = np.r_[np.linspace(0, 10, 51), np.linspace(13, 14, 11), np.linspace(40, 80, 41)]
        signal = np.where(time < 10, .3, .5)
        output = .2 + .08 * -np.expm1(-np.maximum(time-12, 0)/8)
        with self.assertRaisesRegex(ValueError, "редкие"):
            validate_step(time, signal, output, dict(gain=.4, time_constant=8, delay=2))

    def test_quality_zero_noise_bias_and_irregular_time(self):
        time = np.r_[np.linspace(0, 10, 51), np.linspace(10.2, 80, 287)]
        signal = np.where(time < 10, .3, .5)
        output = .2 + .08 * -np.expm1(-np.maximum(time-12, 0)/8)
        parameters = dict(gain=.4, time_constant=8, delay=2)
        exact = validate_step(time, signal, output, parameters)
        self.assertLess(exact["quality"]["response_rmse"], 1e-12)
        self.assertTrue(exact["quality"]["noise_adequate"])
        self.assertGreater(exact["quality"]["duration_T"], 3)
        validate_step_quality(exact)
        for key in ("rmse", "normalized_rmse", "gain", "baseline", "delay"):
            with self.assertRaises(ValueError):
                validate_step_quality(exact | {key: False})
        biased = validate_step(time, signal, output + .02*(time>=10), parameters)
        self.assertAlmostEqual(biased["quality"]["response_bias"], .02, places=12)
        self.assertAlmostEqual(biased["quality"]["response_rmse"], .02, places=12)
        self.assertFalse(biased["quality"]["error_acceptable"])

    def test_unresolved_validation_and_malformed_time_are_rejected(self):
        fit = reference()
        time = np.r_[np.linspace(0, 10, 51), 10.2, 45, np.linspace(45.2, 120, 201)]
        signal = np.where(time < 10, .3, .5)
        output = .2 + .08 * -np.expm1(-np.maximum(time-12, 0)/8)
        with self.assertRaisesRegex(ValueError, "редкие"):
            validate_step(time, signal, output, fit)
        for updates in ({"time": fit["time"][::-1]}, {"time": fit["time"][:-1]}, {"time": fit["time"] * float("nan")}):
            with self.assertRaises(ValueError):
                compare_pi(fit | updates, "η, доля", "Xна, доля", .3, 80, 1, 8)

    def test_time_at_actuator_bound_includes_zero_gain_baseline(self):
        fit = reference()
        fit["input"] = fit["input"] - .3
        result = compare_pi(fit, "η, доля", "Xна, доля", .3, 80, 0, 8)
        self.assertEqual(result["runs"][0]["metrics"]["saturation_duration"], 80)

    def test_independent_validation_keeps_parameters_and_uses_new_baseline_and_step(self):
        fit = reference()
        original = copy.deepcopy(fit)
        time = np.r_[np.linspace(0, 15, 51), np.linspace(15.2, 120, 287)]
        signal = np.where(time < 15, .4, .7)
        output = .35 + fit["gain"] * .3 * -np.expm1(-np.maximum(time-15-fit["delay"], 0)/fit["time_constant"])
        with patch("app.identification.identify_step", side_effect=AssertionError("No refitting")):
            checked = validate_step(time, signal, output, fit)
        self.assertLess(checked["rmse"], 1e-12)
        for key in ("gain", "time_constant", "delay"):
            self.assertEqual(checked[key], original[key])
        np.testing.assert_array_equal(fit["model"], original["model"])
        bad = validate_step(time, signal, output + .1*(time>=15), fit)
        self.assertGreater(bad["normalized_rmse"], .1)
        self.assertTrue(bad["warning"])

    def test_pi_positive_negative_gain_bounds_delay_and_replay(self):
        for gain, setpoint in ((.4, .35), (-.4, .15)):
            fit = reference(gain)
            result = compare_pi(fit, "η, доля", "Xна, доля", setpoint, 160, 1, 8)
            restored = json.loads(json.dumps(result))
            for run in restored["runs"]:
                control = np.array(run["control"])
                self.assertTrue(np.all((control >= 0) & (control <= 1)))
                self.assertLess(abs(run["metrics"]["final_error"]), .001)
                np.testing.assert_allclose(np.array(run["output"])[np.array(run["time"]) <= fit["delay"]], fit["baseline"], atol=1e-14)
                self.assertIsNotNone(run["metrics"]["settling_time"])
            replay = compare_pi(fit, "η, доля", "Xна, доля", setpoint, 160, 1, 8)
            self.assertEqual(restored, replay)

    def test_unreachable_setpoint_is_not_reported_settled(self):
        result = compare_pi(reference(), "η, доля", "Xна, доля", .9, 160, 8, 1)
        for run in result["runs"]:
            self.assertFalse(run["metrics"]["reachable"])
            self.assertIsNone(run["metrics"]["settling_time"])
            self.assertGreater(run["metrics"]["saturation_duration"], 0)
            self.assertTrue(np.isfinite(run["output"]).all())

    def test_unknown_signals_and_invalid_settings_are_rejected(self):
        fit = reference()
        for role in ("Не определён", "Состав", "Расход"):
            with self.assertRaises(ValueError):
                compare_pi(fit, role, "Xна, доля", .3, 80, 1, 8)
        for changes in ({"gain": 0}, {"time_constant": -1}, {"delay": float("nan")}):
            with self.assertRaises(ValueError):
                compare_pi(fit | changes, "η, доля", "Xна, доля", .3, 80, 1, 8)
        with self.assertRaises(ValueError):
            compare_pi(fit, "η, доля", "Xна, доля", .3, 80, 1, 0)


if __name__ == "__main__":
    unittest.main()
