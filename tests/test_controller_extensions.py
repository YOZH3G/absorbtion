import unittest

import numpy as np

from app.calculations import DEFAULT_MODEL_VALUES
from app.controller_extensions import EXTENSION_DEFAULTS, extensions_from_form, normalize_extensions
from app.simulation import LEAN_GAS, RICH_ABSORBENT, run_simulation


DYNAMICS = dict(kind="step", start_time=10., effect_duration=10.,
                simulation_duration=100., time_constant=10., delay=2.)


def settings(chain=LEAN_GAS, kind="PID", **updates):
    return dict(controller_type=kind, controller_gain=5., integral_time=20.,
                derivative_time=1., control_limit=1.,
                setpoint=1 / 6 if chain == LEAN_GAS else .3, **updates)


class ControllerExtensionTests(unittest.TestCase):
    def test_disabled_extensions_preserve_all_base_signals_exactly(self):
        for chain in (LEAN_GAS, RICH_ABSORBENT):
            for kind in ("P", "PI", "PD", "PID"):
                base = settings(chain, kind)
                a = run_simulation(chain, DEFAULT_MODEL_VALUES, .03, 0, DYNAMICS, base)
                b = run_simulation(chain, DEFAULT_MODEL_VALUES, .03, 0, DYNAMICS,
                                   dict(base, **EXTENSION_DEFAULTS))
                for key in ("time", "final_response", "error", "control"):
                    np.testing.assert_array_equal(a[key], b[key])

    def test_noise_repeats_and_filter_reduces_command_variation(self):
        cfg = settings(noise_std=.0005, noise_seed=2026)
        def calculate(**updates):
            return run_simulation(LEAN_GAS, DEFAULT_MODEL_VALUES, .03, 0,
                                  DYNAMICS, dict(cfg, **updates))
        a, b, c = calculate(), calculate(), calculate(noise_seed=2027)
        np.testing.assert_array_equal(a["measurement"], b["measurement"])
        self.assertFalse(np.array_equal(a["measurement"], c["measurement"]))
        filtered = calculate(derivative_filter_time=.5)
        self.assertLess(np.mean(abs(np.diff(filtered["commanded_control"]))),
                        .5 * np.mean(abs(np.diff(a["commanded_control"]))))
        np.testing.assert_array_equal(a["error"], cfg["setpoint"] - a["final_response"])
        np.testing.assert_array_equal(a["measurement_error"], cfg["setpoint"] - a["measurement"])

    def test_actuator_rate_and_bounds_and_step_convergence(self):
        cfg = dict(settings(kind="PI"), controller_gain=2., setpoint=.1,
                   actuator_time_constant=5., actuator_rate_limit=.01)
        results = []
        for step in (.1, .05, .025):
            result = run_simulation(LEAN_GAS, DEFAULT_MODEL_VALUES, 0, 0,
                                    dict(DYNAMICS, delay=0), cfg, max_step=step)
            rate = abs(np.diff(result["control"])) / np.diff(result["time"])
            self.assertLessEqual(float(rate.max()), .01000000001)
            self.assertTrue(np.all((result["control"] >= 0) & (result["control"] <= 1)))
            results.append(result)
        self.assertLess(abs(results[-1]["control"][-1] - results[-2]["control"][-1]), .001)
        self.assertLess(abs(results[-1]["final_response"][-1] - results[-2]["final_response"][-1]), .001)
        self.assertGreater(results[-1]["control"][-1], .85)

    def test_form_units_and_invalid_parameters(self):
        values = extensions_from_form(dict(noise_std="0,05", noise_seed="2026",
                                          actuator_rate_limit="2"), "PID")
        self.assertEqual(values["noise_std"], .0005)
        self.assertEqual(values["actuator_rate_limit"], .02)
        for field in EXTENSION_DEFAULTS:
            for value in (-1, float("nan"), float("inf"), True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    normalize_extensions({field: value})
        with self.assertRaises(ValueError):
            normalize_extensions(dict(noise_seed=1.5))


if __name__ == "__main__":
    unittest.main()
