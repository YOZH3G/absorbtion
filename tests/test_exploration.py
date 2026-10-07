from app.calculations import DEFAULT_MODEL_VALUES, absorption_balance
import unittest

import numpy as np

from app.calculations import STEP
from app.exploration import MAP_CATEGORIES, controller_setting_map, sensitivity_runs
from app.simulation import LEAN_GAS, run_simulation


MODEL_VALUES = DEFAULT_MODEL_VALUES.copy()
DYNAMICS = {
    "kind": STEP,
    "start_time": 10.0,
    "simulation_duration": 100.0,
    "effect_duration": 1.0,
    "time_constant": 10.0,
    "delay": 2.0,
}


class ExplorationTests(unittest.TestCase):
    def test_map_cell_and_single_run_use_identical_resolution(self):
        mapping = controller_setting_map(
            LEAN_GAS, MODEL_VALUES, 0.1, 0.0, DYNAMICS,
            "PID", (0.7,), (0.5,), 0.2, 1 / 6, derivative_time=0.03,
        )
        cell = mapping["results"][0][0]
        single = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0.0,
                                DYNAMICS, cell["controller"])
        np.testing.assert_array_equal(cell["time"], single["time"])
        np.testing.assert_array_equal(cell["final_response"], single["final_response"])

    def test_sensitivity_uses_each_selected_time_constant(self):
        runs = sensitivity_runs(
            LEAN_GAS, MODEL_VALUES, 0.1, 0.0, DYNAMICS, None,
            "Постоянная времени T", (5.0, 20.0),
        )

        self.assertEqual([run["value"] for run in runs], [5.0, 20.0])
        self.assertEqual(runs[0]["result"]["dynamics"]["time_constant"], 5.0)
        self.assertEqual(runs[1]["result"]["dynamics"]["time_constant"], 20.0)
        self.assertNotEqual(
            np.interp(20, runs[0]["result"]["time"], runs[0]["result"]["final_response"]),
            np.interp(20, runs[1]["result"]["time"], runs[1]["result"]["final_response"]),
        )

    def test_p_and_pi_maps_return_grid_categories(self):
        p_map = controller_setting_map(
            LEAN_GAS, MODEL_VALUES, 0.0, 0.1, DYNAMICS,
            "P", (0.2, 1.0, 3.0), (), 0.2, 1 / 6,
        )
        pi_map = controller_setting_map(
            LEAN_GAS, MODEL_VALUES, 0.0, 0.1, DYNAMICS,
            "PI", (0.2, 1.0), (3.0, 10.0), 0.2, 1 / 6,
        )

        self.assertEqual(p_map["categories"].shape, (1, 3))
        self.assertEqual(pi_map["categories"].shape, (2, 2))
        self.assertTrue(all(
            MAP_CATEGORIES[code] in MAP_CATEGORIES
            for code in pi_map["categories"].flat
        ))

    def test_pid_map_keeps_selected_derivative_time(self):
        pid_map = controller_setting_map(
            LEAN_GAS, MODEL_VALUES, 0.0, 0.1, DYNAMICS,
            "PID", (0.2, 1.0), (3.0, 10.0), 0.2, 1 / 6,
            derivative_time=0.75,
        )

        self.assertEqual(pid_map["categories"].shape, (2, 2))
        self.assertEqual(pid_map["derivative_time"], 0.75)
        self.assertEqual(pid_map["results"][0][0]["controller"]["derivative_time"], 0.75)
