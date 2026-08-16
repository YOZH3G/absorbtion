import unittest

from app.calculations import STEP
from app.exploration import MAP_CATEGORIES, controller_setting_map, sensitivity_runs
from app.simulation import LEAN_GAS


MODEL_VALUES = {
    "gna": 7800.0,
    "xa": 0.5,
    "xg": 0.5,
    "gg": 1000.0,
    "xog_initial": 0.8,
    "xna_initial": 30.0,
}
DYNAMICS = {
    "kind": STEP,
    "start_time": 10.0,
    "simulation_duration": 100.0,
    "effect_duration": 1.0,
    "time_constant": 10.0,
    "delay": 2.0,
}


class ExplorationTests(unittest.TestCase):
    def test_sensitivity_uses_each_selected_time_constant(self):
        runs = sensitivity_runs(
            LEAN_GAS, MODEL_VALUES, 0.0, 0.1, DYNAMICS, None,
            "Постоянная времени T", (5.0, 20.0),
        )

        self.assertEqual([run["value"] for run in runs], [5.0, 20.0])
        self.assertEqual(runs[0]["result"]["dynamics"]["time_constant"], 5.0)
        self.assertEqual(runs[1]["result"]["dynamics"]["time_constant"], 20.0)
        self.assertNotEqual(
            runs[0]["result"]["final_response"][100],
            runs[1]["result"]["final_response"][100],
        )

    def test_p_and_pi_maps_return_grid_categories(self):
        p_map = controller_setting_map(
            LEAN_GAS, MODEL_VALUES, 0.0, 0.1, DYNAMICS,
            "P", (0.2, 1.0, 3.0), (), 0.2, 0.8,
        )
        pi_map = controller_setting_map(
            LEAN_GAS, MODEL_VALUES, 0.0, 0.1, DYNAMICS,
            "PI", (0.2, 1.0), (3.0, 10.0), 0.2, 0.8,
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
            "PID", (0.2, 1.0), (3.0, 10.0), 0.2, 0.8,
            derivative_time=0.75,
        )

        self.assertEqual(pid_map["categories"].shape, (2, 2))
        self.assertEqual(pid_map["derivative_time"], 0.75)
        self.assertEqual(pid_map["results"][0][0]["controller"]["derivative_time"], 0.75)
