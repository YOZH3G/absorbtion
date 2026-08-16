import unittest

import numpy as np

from app.calculations import STEP
from app.simulation import LEAN_GAS, RICH_ABSORBENT, run_simulation


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


class SimulationTests(unittest.TestCase):
    def test_open_loop_contains_all_comparison_curves(self):
        result = run_simulation(
            LEAN_GAS,
            MODEL_VALUES,
            component_fraction=0.1,
            flow_fraction=0.1,
            dynamics=DYNAMICS,
        )

        self.assertAlmostEqual(result["baseline"], 0.8)
        self.assertAlmostEqual(result["calculated"], 0.968)
        self.assertAlmostEqual(result["combined_fraction"], 0.21)
        self.assertEqual(result["result_mode"], "Без регулятора")
        self.assertEqual(
            set(result["responses"]),
            {"Исходный режим", "Только состав", "Только расход", "Совместное воздействие"},
        )
        np.testing.assert_array_equal(
            result["final_response"],
            result["responses"]["Совместное воздействие"],
        )

    def test_rich_absorbent_chain_uses_its_material_balance(self):
        result = run_simulation(
            RICH_ABSORBENT,
            MODEL_VALUES,
            component_fraction=-0.15,
            flow_fraction=0.0,
            dynamics=DYNAMICS,
        )

        self.assertEqual(result["baseline"], 30.0)
        self.assertAlmostEqual(result["calculated"], 25.5)

    def test_closed_loop_returns_controller_signals_and_prediction_data(self):
        controller = {
            "controller_type": "PI",
            "controller_gain": 1.4,
            "integral_time": 10.0,
            "derivative_time": 0.0,
            "control_limit": 100.0,
            "setpoint": 0.8,
        }

        result = run_simulation(
            LEAN_GAS,
            MODEL_VALUES,
            component_fraction=0.0,
            flow_fraction=0.1,
            dynamics=DYNAMICS,
            controller=controller,
        )

        self.assertEqual(result["result_mode"], "С PI-регулятором")
        self.assertEqual(result["controlled_response"].shape, result["time"].shape)
        self.assertEqual(result["error"].shape, result["time"].shape)
        self.assertEqual(result["control"].shape, result["time"].shape)
        self.assertTrue(result["prediction_outcome"]["controller_enabled"])

    def test_rejects_unknown_chain(self):
        with self.assertRaises(ValueError):
            run_simulation("unknown", MODEL_VALUES, 0.1, 0.0, DYNAMICS)


if __name__ == "__main__":
    unittest.main()
