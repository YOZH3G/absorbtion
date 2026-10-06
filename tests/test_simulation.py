import unittest

import numpy as np

from app.calculations import IMPULSE, RECTANGLE, STEP
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
    def test_small_time_constants_and_pid_converge_with_off_grid_delay(self):
        dynamics = dict(DYNAMICS, time_constant=0.2, start_time=0.173,
                        simulation_duration=3.0, delay=0.027)
        controller = dict(controller_type="PID", controller_gain=0.5,
                          integral_time=0.1, derivative_time=0.015,
                          control_limit=0.3, setpoint=0.8)
        coarse = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0.0, dynamics, controller)
        fine = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0.0, dynamics, controller,
                              max_step=np.min(np.diff(coarse["time"])) / 2)
        interpolated = np.interp(coarse["time"], fine["time"], fine["final_response"])
        np.testing.assert_allclose(coarse["final_response"], interpolated, atol=2e-4, rtol=0)

    def test_short_rectangle_matches_analytical_peak_with_fractional_delay(self):
        dynamics = dict(DYNAMICS, kind=RECTANGLE, start_time=10.05,
                        effect_duration=0.05, delay=2.037)
        result = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0.0, dynamics)
        expected = 0.08 * (1.0 - np.exp(-0.05 / 10.0))
        self.assertAlmostEqual(result["metrics"]["maximum_deviation"], expected, delta=1e-9)
        self.assertTrue(np.all(result["final_response"][result["time"] <= 12.087] == 0.8))

    def test_impulse_converges_when_step_is_halved(self):
        dynamics = dict(DYNAMICS, kind=IMPULSE, start_time=0.123,
                        effect_duration=0.05, time_constant=0.2, delay=0.017,
                        simulation_duration=2.0)
        results = [run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0.0,
                                  dynamics, max_step=step)
                   for step in (0.001, 0.0005, 0.00025)]
        peaks = [r["metrics"]["maximum_deviation"] for r in results]
        self.assertLess(abs(peaks[1] - peaks[2]), abs(peaks[0] - peaks[2]))
        self.assertAlmostEqual(peaks[1], peaks[2], delta=2e-5)

    def test_excessive_resolution_is_rejected_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "расчётной сетки"):
            run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0.0,
                           dict(DYNAMICS, time_constant=1e-9))

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
