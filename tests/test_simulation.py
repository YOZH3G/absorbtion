from app.calculations import DEFAULT_MODEL_VALUES, absorption_balance
import unittest

import numpy as np

from app.calculations import IMPULSE, RECTANGLE, STEP
from app.simulation import LEAN_GAS, RICH_ABSORBENT, run_simulation


MODEL_VALUES = DEFAULT_MODEL_VALUES.copy()

DYNAMICS = {
    "kind": STEP,
    "start_time": 10.0,
    "simulation_duration": 100.0,
    "effect_duration": 1.0,
    "time_constant": 10.0,
    "delay": 2.0,
}


class SimulationTests(unittest.TestCase):
    def test_unfinished_ramp_uses_future_plateau_and_cannot_claim_settling(self):
        result = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0,
                                dict(DYNAMICS, kind="ramp", effect_duration=100,
                                     simulation_duration=20))
        self.assertAlmostEqual(result["metrics"]["steady_state"], result["calculated"])
        self.assertEqual(result["metrics"]["settling_status"], "Воздействие ещё не завершено")
        self.assertIsNone(result["metrics"]["settling_time"])

    def test_temporary_disturbance_predicts_return_even_before_its_end(self):
        for kind in (RECTANGLE, IMPULSE):
            with self.subTest(kind=kind):
                result = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0,
                                        dict(DYNAMICS, kind=kind, effect_duration=30,
                                             simulation_duration=20))
                self.assertAlmostEqual(result["metrics"]["steady_state"], result["baseline"])
                self.assertIsNone(result["metrics"]["settling_time"])

    def test_final_error_is_distinct_from_theoretical_error_on_short_horizon(self):
        result = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0,
                                dict(DYNAMICS, simulation_duration=13))
        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["final_value"], result["final_response"][-1])
        self.assertAlmostEqual(metrics["final_error"], result["baseline"] - result["final_response"][-1])
        self.assertNotAlmostEqual(metrics["final_error"], metrics["static_error"])
        self.assertIsNone(metrics["iae"])
        self.assertIsNone(metrics["saturation_duration"])

    def test_closed_loop_iae_integrates_actual_error_on_the_calculation_grid(self):
        controller = dict(controller_type="PI", controller_gain=0,
                          integral_time=10, derivative_time=0,
                          control_limit=1, setpoint=1 / 6)
        result = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0, DYNAMICS, controller)
        elapsed = DYNAMICS["simulation_duration"] - DYNAMICS["start_time"] - DYNAMICS["delay"]
        expected = (result["calculated"] - result["baseline"]) * (
            elapsed - DYNAMICS["time_constant"] * (1 - np.exp(-elapsed / DYNAMICS["time_constant"])))
        self.assertAlmostEqual(result["metrics"]["iae"], expected, delta=1e-5)

    def test_entering_band_at_the_end_does_not_confirm_settling(self):
        short = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0,
                               dict(DYNAMICS, simulation_duration=45))
        long = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0,
                              dict(DYNAMICS, simulation_duration=60))
        self.assertLess(abs(short["final_response"][-1] - short["calculated"]),
                        short["metrics"]["settling_tolerance"])
        self.assertIsNone(short["metrics"]["settling_time"])
        self.assertIsNotNone(long["metrics"]["settling_time"])
        self.assertEqual(long["metrics"]["settling_status"], "Установилось за время опыта")

    def test_saturation_duration_counts_intervals_in_seconds(self):
        values = dict(MODEL_VALUES, eta=1)
        controller = dict(controller_type="PI", controller_gain=0,
                          integral_time=10, derivative_time=0,
                          control_limit=1, setpoint=0)
        result = run_simulation(LEAN_GAS, values, 0, 0, DYNAMICS, controller)
        self.assertAlmostEqual(result["metrics"]["saturation_duration"], 100)

    def test_small_time_constants_and_pid_converge_with_off_grid_delay(self):
        dynamics = dict(DYNAMICS, time_constant=0.2, start_time=0.173,
                        simulation_duration=3.0, delay=0.027)
        controller = dict(controller_type="PID", controller_gain=0.5,
                          integral_time=0.1, derivative_time=0.015,
                          control_limit=0.3, setpoint=1 / 6)
        coarse = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0.0, dynamics, controller)
        fine = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0.0, dynamics, controller,
                              max_step=np.min(np.diff(coarse["time"])) / 2)
        interpolated = np.interp(coarse["time"], fine["time"], fine["final_response"])
        np.testing.assert_allclose(coarse["final_response"], interpolated, atol=2e-4, rtol=0)

    def test_short_rectangle_matches_analytical_peak_with_fractional_delay(self):
        dynamics = dict(DYNAMICS, kind=RECTANGLE, start_time=10.05,
                        effect_duration=0.05, delay=2.037)
        result = run_simulation(LEAN_GAS, MODEL_VALUES, 0.1, 0.0, dynamics)
        expected = (float(absorption_balance(**dict(MODEL_VALUES, xg=0.55))["xog"]) - 1 / 6) * (1.0 - np.exp(-0.05 / 10.0))
        self.assertAlmostEqual(result["metrics"]["maximum_deviation"], expected, delta=1e-9)
        self.assertTrue(np.all(result["final_response"][result["time"] <= 12.087] == 1 / 6))

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

        self.assertAlmostEqual(result["baseline"], 1 / 6)
        self.assertAlmostEqual(result["calculated"], 0.55 * 0.2 / (1 - 0.8 * 0.55))
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

        self.assertAlmostEqual(result["baseline"], 0.3)
        self.assertAlmostEqual(result["calculated"], (7400 * (97 / 370) * 0.85 + 400) / 7800)

    def test_closed_loop_returns_controller_signals_and_prediction_data(self):
        controller = {
            "controller_type": "PI",
            "controller_gain": 1.4,
            "integral_time": 10.0,
            "derivative_time": 0.0,
            "control_limit": 1.0,
            "setpoint": 1 / 6,
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
