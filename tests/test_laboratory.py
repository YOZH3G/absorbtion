import unittest

import numpy as np

from calculations import controller_response, disturbance_profile, transition_metrics
from laboratory import (
    SCENARIOS,
    evaluate_prediction,
    expected_direction,
    expected_fastest,
    format_protocol,
)


class ScenarioTests(unittest.TestCase):
    def test_priority_four_scenarios_are_available(self):
        names = {scenario["name"] for scenario in SCENARIOS}

        self.assertEqual(len(SCENARIOS), 6)
        self.assertIn("Увеличение расхода на 10%", names)
        self.assertIn("Снижение состава на 15%", names)
        self.assertIn("Противоположные воздействия", names)
        self.assertIn("Слабое длительное возмущение", names)
        self.assertIn("Кратковременный выброс", names)
        self.assertIn("Неверно настроенный регулятор", names)

    def test_each_scenario_has_a_valid_disturbance(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario["name"]):
                self.assertTrue(
                    scenario["component"] is not None or scenario["flow"] is not None
                )
                self.assertGreater(scenario["simulation_duration"], scenario["start_time"])
                self.assertGreater(scenario["time_constant"], 0.0)
                self.assertGreaterEqual(scenario["delay"], 0.0)

    def test_mistuned_controller_scenario_is_oscillatory_and_unsettled(self):
        scenario = next(
            item for item in SCENARIOS
            if item["name"] == "Неверно настроенный регулятор"
        )
        time = np.linspace(0.0, scenario["simulation_duration"], 501)
        baseline = 0.8
        profile = disturbance_profile(
            time,
            "step",
            scenario["start_time"],
            scenario["effect_duration"],
        )
        target = baseline * (1.0 + scenario["flow"] * profile)
        controller = scenario["controller"]
        response, _error, _control = controller_response(
            time,
            baseline,
            target,
            scenario["time_constant"],
            controller["type"],
            controller["gain"],
            controller["integral_time"],
            controller["derivative_time"],
            controller["control_limit"],
            baseline,
            scenario["delay"],
        )
        deviations = response - baseline
        sign_changes = np.count_nonzero(deviations[1:] * deviations[:-1] < 0.0)
        metrics = transition_metrics(time, response, np.full_like(time, baseline), baseline)

        self.assertGreaterEqual(sign_changes, 4)
        self.assertIsNone(metrics["settling_time"])


class PredictionTests(unittest.TestCase):
    def test_direction_uses_disturbed_value(self):
        self.assertEqual(expected_direction(10.0, 12.0), "Увеличится")
        self.assertEqual(expected_direction(10.0, 8.0), "Уменьшится")
        self.assertEqual(expected_direction(10.0, 10.0), "Не изменится")

    def test_fastest_handles_controller_and_unsettled_response(self):
        self.assertEqual(expected_fastest(20.0, 12.0, True), "С регулятором")
        self.assertEqual(expected_fastest(12.0, 20.0, True), "Без регулятора")
        self.assertEqual(expected_fastest(None, 12.0, True), "С регулятором")
        self.assertEqual(expected_fastest(12.0, None, True), "Без регулятора")
        self.assertEqual(expected_fastest(12.0, 13.0, False), "Без сравнения")

    def test_prediction_score_and_five_percent_steady_tolerance(self):
        outcome = {
            "baseline": 10.0,
            "disturbed_value": 12.0,
            "steady_value": 10.0,
            "open_duration": 20.0,
            "controlled_duration": 12.0,
            "controller_enabled": True,
            "correction": "Да",
        }
        prediction = {
            "direction": "Увеличится",
            "steady": 10.4,
            "fastest": "С регулятором",
            "correction": "Да",
        }

        evaluation = evaluate_prediction(prediction, outcome)

        self.assertEqual(evaluation["score"], 4)
        self.assertEqual(evaluation["total"], 4)


class ProtocolTests(unittest.TestCase):
    def test_protocol_contains_parameters_and_results(self):
        protocol = format_protocol(
            "Свободный расчёт",
            (("T", "10 с"), ("L", "2 с")),
            (("Результат", "0.8"),),
        )

        self.assertIn("ПРОТОКОЛ РАСЧЁТА", protocol)
        self.assertIn("- T: 10 с", protocol)
        self.assertIn("- Результат: 0.8", protocol)
        self.assertTrue(protocol.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
