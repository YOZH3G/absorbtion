import unittest

import numpy as np

from app.calculations import controller_response, disturbance_profile, transition_metrics
from app.laboratory import (
    SCENARIOS,
    VARIANT_COUNT,
    builtin_variant,
    evaluate_prediction,
    expected_direction,
    expected_fastest,
    format_protocol,
    normalize_lesson,
)
from app.scenario_store import normalize_scenario
from app.simulation import run_simulation


MODEL_VALUES = {
    "gna": 7800.0,
    "xa": 0.5,
    "xg": 0.5,
    "gg": 1000.0,
    "xog_initial": 0.8,
    "xna_initial": 30.0,
}


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

    def test_builtin_scenarios_have_thirty_reproducible_variants(self):
        for scenario in SCENARIOS:
            variants = [builtin_variant(scenario, number) for number in range(1, VARIANT_COUNT + 1)]
            with self.subTest(scenario=scenario["name"]):
                self.assertEqual(len(variants), 30)
                self.assertEqual(builtin_variant(scenario), scenario)
                self.assertEqual(builtin_variant(scenario, 17), builtin_variant(scenario, 17))
                inputs = {
                    (
                        item["component"], item["flow"], item["start_time"],
                        item["effect_duration"], item["time_constant"], item["delay"],
                        tuple(sorted((item["controller"] or {}).items())),
                    )
                    for item in variants
                }
                self.assertEqual(len(inputs), VARIANT_COUNT)

    def test_builtin_variant_number_is_validated_and_all_variants_run(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario["name"]):
                with self.assertRaisesRegex(ValueError, "1…30"):
                    builtin_variant(scenario, 0)
                with self.assertRaisesRegex(ValueError, "1…30"):
                    builtin_variant(scenario, 31)
                response_signatures = set()
                for number in range(1, VARIANT_COUNT + 1):
                    variant = normalize_scenario(builtin_variant(scenario, number))
                    controller = variant["controller"]
                    controller_data = None if controller is None else {
                        "controller_type": controller["type"],
                        "controller_gain": controller["gain"],
                        "integral_time": controller["integral_time"],
                        "derivative_time": controller["derivative_time"],
                        "control_limit": controller["control_limit"],
                        "setpoint": (
                            MODEL_VALUES["xog_initial"] if variant["chain"] == "lean_gas"
                            else MODEL_VALUES["xna_initial"]
                        ) if controller.get("setpoint") is None else controller["setpoint"],
                    }
                    result = run_simulation(
                        variant["chain"], MODEL_VALUES, variant["component"] or 0.0,
                        variant["flow"] or 0.0,
                        {
                            "kind": {
                                "Ступенчатое": "step",
                                "Импульсное": "impulse",
                                "Временное прямоугольное": "rectangle",
                                "Плавно нарастающее": "ramp",
                            }[variant["disturbance_type"]],
                            "start_time": variant["start_time"],
                            "simulation_duration": variant["simulation_duration"],
                            "effect_duration": variant["effect_duration"],
                            "time_constant": variant["time_constant"],
                            "delay": variant["delay"],
                        },
                        controller_data,
                    )
                    self.assertTrue(np.all(np.isfinite(result["final_response"])))
                    response_signatures.add(tuple(np.round(result["final_response"], 12)))
                self.assertEqual(len(response_signatures), VARIANT_COUNT)

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
    def test_normalizes_hidden_answer_choices(self):
        lesson = normalize_lesson({
            "hidden_answers": {
                "direction": "Увеличится",
                "fastest": "С регулятором",
                "correction": "Да",
            },
        })

        self.assertEqual(
            lesson["hidden_answers"],
            {
                "direction": "Увеличится",
                "fastest": "С регулятором",
                "correction": "Да",
            },
        )

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

    def test_prediction_uses_scenario_tolerance(self):
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
            "steady": 10.8,
            "fastest": "С регулятором",
            "correction": "Да",
        }

        evaluation = evaluate_prediction(
            prediction,
            outcome,
            steady_tolerance_percent=10.0,
        )

        self.assertEqual(evaluation["score"], 4)

    def test_teacher_criteria_include_controller_settings(self):
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
            "steady": 10.0,
            "fastest": "С регулятором",
            "correction": "Да",
        }
        lesson = {
            "attempt_limit": 2,
            "controller_target": {
                "type": "PI",
                "gain_min": 1.5,
                "gain_max": 2.5,
                "integral_time_min": 15.0,
                "integral_time_max": 25.0,
            },
        }
        controller = {
            "controller_type": "PI",
            "controller_gain": 2.0,
            "integral_time": 20.0,
            "derivative_time": 0.0,
        }

        evaluation = evaluate_prediction(prediction, outcome, lesson=lesson, controller=controller)

        self.assertEqual(evaluation["score"], 5)
        self.assertEqual(evaluation["total"], 5)
        self.assertTrue(evaluation["criteria"][-1]["passed"])


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
