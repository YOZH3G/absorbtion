SCENARIOS = (
    {
        "name": "Увеличение расхода на 10%",
        "description": "Ступенчатое увеличение расхода газовой смеси без регулятора.",
        "chain": "lean_gas",
        "component": None,
        "flow": 0.10,
        "disturbance_type": "Ступенчатое",
        "start_time": 10.0,
        "simulation_duration": 100.0,
        "effect_duration": 10.0,
        "time_constant": 10.0,
        "delay": 2.0,
        "controller": None,
    },
    {
        "name": "Снижение состава на 15%",
        "description": "Ступенчатое снижение состава исходного абсорбента.",
        "chain": "rich_absorbent",
        "component": -0.15,
        "flow": None,
        "disturbance_type": "Ступенчатое",
        "start_time": 10.0,
        "simulation_duration": 100.0,
        "effect_duration": 10.0,
        "time_constant": 10.0,
        "delay": 2.0,
        "controller": None,
    },
    {
        "name": "Противоположные воздействия",
        "description": "Состав увеличивается на 20%, а расход одновременно снижается на 10%.",
        "chain": "lean_gas",
        "component": 0.20,
        "flow": -0.10,
        "disturbance_type": "Ступенчатое",
        "start_time": 10.0,
        "simulation_duration": 100.0,
        "effect_duration": 10.0,
        "time_constant": 10.0,
        "delay": 2.0,
        "controller": None,
    },
    {
        "name": "Слабое длительное возмущение",
        "description": "Увеличение состава на 3% в течение 40 секунд.",
        "chain": "rich_absorbent",
        "component": 0.03,
        "flow": None,
        "disturbance_type": "Временное прямоугольное",
        "start_time": 10.0,
        "simulation_duration": 100.0,
        "effect_duration": 40.0,
        "time_constant": 10.0,
        "delay": 2.0,
        "controller": None,
    },
    {
        "name": "Кратковременный выброс",
        "description": "Импульсное увеличение состава газа на 50% длительностью 6 секунд.",
        "chain": "lean_gas",
        "component": 0.50,
        "flow": None,
        "disturbance_type": "Импульсное",
        "start_time": 10.0,
        "simulation_duration": 80.0,
        "effect_duration": 6.0,
        "time_constant": 10.0,
        "delay": 2.0,
        "controller": None,
    },
    {
        "name": "Неверно настроенный регулятор",
        "description": "Агрессивный PI-регулятор демонстрирует колебательную реакцию.",
        "chain": "lean_gas",
        "component": None,
        "flow": 0.20,
        "disturbance_type": "Ступенчатое",
        "start_time": 10.0,
        "simulation_duration": 100.0,
        "effect_duration": 10.0,
        "time_constant": 10.0,
        "delay": 2.0,
        "controller": {
            "type": "PI",
            "gain": 6.0,
            "integral_time": 2.0,
            "derivative_time": 0.0,
            "control_limit": 100.0,
        },
    },
)

SCENARIOS_BY_NAME = {scenario["name"]: scenario for scenario in SCENARIOS}

DIRECTION_OPTIONS = ("Увеличится", "Уменьшится", "Не изменится")
FASTEST_OPTIONS = ("Без регулятора", "С регулятором", "Одинаково", "Без сравнения")
CORRECTION_OPTIONS = ("Да", "Нет", "Регулятор выключен")


def expected_direction(baseline, disturbed_value):
    tolerance = max(abs(baseline), 1.0) * 1e-9
    if disturbed_value > baseline + tolerance:
        return "Увеличится"
    if disturbed_value < baseline - tolerance:
        return "Уменьшится"
    return "Не изменится"


def expected_fastest(open_duration, controlled_duration, controller_enabled):
    if not controller_enabled:
        return "Без сравнения"
    if open_duration is None and controlled_duration is None:
        return "Одинаково"
    if open_duration is None:
        return "С регулятором"
    if controlled_duration is None:
        return "Без регулятора"
    if abs(open_duration - controlled_duration) <= 0.5:
        return "Одинаково"
    return "Без регулятора" if open_duration < controlled_duration else "С регулятором"


def evaluate_prediction(prediction, outcome):
    expected = {
        "direction": expected_direction(outcome["baseline"], outcome["disturbed_value"]),
        "steady": outcome["steady_value"],
        "fastest": expected_fastest(
            outcome["open_duration"],
            outcome["controlled_duration"],
            outcome["controller_enabled"],
        ),
        "correction": outcome["correction"],
    }
    steady_tolerance = max(abs(expected["steady"]) * 0.05, 0.01)
    checks = (
        (prediction["direction"] == expected["direction"], "Направление", expected["direction"]),
        (
            abs(prediction["steady"] - expected["steady"]) <= steady_tolerance,
            "Установившееся значение",
            f"{expected['steady']:.4f}".rstrip("0").rstrip("."),
        ),
        (prediction["fastest"] == expected["fastest"], "Быстрее", expected["fastest"]),
        (prediction["correction"] == expected["correction"], "Устранение отклонения", expected["correction"]),
    )
    lines = [
        f"{'Верно' if passed else 'Неверно'}: {label}. Ответ: {answer}."
        for passed, label, answer in checks
    ]
    return {
        "score": sum(passed for passed, _label, _answer in checks),
        "total": len(checks),
        "lines": lines,
    }


def format_protocol(title, parameters, results):
    lines = ["ПРОТОКОЛ РАСЧЁТА", title, "", "Параметры:"]
    lines.extend(f"- {label}: {value}" for label, value in parameters)
    lines.extend(("", "Результаты:"))
    lines.extend(f"- {label}: {value}" for label, value in results)
    return "\n".join(lines) + "\n"
