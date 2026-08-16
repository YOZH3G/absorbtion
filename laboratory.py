import json
from pathlib import Path


def _load_builtin_scenarios():
    path = Path(__file__).with_name("builtin_scenarios.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not isinstance(payload.get("scenarios"), list):
        raise ValueError("Некорректный формат встроенных сценариев.")
    return tuple(payload["scenarios"])


SCENARIOS = _load_builtin_scenarios()

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


def evaluate_prediction(prediction, outcome, steady_tolerance_percent=5.0):
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
    steady_tolerance = max(
        abs(expected["steady"]) * steady_tolerance_percent / 100.0,
        0.01,
    )
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
