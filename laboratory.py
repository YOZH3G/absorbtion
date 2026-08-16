import json
import math
from pathlib import Path

from calculations import CONTROLLER_TYPES


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
PREDICTION_KEYS = ("direction", "steady", "fastest", "correction")
DEFAULT_LESSON = {
    "task": "Спрогнозируйте реакцию объекта, выполните расчёт и сформулируйте вывод.",
    "guidance": "Сначала оцените направление и скорость реакции, затем сравните прогноз с графиком.",
    "questions": (),
    "attempt_limit": 1,
    "weights": {
        "direction": 1.0,
        "steady": 1.0,
        "fastest": 1.0,
        "correction": 1.0,
        "controller_settings": 1.0,
    },
    "hidden_answers": {},
    "controller_target": None,
}


def normalize_lesson(lesson):
    """Normalize optional teacher data while retaining old scenario compatibility."""
    if lesson is None:
        return {
            **DEFAULT_LESSON,
            "questions": list(DEFAULT_LESSON["questions"]),
            "weights": DEFAULT_LESSON["weights"].copy(),
            "hidden_answers": {},
        }
    if not isinstance(lesson, dict):
        raise ValueError("Учебное задание должно быть объектом JSON.")
    task = _optional_text(lesson.get("task", DEFAULT_LESSON["task"]), "Формулировка задания")
    guidance = _optional_text(lesson.get("guidance", DEFAULT_LESSON["guidance"]), "Методические указания")
    questions = lesson.get("questions", ())
    if not isinstance(questions, (list, tuple)):
        raise ValueError("Вопросы студенту должны быть списком.")
    normalized_questions = [_required_text(question, "Вопрос студенту") for question in questions]
    attempt_limit = _integer(lesson.get("attempt_limit", 1), "Количество попыток", 1, 20)
    weights = _normalize_weights(lesson.get("weights", {}))
    hidden_answers = _normalize_hidden_answers(lesson.get("hidden_answers", {}))
    controller_target = _normalize_controller_target(lesson.get("controller_target"))
    return {
        "task": task,
        "guidance": guidance,
        "questions": normalized_questions,
        "attempt_limit": attempt_limit,
        "weights": weights,
        "hidden_answers": hidden_answers,
        "controller_target": controller_target,
    }


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


def evaluate_prediction(
    prediction,
    outcome,
    steady_tolerance_percent=5.0,
    lesson=None,
    controller=None,
):
    lesson = normalize_lesson(lesson)
    hidden_answers = lesson["hidden_answers"]
    expected = {
        "direction": hidden_answers.get(
            "direction",
            expected_direction(outcome["baseline"], outcome["disturbed_value"]),
        ),
        "steady": hidden_answers.get("steady", outcome["steady_value"]),
        "fastest": hidden_answers.get("fastest", expected_fastest(
            outcome["open_duration"],
            outcome["controlled_duration"],
            outcome["controller_enabled"],
        )),
        "correction": hidden_answers.get("correction", outcome["correction"]),
    }
    steady_tolerance = max(
        abs(expected["steady"]) * steady_tolerance_percent / 100.0,
        0.01,
    )
    checks = [
        ("direction", prediction["direction"] == expected["direction"], "Направление", expected["direction"]),
        (
            "steady",
            abs(prediction["steady"] - expected["steady"]) <= steady_tolerance,
            "Установившееся значение",
            f"{expected['steady']:.4f}".rstrip("0").rstrip("."),
        ),
        ("fastest", prediction["fastest"] == expected["fastest"], "Скорость реакции", expected["fastest"]),
        ("correction", prediction["correction"] == expected["correction"], "Действие регулятора", expected["correction"]),
    ]
    controller_check = _evaluate_controller_settings(controller, lesson["controller_target"])
    if controller_check is not None:
        checks.append(("controller_settings", *controller_check))
    lines = [
        f"{'Верно' if passed else 'Неверно'}: {label}. Ответ: {answer}."
        for _key, passed, label, answer in checks
    ]
    criteria = [
        {
            "key": key,
            "label": label,
            "passed": passed,
            "answer": answer,
            "points": lesson["weights"][key] if passed else 0.0,
            "maximum": lesson["weights"][key],
        }
        for key, passed, label, answer in checks
    ]
    return {
        "score": sum(item["points"] for item in criteria),
        "total": sum(item["maximum"] for item in criteria),
        "lines": lines,
        "criteria": criteria,
    }


def format_protocol(title, parameters, results):
    lines = ["ПРОТОКОЛ РАСЧЁТА", title, "", "Параметры:"]
    lines.extend(f"- {label}: {value}" for label, value in parameters)
    lines.extend(("", "Результаты:"))
    lines.extend(f"- {label}: {value}" for label, value in results)
    return "\n".join(lines) + "\n"


def _optional_text(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} должно быть текстом.")
    return value.strip()


def _required_text(value, label):
    text = _optional_text(value, label)
    if not text:
        raise ValueError(f"{label}: заполните поле.")
    return text


def _integer(value, label, minimum, maximum):
    if isinstance(value, bool):
        raise ValueError(f"{label}: введите целое число.")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: введите целое число.") from error
    if isinstance(value, float) and number != value:
        raise ValueError(f"{label}: введите целое число.")
    if not minimum <= number <= maximum:
        raise ValueError(f"{label}: допустимый диапазон {minimum}…{maximum}.")
    return number


def _normalize_weights(weights):
    if not isinstance(weights, dict):
        raise ValueError("Критерии оценки должны быть объектом JSON.")
    normalized = {}
    for key, default in DEFAULT_LESSON["weights"].items():
        value = weights.get(key, default)
        if isinstance(value, bool):
            raise ValueError("Вес критерия должен быть числом.")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Вес критерия должен быть числом.") from error
        if not math.isfinite(number) or number < 0 or number > 100:
            raise ValueError("Вес критерия должен быть в диапазоне 0…100.")
        normalized[key] = number
    return normalized


def _normalize_hidden_answers(answers):
    if not isinstance(answers, dict):
        raise ValueError("Скрытые ответы должны быть объектом JSON.")
    normalized = {}
    if "direction" in answers:
        if answers["direction"] not in DIRECTION_OPTIONS:
            raise ValueError("Скрытый ответ направления недопустим.")
        normalized["direction"] = answers["direction"]
    if "fastest" in answers:
        if answers["fastest"] not in FASTEST_OPTIONS:
            raise ValueError("Скрытый ответ скорости недопустим.")
        normalized["fastest"] = answers["fastest"]
    if "correction" in answers:
        if answers["correction"] not in CORRECTION_OPTIONS:
            raise ValueError("Скрытый ответ действия регулятора недопустим.")
        normalized["correction"] = answers["correction"]
    if "steady" in answers:
        try:
            steady = float(answers["steady"])
        except (TypeError, ValueError) as error:
            raise ValueError("Скрытое установившееся значение должно быть числом.") from error
        if not math.isfinite(steady):
            raise ValueError("Скрытое установившееся значение должно быть конечным.")
        normalized["steady"] = steady
    return normalized


def _normalize_controller_target(target):
    if target is None:
        return None
    if not isinstance(target, dict):
        raise ValueError("Целевые настройки регулятора должны быть объектом JSON.")
    controller_type = target.get("type")
    if controller_type not in CONTROLLER_TYPES:
        raise ValueError("Выберите допустимый тип целевого регулятора.")
    normalized = {"type": controller_type}
    for key in ("gain", "integral_time", "derivative_time"):
        minimum = target.get(f"{key}_min")
        maximum = target.get(f"{key}_max")
        if minimum is None and maximum is None:
            continue
        try:
            minimum = float(minimum)
            maximum = float(maximum)
        except (TypeError, ValueError) as error:
            raise ValueError("Границы настройки регулятора должны быть числами.") from error
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
            raise ValueError("Нижняя граница настройки не должна превышать верхнюю.")
        normalized[f"{key}_min"] = minimum
        normalized[f"{key}_max"] = maximum
    return normalized


def _evaluate_controller_settings(controller, target):
    if target is None:
        return None
    if controller is None:
        return False, "Настройки регулятора", target["type"]
    if controller["controller_type"] != target["type"]:
        return False, "Настройки регулятора", target["type"]
    labels = {"gain": "K", "integral_time": "Ti", "derivative_time": "Td"}
    for key, label in labels.items():
        minimum = target.get(f"{key}_min")
        maximum = target.get(f"{key}_max")
        if minimum is None:
            continue
        value = controller["controller_gain"] if key == "gain" else controller[key]
        if not minimum <= value <= maximum:
            return False, "Настройки регулятора", f"{label} = {minimum:g}…{maximum:g}"
    return True, "Настройки регулятора", target["type"]
