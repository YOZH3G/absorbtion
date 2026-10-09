"""Identification errors retune PI; every trial controls the same physical tank."""

import copy
import itertools

import numpy as np

from .background_tasks import check_cancelled
from .calculations import tune_controller_parameters
from .experiment_control import model_parameters
from .physical_models import run_model
from .tank import MODEL_ID, MODEL_VERSION, equilibrium, pi_metrics, validate_inputs


FIELD_DEFAULTS = dict(k_errors="-20;0;20", t_errors="-20;0;20", l_errors="0;5",
                     final_error="0.02", iae="100", overshoot_percent="10",
                     settling_time="600", saturation_duration="40")
REQUIREMENTS = ("final_error", "iae", "overshoot_percent", "settling_time", "saturation_duration")
REASONS = dict(physical_stop="Физическая остановка", unreachable="Задание недостижимо",
              final_error="Конечная ошибка", iae="Интегральная ошибка IAE",
              overshoot_percent="Перерегулирование", settling_time="Установление",
              saturation_duration="Насыщение насоса")


def finite(value):
    if type(value) not in (int, float):
        raise ValueError("Чувствительность PI: нужны конечные числа.")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Чувствительность PI: неверное число.") from error
    if not np.isfinite(number):
        raise ValueError("Чувствительность PI: нужны конечные числа.")
    return number


def parse_fields(fields):
    if not isinstance(fields, dict) or set(fields) != set(FIELD_DEFAULTS) or any(not isinstance(v, str) for v in fields.values()):
        raise ValueError("Чувствительность PI: неполные поля формы.")
    try:
        grid = {key: [float(v.strip().replace(",", ".")) for v in fields[source].split(";")]
                for key, source in (("k", "k_errors"), ("t", "t_errors"), ("l", "l_errors"))}
        requirements = {key: float(fields[key].replace(",", ".")) for key in REQUIREMENTS}
    except ValueError as error:
        raise ValueError("Отклонения разделяйте точкой с запятой; требования должны быть числами.") from error
    validate_grid(grid, requirements)
    return grid, requirements


def validate_grid(grid, requirements):
    if not isinstance(grid, dict) or set(grid) != {"k", "t", "l"}:
        raise ValueError("Чувствительность PI: нужны отклонения K, T и L.")
    for key, values in grid.items():
        if not isinstance(values, list) or not 1 <= len(values) <= 27:
            raise ValueError("Нужен непустой список отклонений.")
        numbers = [finite(v) for v in values]
        if len(set(numbers)) != len(numbers) or (key in ("k", "t") and min(numbers) <= -100):
            raise ValueError("Отклонения должны быть различны; ошибки K/T должны быть больше −100%.")
    if np.prod([len(v) for v in grid.values()]) > 27:
        raise ValueError("Допускается не больше 27 сочетаний ошибок.")
    if not isinstance(requirements, dict) or set(requirements) != set(REQUIREMENTS):
        raise ValueError("Задайте все пять требований.")
    if any(finite(v) < 0 for v in requirements.values()):
        raise ValueError("Пороги требований не могут быть отрицательными.")


def estimate(nominal, offsets):
    k, t, delay = model_parameters(nominal)
    result = dict(gain=k*(1+offsets[0]/100), time_constant=t*(1+offsets[1]/100), delay=delay+offsets[2])
    k, _, _ = model_parameters(result)
    if k <= 0:
        raise ValueError("Для подачи насоса нужна положительная оценка K.")
    return result


def tuning(local, setpoint):
    k, t, delay = model_parameters(local)
    settings = tune_controller_parameters("PI", t, delay)
    return dict(gain=settings["proportional_gain"]/k, integral_time=settings["integral_time"], setpoint=setpoint)


def assess(run, requirements):
    metrics = run["metrics"]
    reasons = []
    if run["event"] is not None:
        reasons.append("physical_stop")
    if not metrics["reachable"]:
        reasons.append("unreachable")
    for key in REQUIREMENTS:
        value = metrics[key]
        if value is None or (abs(value) if key == "final_error" else value) > requirements[key]:
            reasons.append(key)
    return dict(passed=not reasons, reasons=reasons)


def calculate(values, local, controller, grid, requirements, *, cancel=None, progress=None):
    p = validate_inputs(values)
    validate_grid(grid, requirements)
    k, t, delay = model_parameters(local)
    nominal = dict(gain=k, time_constant=t, delay=delay)
    if k <= 0:
        raise ValueError("Нужна положительная локальная оценка K.")
    controller = {key: finite(controller[key]) for key in ("gain", "integral_time", "setpoint")}
    if controller["gain"] < 0 or controller["integral_time"] <= 0 or not 0 < controller["setpoint"] < p["height"]:
        raise ValueError("Нужны Kp ≥ 0, Ti > 0 и задание внутри бака.")
    if abs(controller["setpoint"]-p["initial_level"]) <= 1e-8:
        raise ValueError("Для сравнения перерегулирования задание должно отличаться от исходного уровня.")
    offsets = list(itertools.product(grid["k"], grid["t"], grid["l"]))
    estimates = [estimate(nominal, offset) for offset in offsets]
    settings = [controller, tuning(nominal, controller["setpoint"])]
    if (len(offsets)+2)*(int(np.ceil(p["horizon"]/p["sample_time"]))+2) > 300000:
        raise ValueError("Опыт слишком велик: допускается 300000 суммарных интервалов.")
    controls, cases = [], []
    for index in range(len(offsets)+2):
        check_cancelled(cancel)
        setting = settings[index] if index < 2 else tuning(estimates[index-2], controller["setpoint"])
        run = run_model(MODEL_ID, dict(parameters=p, controller=setting), cancel=cancel)
        case = dict(run=run, assessment=assess(run, requirements))
        if index < 2:
            case["name"] = ("Исходный PI", "PI по номинальной оценке")[index]
            controls.append(case)
        else:
            case.update(offsets=list(offsets[index-2]), estimate=estimates[index-2])
            cases.append(case)
        if progress:
            progress((index+1)/(len(offsets)+2))
    return dict(format_version=1, model_id=MODEL_ID, model_version=MODEL_VERSION,
                nominal_parameters=p, local_model=nominal, grid=copy.deepcopy(grid),
                requirements=dict(requirements), controls=controls, cases=cases,
                method="identification errors; retuned PI; unchanged nonlinear tank")


def validate_result(result):
    try:
        return _validate_result(result)
    except (TypeError, OverflowError, KeyError, IndexError) as error:
        raise ValueError("Чувствительность PI: повреждённый результат.") from error


def _validate_result(result):
    from .tank_state import validate_run
    if (not isinstance(result, dict) or type(result.get("format_version")) is not int
            or result["format_version"] != 1 or result.get("model_id") != MODEL_ID or type(result.get("model_version")) is not int
            or result.get("model_version") != MODEL_VERSION):
        raise ValueError("Чувствительность PI: неизвестный формат результата.")
    p = validate_inputs(result.get("nominal_parameters"))
    validate_grid(result.get("grid"), result.get("requirements"))
    local = result.get("local_model")
    if not isinstance(local, dict) or set(local) != {"gain", "time_constant", "delay"}:
        raise ValueError("Чувствительность PI: неполная номинальная оценка.")
    for value in local.values(): finite(value)
    k, t, delay = model_parameters(local)
    if k <= 0:
        raise ValueError("Чувствительность PI: неверная номинальная оценка.")
    controls, cases = result.get("controls"), result.get("cases")
    offsets = list(itertools.product(*(result["grid"][key] for key in ("k", "t", "l"))))
    if not isinstance(controls, list) or len(controls) != 2 or not isinstance(cases, list) or len(cases) != len(offsets):
        raise ValueError("Чувствительность PI: неполная сетка результатов.")
    if (len(offsets)+2)*(int(np.ceil(p["horizon"]/p["sample_time"]))+2) > 300000:
        raise ValueError("Чувствительность PI: снимок превышает размер опыта.")
    expected_p = dict(p, pump=equilibrium(p, p["initial_level"]))
    expected_time = np.unique(np.r_[np.arange(0, p["horizon"], p["sample_time"]), p["step_time"], p["horizon"]])
    setpoint = None
    for index, case in enumerate(controls+cases):
        if not isinstance(case, dict):
            raise ValueError("Чувствительность PI: неверный опыт.")
        run = case.get("run")
        validate_run(run)
        if run["controller"] is None or run["parameters"] != expected_p:
            raise ValueError("Во всех опытах нужен один физический бак и регулятор PI.")
        if run["event"] is None and (len(run["time"]) != len(expected_time)
                or not np.allclose(run["time"], expected_time, rtol=1e-10, atol=1e-10)):
            raise ValueError("Чувствительность PI: неполный горизонт или неверный период опыта.")
        sp = run["controller"]["setpoint"]
        if setpoint is None:
            setpoint = sp
        if sp != setpoint or abs(sp-p["initial_level"]) <= 1e-8:
            raise ValueError("Чувствительность PI: неверное задание.")
        if index in (0, 1):
            if case.get("name") != ("Исходный PI", "PI по номинальной оценке")[index]:
                raise ValueError("Чувствительность PI: неверный контрольный опыт.")
            expected_controller = run["controller"] if index == 0 else tuning(result["local_model"], sp)
        else:
            offset = list(offsets[index-2])
            local = estimate(result["local_model"], offset)
            if case.get("offsets") != offset or case.get("estimate") != local:
                raise ValueError("Чувствительность PI: оценка не соответствует отклонениям.")
            expected_controller = tuning(local, sp)
        if run["controller"] != expected_controller or case.get("assessment") != assess(run, result["requirements"]):
            raise ValueError("Чувствительность PI: настройки или оценка требований повреждены.")
        pump = np.asarray(run["pump"][:-1])
        saturation = float(np.dot((pump == 0) | (pump == p["pump_limit"]), np.diff(run["time"])))
        expected_metrics = pi_metrics(expected_p, run["time"], run["level"], run["withdrawal"], run["event"], sp, saturation)
        for key, value in expected_metrics.items():
            actual = run["metrics"].get(key)
            if ((value is None or type(value) in (str, bool)) and actual != value
                    or (type(value) in (int, float) and (type(actual) not in (int, float) or not np.isclose(actual, value, rtol=1e-9, atol=1e-9)))):
                raise ValueError("Чувствительность PI: показатели не соответствуют сигналам.")
    return result
