"""Educational tools for sensitivity analysis and controller setting maps."""

import copy

import numpy as np

from .simulation import run_simulation


SENSITIVITY_PARAMETERS = (
    "Постоянная времени T",
    "Запаздывание L",
    "Возмущение состава",
    "Возмущение расхода",
)

MAP_CATEGORIES = (
    "Установилось за время опыта",
    "Затухающие колебания",
    "Незатухающие колебания",
    "Установление не подтверждено",
)
MAP_CATEGORY_CODES = {category: index for index, category in enumerate(MAP_CATEGORIES)}


def sensitivity_runs(
    chain,
    model_values,
    component_fraction,
    flow_fraction,
    dynamics,
    controller,
    parameter,
    values,
):
    """Return a simulation for every selected value of one parameter."""
    if parameter not in SENSITIVITY_PARAMETERS:
        raise ValueError("Неизвестный параметр анализа чувствительности.")
    if not values:
        raise ValueError("Укажите хотя бы одно значение параметра.")
    runs = []
    for value in values:
        adjusted_dynamics = copy.deepcopy(dynamics)
        adjusted_component = component_fraction
        adjusted_flow = flow_fraction
        if parameter == "Постоянная времени T":
            adjusted_dynamics["time_constant"] = value
        elif parameter == "Запаздывание L":
            adjusted_dynamics["delay"] = value
        elif parameter == "Возмущение состава":
            adjusted_component = value
        else:
            adjusted_flow = value
        result = run_simulation(
            chain,
            model_values,
            adjusted_component,
            adjusted_flow,
            adjusted_dynamics,
            controller,
        )
        runs.append({"value": value, "result": result})
    return runs


def controller_setting_map(
    chain,
    model_values,
    component_fraction,
    flow_fraction,
    dynamics,
    controller_type,
    gains,
    integral_times,
    control_limit,
    setpoint,
    derivative_time=0.0,
):
    """Classify a grid of P, PI, or PID controller settings."""
    if controller_type not in ("P", "PI", "PID"):
        raise ValueError("Карта настроек доступна только для P-, PI- и PID-регуляторов.")
    if not gains:
        raise ValueError("Укажите значения коэффициента Kp.")
    if controller_type in ("PI", "PID") and not integral_times:
        raise ValueError("Укажите значения времени интегрирования Ti.")
    if controller_type == "PID" and derivative_time < 0:
        raise ValueError("Время дифференцирования Td не может быть отрицательным.")
    rows = (None,) if controller_type == "P" else tuple(integral_times)
    categories = np.empty((len(rows), len(gains)), dtype=int)
    results = [[None for _gain in gains] for _row in rows]
    assessments = [[None for _gain in gains] for _row in rows]
    for row, integral_time in enumerate(rows):
        for column, gain in enumerate(gains):
            controller = {
                "controller_type": controller_type,
                "controller_gain": gain,
                "integral_time": 1.0 if integral_time is None else integral_time,
                "derivative_time": derivative_time if controller_type == "PID" else 0.0,
                "control_limit": control_limit,
                "setpoint": setpoint,
            }
            result = run_simulation(
                chain,
                model_values,
                component_fraction,
                flow_fraction,
                dynamics,
                controller,
            )
            results[row][column] = result
            assessment = assess_controller_result(result)
            assessments[row][column] = assessment
            categories[row, column] = MAP_CATEGORY_CODES[assessment["category"]]
    return {
        "controller_type": controller_type,
        "gains": tuple(gains),
        "integral_times": rows,
        "derivative_time": derivative_time if controller_type == "PID" else None,
        "categories": categories,
        "results": results,
        "assessments": assessments,
    }


def classify_controller_result(result):
    """Return the primary map colour; independent features remain in the assessment."""
    return assess_controller_result(result)["category"]


def assess_controller_result(result):
    """Describe observations on the finite horizon without claiming analytic stability."""
    response = np.asarray(result["final_response"], dtype=float)
    metrics = result["metrics"]
    start_index = int(np.searchsorted(result["time"], result["response_start"], side="left"))
    oscillation = "Колебания не выявлены"
    ratio = None
    if np.all(np.isfinite(response)) and start_index < response.size:
        deviation = response[start_index:] - metrics["steady_state"]
        scale = max(float(np.max(np.abs(deviation))),
                    abs(metrics["initial_value"] - metrics["steady_state"]), 1e-12)
        significant = deviation[np.abs(deviation) > 0.02 * scale]
        boundaries = np.flatnonzero(np.diff(np.signbit(significant))) + 1
        if len(boundaries) >= 2:
            # Exclude the first and last lobes: the horizon can cut either one short.
            peaks = [float(np.max(np.abs(lobe))) for lobe in np.split(significant, boundaries)[1:-1]]
            if len(peaks) < 2:
                oscillation = "Колебания: недостаточно циклов для оценки затухания"
            else:
                ratio = peaks[-1] / peaks[0]
                oscillation = ("Затухающие колебания" if ratio < 0.8 else
                               "Растущие колебания" if ratio > 1.2 else "Незатухающие колебания")
    else:
        oscillation = "Недостаточно данных для оценки колебаний"
    saturated = False
    saturation_duration = 0.0
    controller = result["controller"]
    if controller is not None:
        control = np.asarray(result["control"], dtype=float)
        eta = result["model_values"]["eta"]
        low = max(0.0, eta - controller["control_limit"])
        high = min(1.0, eta + controller["control_limit"])
        at_limit = (np.isclose(control, low, rtol=0.0, atol=1e-8)
                    | np.isclose(control, high, rtol=0.0, atol=1e-8))
        saturated = bool(np.any(at_limit))
        saturation_duration = float(np.sum(np.diff(result["time"])[at_limit[:-1]]))
    settled = metrics["settling_time"] is not None and bool(np.all(np.isfinite(response)))
    category = (MAP_CATEGORIES[0] if settled else MAP_CATEGORIES[1]
                if oscillation == "Затухающие колебания" else MAP_CATEGORIES[2]
                if oscillation in ("Растущие колебания", "Незатухающие колебания") else MAP_CATEGORIES[3])
    status = metrics.get("settling_status", "Установилось за время опыта" if settled
                         else "Установление не подтверждено")
    explanation = f"{status}. {oscillation}. "
    if ratio is not None:
        explanation += f"Отношение последней полной амплитуды к первой: {ratio:.3g}. "
    explanation += (f"Насыщение η: {saturation_duration:.3g} с. " if saturated else "Насыщение η не выявлено. ")
    explanation += "Вывод относится только к длительности этого опыта."
    return {"category": category, "settled": settled, "oscillation": oscillation,
            "amplitude_ratio": ratio, "saturated": saturated,
            "saturation_duration": saturation_duration, "explanation": explanation}
