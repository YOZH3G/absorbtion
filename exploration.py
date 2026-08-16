"""Educational tools for sensitivity analysis and controller setting maps."""

import copy

import numpy as np

from simulation import run_simulation


SENSITIVITY_PARAMETERS = (
    "Постоянная времени T",
    "Запаздывание L",
    "Возмущение состава",
    "Возмущение расхода",
)

MAP_CATEGORIES = (
    "Устойчиво",
    "Колебания",
    "Не установилось",
    "Ограничение управления",
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
                point_count=301,
            )
            results[row][column] = result
            categories[row, column] = MAP_CATEGORY_CODES[classify_controller_result(result)]
    return {
        "controller_type": controller_type,
        "gains": tuple(gains),
        "integral_times": rows,
        "derivative_time": derivative_time if controller_type == "PID" else None,
        "categories": categories,
        "results": results,
    }


def classify_controller_result(result):
    """Categorize a simulation for the simplified educational map."""
    response = np.asarray(result["final_response"], dtype=float)
    metrics = result["metrics"]
    if not np.all(np.isfinite(response)):
        return "Не установилось"
    start_index = int(np.searchsorted(result["time"], result["response_start"], side="left"))
    if start_index >= response.size:
        return "Не установилось"
    deviation = response[start_index:] - metrics["steady_state"]
    scale = max(float(np.max(np.abs(deviation))), abs(metrics["initial_value"] - metrics["steady_state"]), 1e-6)
    significant = deviation[np.abs(deviation) > 0.02 * scale]
    crossings = int(np.count_nonzero(np.diff(np.signbit(significant))))
    if metrics["settling_time"] is None:
        return "Колебания" if crossings >= 2 else "Не установилось"
    if crossings >= 3:
        return "Колебания"
    control = result["control"]
    controller = result["controller"]
    if controller is not None and np.any(np.isclose(np.abs(control), controller["control_limit"], rtol=0.0, atol=1e-8)):
        return "Ограничение управления"
    return "Устойчиво"
