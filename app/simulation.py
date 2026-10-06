import numpy as np

from .calculations import (
    calculate_xna,
    calculate_xog,
    combine_fractions,
    controller_response,
    controller_steady_state,
    disturbance_profile,
    first_order_response,
    transition_metrics,
)


LEAN_GAS = "lean_gas"
RICH_ABSORBENT = "rich_absorbent"
MAX_TIME_POINTS = 200000


def simulation_time_grid(dynamics, controller=None, max_step=None):
    """Resolve dynamics and switching events independently of plot resolution."""
    horizon = dynamics["simulation_duration"]
    time_constant = dynamics["time_constant"]
    start = dynamics["start_time"]
    duration = dynamics["effect_duration"]
    delay = dynamics["delay"]
    if not np.all(np.isfinite([horizon, time_constant, start, duration, delay])):
        raise ValueError("Параметры времени должны быть конечными числами.")
    if horizon <= 0 or time_constant <= 0 or delay < 0 or start < 0:
        raise ValueError("Проверьте горизонт, постоянную времени, начало и запаздывание.")
    step = min(horizon / 500, time_constant / 100)
    events = [start, start + delay, delay]
    if dynamics["kind"] != "step":
        if duration <= 0:
            raise ValueError("Длительность воздействия должна быть больше нуля.")
        step = min(step, duration / 40)
        events.extend([start + duration, start + duration + delay])
    if controller is not None:
        gain = controller["controller_gain"]
        step = min(step, time_constant / (100 * max(1.0, gain)))
        if "I" in controller["controller_type"]:
            step = min(step, controller["integral_time"] / (100 * max(1.0, gain)))
        if "D" in controller["controller_type"] and controller["derivative_time"] > 0:
            step = min(step, controller["derivative_time"] / 40)
        if delay > 0:
            step = min(step, delay / 20)
    if max_step is not None:
        if not np.isfinite(max_step) or max_step <= 0:
            raise ValueError("Максимальный шаг должен быть положительным конечным числом.")
        step = min(step, max_step)
    if step <= 0 or horizon / step + len(events) + 2 > MAX_TIME_POINTS:
        raise ValueError("Требуется слишком много точек расчётной сетки (предел 200000). "
                         "Увеличьте T, Ti или длительность воздействия либо сократите время опыта.")
    count = int(np.ceil(horizon / step))
    grid = np.linspace(0.0, horizon, count + 1)
    # Replace adjacent floating-point duplicates with the exact event time.
    for event in events:
        if 0 <= event <= horizon:
            grid = grid[~np.isclose(grid, event, rtol=0.0, atol=1e-12 * max(1.0, horizon))]
            grid = np.append(grid, event)
    return np.sort(grid)


def _chain_calculator(chain, model_values):
    if chain == LEAN_GAS:
        baseline = model_values["xog_initial"]
        outlet_flow = model_values["gg"] * model_values["xg"] / baseline

        def calculate(component, flow):
            return calculate_xog(
                model_values["gg"],
                model_values["xg"],
                outlet_flow,
                component,
                flow,
            )

        return baseline, calculate
    if chain == RICH_ABSORBENT:
        baseline = model_values["xna_initial"]
        absorbent_flow = model_values["gna"] * baseline / model_values["xa"]

        def calculate(component, flow):
            return calculate_xna(
                absorbent_flow,
                model_values["gna"],
                model_values["xa"],
                component,
                flow,
            )

        return baseline, calculate
    raise ValueError(f"Неизвестная цепь управления: {chain}")


def run_simulation(
    chain,
    model_values,
    component_fraction,
    flow_fraction,
    dynamics,
    controller=None,
    *,
    max_step=None,
):
    """Calculate all open- and closed-loop signals without touching the GUI."""
    baseline, calculate = _chain_calculator(chain, model_values)

    combined_fraction = combine_fractions(component_fraction, flow_fraction)
    calculated = calculate(component_fraction, flow_fraction)
    time = simulation_time_grid(dynamics, controller, max_step)
    profile = disturbance_profile(
        time,
        dynamics["kind"],
        dynamics["start_time"],
        dynamics["effect_duration"],
    )
    targets = {
        "Исходный режим": np.full_like(time, baseline),
        "Только состав": calculate(component_fraction * profile, 0.0),
        "Только расход": calculate(0.0, flow_fraction * profile),
        "Совместное воздействие": calculate(
            component_fraction * profile,
            flow_fraction * profile,
        ),
    }
    delayed_profile = disturbance_profile(
        time, dynamics["kind"], dynamics["start_time"] + dynamics["delay"],
        dynamics["effect_duration"],
    )
    delayed_targets = {
        "Исходный режим": np.full_like(time, baseline),
        "Только состав": calculate(component_fraction * delayed_profile, 0.0),
        "Только расход": calculate(0.0, flow_fraction * delayed_profile),
        "Совместное воздействие": calculate(component_fraction * delayed_profile,
                                           flow_fraction * delayed_profile),
    }
    responses = {
        label: first_order_response(
            time,
            baseline,
            target,
            dynamics["time_constant"],
            0.0,
        )
        for label, target in delayed_targets.items()
    }
    open_metrics = transition_metrics(
        time,
        responses["Совместное воздействие"],
        targets["Совместное воздействие"],
        baseline,
    )

    controlled_response = None
    error = None
    control = None
    if controller is None:
        final_response = responses["Совместное воздействие"]
        metrics = open_metrics
        response_start = dynamics["start_time"] + dynamics["delay"]
        result_mode = "Без регулятора"
    else:
        controlled_response, error, control = controller_response(
            time,
            baseline,
            targets["Совместное воздействие"],
            dynamics["time_constant"],
            controller["controller_type"],
            controller["controller_gain"],
            controller["integral_time"],
            controller["derivative_time"],
            controller["control_limit"],
            controller["setpoint"],
            dynamics["delay"],
            delayed_disturbance=delayed_targets["Совместное воздействие"],
        )
        final_response = controlled_response
        steady_state = controller_steady_state(
            targets["Совместное воздействие"][-1],
            controller["setpoint"],
            controller["controller_type"],
            controller["controller_gain"],
            controller["control_limit"],
        )
        metrics = transition_metrics(
            time,
            controlled_response,
            np.full_like(time, steady_state),
            baseline,
        )
        metrics["static_error"] = controller["setpoint"] - steady_state
        response_start = (
            dynamics["delay"]
            if controller["setpoint"] != baseline
            else dynamics["start_time"] + dynamics["delay"]
        )
        result_mode = f"С {controller['controller_type']}-регулятором"

    open_duration = _settling_duration(
        open_metrics["settling_time"],
        dynamics["start_time"] + dynamics["delay"],
    )
    controlled_duration = (
        None
        if controller is None
        else _settling_duration(metrics["settling_time"], response_start)
    )
    correction = "Регулятор выключен"
    if controller is not None:
        tolerance = max(abs(controller["setpoint"]) * 0.01, 1e-6)
        correction = (
            "Да"
            if metrics["settling_time"] is not None
            and abs(metrics["static_error"]) <= tolerance
            else "Нет"
        )

    return {
        "chain": chain,
        "component_fraction": component_fraction,
        "flow_fraction": flow_fraction,
        "combined_fraction": combined_fraction,
        "baseline": baseline,
        "calculated": calculated,
        "dynamics": dynamics,
        "controller": controller,
        "time": time,
        "profile": profile,
        "targets": targets,
        "responses": responses,
        "controlled_response": controlled_response,
        "error": error,
        "control": control,
        "final_response": final_response,
        "metrics": metrics,
        "response_start": response_start,
        "result_mode": result_mode,
        "prediction_outcome": {
            "baseline": baseline,
            "disturbed_value": calculated,
            "steady_value": metrics["steady_state"],
            "open_duration": open_duration,
            "controlled_duration": controlled_duration,
            "controller_enabled": controller is not None,
            "correction": correction,
        },
    }


def _settling_duration(settling_time, response_start):
    if settling_time is None:
        return None
    return max(0.0, settling_time - response_start)
