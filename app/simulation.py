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
    point_count=501,
):
    """Calculate all open- and closed-loop signals without touching the GUI."""
    baseline, calculate = _chain_calculator(chain, model_values)

    combined_fraction = combine_fractions(component_fraction, flow_fraction)
    calculated = calculate(component_fraction, flow_fraction)
    time = np.linspace(0.0, dynamics["simulation_duration"], point_count)
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
    responses = {
        label: first_order_response(
            time,
            baseline,
            target,
            dynamics["time_constant"],
            dynamics["delay"],
        )
        for label, target in targets.items()
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
