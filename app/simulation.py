import numpy as np
from .background_tasks import check_cancelled
from .controller_extensions import normalize_extensions

from .calculations import (
    absorption_balance,
    MODEL_FORMAT_VERSION,
    combine_fractions,
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
        gain = controller["controller_gain"] * controller.get("process_gain", 1.0)
        step = min(step, time_constant / (100 * max(1.0, gain)))
        if "I" in controller["controller_type"]:
            step = min(step, controller["integral_time"] / (100 * max(1.0, gain)))
        if "D" in controller["controller_type"] and controller["derivative_time"] > 0:
            step = min(step, controller["derivative_time"] / 40)
        extensions = normalize_extensions(controller)
        for key in ("derivative_filter_time", "actuator_time_constant"):
            if extensions[key] > 0 and (key != "derivative_filter_time" or "D" in controller["controller_type"]):
                step = min(step, extensions[key] / 40)
        if extensions["noise_std"] > 0:
            step = min(step, 0.01)
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


def disturbed_inputs(chain, model_values, component, flow):
    if chain not in (LEAN_GAS, RICH_ABSORBENT):
        raise ValueError(f"Неизвестная цепь управления: {chain}")
    values = dict(model_values)
    values["xg" if chain == LEAN_GAS else "xa"] *= 1 + component
    values["gg" if chain == LEAN_GAS else "ga"] *= 1 + flow
    return values


def _chain_calculator(chain, model_values):
    output = "xog" if chain == LEAN_GAS else "xna"
    baseline = float(absorption_balance(**model_values)[output])

    def calculate(component, flow):
        return absorption_balance(**disturbed_inputs(chain, model_values, component, flow))[output]

    # Validate the chain even when both disturbances are zero.
    calculate(0.0, 0.0)
    return baseline, calculate


def eta_gain(chain, values):
    gg, xg, ga, xa, eta = (values[key] for key in ("gg", "xg", "ga", "xa", "eta"))
    if chain == LEAN_GAS:
        return -xg * (1 - xg) / (1 - eta * xg) ** 2
    return gg * xg * ga * (1 - xa) / (ga + eta * gg * xg) ** 2


def tune_balanced_controller(chain, values, controller_type, time_constant, delay):
    from .calculations import tune_controller_parameters

    absorption_balance(**values)
    gain = abs(eta_gain(chain, values))
    if gain <= 1e-12:
        raise ValueError("В этом режиме выбранная концентрация не зависит от η; автоподбор невозможен.")
    tuning = tune_controller_parameters(controller_type, time_constant, delay)
    tuning["proportional_gain"] /= gain
    return tuning


def _eta_bounds(values, controller):
    limit = controller["control_limit"]
    if not np.isfinite(limit) or not 0 < limit <= 1:
        raise ValueError("Ограничение изменения η должно быть больше 0 и не больше 1.")
    return max(0.0, values["eta"] - limit), min(1.0, values["eta"] + limit)


def eta_steady_state(chain, values, controller):
    output = "xog" if chain == LEAN_GAS else "xna"
    direction = -1 if chain == LEAN_GAS else 1
    low, high = _eta_bounds(values, controller)

    def value(eta):
        return float(absorption_balance(**dict(values, eta=eta))[output])

    if controller["controller_gain"] == 0:
        return value(values["eta"])
    setpoint = controller["setpoint"]
    if "I" in controller["controller_type"]:
        return float(np.clip(setpoint, min(value(low), value(high)), max(value(low), value(high))))
    # Unique static P/PD equilibrium, including a saturated actuator.
    for _ in range(60):
        eta = (low + high) / 2
        commanded = values["eta"] + direction * controller["controller_gain"] * (setpoint - value(eta))
        if eta < commanded:
            low = eta
        else:
            high = eta
    return value((low + high) / 2)


def eta_controller_response(time, chain, values, component, flow, dynamics, controller, delayed_profile,
                            *, cancel=None, progress=None):
    """One controller changes η; both phase targets follow the common balance."""
    kind = controller["controller_type"]
    gain = controller["controller_gain"]
    ti, td = controller["integral_time"], controller["derivative_time"]
    setpoint = controller["setpoint"]
    if kind not in ("P", "PI", "PD", "PID"):
        raise ValueError("Неизвестный тип регулятора.")
    if not np.all(np.isfinite([gain, ti, td, setpoint])) or gain < 0 or td < 0:
        raise ValueError("Параметры регулятора должны быть конечными и неотрицательными.")
    if "I" in kind and ti <= 0:
        raise ValueError("Время интегрирования Ti должно быть больше нуля.")
    if not 0 <= setpoint <= 1:
        raise ValueError("Задание концентрации должно быть в диапазоне 0…1.")
    extensions = normalize_extensions(controller)
    low, high = _eta_bounds(values, controller)
    inputs = disturbed_inputs(chain, values, component * delayed_profile, flow * delayed_profile)
    # Validate every disturbed concentration and both ends of the actuator range.
    absorption_balance(**dict(inputs, eta=low))
    absorption_balance(**dict(inputs, eta=high))
    gg, xg, ga, xa = [np.broadcast_to(inputs[key], time.shape) for key in ("gg", "xg", "ga", "xa")]
    baseline = absorption_balance(**values)
    phases = {key: np.empty_like(time) for key in ("xog", "xna")}
    for key in phases:
        phases[key][0] = baseline[key]
    output = "xog" if chain == LEAN_GAS else "xna"
    direction = -1 if chain == LEAN_GAS else 1
    control = np.empty_like(time)
    commanded = np.empty_like(time)
    measurement = np.empty_like(time)
    rate_limited = np.zeros(time.size, dtype=bool)
    noise = np.zeros_like(time)
    if extensions["noise_std"] > 0:
        samples = int(np.ceil(time[-1] / 0.1)) + 1
        if samples > MAX_TIME_POINTS:
            raise ValueError("Шум измерения: слишком длинный опыт для интервала 0.1 с.")
        source_noise = np.random.default_rng(extensions["noise_seed"]).normal(0, extensions["noise_std"], samples)
        noise = np.interp(time, np.arange(samples) * 0.1, source_noise)
    actuator = extensions["actuator_time_constant"] > 0 or extensions["actuator_rate_limit"] > 0
    filtered_derivative = 0.0
    integral = 0.0
    steps = np.diff(time)
    decays = np.exp(-steps / dynamics["time_constant"])
    for index in range(time.size):
        if index % 256 == 0:
            check_cancelled(cancel)
            if progress is not None:
                progress(index / time.size)
        measurement[index] = phases[output][index] + noise[index]
        error = setpoint - measurement[index]
        derivative = (0.0 if index == 0 else
                      -(measurement[index] - measurement[index - 1]) / steps[index - 1])
        if "D" in kind and extensions["derivative_filter_time"] > 0 and index > 0:
            decay = np.exp(-steps[index - 1] / extensions["derivative_filter_time"])
            filtered_derivative = decay * filtered_derivative + (1 - decay) * derivative
            derivative = filtered_derivative
        if actuator:
            if index == 0:
                control[index] = values["eta"]
            else:
                interval = steps[index - 1]
                change = commanded[index - 1] - control[index - 1]
                if extensions["actuator_time_constant"] > 0:
                    change *= -np.expm1(-interval / extensions["actuator_time_constant"])
                if extensions["actuator_rate_limit"] > 0:
                    maximum = extensions["actuator_rate_limit"] * interval
                    rate_limited[index - 1] = abs(change) > maximum
                    change = np.clip(change, -maximum, maximum)
                control[index] = np.clip(control[index - 1] + change, low, high)
        step = steps[index] if index < steps.size else 0.0
        old_integral = integral
        candidate_integral = integral + error * step if "I" in kind else 0.0

        def command(accumulated):
            return values["eta"] + direction * gain * (
                error + (accumulated / ti if "I" in kind else 0.0)
                + (td * derivative if "D" in kind else 0.0))

        candidate = command(candidate_integral)
        clipped = np.clip(candidate, low, high)
        if "I" in kind and (candidate - clipped) * direction * error <= 0:
            integral = candidate_integral
        elif "I" in kind and gain > 0:
            integral = ti * ((clipped - values["eta"]) / (direction * gain)
                             - error - (td * derivative if "D" in kind else 0.0))
        commanded[index] = clipped
        if not actuator:
            control[index] = clipped
        elif "I" in kind and index > 0 and rate_limited[index - 1]:
            # Freeze accumulation only when error pushes into the active rate limit.
            # Keep the commanded target independent of the numerical time step.
            if (clipped - control[index]) * direction * error > 0:
                integral = old_integral
                commanded[index] = np.clip(command(integral), low, high)
        if index == steps.size:
            break
        delayed_eta = np.interp(time[index] - dynamics["delay"], time[:index + 1],
                                control[:index + 1], left=values["eta"])
        transfer = delayed_eta * gg[index] * xg[index]
        targets = {"xog": (gg[index] * xg[index] - transfer) / (gg[index] - transfer),
                   "xna": (ga[index] * xa[index] + transfer) / (ga[index] + transfer)}
        for key in phases:
            phases[key][index + 1] = targets[key] + (phases[key][index] - targets[key]) * decays[index]
    return phases, setpoint - phases[output], control, measurement, commanded, rate_limited


def run_simulation(
    chain,
    model_values,
    component_fraction,
    flow_fraction,
    dynamics,
    controller=None,
    *,
    max_step=None,
    cancel=None,
    progress=None,
):
    """Calculate all open- and closed-loop signals without touching the GUI."""
    check_cancelled(cancel)
    baseline, calculate = _chain_calculator(chain, model_values)

    combined_fraction = combine_fractions(component_fraction, flow_fraction)
    calculated = calculate(component_fraction, flow_fraction)
    if controller is not None:
        controller = dict(controller, process_gain=abs(eta_gain(chain, model_values)))
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
    responses = {}
    for index, (label, target) in enumerate(delayed_targets.items()):
        responses[label] = first_order_response(
            time,
            baseline,
            target,
            dynamics["time_constant"],
            0.0,
            cancel=cancel,
        )
        if progress is not None:
            progress((index + 1) / 7)
    permanent = dynamics["kind"] in ("step", "ramp")
    terminal_profile = 1.0 if permanent else 0.0
    stationary_from = (dynamics["start_time"] + dynamics["delay"]
                       + (0.0 if dynamics["kind"] == "step" else dynamics["effect_duration"]))
    open_metrics = transition_metrics(
        time,
        responses["Совместное воздействие"],
        targets["Совместное воздействие"],
        baseline,
        steady_value=float(calculate(component_fraction * terminal_profile, flow_fraction * terminal_profile)),
        stationary_from=stationary_from,
        confirmation_time=dynamics["time_constant"],
    )

    controlled_response = None
    controlled_phases = None
    error = None
    control = None
    measurement = commanded_control = rate_limited = None
    if controller is None:
        final_response = responses["Совместное воздействие"]
        metrics = open_metrics
        response_start = dynamics["start_time"] + dynamics["delay"]
        result_mode = "Без регулятора"
    else:
        controlled_phases, error, control, measurement, commanded_control, rate_limited = eta_controller_response(
            time, chain, model_values, component_fraction, flow_fraction,
            dynamics, controller, delayed_profile,
            cancel=cancel,
            progress=None if progress is None else lambda value: progress((4 + 2 * value) / 7),
        )
        controlled_response = controlled_phases["xog" if chain == LEAN_GAS else "xna"]
        final_response = controlled_response
        terminal_inputs = disturbed_inputs(chain, model_values,
                                           component_fraction * terminal_profile,
                                           flow_fraction * terminal_profile)
        steady_state = eta_steady_state(chain, terminal_inputs, controller)
        metrics = transition_metrics(
            time,
            controlled_response,
            np.full_like(time, steady_state),
            baseline,
            stationary_from=stationary_from,
            confirmation_time=dynamics["time_constant"],
            reference=controller["setpoint"],
        )
        metrics["iae"] = float(np.trapezoid(np.abs(error), time))
        low, high = _eta_bounds(model_values, controller)
        saturated = (np.isclose(control[:-1], low, rtol=0, atol=1e-8)
                     | np.isclose(control[:-1], high, rtol=0, atol=1e-8))
        metrics["saturation_duration"] = float(np.sum(np.diff(time)[saturated]))
        metrics["rate_limit_duration"] = float(np.sum(np.diff(time)[rate_limited[:-1]]))
        response_start = (
            dynamics["delay"]
            if controller["setpoint"] != baseline
            else dynamics["start_time"] + dynamics["delay"]
        )
        result_mode = f"С {controller['controller_type']}-регулятором"

    reachable = None
    if controller is not None:
        low, high = _eta_bounds(terminal_inputs, controller)
        output = "xog" if chain == LEAN_GAS else "xna"
        bounds = [float(absorption_balance(**dict(terminal_inputs, eta=eta))[output])
                  for eta in (low, high)]
        reachable = min(bounds) <= controller["setpoint"] <= max(bounds)

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
            and abs(metrics["final_error"]) <= tolerance
            else "Нет"
        )

    check_cancelled(cancel)
    if progress is not None:
        progress(6 / 7)
    result = {
        "chain": chain,
        "model_version": MODEL_FORMAT_VERSION,
        "model_values": dict(model_values),
        "setpoint_reachable": reachable,
        "stationary_balance": absorption_balance(**disturbed_inputs(chain, model_values, component_fraction, flow_fraction)),
        "phase_responses": controlled_phases if controlled_phases is not None else {
            key: first_order_response(time, float(absorption_balance(**model_values)[key]),
                absorption_balance(**disturbed_inputs(chain, model_values,
                    component_fraction * delayed_profile, flow_fraction * delayed_profile))[key],
                dynamics["time_constant"], cancel=cancel) for key in ("xog", "xna")
        },
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
        "measurement": measurement,
        "measurement_error": None if controller is None else controller["setpoint"] - measurement,
        "commanded_control": commanded_control,
        "rate_limited": rate_limited,
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
            "settling_status": metrics["settling_status"],
            "iae": metrics["iae"],
            "saturation_duration": metrics["saturation_duration"],
        },
    }
    check_cancelled(cancel)
    if progress is not None:
        progress(1.0)
    return result


def _settling_duration(settling_time, response_start):
    if settling_time is None:
        return None
    return max(0.0, settling_time - response_start)
