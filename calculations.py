import numpy as np


STEP = "step"
IMPULSE = "impulse"
RECTANGLE = "rectangle"
RAMP = "ramp"

CONTROLLER_TYPES = ("P", "PI", "PID", "PD")


def combine_fractions(first, second):
    """Return the combined relative change for two independent fractions."""
    return (1 + first) * (1 + second) - 1


def calculate_xog(gg, xg, gog, k_xg=0.0, k_gg=0.0):
    """Calculate lean-gas concentration after composition and flow changes."""
    return (gg * (1 + k_gg) * xg * (1 + k_xg)) / gog


def calculate_xna(ga, gna, xa, k_xa=0.0, k_ga=0.0):
    """Calculate rich-absorbent concentration after composition and flow changes."""
    return (ga * (1 + k_ga) / gna) * xa * (1 + k_xa)


def disturbance_profile(time, kind, start_time, duration):
    """Return a normalized disturbance profile for the requested signal shape."""
    time = np.asarray(time, dtype=float)

    if kind == STEP:
        return (time >= start_time).astype(float)

    if duration <= 0:
        raise ValueError("Длительность воздействия должна быть больше нуля.")

    elapsed = time - start_time
    if kind == IMPULSE:
        profile = np.zeros_like(time)
        active = (elapsed >= 0) & (elapsed <= duration)
        profile[active] = np.sin(np.pi * elapsed[active] / duration)
        return profile
    if kind == RECTANGLE:
        return ((elapsed >= 0) & (elapsed < duration)).astype(float)
    if kind == RAMP:
        return np.clip(elapsed / duration, 0.0, 1.0)

    raise ValueError(f"Неизвестный вид воздействия: {kind}")


def first_order_response(
    time,
    baseline,
    target,
    time_constant,
    delay=0.0,
):
    """Simulate T·dy/dt + y = yуст for a time-varying target value."""
    time = np.asarray(time, dtype=float)
    target = np.asarray(target, dtype=float)
    if time.ndim != 1 or target.shape != time.shape or time.size == 0:
        raise ValueError("Время и целевое значение должны быть непустыми одномерными массивами одинаковой длины.")
    if time_constant <= 0:
        raise ValueError("Постоянная времени должна быть больше нуля.")
    if delay < 0:
        raise ValueError("Запаздывание не может быть отрицательным.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Значения времени должны строго возрастать.")

    delayed_target = np.interp(time - delay, time, target, left=baseline, right=target[-1])
    response = np.empty_like(time)
    response[0] = baseline

    for index in range(1, time.size):
        decay = np.exp(-(time[index] - time[index - 1]) / time_constant)
        interval_target = delayed_target[index - 1]
        response[index] = interval_target + (response[index - 1] - interval_target) * decay

    return response


def transition_metrics(time, response, target, baseline, settling_band=0.05):
    """Return teaching metrics for a simulated transition process."""
    time = np.asarray(time, dtype=float)
    response = np.asarray(response, dtype=float)
    target = np.asarray(target, dtype=float)
    if (
        time.ndim != 1
        or response.shape != time.shape
        or target.shape != time.shape
        or time.size == 0
    ):
        raise ValueError("Время, отклик и целевое значение должны быть непустыми одномерными массивами одинаковой длины.")
    if not 0 < settling_band < 1:
        raise ValueError("Полоса регулирования должна быть долей от нуля до единицы.")

    steady_state = float(target[-1])
    maximum_deviation = float(np.max(np.abs(response - baseline)))
    relative_deviation = (
        maximum_deviation / abs(baseline) * 100.0
        if baseline != 0
        else None
    )
    reference_deviation = max(maximum_deviation, abs(steady_state - baseline))
    tolerance = settling_band * reference_deviation
    outside = np.flatnonzero(np.abs(response - steady_state) > tolerance)
    if outside.size == 0:
        settling_time = float(time[0])
    elif outside[-1] + 1 < time.size:
        settling_time = float(time[outside[-1] + 1])
    else:
        settling_time = None

    return {
        "initial_value": float(response[0]),
        "steady_state": steady_state,
        "maximum_deviation": maximum_deviation,
        "relative_deviation": relative_deviation,
        "settling_time": settling_time,
        "static_error": float(baseline - steady_state),
    }


def controller_response(
    time,
    baseline,
    disturbance_target,
    time_constant,
    controller_type,
    controller_gain,
    integral_time,
    derivative_time,
    control_limit,
    setpoint,
    delay=0.0,
):
    """Simulate a first-order object controlled by a selected ideal controller."""
    time = np.asarray(time, dtype=float)
    disturbance_target = np.asarray(disturbance_target, dtype=float)
    if time.ndim != 1 or disturbance_target.shape != time.shape or time.size == 0:
        raise ValueError("Время и целевое значение должны быть непустыми одномерными массивами одинаковой длины.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Значения времени должны строго возрастать.")
    if time_constant <= 0:
        raise ValueError("Постоянная времени должна быть больше нуля.")
    if controller_type not in CONTROLLER_TYPES:
        raise ValueError(f"Неизвестный тип регулятора: {controller_type}")
    if controller_gain < 0:
        raise ValueError("Коэффициент K не может быть отрицательным.")
    uses_integral = "I" in controller_type
    uses_derivative = "D" in controller_type
    if uses_integral and integral_time <= 0:
        raise ValueError("Время интегрирования Ti должно быть больше нуля.")
    if uses_derivative and derivative_time < 0:
        raise ValueError("Время дифференцирования Td не может быть отрицательным.")
    if control_limit <= 0:
        raise ValueError("Ограничение управляющего воздействия должно быть больше нуля.")
    if delay < 0:
        raise ValueError("Запаздывание не может быть отрицательным.")
    if not np.all(np.isfinite(disturbance_target)) or not np.all(
        np.isfinite([
            baseline,
            controller_gain,
            integral_time,
            derivative_time,
            control_limit,
            setpoint,
            delay,
        ])
    ):
        raise ValueError("Параметры модели регулятора должны быть конечными числами.")

    response = np.empty_like(time)
    error = np.empty_like(time)
    control = np.empty_like(time)
    response[0] = baseline
    integral = 0.0

    def calculate_control(current_error, integral_value, derivative_value):
        proportional_term = current_error if "P" in controller_type else 0.0
        integral_term = integral_value / integral_time if uses_integral else 0.0
        derivative_term = derivative_time * derivative_value if uses_derivative else 0.0
        return controller_gain * (proportional_term + integral_term + derivative_term)

    for index in range(1, time.size):
        step = time[index] - time[index - 1]
        error[index - 1] = setpoint - response[index - 1]
        derivative = (
            0.0
            if index == 1
            else -(response[index - 1] - response[index - 2])
            / (time[index - 1] - time[index - 2])
        )
        candidate_integral = (
            integral + error[index - 1] * step
            if uses_integral
            else integral
        )
        candidate_control = calculate_control(
            error[index - 1],
            candidate_integral,
            derivative,
        )
        saturated_control = float(np.clip(candidate_control, -control_limit, control_limit))
        if uses_integral and (
            candidate_control == saturated_control
            or candidate_control * error[index - 1] < 0
        ):
            integral = candidate_integral

        control[index - 1] = float(np.clip(
            calculate_control(error[index - 1], integral, derivative),
            -control_limit,
            control_limit,
        ))
        delayed_time = time[index - 1] - delay
        delayed_disturbance = np.interp(
            delayed_time,
            time,
            disturbance_target,
            left=baseline,
            right=disturbance_target[-1],
        )
        delayed_control = np.interp(
            delayed_time,
            time[:index],
            control[:index],
            left=0.0,
            right=control[index - 1],
        )
        interval_target = delayed_disturbance + delayed_control
        decay = np.exp(-step / time_constant)
        response[index] = interval_target + (response[index - 1] - interval_target) * decay

    error[-1] = setpoint - response[-1]
    final_derivative = (
        0.0
        if time.size == 1
        else -(response[-1] - response[-2]) / (time[-1] - time[-2])
    )
    control[-1] = float(np.clip(
        calculate_control(error[-1], integral, final_derivative),
        -control_limit,
        control_limit,
    ))
    return response, error, control


def pi_control_response(
    time,
    baseline,
    disturbance_target,
    time_constant,
    proportional_gain,
    integral_time,
    control_limit,
    setpoint,
    delay=0.0,
):
    """Simulate the PI variant while preserving the existing public function."""
    return controller_response(
        time,
        baseline,
        disturbance_target,
        time_constant,
        "PI",
        proportional_gain,
        integral_time,
        0.0,
        control_limit,
        setpoint,
        delay,
    )


def controller_steady_state(
    disturbance_steady_state,
    setpoint,
    controller_type,
    controller_gain,
    control_limit,
):
    """Return the theoretical steady output for the selected controller type."""
    if controller_type not in CONTROLLER_TYPES:
        raise ValueError(f"Неизвестный тип регулятора: {controller_type}")
    if controller_gain < 0:
        raise ValueError("Коэффициент K не может быть отрицательным.")
    if control_limit <= 0:
        raise ValueError("Ограничение управляющего воздействия должно быть больше нуля.")
    if not np.all(np.isfinite([
        disturbance_steady_state,
        setpoint,
        controller_gain,
        control_limit,
    ])):
        raise ValueError("Параметры установившегося режима должны быть конечными числами.")

    if controller_gain == 0:
        return float(disturbance_steady_state)
    if "I" in controller_type:
        required_control = setpoint - disturbance_steady_state
    else:
        unconstrained_output = (
            disturbance_steady_state + controller_gain * setpoint
        ) / (1.0 + controller_gain)
        required_control = controller_gain * (setpoint - unconstrained_output)
    return float(
        disturbance_steady_state
        + np.clip(required_control, -control_limit, control_limit)
    )


def tune_pi_parameters(time_constant, delay):
    """Tune a PI controller for the unity-gain first-order object using an IMC rule."""
    return tune_controller_parameters("PI", time_constant, delay)


def tune_controller_parameters(controller_type, time_constant, delay):
    """Tune P, PI, PID, or PD settings for the unity-gain first-order object."""
    if controller_type not in CONTROLLER_TYPES:
        raise ValueError(f"Неизвестный тип регулятора: {controller_type}")
    if not np.all(np.isfinite([time_constant, delay])):
        raise ValueError("Параметры автоподбора должны быть конечными числами.")
    if time_constant <= 0:
        raise ValueError("Постоянная времени должна быть больше нуля.")
    if delay < 0:
        raise ValueError("Запаздывание не может быть отрицательным.")

    base_closed_loop_time = max(0.5 * time_constant, delay)
    uses_derivative = "D" in controller_type
    closed_loop_time = (
        0.5 * base_closed_loop_time
        if uses_derivative and delay > 0
        else base_closed_loop_time
    )
    proportional_gain = time_constant / (closed_loop_time + delay)
    integral_time = (
        min(time_constant, 4.0 * (closed_loop_time + delay))
        if "I" in controller_type
        else None
    )
    derivative_time = delay / 3.0 if uses_derivative else None
    return {
        "controller_type": controller_type,
        "proportional_gain": proportional_gain,
        "integral_time": integral_time,
        "derivative_time": derivative_time,
        "closed_loop_time": closed_loop_time,
    }
