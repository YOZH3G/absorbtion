import numpy as np


STEP = "step"
IMPULSE = "impulse"
RECTANGLE = "rectangle"
RAMP = "ramp"

CONTROLLER_TYPES = ("P", "PI", "PID", "PD")
MODEL_FORMAT_VERSION = 2
DEFAULT_MODEL_VALUES = {"gg": 1000.0, "xg": 0.5, "ga": 7400.0,
                        "xa": 97.0 / 370.0, "eta": 0.8}


def absorption_balance(gg, xg, ga, xa, eta):
    """Return the common stationary mass balance; fractions are always 0..1."""
    gg, xg, ga, xa, eta = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in (gg, xg, ga, xa, eta)]
    )
    if not all(np.all(np.isfinite(value)) for value in (gg, xg, ga, xa, eta)):
        raise ValueError("Параметры баланса должны быть конечными числами.")
    if np.any(gg <= 0) or np.any(ga <= 0):
        raise ValueError("Входные расходы газа и абсорбента должны быть больше нуля.")
    if any(np.any((value < 0) | (value > 1)) for value in (xg, xa, eta)):
        raise ValueError("Концентрации и степень извлечения должны быть в диапазоне 0…1.")
    transfer = eta * gg * xg
    gog = gg - transfer
    gna = ga + transfer
    if np.any(gog <= 0) or np.any(gna <= 0):
        raise ValueError("Выходной расход должен быть больше нуля: при Xг=100% и η=100% "
                         "концентрация остаточного газа не определена.")
    xog = (gg * xg - transfer) / gog
    xna = (ga * xa + transfer) / gna
    return {"j": transfer, "gog": gog, "gna": gna, "xog": xog, "xna": xna,
            "mass_residual": gg + ga - gog - gna,
            "component_residual": gg * xg + ga * xa - gog * xog - gna * xna}


def _validate_time_series(time, target):
    time = np.asarray(time, dtype=float)
    target = np.asarray(target, dtype=float)
    if time.ndim != 1 or target.shape != time.shape or time.size == 0:
        raise ValueError("Время и целевое значение должны быть непустыми одномерными массивами одинаковой длины.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Значения времени должны строго возрастать.")
    return time, target


def _controller_features(controller_type):
    if controller_type not in CONTROLLER_TYPES:
        raise ValueError(f"Неизвестный тип регулятора: {controller_type}")
    return "I" in controller_type, "D" in controller_type


def combine_fractions(first, second):
    """Return the combined relative change for two independent fractions."""
    return (1 + first) * (1 + second) - 1


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
    time, target = _validate_time_series(time, target)
    if time_constant <= 0:
        raise ValueError("Постоянная времени должна быть больше нуля.")
    if delay < 0:
        raise ValueError("Запаздывание не может быть отрицательным.")

    delayed_target = np.interp(time - delay, time, target, left=baseline, right=target[-1])
    response = np.empty_like(time)
    response[0] = baseline

    for index in range(1, time.size):
        decay = np.exp(-(time[index] - time[index - 1]) / time_constant)
        interval_target = delayed_target[index - 1]
        response[index] = interval_target + (response[index - 1] - interval_target) * decay

    return response


def transition_metrics(time, response, target, baseline, settling_band=0.05, *,
                       steady_value=None, stationary_from=None, confirmation_time=0.0,
                       reference=None):
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

    if not (np.isfinite(time).all() and np.isfinite(response).all()
            and np.isfinite(target).all()) or np.any(np.diff(time) <= 0):
        raise ValueError("Сигналы должны быть конечными, а время — строго возрастать.")
    steady_state = float(target[-1] if steady_value is None else steady_value)
    stationary_from = float(time[0] if stationary_from is None else stationary_from)
    reference = float(baseline if reference is None else reference)
    if not np.all(np.isfinite([steady_state, stationary_from, confirmation_time, reference])) or confirmation_time < 0:
        raise ValueError("Параметры установления должны быть конечными; время подтверждения неотрицательно.")
    maximum_deviation = float(np.max(np.abs(response - baseline)))
    relative_deviation = (
        maximum_deviation / abs(baseline) * 100.0
        if baseline != 0
        else None
    )
    reference_deviation = max(maximum_deviation, abs(steady_state - baseline))
    tolerance = max(settling_band * reference_deviation, 1e-12)
    outside = np.flatnonzero(np.abs(response - steady_state) > tolerance)
    candidate = max(int(np.searchsorted(time, stationary_from)),
                    int(outside[-1] + 1) if outside.size else 0)
    settling_time = (float(time[candidate]) if candidate < time.size
                     and time[-1] - time[candidate] >= confirmation_time else None)
    settling_status = ("Воздействие ещё не завершено" if stationary_from > time[-1]
                       else "Установилось за время опыта" if settling_time is not None
                       else "Установление не подтверждено")

    return {
        "initial_value": float(response[0]),
        "steady_state": steady_state,
        "final_value": float(response[-1]),
        "final_error": reference - float(response[-1]),
        "settling_status": settling_status,
        "settling_tolerance": tolerance,
        "maximum_deviation": maximum_deviation,
        "relative_deviation": relative_deviation,
        "settling_time": settling_time,
        "static_error": reference - steady_state,
        "iae": None,
        "saturation_duration": None,
    }


def tune_controller_parameters(controller_type, time_constant, delay):
    """Tune P, PI, PID, or PD settings for the unity-gain first-order object."""
    uses_integral, uses_derivative = _controller_features(controller_type)
    if not np.all(np.isfinite([time_constant, delay])):
        raise ValueError("Параметры автоподбора должны быть конечными числами.")
    if time_constant <= 0:
        raise ValueError("Постоянная времени должна быть больше нуля.")
    if delay < 0:
        raise ValueError("Запаздывание не может быть отрицательным.")

    base_closed_loop_time = max(0.5 * time_constant, delay)
    closed_loop_time = (
        0.5 * base_closed_loop_time
        if uses_derivative and delay > 0
        else base_closed_loop_time
    )
    proportional_gain = time_constant / (closed_loop_time + delay)
    integral_time = (
        min(time_constant, 4.0 * (closed_loop_time + delay))
        if uses_integral
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
