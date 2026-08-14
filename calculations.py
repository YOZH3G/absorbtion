import numpy as np


STEP = "step"
IMPULSE = "impulse"
RECTANGLE = "rectangle"
RAMP = "ramp"


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
