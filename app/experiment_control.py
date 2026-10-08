"""Independent validation and PI experiments for an identified local model."""

import copy
import numpy as np

from .background_tasks import check_cancelled
from .calculations import tune_controller_parameters
from .pi_control import pi_update


def validate_step(time, signal, output, parameters, *, cancel=None):
    check_cancelled(cancel)
    time, signal, output = (np.asarray(a, dtype=float) for a in (time, signal, output))
    if (any(a.ndim != 1 or not np.isfinite(a).all() for a in (time, signal, output))
            or not time.size == signal.size == output.size or time.size < 17
            or np.any(np.diff(time) <= 0)):
        raise ValueError("Проверочная запись: нужны минимум 17 точек с возрастающим временем без пропусков.")
    gain, constant, delay = model_parameters(parameters)
    before = float(np.median(signal[:5]))
    span = float(np.ptp(signal))
    if span <= 1e-12:
        raise ValueError("Проверочная запись: вход не изменяется.")
    changed = np.flatnonzero(abs(signal - before) > .1 * span)
    index = int(changed[0])
    if index < 5 or time.size - index < 12:
        raise ValueError("Нужны 5 точек до ступени и 12 после неё.")
    amplitude = float(np.median(signal[index:]) - before)
    if (abs(amplitude) < .5 * span or np.max(abs(signal[:index] - before)) > .01 * abs(amplitude)
            or np.max(abs(signal[index:] - before - amplitude)) > .01 * abs(amplitude)):
        raise ValueError("Проверочная запись должна содержать одну постоянную ступень.")
    base = output[:index]
    centered = time[:index] - np.mean(time[:index])
    slope = float(np.dot(centered, base - np.mean(base)) / np.dot(centered, centered))
    noise = base - np.mean(base) - slope * centered
    slope_error = np.sqrt(np.dot(noise, noise) / (index - 2) / np.dot(centered, centered))
    if abs(slope) > 3 * slope_error and abs(slope) * np.ptp(time) > .05 * abs(gain * amplitude):
        raise ValueError("Исходный режим проверочной записи дрейфует.")
    if time[-1] - time[index] - delay < 3 * constant:
        raise ValueError("Проверочная запись слишком короткая: нужны три T после запаздывания.")
    response = -np.expm1(-np.maximum(time - time[index] - delay, 0) / constant)
    transition = time[(response >= .1) & (response <= .9)]
    if transition.size < 5 or constant < 2 * np.median(np.diff(transition)):
        raise ValueError("Измерения проверочной записи слишком редкие.")
    baseline = float(np.mean(base))
    model = baseline + gain * amplitude * -np.expm1(-np.maximum(time - time[index] - delay, 0) / constant)
    residual = output - model
    rmse = float(np.sqrt(np.trapezoid(residual**2, time) / np.ptp(time)))
    check_cancelled(cancel)
    return dict(time=time.tolist(), input=signal.tolist(), output=output.tolist(), model=model.tolist(),
                residual=residual.tolist(), gain=gain, time_constant=constant, delay=delay,
                baseline=baseline, step_time=float(time[index]), rmse=rmse,
                normalized_rmse=rmse / abs(gain * amplitude),
                warning="Ошибка выше 10% ожидаемого отклика." if rmse > .1 * abs(gain * amplitude) else "",
                method="independent fixed-parameter step validation")


def model_parameters(parameters):
    try:
        gain, constant, delay = (float(parameters[k]) for k in ("gain", "time_constant", "delay"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Нужны числовые K, T и L.") from error
    if not np.isfinite([gain, constant, delay]).all() or gain == 0 or constant <= 0 or delay < 0:
        raise ValueError("Нужны конечное ненулевое K, положительное T и неотрицательное L.")
    return gain, constant, delay


def compare_pi(result, input_role, output_role, setpoint, horizon, gain, integral_time, *, cancel=None, progress=None):
    if input_role != "η, доля" or output_role not in ("Xог, доля", "Xна, доля"):
        raise ValueError("Для проверки регулятора подтвердите вход η и выход концентрации в долях.")
    k, constant, delay = model_parameters(result)
    t, u, y = (np.asarray(result[key], dtype=float) for key in ("time", "input", "output"))
    if (any(a.ndim != 1 or not np.isfinite(a).all() for a in (t, u, y))
            or not t.size == u.size == y.size or t.size < 17 or np.any(np.diff(t) <= 0)
            or not np.isfinite(result["step_time"]) or np.count_nonzero(t < result["step_time"]) < 5):
        raise ValueError("Эксперимент: нужны согласованные сигналы и возрастающее время с исходным участком.")
    if not np.isfinite(u).all() or not np.isfinite(y).all() or np.any((u < 0) | (u > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("Экспериментальные η и концентрация должны находиться в диапазоне 0…1.")
    eta0 = float(np.median(u[t < result["step_time"]]))
    baseline = float(result["baseline"])
    if (not np.isfinite([setpoint, horizon, gain, integral_time, eta0, baseline]).all()
            or not 0 <= setpoint <= 1 or not 0 <= baseline <= 1 or horizon <= 0 or gain < 0 or integral_time <= 0):
        raise ValueError("Нужны концентрация 0…1, положительные горизонт и Ti, неотрицательное Kp.")
    if abs(setpoint - baseline) <= 1e-12:
        raise ValueError("Задание должно отличаться от исходного выхода.")
    tuning = tune_controller_parameters("PI", constant, delay)
    proposed = float(tuning["proportional_gain"] / abs(k))
    settings = (("Исходные настройки", float(gain), float(integral_time)),
                ("Предложенные настройки", proposed, float(tuning["integral_time"])))
    step = min(constant / 100, horizon / 1000, constant / (100 * max(1, abs(k) * max(gain, proposed))),
               min(integral_time, tuning["integral_time"]) / 100)
    count = int(np.ceil(horizon / step))
    if count > 200000:
        raise ValueError("Слишком большой горизонт или усиление: нужно больше 200000 шагов.")
    time = np.linspace(0, horizon, count + 1)
    if 0 < delay < horizon:
        time = np.unique(np.r_[time, delay])
    runs = []
    for number, (name, kp, ti) in enumerate(settings):
        output = np.full(time.size, baseline)
        control = np.full(time.size, eta0)
        integral = 0.
        signed_gain = np.sign(k) * kp
        for index, dt in enumerate(np.diff(time)):
            if index % 256 == 0:
                check_cancelled(cancel)
            error = setpoint - output[index]
            clipped, integral = pi_update(error, signed_gain, ti, dt, eta0, 0., 1., integral)
            control[index] = clipped
            delayed_time = time[index] - delay
            delayed = eta0 if delayed_time < 0 else float(np.interp(delayed_time, time[:index+1], control[:index+1]))
            target = baseline + k * (delayed - eta0)
            output[index+1] = target + (output[index] - target) * np.exp(-dt / constant)
        control[-1] = control[-2]
        error = setpoint - output
        amplitude = abs(setpoint - baseline)
        outside = np.flatnonzero(abs(error) > .05 * amplitude)
        start = int(outside[-1] + 1) if outside.size else 0
        settled = float(time[start]) if start < time.size and time[-1] - time[start] >= constant else None
        reach = sorted((baseline + k * (0 - eta0), baseline + k * (1 - eta0)))
        reachable = reach[0] <= setpoint <= reach[1]
        if not reachable:
            settled = None
        metrics = dict(final_error=float(error[-1]), iae=float(np.trapezoid(abs(error), time)),
                       overshoot_percent=float(max(0, np.max(np.sign(setpoint-baseline)*(output-setpoint))) / amplitude * 100),
                       settling_time=settled, reachable=reachable,
                       saturation_duration=float(np.dot(np.isclose(control[:-1], 0, atol=1e-12, rtol=0)
                           | np.isclose(control[:-1], 1, atol=1e-12, rtol=0), np.diff(time))),
                       output_out_of_range=bool(np.any((output < 0) | (output > 1))))
        runs.append(dict(name=name, gain=kp, integral_time=ti, time=time.tolist(), output=output.tolist(),
                         control=control.tolist(), error=error.tolist(), metrics=metrics))
        if progress:
            progress((number+1)/2)
    return dict(model={key: copy.deepcopy(result[key]) for key in ("gain", "time_constant", "delay", "baseline")},
                input_role=input_role, output_role=output_role, eta0=eta0, baseline=baseline,
                setpoint=float(setpoint), horizon=float(horizon), runs=runs,
                method="identified FOPDT PI; exact decay; conditional integration; eta bounds 0..1")
