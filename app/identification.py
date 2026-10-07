"""Fit one permanent experimental step to a first-order model with delay."""

import csv
import json
from pathlib import Path

import numpy as np

from .background_tasks import check_cancelled


def read_experiment(path, delimiter=";"):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=delimiter)
        headers = next(reader, [])
        if len(headers) < 3 or len(set(headers)) != len(headers) or any(not h.strip() for h in headers):
            raise ValueError("CSV: нужны хотя бы три непустых разных заголовка.")
        rows = []
        for row_number, row in enumerate(reader, 2):
            if len(row) != len(headers):
                raise ValueError(f"CSV, строка {row_number}: число столбцов не совпадает с заголовком.")
            rows.append(row)
            if len(rows) > 200000:
                raise ValueError("CSV: допускается не больше 200000 строк данных.")
    return headers, rows


def experiment_columns(headers, rows, columns, units):
    if len(set(columns)) != 3 or any(c not in headers for c in columns):
        raise ValueError("Выберите три разных столбца: время, вход и выход.")
    factors = (60 if units[0] == "мин" else 1,
               .01 if units[1] == "%" else 1, .01 if units[2] == "%" else 1)
    arrays = []
    for column, factor in zip(columns, factors):
        values = []
        for number, row in enumerate(rows, 2):
            try:
                value = float(row[headers.index(column)].strip().replace(",", ".")) * factor
                if not np.isfinite(value):
                    raise ValueError()
            except ValueError as error:
                raise ValueError(f"CSV, строка {number}, столбец «{column}»: нужно конечное число без пропусков.") from error
            values.append(value)
        arrays.append(np.asarray(values))
    return tuple(arrays)


def identify_step(time, input_signal, output_signal, *, cancel=None, progress=None):
    check_cancelled(cancel)
    time, input_signal, output_signal = (np.asarray(a, dtype=float) for a in
                                         (time, input_signal, output_signal))
    if any(a.ndim != 1 for a in (time, input_signal, output_signal)) or not (
            time.size == input_signal.size == output_signal.size >= 17):
        raise ValueError("Нужны одинаковые одномерные массивы: минимум 17 точек.")
    if not all(np.all(np.isfinite(a)) for a in (time, input_signal, output_signal)):
        raise ValueError("Данные содержат пропуски или неконечные числа.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Время должно строго возрастать: сортировка и повторные отметки недопустимы.")
    span = float(np.ptp(input_signal))
    if span <= 1e-12 * max(1, float(np.max(abs(input_signal)))):
        raise ValueError("Недостаточное возбуждение: вход не изменяется.")
    baseline_input = float(np.median(input_signal[:5]))
    changed = np.flatnonzero(abs(input_signal - baseline_input) > .1 * span)
    index = int(changed[0]) if changed.size else 0
    if index < 5 or time.size - index < 12:
        raise ValueError("Нужны минимум 5 точек до ступени и 12 после неё.")
    final_input = float(np.median(input_signal[index:]))
    amplitude = final_input - baseline_input
    if abs(amplitude) < .5 * span or np.max(abs(input_signal[:index] - baseline_input)) > .01 * abs(amplitude) or np.max(abs(input_signal[index:] - final_input)) > .01 * abs(amplitude):
        raise ValueError("Поддерживается одна постоянная ступень: вход должен быть постоянным до и после неё (допуск 1%).")
    baseline_output = output_signal[:index]
    if abs(np.mean(output_signal[-max(5, time.size // 10):]) - np.mean(baseline_output)) <= max(5 * np.std(baseline_output), 1e-10):
        raise ValueError("Недостаточное возбуждение: изменение выхода не превышает шум исходного режима.")
    step_time = float(time[index])
    duration = float(time[-1] - step_time)
    minimum = float(np.min(np.diff(time))) / 10
    # Bound the fitting cost; keep the full original arrays for results and exports.
    selected = np.unique(np.r_[np.linspace(0, time.size - 1, min(2000, time.size)).astype(int), index - 1, index])
    t, y = time[selected], output_signal[selected]
    weights = np.r_[np.diff(t)[0] / 2, (t[2:] - t[:-2]) / 2, np.diff(t)[-1] / 2]
    weights /= weights.sum()
    mean_y = float(np.dot(weights, y))
    best = None

    def search(constants, delays):
        nonlocal best
        for constant in constants:
            check_cancelled(cancel)
            for delay in delays:
                shape = amplitude * -np.expm1(-np.maximum(t - step_time - delay, 0) / constant)
                mean = np.dot(weights, shape)
                variance = np.dot(weights, (shape - mean) ** 2)
                if variance <= 0:
                    continue
                gain = np.dot(weights, (shape - mean) * (y - mean_y)) / variance
                offset = mean_y - gain * mean
                loss = float(np.dot(weights, (y - offset - gain * shape) ** 2))
                if best is None or loss < best[0]:
                    best = (loss, float(constant), float(delay), float(gain), float(offset))

    search(np.geomspace(minimum, duration * 10, 50), np.linspace(0, duration * .8, 61))
    log_radius = np.log(duration * 10 / minimum) / 49
    delay_radius = duration * .8 / 60
    for iteration in range(6):
        check_cancelled(cancel)
        search(np.exp(np.linspace(max(np.log(minimum), np.log(best[1]) - log_radius),
                                  min(np.log(duration * 10), np.log(best[1]) + log_radius), 11)),
               np.linspace(max(0, best[2] - delay_radius), min(duration * .8, best[2] + delay_radius), 11))
        log_radius /= 4
        delay_radius /= 4
        if progress:
            progress((iteration + 1) / 6)
    _loss, constant, delay, gain, offset = best
    model = offset + gain * amplitude * -np.expm1(-np.maximum(time - step_time - delay, 0) / constant)
    residual = output_signal - model
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    scale = abs(gain * amplitude)
    transition = (time >= step_time + delay - constant * np.log(.9)) & (time <= step_time + delay - constant * np.log(.1))
    transition_times = time[transition]
    if transition_times.size < 5 or constant < 2 * np.median(np.diff(transition_times)):
        raise ValueError("Измерения слишком редкие: нужны минимум 5 точек между 10% и 90% отклика и T не меньше двух интервалов измерения.")
    initial_time = time[:index] - np.mean(time[:index])
    slope = float(np.dot(initial_time, baseline_output - np.mean(baseline_output)) / np.dot(initial_time, initial_time))
    initial_residual = baseline_output - np.mean(baseline_output) - slope * initial_time
    slope_error = np.sqrt(np.dot(initial_residual, initial_residual) / (index - 2) / np.dot(initial_time, initial_time))
    if abs(slope) > 3 * slope_error and abs(slope) * (time[-1] - time[0]) > .05 * scale:
        raise ValueError("Исходный режим дрейфует: для этого метода нужен установившийся участок до ступени.")
    if duration - delay < 3 * constant:
        raise ValueError("Слишком короткий опыт: после запаздывания нужны минимум три найденные постоянные времени T.")
    if constant <= minimum * 1.01 or delay >= duration * .799:
        raise ValueError("Параметры достигли границы поиска: нужны более частые или более длинные измерения.")
    if scale <= 1e-12 or rmse / scale > .1:
        raise ValueError("Модель не описывает опыт: ошибка превышает 10% амплитуды отклика.")
    check_cancelled(cancel)
    return dict(time=time.copy(), input=input_signal.copy(), output=output_signal.copy(),
                model=model, residual=residual, gain=gain, time_constant=constant, delay=delay,
                baseline=offset, step_time=step_time, step_interval=[float(time[index - 1]), step_time],
                amplitude=amplitude, rmse=rmse, normalized_rmse=rmse / scale,
                method="one-step FOPDT; time-weighted least squares")


def export_identification(path, result, format_name):
    path = Path(path)
    if format_name == "csv":
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(("Время, с", "Вход", "Эксперимент", "Модель", "Остаток"))
            writer.writerows(zip(*(result[key] for key in ("time", "input", "output", "model", "residual"))))
    else:
        payload = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
