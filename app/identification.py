"""Fit one permanent experimental step to a first-order model with delay."""

import csv
import copy
import json
from pathlib import Path

import numpy as np

from .background_tasks import check_cancelled


def validate_experiment_origin(origin):
    if origin is None:
        return None
    if not isinstance(origin, dict) or not isinstance(origin.get("experiment_id"), str) or not isinstance(origin.get("source"), str):
        raise ValueError("Эксперимент: некорректное происхождение параметров.")
    for key in ("data_revision", "result_revision"):
        value = origin.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Эксперимент: некорректная версия параметров.")
    from .experiment_control import model_parameters
    model_parameters(origin.get("parameters", {}))
    if not isinstance(origin.get("metadata"), dict) or any(not isinstance(v, str) for v in origin["metadata"].values()):
        raise ValueError("Эксперимент: некорректные сведения о параметрах.")
    return copy.deepcopy(origin)


def format_experiment_origin(origin):
    if origin is None:
        return ""
    p = origin["parameters"]
    return (f"Источник T/L: {origin['source']}; версия идентификации {origin['result_revision']}; "
            f"K={p['gain']:.6g}, T={p['time_constant']:.6g} с, L={p['delay']:.6g} с. "
            "В балансе абсорбера использованы только T/L; его усиление сохранено.")


def validate_experiment_state(state):
    """Validate a portable page snapshot, including the last successful fit."""
    if state is None:
        return None
    if not isinstance(state, dict):
        raise ValueError("Эксперимент должен быть объектом JSON.")
    state = copy.deepcopy(state)
    headers, rows = state.get("headers"), state.get("rows")
    if (not isinstance(headers, list) or any(not isinstance(h, str) or not h.strip() for h in headers)
            or len(set(headers)) != len(headers) or (headers and len(headers) < 3)):
        raise ValueError("Эксперимент: нужны разные непустые заголовки.")
    if (not isinstance(rows, list) or len(rows) > 200000 or any(
            not isinstance(row, list) or len(row) != len(headers)
            or any(not isinstance(value, str) for value in row) for row in rows)):
        raise ValueError("Эксперимент: некорректные строки или больше 200000 строк.")
    columns, units = state.get("columns"), state.get("units")
    if (not isinstance(columns, list) or len(columns) != 3 or any(
            not isinstance(c, str) or (c and c not in headers) for c in columns)):
        raise ValueError("Эксперимент: некорректный выбор столбцов.")
    if (not isinstance(units, list) or len(units) != 3 or units[0] not in ("с", "мин")
            or any(u not in ("как в\u00a0файле", "%") for u in units[1:])):
        raise ValueError("Эксперимент: неизвестные единицы.")
    if (not isinstance(state.get("source"), str) or state.get("delimiter") not in (";", ",", "табуляция")
            or state.get("view") not in ("Отклик", "Остаток", "Проверка", "Регулирование")
            or not isinstance(state.get("result_current"), bool)):
        raise ValueError("Эксперимент: некорректные настройки страницы.")
    result = state.get("result")
    if result is not None:
        if not isinstance(result, dict):
            raise ValueError("Эксперимент: некорректный результат идентификации.")
        arrays = []
        for key in ("time", "input", "output", "model", "residual"):
            try:
                a = np.asarray(result.get(key), dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError("Эксперимент: сигналы должны быть числовыми.") from error
            if a.ndim != 1 or not 17 <= a.size <= 200000 or not np.isfinite(a).all():
                raise ValueError("Эксперимент: нужны конечные сигналы длиной 17–200000.")
            arrays.append(a)
            result[key] = a.tolist()
        if len({a.size for a in arrays}) != 1 or np.any(np.diff(arrays[0]) <= 0):
            raise ValueError("Эксперимент: длины сигналов или порядок времени нарушены.")
        for key in ("gain", "time_constant", "delay", "baseline", "step_time", "amplitude", "rmse", "normalized_rmse"):
            value = result.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Эксперимент: некорректный параметр {key}.")
        if (result["time_constant"] <= 0 or result["delay"] < 0 or result["rmse"] < 0
                or result["normalized_rmse"] < 0 or result["gain"] == 0 or result["amplitude"] == 0):
            raise ValueError("Эксперимент: параметры модели вне допустимых границ.")
        interval = result.get("step_interval")
        if (not isinstance(interval, list) or len(interval) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in interval)
                or not arrays[0][0] <= interval[0] <= interval[1] == result["step_time"] <= arrays[0][-1]):
            raise ValueError("Эксперимент: некорректный интервал ступени.")
        if (not isinstance(result.get("source"), str) or not isinstance(result.get("method"), str)
                or any(not isinstance(result.get(key), (list, tuple)) or len(result[key]) != 3
                       or any(not isinstance(v, str) for v in result[key]) for key in ("columns", "units"))
                or result["units"][0] not in ("с", "мин")
                or any(v.replace("\u00a0", " ") not in ("как в файле", "%") for v in result["units"][1:])):
            raise ValueError("Эксперимент: отсутствуют сведения о результате.")
        result["columns"] = list(result["columns"])
        result["units"] = list(result["units"])
        validate_experiment_origin(result.get("origin"))
        if not np.allclose(arrays[2] - arrays[3], arrays[4], rtol=1e-12, atol=1e-12):
            raise ValueError("Эксперимент: остаток не соответствует данным.")
    elif state["result_current"]:
        raise ValueError("Эксперимент: нет актуального результата.")
    for key in ("data_revision", "result_revision"):
        value = state.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Эксперимент: некорректная версия данных.")
    if not isinstance(state.get("experiment_id", ""), str):
        raise ValueError("Эксперимент: некорректный идентификатор.")
    for key in ("metadata", "pi_settings"):
        value = state.get(key, {})
        if not isinstance(value, dict) or any(not isinstance(v, str) for v in value.values()):
            raise ValueError("Эксперимент: сведения и черновые настройки должны быть текстом.")
    for key in ("validation", "pi_result", "applied_origin"):
        if state.get(key) is not None and not isinstance(state[key], dict):
            raise ValueError("Эксперимент: некорректный снимок проверки.")
    validate_experiment_origin(state.get("applied_origin"))
    for key in ("validation", "pi_result"):
        record = state.get(key)
        if record is None:
            continue
        validate_experiment_origin(record.get("origin"))
        if record.get("origin") is None:
            raise ValueError("Эксперимент: проверка должна содержать происхождение параметров.")
        if key == "validation":
            required = ("time", "input", "output", "model", "residual")
            records = [record]
            from .experiment_control import model_parameters
            model_parameters(record)
            for name in ("source", "warning", "method"):
                if not isinstance(record.get(name), str):
                    raise ValueError("Эксперимент: отсутствуют сведения о проверке.")
            for name in ("rmse", "normalized_rmse"):
                value = record.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                    raise ValueError("Эксперимент: некорректная ошибка проверки.")
        else:
            required = ("time", "output", "error", "control")
            records = record.get("runs")
            if not isinstance(records, list) or len(records) != 2:
                raise ValueError("Эксперимент: нужны два опыта проверки PI.")
            for name in ("eta0", "baseline", "setpoint", "horizon"):
                value = record.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError("Эксперимент: некорректные параметры проверки PI.")
            from .experiment_control import model_parameters
            model_parameters(record.get("model", {}))
        for run in records:
            if not isinstance(run, dict):
                raise ValueError("Эксперимент: некорректный опыт проверки.")
            try:
                arrays = [np.asarray(run.get(name), dtype=float) for name in required]
            except (TypeError, ValueError) as error:
                raise ValueError("Эксперимент: некорректные сигналы проверки.") from error
            if (any(a.ndim != 1 or not 2 <= a.size <= 200002 or not np.isfinite(a).all() for a in arrays)
                    or len({a.size for a in arrays}) != 1 or np.any(np.diff(arrays[0]) <= 0)):
                raise ValueError("Эксперимент: нарушены сигналы проверки.")
            if key == "pi_result":
                if (not isinstance(run.get("name"), str) or not isinstance(run.get("metrics"), dict)
                        or any(name not in run["metrics"] for name in ("final_error", "iae", "overshoot_percent", "settling_time", "reachable", "saturation_duration"))
                        or any(name not in run for name in ("gain", "integral_time"))
                        or np.any((arrays[3] < 0) | (arrays[3] > 1))):
                    raise ValueError("Эксперимент: некорректный результат PI.")
                for name, value in {**{name: run[name] for name in ("gain", "integral_time")}, **run["metrics"]}.items():
                    if name in ("reachable", "output_out_of_range"):
                        if not isinstance(value, bool):
                            raise ValueError("Эксперимент: некорректный признак проверки PI.")
                    elif name == "settling_time" and value is None:
                        continue
                    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                        raise ValueError("Эксперимент: показатели проверки PI должны быть конечными числами.")
                if (run["gain"] < 0 or run["integral_time"] <= 0 or any(run["metrics"][name] < 0 for name in
                        ("iae", "overshoot_percent", "saturation_duration"))):
                    raise ValueError("Эксперимент: показатели проверки PI вне допустимых границ.")
        if key == "pi_result" and (record.get("input_role") != "η, доля" or record.get("output_role") not in ("Xог, доля", "Xна, доля")):
            raise ValueError("Эксперимент: неизвестное назначение сигналов PI.")
    if not isinstance(state.get("same_signals", False), bool):
        raise ValueError("Эксперимент: подтверждение сигналов должно быть логическим.")
    return state


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
