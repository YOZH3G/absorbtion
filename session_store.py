"""Portable JSON sessions for the comparison history."""

import copy
import json
import math
from pathlib import Path

import numpy as np


FORMAT_VERSION = 1


def write_session(path, runs, counter):
    destination = Path(path)
    payload = {
        "version": FORMAT_VERSION,
        "comparison_counter": int(counter),
        "runs": [_serialize_run(run) for run in runs],
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def read_session(path):
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Не удалось прочитать учебный сеанс: {error}") from error
    if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"Поддерживается формат учебного сеанса версии {FORMAT_VERSION}.")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Поле runs должно содержать список опытов.")
    restored = [_deserialize_run(run) for run in runs]
    counter = payload.get("comparison_counter", len(restored))
    if isinstance(counter, bool) or not isinstance(counter, int) or counter < len(restored):
        counter = len(restored)
    return restored, counter


def _serialize_run(run):
    serialized = copy.deepcopy(run)
    serialized["time"] = _serialize_signal(run.get("time"), "time")
    serialized["response"] = _serialize_signal(run.get("response"), "response")
    return serialized


def _deserialize_run(run):
    if not isinstance(run, dict):
        raise ValueError("Опыт должен быть объектом JSON.")
    required_text = ("id", "name", "chain", "controller_type")
    for key in required_text:
        if not isinstance(run.get(key), str) or not run[key].strip():
            raise ValueError(f"Опыт: поле {key} должно быть непустым текстом.")
    restored = copy.deepcopy(run)
    restored["time"] = _deserialize_signal(run.get("time"), "time")
    restored["response"] = _deserialize_signal(run.get("response"), "response")
    if restored["time"].shape != restored["response"].shape:
        raise ValueError("Опыт: массивы time и response должны иметь одинаковую длину.")
    for key in ("time_constant", "delay", "maximum_deviation", "static_error"):
        restored[key] = _finite_number(run.get(key), key)
    relative = run.get("relative_deviation")
    restored["relative_deviation"] = None if relative is None else _finite_number(relative, "relative_deviation")
    settling = run.get("settling_duration")
    restored["settling_duration"] = None if settling is None else _finite_number(settling, "settling_duration")
    input_state = run.get("input_state")
    if input_state is not None and not isinstance(input_state, dict):
        raise ValueError("Опыт: input_state должен быть объектом JSON.")
    return restored


def _serialize_signal(values, label):
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        raise ValueError(f"Опыт: {label} должен быть непустым конечным одномерным сигналом.")
    return signal.tolist()


def _deserialize_signal(values, label):
    try:
        signal = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Опыт: {label} должен быть числовым сигналом.") from error
    if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
        raise ValueError(f"Опыт: {label} должен быть непустым конечным одномерным сигналом.")
    return signal


def _finite_number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"Опыт: {label} должно быть числом.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Опыт: {label} должно быть числом.") from error
    if not math.isfinite(number):
        raise ValueError(f"Опыт: {label} должно быть конечным числом.")
    return number
