import csv
import copy
from pathlib import Path

import numpy as np


MAX_COMPARISON_RUNS = 6


def build_comparison_run(result, name, input_state=None):
    metrics = result["metrics"]
    settling_time = metrics["settling_time"]
    settling_duration = (
        None
        if settling_time is None
        else max(0.0, settling_time - result["response_start"])
    )
    controller = result["controller"]
    run = {
        "name": name,
        "chain": result["chain"],
        "time": np.array(result["time"], dtype=float, copy=True),
        "response": np.array(result["final_response"], dtype=float, copy=True),
        "time_constant": float(result["dynamics"]["time_constant"]),
        "delay": float(result["dynamics"]["delay"]),
        "controller_type": "—" if controller is None else controller["controller_type"],
        "maximum_deviation": float(metrics["maximum_deviation"]),
        "relative_deviation": metrics["relative_deviation"],
        "settling_duration": settling_duration,
        "static_error": float(metrics["static_error"]),
    }
    if input_state is not None:
        run["input_state"] = copy.deepcopy(input_state)
    return run


def write_comparison_csv(path, runs):
    if not runs:
        raise ValueError("Нет опытов для экспорта.")
    destination = Path(path)
    with destination.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow((
            "Опыт",
            "Время, с",
            "Выход",
            "T, с",
            "L, с",
            "Регулятор",
            "Максимальное отклонение",
            "Относительное отклонение, %",
            "Длительность установления, с",
            "Статическая ошибка",
        ))
        for run in runs:
            for time_value, response_value in zip(run["time"], run["response"], strict=True):
                writer.writerow((
                    run["name"],
                    _format_number(time_value),
                    _format_number(response_value),
                    _format_number(run["time_constant"]),
                    _format_number(run["delay"]),
                    run["controller_type"],
                    _format_number(run["maximum_deviation"]),
                    _format_optional(run["relative_deviation"]),
                    _format_optional(run["settling_duration"]),
                    _format_number(run["static_error"]),
                ))
    return destination


def _format_optional(value):
    return "" if value is None else _format_number(value)


def _format_number(value):
    return f"{float(value):.10g}"
