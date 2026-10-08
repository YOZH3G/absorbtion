"""Model-independent plots and CSV/HTML/PDF exports for named signals."""

import csv
import html
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .exporting import _figures_markup, _text_pages, _write_pdf
from .plotting import adaptive_legend


def signal_groups(result, registry):
    model = registry.get(result["model_id"])
    groups = {}
    for signal in model.signals:
        if signal.key in result["signals"]:
            groups.setdefault((signal.role, signal.display_unit or signal.unit), []).append(signal)
    return groups


def draw_signals(axis, result, definitions):
    for signal in definitions:
        axis.plot(result["time"], np.asarray(result["signals"][signal.key]) * signal.scale, label=signal.label)
    keys = {s.key: s for s in definitions}
    for limit in result.get("limits", []):
        if limit["signal"] in keys:
            axis.axhline(limit["value"] * keys[limit["signal"]].scale, linestyle="--", label=limit["label"])
    for event in result["events"]:
        axis.axvline(event["time"], color="#B91C1C", linestyle=":", label=event["kind"])
    axis.set_xlabel("Время, с")
    if definitions:
        axis.set_ylabel(definitions[0].display_unit or definitions[0].unit)
    axis.grid(alpha=.2)


def result_figures(result, registry):
    data = registry.validate(result)
    figures = []
    model = registry.get(data["model_id"])
    for (role, unit), signals in signal_groups(data, registry).items():
        figure = Figure(figsize=(9, 5), layout="constrained")
        axis = figure.add_subplot()
        draw_signals(axis, data, signals)
        heading = dict(input="Входы", output="Выходы", error="Ошибка", state="Состояния").get(role, role)
        figure.suptitle(f"{model.name}: {heading}, {unit}")
        adaptive_legend(axis)
        figures.append(figure)
    return figures


def write_model_report(path, result, format_name, registry):
    data = registry.validate(result)
    model = registry.get(data["model_id"])
    destination = Path(path)
    if format_name == "csv":
        definitions = [s for s in model.signals if s.key in data["signals"]]
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["Время, с", *[f"{s.label}, {s.unit}" for s in definitions]])
            writer.writerows(zip(data["time"], *[data["signals"][s.key] for s in definitions]))
        return destination
    if format_name not in ("html", "pdf"):
        raise ValueError("Поддерживаются CSV, HTML и PDF.")
    title = f"Результат модели: {model.name}"
    sections = [("Физическая модель", f"{model.id}; версия расчёта {model.version}.\nСнимок результата; полная учебная работа сохраняется отдельным сеансом."),
                ("Параметры", "\n".join(f"{p.label}: {data['parameters'][p.key]:.8g} {p.unit}" for p in model.parameters)),
                ("Конфигурация опыта", str(data.get("configuration", {}))),
                ("Сигналы", "\n".join(f"{s.label}: хранение {s.unit}, график {s.display_unit or s.unit}" for s in model.signals if s.key in data["signals"])),
                ("События", str(data["events"])), ("Показатели", str(data.get("metrics", {})))]
    figures = result_figures(data, registry)
    if format_name == "pdf":
        return _write_pdf(destination, [*_text_pages(title, sections), *figures])
    markup = "".join(f"<h2>{html.escape(name)}</h2><pre>{html.escape(content)}</pre>" for name, content in sections)
    destination.write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'
        f'<title>{html.escape(title)}</title><style>body{{font:16px sans-serif;max-width:1000px;margin:32px auto}}pre{{white-space:pre-wrap}}img{{max-width:100%}}</style>'
        f'<h1>{html.escape(title)}</h1>{markup}' + _figures_markup(figures, "Сигналы модели") + '</html>', encoding="utf-8")
    return destination
