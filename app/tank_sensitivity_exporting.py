"""Full grid and selected physical traces for identification-error experiments."""

import csv
import html
from pathlib import Path

from matplotlib.figure import Figure

from .exporting import _figures_markup, _text_pages, _write_pdf
from .plotting import adaptive_legend
from .tank_exporting import PARAMETER_TEXT
from .tank_sensitivity import REASONS, REQUIREMENTS, validate_result


def case_label(case):
    if "name" in case:
        return case["name"]
    k, t, delay = case["offsets"]
    return f"δK={k:g}%; δT={t:g}%; ΔL={delay:g} с"


def assessment_text(case):
    return "Требования выполнены" if case["assessment"]["passed"] else "; ".join(REASONS[key] for key in case["assessment"]["reasons"])


def draw_region(axis, result, delay):
    for passed, color, marker, label in ((True, "#15803D", "o", "Требования выполнены"),
                                         (False, "#B91C1C", "x", "Требования не выполнены")):
        points = [c["offsets"] for c in result["cases"] if c["offsets"][2] == delay and c["assessment"]["passed"] == passed]
        if points:
            axis.scatter([p[0] for p in points], [p[1] for p in points], color=color, marker=marker, s=65, label=label)
    axis.set_xlabel("δK, %"); axis.set_ylabel("δT, %")
    axis.grid(alpha=.2)


def draw_responses(axis, result, selected):
    for case in [*result["controls"], result["cases"][selected]]:
        run = case["run"]
        axis.plot(run["time"], run["level"], label=case_label(case))
    axis.axhline(result["controls"][0]["run"]["controller"]["setpoint"], linestyle="--", color="#64748B", label="Задание")
    axis.set_xlabel("Время, с"); axis.set_ylabel("Уровень, м"); axis.grid(alpha=.2)


def report_parts(result, selected):
    validate_result(result)
    if type(selected) is not int or not 0 <= selected < len(result["cases"]):
        raise ValueError("Выберите существующий опыт чувствительности.")
    p, local = result["nominal_parameters"], result["local_model"]
    sections = [("Методика", "Ошибочные оценки K/T/L используются только для настройки PI. Все регуляторы управляют одним неизменным нелинейным баком.\n"
                 "λ=max(0.5T,L); Kp=T/[K(λ+L)]; Ti=min(T,4(λ+L)). Изменение оценки L не добавляет задержку физическому баку.\n"
                 "Проверены только узлы заданной сетки; это не доказательство устойчивости и не доверительная область."),
                ("Условия", f"Номинальная оценка: K={local['gain']:.8g} с/м²; T={local['time_constant']:.8g} с; L={local['delay']:.8g} с.\n"
                 + "\n".join(f"{PARAMETER_TEXT[key]}: {value:.8g}" for key, value in p.items())),
                ("Требования", "\n".join(f"{REASONS[key]} ≤ {result['requirements'][key]:.8g} "
                 + dict(final_error="м (по модулю)", iae="м·с", overshoot_percent="%", settling_time="с", saturation_duration="с")[key] for key in REQUIREMENTS)),
                ("Источник", "\n".join((f"Запись: {result.get('origin', {}).get('source', 'не указана')}",
                 f"Версия данных: {result.get('revision', 'не указана')}",
                 "Столбцы: " + "; ".join(result.get("origin", {}).get("columns", [])),
                 "Единицы записи: " + "; ".join(result.get("origin", {}).get("units", []))))),
                ("Выбранный опыт", case_label(result["cases"][selected]) + "\n" + assessment_text(result["cases"][selected]))]
    if result.get("exported_stale"):
        sections.append(("Актуальность", "Снимок прежних настроек; текущие поля формы в этот расчёт не входят."))
    rows = []
    for case in [*result["controls"], *result["cases"]]:
        run = case["run"]; m = run["metrics"]; c = run["controller"]
        settled = "не подтверждено" if m["settling_time"] is None else f"{m['settling_time']:.6g} с"
        rows.append(f"{case_label(case)}: Kp={c['gain']:.6g}; Ti={c['integral_time']:.6g}; "
                    f"e={m['final_error']:.6g} м; IAE={m['iae']:.6g} м·с; выброс={m['overshoot_percent']:.5g}%; "
                    f"tуст={settled}; "
                    f"tнас={m['saturation_duration']:.5g} с. {assessment_text(case)}")
    sections.append(("Все проверенные сочетания", "\n".join(rows)))
    figures = []
    for delay in result["grid"]["l"]:
        figure = Figure(figsize=(9, 5), layout="constrained"); axis = figure.add_subplot()
        draw_region(axis, result, delay); adaptive_legend(axis)
        figure.suptitle(f"Проверенные сочетания: ΔL={delay:g} с"); figures.append(figure)
    figure = Figure(figsize=(9, 5), layout="constrained"); axis = figure.add_subplot()
    draw_responses(axis, result, selected); adaptive_legend(axis)
    figure.suptitle("Три PI на одном нелинейном баке"); figures.append(figure)
    return sections, figures


def write_report(path, result, selected, format_name):
    sections, figures = report_parts(result, selected)
    destination = Path(path)
    if format_name == "csv":
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            metric_headers = ("Конечная ошибка, м", "IAE, м·с", "Перерегулирование, %", "Установление, с", "Насыщение, с")
            writer.writerow(["Опыт", "K, с/м²", "T, с", "L, с", "Kp, м²/с", "Ti, с", "Время, с", "Подача, м³/с", "Уровень, м", "Отбор, м³/с", "Ошибка, м", "Требования выполнены", "Причины", "Снимок прежних настроек", *metric_headers, *["Порог: "+key for key in metric_headers]])
            for case in [*result["controls"], *result["cases"]]:
                run = case["run"]; local = case.get("estimate", result["local_model"])
                prefix = [case_label(case), *[local[key] for key in ("gain", "time_constant", "delay")], run["controller"]["gain"], run["controller"]["integral_time"]]
                for i, time in enumerate(run["time"]):
                    writer.writerow([*prefix, time, run["pump"][i], run["level"][i], run["withdrawal"][i], run["error"][i],
                                     case["assessment"]["passed"], assessment_text(case), result.get("exported_stale", False), *[run["metrics"][key] for key in REQUIREMENTS], *[result["requirements"][key] for key in REQUIREMENTS]])
        return destination
    if format_name == "pdf":
        return _write_pdf(destination, [*_text_pages("Ошибки идентификации и PI бака", sections), *figures])
    if format_name != "html":
        raise ValueError("Поддерживаются CSV, HTML и PDF.")
    markup = "".join(f"<h2>{html.escape(name)}</h2><pre>{html.escape(text)}</pre>" for name, text in sections)
    destination.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Ошибки идентификации и PI бака</title>'
        '<style>body{font:16px sans-serif;max-width:1000px;margin:32px auto}pre{white-space:pre-wrap}img{max-width:100%}</style>'
        '<h1>Ошибки идентификации и PI бака</h1>' + markup + _figures_markup(figures, "Проверка PI") + '</html>', encoding="utf-8")
    return destination
