import csv
import base64
import html
import textwrap
from io import BytesIO
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .laboratory import format_protocol
from .simulation import LEAN_GAS


def save_graphs(signal_figure, response_figure, selected_path):
    base = Path(selected_path)
    stem = base.with_suffix("")
    signal_path = stem.with_name(f"{stem.name}_signals.png")
    response_path = stem.with_name(f"{stem.name}_response.png")
    signal_figure.savefig(signal_path, dpi=160, bbox_inches="tight")
    response_figure.savefig(response_path, dpi=160, bbox_inches="tight")
    return signal_path, response_path


def write_csv(selected_path, result):
    columns = [
        ("Время, с", result["time"]),
        ("Профиль воздействия", result["profile"]),
        ("Цель: совместное воздействие", result["targets"]["Совместное воздействие"]),
    ]
    columns.extend(
        (f"Отклик: {label}", values)
        for label, values in result["responses"].items()
    )
    if result["controlled_response"] is not None:
        columns.extend((
            ("Регулируемый выход", result["controlled_response"]),
            ("Ошибка e(t)", result["error"]),
            ("Управляющее воздействие u(t)", result["control"]),
        ))
    path = Path(selected_path)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(label for label, _values in columns)
        writer.writerows(zip(*(values for _label, values in columns)))
    return path


def build_protocol(result, title):
    controller = result["controller"]
    parameters = [
        ("Цепь", "Обеднённый газ" if result["chain"] == LEAN_GAS else "Насыщенный абсорбент"),
        ("Возмущение состава", f"{result['component_fraction'] * 100:+.1f}%"),
        ("Возмущение расхода", f"{result['flow_fraction'] * 100:+.1f}%"),
        ("Вид воздействия", result["disturbance_type"]),
        ("Начало воздействия", f"{result['dynamics']['start_time']:g} с"),
        ("Длительность моделирования", f"{result['dynamics']['simulation_duration']:g} с"),
        ("Постоянная времени T", f"{result['dynamics']['time_constant']:g} с"),
        ("Запаздывание L", f"{result['dynamics']['delay']:g} с"),
        ("Режим", result["result_mode"]),
    ]
    if controller is not None:
        parameters.extend((
            ("K", f"{controller['controller_gain']:g}"),
            (
                "Ti",
                f"{controller['integral_time']:g} с"
                if "I" in controller["controller_type"]
                else "не используется",
            ),
            (
                "Td",
                f"{controller['derivative_time']:g} с"
                if "D" in controller["controller_type"]
                else "не используется",
            ),
            ("Ограничение |u|", f"{controller['control_limit']:g}"),
            ("Задание", f"{controller['setpoint']:g}"),
        ))

    metrics = result["metrics"]
    relative_deviation = metrics["relative_deviation"]
    settling_time = metrics["settling_time"]
    settling_duration = (
        None
        if settling_time is None
        else max(0.0, settling_time - result["response_start"])
    )
    results = [
        ("Базовое значение", _format_number(result["baseline"])),
        ("Суммарная доля", f"{_format_number(result['combined_fraction'])} ({result['combined_fraction'] * 100:+.1f}%)"),
        ("Расчётное значение", _format_number(result["calculated"])),
        ("В конце моделирования", _format_number(result["final_response"][-1])),
        ("Установившееся значение", _format_number(metrics["steady_state"])),
        ("Максимальное отклонение", _format_number(metrics["maximum_deviation"])),
        (
            "Относительное отклонение",
            f"{relative_deviation:.2f}%" if relative_deviation is not None else "не определено",
        ),
        (
            "Длительность регулирования (±5%)"
            if controller is not None
            else "Длительность установления (±5%)",
            "не достигнуто" if settling_duration is None else f"{settling_duration:.1f} с",
        ),
        ("Статическая ошибка", _format_signed_number(metrics["static_error"])),
    ]
    return format_protocol(title, parameters, results)


def write_protocol(selected_path, protocol):
    path = Path(selected_path)
    path.write_text(protocol, encoding="utf-8")
    return path


def write_html_report(
    selected_path,
    result,
    title,
    lesson,
    evaluation,
    student_name,
    conclusion,
    figures,
    prediction=None,
    comparison_runs=(),
):
    images = _figures_markup(figures, "График расчёта")
    score = "Не оценивалось" if evaluation is None else f"{evaluation['score']} из {evaluation['total']}"
    questions = "".join(f"<li>{html.escape(question)}</li>" for question in lesson["questions"])
    prediction_markup = _prediction_markup(prediction)
    criteria_markup = _criteria_markup(evaluation)
    comparison_markup = _comparison_markup(comparison_runs)
    report = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:16px Segoe UI,Arial,sans-serif;color:#1f2937;max-width:1040px;margin:32px auto;line-height:1.5}}h1,h2{{color:#10243e}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:16px}}img{{max-width:100%;margin:18px 0;border:1px solid #d7dce2}}</style></head>
<body><h1>Отчёт по лабораторной работе</h1><p><b>Студент:</b> {html.escape(student_name or 'не указан')}</p>
<h2>{html.escape(title)}</h2><p>{html.escape(lesson['task'])}</p><p>{html.escape(lesson['guidance'])}</p>
<h2>Вопросы</h2><ul>{questions or '<li>Не заданы</li>'}</ul><h2>Оценка</h2><p>{score}</p>
{criteria_markup}<h2>Прогноз студента</h2>{prediction_markup}<h2>Сравнение опытов</h2>{comparison_markup}
<pre>{html.escape(build_protocol(result, title))}</pre><h2>Графики</h2>{images}
<h2>Вывод студента</h2><p>{html.escape(conclusion or 'не указан')}</p></body></html>"""
    path = Path(selected_path)
    path.write_text(report, encoding="utf-8")
    return path


def write_pdf_report(
    selected_path,
    result,
    title,
    lesson,
    evaluation,
    student_name,
    conclusion,
    figures,
    prediction=None,
    comparison_runs=(),
):
    score = "Не оценивалось" if evaluation is None else f"{evaluation['score']} из {evaluation['total']}"
    sections = (
        ("Студент", student_name or "не указан"),
        ("Сценарий", title),
        ("Задание", lesson["task"]),
        ("Методические указания", lesson["guidance"]),
        ("Оценка", score),
        ("Прогноз", _prediction_text(prediction)),
        ("Показатели расчёта", build_protocol(result, title)),
        ("Закреплённые опыты", _comparison_text(comparison_runs)),
        ("Вывод студента", conclusion or "не указан"),
    )
    return _write_pdf(selected_path, (_text_page("Отчёт по лабораторной работе", sections), *figures))


def write_comparison_html_report(selected_path, runs):
    if not runs:
        raise ValueError("Нет закреплённых опытов для отчёта.")
    figures = build_comparison_figures(runs)
    images = _figures_markup(figures, "График сравнения")
    report = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Отчёт по сравнению опытов</title>
<style>body{{font:16px Segoe UI,Arial,sans-serif;color:#1f2937;max-width:1040px;margin:32px auto;line-height:1.5}}h1,h2{{color:#10243e}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border:1px solid #d7dce2;text-align:left}}th{{background:#f4f6f8}}img{{max-width:100%;margin:18px 0;border:1px solid #d7dce2}}</style></head>
<body><h1>Отчёт по сравнению закреплённых опытов</h1><p>Всего опытов: {len(runs)}</p>
{_comparison_markup(runs)}<h2>Графики</h2>{images}</body></html>"""
    path = Path(selected_path)
    path.write_text(report, encoding="utf-8")
    return path


def write_comparison_pdf_report(selected_path, runs):
    if not runs:
        raise ValueError("Нет закреплённых опытов для отчёта.")
    sections = (
        ("Количество закреплённых опытов", str(len(runs))),
        ("Сводные показатели", _comparison_text(runs)),
    )
    figures = build_comparison_figures(runs)
    return _write_pdf(
        selected_path,
        (_text_page("Отчёт по сравнению закреплённых опытов", sections), *figures),
    )


def build_comparison_figures(runs):
    if not runs:
        raise ValueError("Нет закреплённых опытов для отчёта.")
    colors = ("#2563EB", "#F59E0B", "#16A34A", "#7C3AED", "#DB2777", "#0891B2")
    response_figure = Figure(figsize=(10, 4.8), constrained_layout=True)
    response_axis = response_figure.add_subplot(111)
    for index, run in enumerate(runs):
        response_axis.plot(
            run["time"], run["response"], color=colors[index % len(colors)],
            linewidth=2.2, label=run["name"],
        )
    response_axis.set_title("Сравнение переходных процессов")
    response_axis.set_xlabel("Время, с")
    response_axis.set_ylabel("Концентрация")
    response_axis.grid(color="#D7DCE2", linewidth=0.8)
    response_axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(runs), frameon=False)

    metrics_figure = Figure(figsize=(10, 4.8), constrained_layout=True)
    metrics_axis = metrics_figure.add_subplot(111)
    durations = [0.0 if run["settling_duration"] is None else run["settling_duration"] for run in runs]
    positions = np.arange(len(runs))
    bars = metrics_axis.bar(positions, durations, color=[colors[index % len(colors)] for index in positions])
    metrics_axis.set_title("Длительность установления")
    metrics_axis.set_ylabel("Время, с")
    metrics_axis.set_xticks(positions, [run["name"] for run in runs], rotation=20, ha="right")
    metrics_axis.grid(axis="y", color="#D7DCE2", linewidth=0.8)
    for bar, run in zip(bars, runs, strict=True):
        label = "не достигнуто" if run["settling_duration"] is None else f"{run['settling_duration']:.1f} с"
        metrics_axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom")
    metrics_axis.margins(y=0.18)
    return response_figure, metrics_figure


def _figure_base64(figure):
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _figures_markup(figures, alt_text):
    return "".join(
        f'<img src="data:image/png;base64,{_figure_base64(figure)}" alt="{alt_text}">'
        for figure in figures
    )


def _write_pdf(selected_path, figures):
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(selected_path)
    with PdfPages(path) as pdf:
        for figure in figures:
            pdf.savefig(figure, bbox_inches="tight")
    return path


def _text_page(title, sections):
    figure = Figure(figsize=(8.27, 11.69))
    axis = figure.add_axes((0.08, 0.06, 0.84, 0.88))
    axis.set_axis_off()
    y = 0.98
    axis.text(0, y, title, fontsize=18, fontweight="bold", va="top")
    y -= 0.06
    for heading, content in sections:
        axis.text(0, y, heading, fontsize=11, fontweight="bold", va="top")
        y -= 0.025
        for line in _wrapped_lines(content):
            axis.text(0, y, line, fontsize=9, va="top", family="DejaVu Sans")
            y -= 0.018
            if y < 0.05:
                return figure
        y -= 0.018
    return figure


def _wrapped_lines(value):
    lines = []
    for paragraph in str(value).splitlines() or ("",):
        lines.extend(textwrap.wrap(paragraph, width=100) or [""])
    return lines


def _prediction_text(prediction):
    if prediction is None:
        return "Режим задания не использовался."
    labels = {
        "direction": "Направление изменения",
        "steady": "Установившееся значение",
        "fastest": "Самая быстрая кривая",
        "correction": "Действие регулятора",
    }
    return "\n".join(f"{labels[key]}: {value}" for key, value in prediction.items() if key in labels)


def _comparison_text(runs):
    if not runs:
        return "Закреплённые опыты отсутствуют."
    return "\n".join(
        f"{run['name']}: регулятор {run['controller_type']}; "
        f"максимальное отклонение {_format_number(run['maximum_deviation'])}; "
        f"установление {_format_optional_number(run['settling_duration'])} с."
        for run in runs
    )


def _prediction_markup(prediction):
    if prediction is None:
        return "<p>Режим задания не использовался.</p>"
    labels = {
        "direction": "Направление изменения",
        "steady": "Установившееся значение",
        "fastest": "Самая быстрая кривая",
        "correction": "Действие регулятора",
    }
    items = "".join(
        f"<li><b>{labels[key]}:</b> {html.escape(str(value))}</li>"
        for key, value in prediction.items() if key in labels
    )
    return f"<ul>{items or '<li>Не указан</li>'}</ul>"


def _criteria_markup(evaluation):
    if evaluation is None or not evaluation.get("criteria"):
        return ""
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['label']))}</td>"
        f"<td>{'Верно' if item['passed'] else 'Неверно'}</td>"
        f"<td>{item['points']} / {item['maximum']}</td>"
        "</tr>"
        for item in evaluation["criteria"]
    )
    return (
        "<table><thead><tr><th>Критерий</th><th>Результат</th><th>Баллы</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _comparison_markup(runs):
    if not runs:
        return "<p>Закреплённые опыты отсутствуют.</p>"
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(run['name']))}</td>"
        f"<td>{html.escape(str(run['controller_type']))}</td>"
        f"<td>{_format_number(run['maximum_deviation'])}</td>"
        f"<td>{_format_optional_number(run['settling_duration'])}</td>"
        "</tr>"
        for run in runs
    )
    return (
        "<table><thead><tr><th>Опыт</th><th>Регулятор</th>"
        "<th>Максимальное отклонение</th><th>Установление, с</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _format_optional_number(value):
    return "—" if value is None else _format_number(value)


def _format_number(value):
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_signed_number(value):
    if abs(value) < 0.00005:
        return "0"
    return f"{value:+.4f}".rstrip("0").rstrip(".").replace("-", "−")
