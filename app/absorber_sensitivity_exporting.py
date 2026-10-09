"""Portable reports with all trials, physical conditions and both phases."""

import csv
import html
from pathlib import Path

from matplotlib.figure import Figure

from .absorber_sensitivity import REASONS, REQUIREMENTS, validate_result
from .exporting import _figures_markup, _text_pages, _write_pdf
from .plotting import adaptive_legend
from .tank_sensitivity_exporting import case_label, draw_region


def assessment_text(case):
    return "Требования выполнены" if case["assessment"]["passed"] else "; ".join(REASONS[key] for key in case["assessment"]["reasons"])


def draw_responses(axis, result, selected):
    output = "xog" if result["request"]["chain"] == "lean_gas" else "xna"
    for case in [*result["controls"], result["cases"][selected]]:
        run = case["run"]
        axis.plot(run["time"], run["signals"][output], label=case_label(case))
    axis.axhline(result["request"]["controller"]["setpoint"], linestyle="--", color="#64748B", label="Задание")
    axis.set_xlabel("Время, с"); axis.set_ylabel(("Xог" if output == "xog" else "Xна")+", доля"); axis.grid(alpha=.2)


def report_parts(result, selected):
    validate_result(result)
    if type(selected) is not int or not 0 <= selected < len(result["cases"]):
        raise ValueError("Выберите существующий опыт.")
    request, local = result["request"], result["local_model"]
    sections = [("Методика", "Оценки K/T/L из CSV используются только для настройки PI. Один неизменный абсорбер проверяет все настройки.\n"
        "λ=max(0.5T,L); Kp=T/[|K|(λ+L)]; Ti=min(T,4(λ+L)). Направление управления задаёт физический канал.\n"
        "Ошибки оценки не меняют реальную динамику объекта. Полоса установления: 5% модуля ступени задания; подтверждение не меньше реального T после окончания возмущения.\n"
        "Проверены только узлы сетки. Это не доказательство устойчивости между ними и не доверительная область."),
        ("Оценка и источник", f"K={local['gain']:.8g}; T={local['time_constant']:.8g} с; L={local['delay']:.8g} с.\n"
         + f"CSV: {result.get('origin', {}).get('source', 'не указан')}; версия идентификации: {result.get('origin', {}).get('result_revision', 'не указана')}."),
        ("Неизменный объект", "Канал: η → " + ("Xог" if request["chain"] == "lean_gas" else "Xна") + ".\n"
         + "\n".join(f"{label}: {request['parameters'][key]:.8g}" for key, label in
              (("gg", "Расход газа, кг/ч"), ("ga", "Расход абсорбента, кг/ч"), ("xg", "Вход газа, доля"),
               ("xa", "Вход абсорбента, доля"), ("eta", "Исходная η, доля")))
         + "\n" + "\n".join(f"{label}: {request['dynamics'][key]}" for key, label in
              (("time_constant", "Реальное T, с"), ("delay", "Реальное L, с"), ("kind", "Профиль возмущения"),
               ("start_time", "Начало, с"), ("effect_duration", "Длительность воздействия, с"), ("simulation_duration", "Горизонт, с")))
         + f"\nВозмущение состава: {request['component']:.8g}; расхода: {request['flow']:.8g}, относительные доли.\n"
         + "\n".join(f"{label}: {request['controller'][key]}" for key, label in
              (("setpoint", "Задание, доля"), ("control_limit", "Предел Δη, доля"), ("noise_std", "Шум, доля"),
               ("noise_seed", "Начальное число шума"), ("actuator_time_constant", "Инерция механизма, с"),
               ("actuator_rate_limit", "Скорость η, доля/с")))),
        ("Требования", "\n".join(f"{REASONS[key]} ≤ {result['requirements'][key]:.8g} "
         + dict(final_error="доля (по модулю)", iae="доля·с", overshoot_percent="%", settling_time="с", saturation_duration="с")[key] for key in REQUIREMENTS)),
        ("Выбранный опыт", case_label(result["cases"][selected])+"\n"+assessment_text(result["cases"][selected]))]
    if result.get("exported_stale"):
        sections.append(("Актуальность", "Снимок прежних данных или настроек; текущая форма в этот расчёт не входит."))
    lines = []
    for case in result["controls"]+result["cases"]:
        c, m = case["run"]["configuration"]["controller"], case["metrics"]
        settled = "не подтверждено" if m["settling_time"] is None else f"{m['settling_time']:.6g} с"
        lines.append(f"{case_label(case)}: Kp={c['controller_gain']:.6g}; Ti={c['integral_time']:.6g} с; "
            f"e={m['final_error']:.6g}; IAE={m['iae']:.6g} доля·с; выброс={m['overshoot_percent']:.5g}%; "
            f"tуст={settled}; tнас={m['saturation_duration']:.5g} с. {assessment_text(case)}")
    sections.append(("Все проверенные сочетания", "\n".join(lines)))
    figures = []
    for delay in result["grid"]["l"]:
        figure = Figure(figsize=(9, 5), layout="constrained"); axis = figure.add_subplot()
        draw_region(axis, result, delay); adaptive_legend(axis)
        figure.suptitle(f"Проверенные сочетания: ΔL={delay:g} с"); figures.append(figure)
    figure = Figure(figsize=(9, 5), layout="constrained"); axis = figure.add_subplot()
    draw_responses(axis, result, selected); adaptive_legend(axis)
    figure.suptitle("Три PI на одном абсорбере"); figures.append(figure)
    return sections, figures


def write_report(path, result, selected, format_name):
    sections, figures = report_parts(result, selected)
    destination = Path(path)
    if format_name == "csv":
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(["Опыт", "K, доля/доля", "T, с", "L, с", "Kp", "Ti, с", "Время, с",
                "η, доля", "Команда η, доля", "Xог, доля", "Xна, доля", "Ошибка, доля", "Измеренный выход, доля",
                "Состав, относительная доля", "Расход, относительная доля", "Требования выполнены", "Причины",
                "Снимок прежних настроек", *REQUIREMENTS, *["Порог: "+key for key in REQUIREMENTS]])
            for case in result["controls"]+result["cases"]:
                run = case["run"]; local = case.get("estimate", result["local_model"]); c = run["configuration"]["controller"]
                for i, time in enumerate(run["time"]):
                    writer.writerow([case_label(case), *[local[key] for key in ("gain", "time_constant", "delay")],
                        c["controller_gain"], c["integral_time"], time,
                        *[run["signals"][key][i] for key in ("control", "commanded_control", "xog", "xna", "error", "measurement", "component", "flow")],
                        case["assessment"]["passed"], assessment_text(case), result.get("exported_stale", False),
                        *[case["metrics"][key] for key in REQUIREMENTS], *[result["requirements"][key] for key in REQUIREMENTS]])
        return destination
    if format_name == "pdf":
        return _write_pdf(destination, [*_text_pages("Ошибки идентификации и PI абсорбера", sections), *figures])
    if format_name != "html": raise ValueError("Поддерживаются CSV, HTML и PDF.")
    markup = "".join(f"<h2>{html.escape(name)}</h2><pre>{html.escape(text)}</pre>" for name, text in sections)
    destination.write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Ошибки идентификации и PI абсорбера</title>'
        '<style>body{font:16px sans-serif;max-width:1000px;margin:32px auto}pre{white-space:pre-wrap}img{max-width:100%}</style>'
        '<h1>Ошибки идентификации и PI абсорбера</h1>'+markup+_figures_markup(figures, "Проверка PI")+'</html>', encoding="utf-8")
    return destination
