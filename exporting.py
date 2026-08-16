import csv
from pathlib import Path

from laboratory import format_protocol
from simulation import LEAN_GAS


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


def _format_number(value):
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _format_signed_number(value):
    if abs(value) < 0.00005:
        return "0"
    return f"{value:+.4f}".rstrip("0").rstrip(".").replace("-", "−")
