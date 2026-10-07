"""Optional sensor and actuator settings shared by forms, scenarios and simulation."""

import math

EXTENSION_DEFAULTS = {"noise_std": 0.0, "noise_seed": 0, "derivative_filter_time": 0.0,
                      "actuator_time_constant": 0.0, "actuator_rate_limit": 0.0}

EXTENSION_FIELDS = (("noise_std", "Шум измерения, п.п.", 100),
                    ("noise_seed", "Начальное число шума", 1),
                    ("derivative_filter_time", "Фильтр D, с", 1),
                    ("actuator_time_constant", "Инерция механизма, с", 1),
                    ("actuator_rate_limit", "Скорость η, п.п./с", 100))


def normalize_extensions(controller):
    result = {}
    for key, default in EXTENSION_DEFAULTS.items():
        label = next(label for field, label, _scale in EXTENSION_FIELDS if field == key)
        value = controller.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"{label}: введите число.")
        try:
            number = float(value.replace(",", ".") if isinstance(value, str) else value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}: введите число.") from error
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{label}: нужно конечное неотрицательное число.")
        if key == "noise_std" and number > 1:
            raise ValueError("Шум измерения: не больше 100 п.п.")
        if key == "noise_seed":
            if not number.is_integer() or number > 4294967295:
                raise ValueError("Начальное число шума: целое от 0 до 4294967295.")
            number = int(number)
        result[key] = number
    return result


def extensions_from_form(state, kind):
    values = {}
    for key, label, scale in EXTENSION_FIELDS:
        if key == "derivative_filter_time" and "D" not in kind:
            values[key] = 0
            continue
        if key == "noise_seed" and values["noise_std"] == 0:
            values[key] = 0
            continue
        try:
            values[key] = float(state.get(key, "0").replace(",", ".")) / scale
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"{label}: введите число.") from error
    return normalize_extensions(values)
