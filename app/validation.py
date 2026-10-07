import math
from decimal import Decimal


MIN_FRACTION = -0.99
MAX_FRACTION = 9.99


def _parse_number(value, empty_message, invalid_message):
    text = value.strip()
    if not text:
        raise ValueError(empty_message)
    try:
        return float(text.replace(",", "."))
    except ValueError as error:
        raise ValueError(invalid_message) from error


def parse_fraction(value):
    """Parse a finite disturbance fraction within the supported range."""
    fraction = _parse_number(
        value,
        "Введите значение. Поле возмущающего воздействия не может быть пустым.",
        "Некорректное значение. Введите число от −0.99 до 9.99.",
    )

    if not math.isfinite(fraction) or not MIN_FRACTION <= fraction <= MAX_FRACTION:
        raise ValueError("Значение должно быть в диапазоне от −0.99 до 9.99.")

    return fraction


def parse_positive_number(value):
    """Parse a finite number greater than zero."""
    number = _parse_number(value, "Введите значение.", "Введите корректное число.")

    if not math.isfinite(number) or number <= 0:
        raise ValueError("Значение должно быть больше нуля.")

    return number


def parse_nonnegative_number(value):
    """Parse a finite number greater than or equal to zero."""
    number = _parse_number(value, "Введите значение.", "Введите корректное число.")

    if not math.isfinite(number) or number < 0:
        raise ValueError("Значение не может быть отрицательным.")

    return number


def parse_percentage(value):
    number = parse_nonnegative_number(value)
    if number > 100:
        raise ValueError("Значение должно быть в диапазоне от 0 до 100%.")
    return number / 100.0


def parse_disturbance(value, units="fraction"):
    if units == "fraction":
        return parse_fraction(value)
    if units != "percent":
        raise ValueError("Неизвестные единицы возмущения.")
    number = _parse_number(value, "Введите значение.", "Введите число от −99 до 999%.")
    if not math.isfinite(number) or not -99 <= number <= 999:
        raise ValueError("Значение должно быть в диапазоне от −99 до 999%.")
    return float(Decimal(value.strip().replace(",", ".")).scaleb(-2))


def parse_value_list(value, parser):
    """A decimal comma belongs to one number; semicolons/newlines separate values."""
    source = value.replace("\n", ";").split(";")
    if not 2 <= len(source) <= 6 or any(not item.strip() for item in source):
        raise ValueError("Укажите от 2 до 6 значений через точку с запятой или с новой строки.")
    values = tuple(parser(item.strip()) for item in source)
    if len(set(values)) != len(values):
        raise ValueError("Значения параметра не должны повторяться.")
    return values
