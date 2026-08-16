import math


MIN_FRACTION = -0.99
MAX_FRACTION = 9.99


def _parse_number(value, empty_message, invalid_message):
    text = value.strip()
    if not text:
        raise ValueError(empty_message)
    try:
        return float(text)
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
