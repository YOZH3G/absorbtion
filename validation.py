import math


MIN_FRACTION = 0.01
MAX_FRACTION = 9.99


def parse_fraction(value):
    """Parse a finite disturbance fraction within the supported range."""
    text = value.strip()
    if not text:
        raise ValueError("Введите значение. Поле возмущающего воздействия не может быть пустым.")

    try:
        fraction = float(text)
    except ValueError as error:
        raise ValueError("Некорректное значение. Введите число от 0.01 до 9.99.") from error

    if not math.isfinite(fraction) or not MIN_FRACTION <= fraction <= MAX_FRACTION:
        raise ValueError("Значение должно быть в диапазоне от 0.01 до 9.99.")

    return fraction


def parse_positive_number(value):
    """Parse a finite number greater than zero."""
    text = value.strip()
    if not text:
        raise ValueError("Введите значение.")

    try:
        number = float(text)
    except ValueError as error:
        raise ValueError("Введите корректное число.") from error

    if not math.isfinite(number) or number <= 0:
        raise ValueError("Значение должно быть больше нуля.")

    return number
