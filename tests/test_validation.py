import unittest

from app.validation import parse_fraction, parse_nonnegative_number, parse_positive_number


class FractionValidationTests(unittest.TestCase):
    def test_accepts_decimal_comma(self):
        self.assertEqual(parse_fraction("0,15"), 0.15)

    def test_accepts_range_boundaries(self):
        self.assertEqual(parse_fraction("-0.99"), -0.99)
        self.assertEqual(parse_fraction("9.99"), 9.99)

    def test_accepts_zero_and_values_within_range(self):
        self.assertEqual(parse_fraction("0"), 0.0)
        self.assertEqual(parse_fraction("-0.1"), -0.1)
        self.assertEqual(parse_fraction("1"), 1.0)

    def test_rejects_empty_and_non_numeric_values(self):
        for value in ("", " ", "abc"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_fraction(value)

    def test_rejects_non_finite_and_out_of_range_values(self):
        for value in ("nan", "inf", "-inf", "-1", "-1.01", "10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_fraction(value)


class PositiveNumberValidationTests(unittest.TestCase):
    def test_accepts_positive_finite_numbers(self):
        self.assertEqual(parse_positive_number("0.01"), 0.01)
        self.assertEqual(parse_positive_number("7800"), 7800.0)

    def test_rejects_empty_non_numeric_non_finite_and_non_positive_values(self):
        for value in ("", " ", "abc", "nan", "inf", "-1", "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_positive_number(value)


class NonnegativeNumberValidationTests(unittest.TestCase):
    def test_accepts_zero_and_positive_finite_numbers(self):
        self.assertEqual(parse_nonnegative_number("0"), 0.0)
        self.assertEqual(parse_nonnegative_number("12.5"), 12.5)

    def test_rejects_empty_non_numeric_non_finite_and_negative_values(self):
        for value in ("", " ", "abc", "nan", "inf", "-0.01"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_nonnegative_number(value)

class DisturbanceUnitTests(unittest.TestCase):
    def test_percent_and_decimal_fraction_are_equivalent(self):
        from app.validation import parse_disturbance
        for fraction in ("0,1", "0.12345678901234567", "-0.99", "9.99"):
            from decimal import Decimal
            percent = str(Decimal(fraction.replace(",", ".")).scaleb(2))
            self.assertEqual(parse_disturbance(fraction), parse_disturbance(percent, "percent"))
        for value in ("1000", "-100", "nan", "inf", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_disturbance(value, "percent")

    def test_list_uses_semicolon_or_newline_without_splitting_decimal_comma(self):
        from app.validation import parse_value_list
        self.assertEqual(parse_value_list("0,1; 0.25", parse_fraction), (0.1, 0.25))
        self.assertEqual(parse_value_list("0,1\n0,25", parse_fraction), (0.1, 0.25))
        for value in ("0,1", "0.1, 0.2", "0.1;", "0.1;;0.2", "0.1;0,10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_value_list(value, parse_fraction)


if __name__ == "__main__":
    unittest.main()
