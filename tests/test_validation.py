import unittest

from app.validation import parse_fraction, parse_nonnegative_number, parse_positive_number


class FractionValidationTests(unittest.TestCase):
    def test_accepts_range_boundaries(self):
        self.assertEqual(parse_fraction("-0.99"), -0.99)
        self.assertEqual(parse_fraction("9.99"), 9.99)

    def test_accepts_zero_and_values_within_range(self):
        self.assertEqual(parse_fraction("0"), 0.0)
        self.assertEqual(parse_fraction("-0.1"), -0.1)
        self.assertEqual(parse_fraction("1"), 1.0)

    def test_rejects_empty_and_non_numeric_values(self):
        for value in ("", " ", "abc", "1,5"):
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

if __name__ == "__main__":
    unittest.main()
