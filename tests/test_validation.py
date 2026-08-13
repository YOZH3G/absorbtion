import unittest

from validation import parse_fraction, parse_positive_number


class FractionValidationTests(unittest.TestCase):
    def test_accepts_range_boundaries(self):
        self.assertEqual(parse_fraction("0.01"), 0.01)
        self.assertEqual(parse_fraction("9.99"), 9.99)

    def test_accepts_single_digit_within_range(self):
        self.assertEqual(parse_fraction("1"), 1.0)

    def test_rejects_empty_and_non_numeric_values(self):
        for value in ("", " ", "abc", "1,5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_fraction(value)

    def test_rejects_non_finite_and_out_of_range_values(self):
        for value in ("nan", "inf", "-inf", "0", "0.009", "10"):
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

if __name__ == "__main__":
    unittest.main()
