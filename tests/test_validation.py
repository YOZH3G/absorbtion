import unittest

from validation import parse_fraction


class FractionValidationTests(unittest.TestCase):
    def test_accepts_range_boundaries(self):
        self.assertEqual(parse_fraction("0.1"), 0.1)
        self.assertEqual(parse_fraction("9.9"), 9.9)

    def test_accepts_single_digit_within_range(self):
        self.assertEqual(parse_fraction("1"), 1.0)

    def test_rejects_empty_and_non_numeric_values(self):
        for value in ("", " ", "abc", "1,5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_fraction(value)

    def test_rejects_non_finite_and_out_of_range_values(self):
        for value in ("nan", "inf", "-inf", "0", "0.09", "10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_fraction(value)


if __name__ == "__main__":
    unittest.main()
