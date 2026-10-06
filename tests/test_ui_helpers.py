import unittest

import numpy as np

from app.calculations import CONTROLLER_TYPES
from ui.ui_helpers import DISTURBANCE_HELP, FORMULAS, ICON_PATTERNS, plot_samples


class UiResourceTests(unittest.TestCase):
    def test_plot_sampling_preserves_a_short_pulse_and_full_input(self):
        time = np.linspace(0, 100, 100001)
        values = np.zeros_like(time)
        values[10051:10055] = 1
        shown_time, shown_values = plot_samples(time, values)
        self.assertLessEqual(len(shown_time), 2000)
        self.assertEqual(np.max(shown_values), 1)
        self.assertEqual(shown_time[0], time[0])
        self.assertEqual(shown_time[-1], time[-1])
        self.assertEqual(len(values), 100001)

    def test_icons_are_sixteen_pixel_square_patterns(self):
        for name, pattern in ICON_PATTERNS.items():
            with self.subTest(icon=name):
                self.assertEqual(len(pattern), 16)
                self.assertTrue(all(len(row) == 16 for row in pattern))
                self.assertTrue(all(pixel in ".#" for row in pattern for pixel in row))

    def test_every_controller_has_mathtext_formula(self):
        self.assertEqual(set(FORMULAS), set(CONTROLLER_TYPES))
        self.assertTrue(all(formula.startswith("$") for formula in FORMULAS.values()))

    def test_every_disturbance_has_help_and_preview_shape(self):
        self.assertEqual(
            set(DISTURBANCE_HELP),
            {"Ступенчатое", "Импульсное", "Временное прямоугольное", "Плавно нарастающее"},
        )
        self.assertTrue(all(len(content) == 3 for content in DISTURBANCE_HELP.values()))


if __name__ == "__main__":
    unittest.main()
