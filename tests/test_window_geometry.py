import unittest

from ui.window_geometry import fit_geometry


class GeometryTests(unittest.TestCase):
    def test_disconnected_monitor_and_large_window_return_to_work_area(self):
        self.assertEqual(fit_geometry("1600x1000+2000+1000", (0, 0, 1366, 728)),
                         "1350x680+8+8")

    def test_negative_offscreen_position_and_small_window_are_clamped(self):
        self.assertEqual(fit_geometry("900x540-1000-500", (0, 0, 1920, 1040)),
                         "900x540+0+0")
