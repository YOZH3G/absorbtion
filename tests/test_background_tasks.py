import unittest

import numpy as np

from app.background_tasks import BackgroundTask, CalculationCancelled
from app.calculations import first_order_response


class BackgroundTests(unittest.TestCase):
    def test_result_progress_and_error_are_delivered_without_ui(self):
        def calculate(cancel, progress):
            progress(0.5)
            return 42
        task = BackgroundTask(calculate)
        task.thread.join(2)
        self.assertEqual(task.progress, 0.5)
        self.assertEqual(task.poll(), (42, None))
        task = BackgroundTask(lambda cancel, progress: 1 / 0)
        task.thread.join(2)
        self.assertIsInstance(task.poll()[1], ZeroDivisionError)

    def test_cancelled_loop_stops_inside_a_single_large_response(self):
        class CancelDuringLoop:
            calls = 0
            def is_set(self):
                self.calls += 1
                return self.calls >= 3
        probe = CancelDuringLoop()
        time = np.linspace(0, 100, 200000)
        with self.assertRaises(CalculationCancelled):
            first_order_response(time, 0, np.ones_like(time), 10, cancel=probe)
        self.assertEqual(probe.calls, 3)
