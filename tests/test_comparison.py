import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from comparison import build_comparison_run, write_comparison_csv


def calculation_result(settling_time=12.0):
    return {
        "chain": "lean_gas",
        "time": np.array([0.0, 1.0, 2.0]),
        "final_response": np.array([0.8, 0.82, 0.81]),
        "response_start": 2.0,
        "dynamics": {"time_constant": 10.0, "delay": 2.0},
        "controller": {
            "controller_type": "PI",
        },
        "metrics": {
            "settling_time": settling_time,
            "maximum_deviation": 0.02,
            "relative_deviation": 2.5,
            "static_error": 0.0,
        },
    }


class ComparisonTests(unittest.TestCase):
    def test_build_run_copies_signals_and_calculates_duration(self):
        result = calculation_result()

        run = build_comparison_run(result, "Опыт 1")
        result["final_response"][0] = 99.0

        self.assertEqual(run["settling_duration"], 10.0)
        self.assertEqual(run["controller_type"], "PI")
        self.assertEqual(run["response"][0], 0.8)

    def test_unsettled_run_has_no_duration(self):
        run = build_comparison_run(calculation_result(None), "Опыт 1")

        self.assertIsNone(run["settling_duration"])

    def test_export_uses_long_csv_format(self):
        run = build_comparison_run(calculation_result(), "Опыт 1")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"

            write_comparison_csv(path, [run])
            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file, delimiter=";"))

        self.assertEqual(rows[0][0], "Опыт")
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1][0], "Опыт 1")

    def test_export_rejects_empty_collection(self):
        with self.assertRaisesRegex(ValueError, "Нет опытов"):
            write_comparison_csv("unused.csv", [])


if __name__ == "__main__":
    unittest.main()
