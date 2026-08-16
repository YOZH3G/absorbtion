import tempfile
import unittest
from pathlib import Path

from matplotlib.figure import Figure

from calculations import STEP
from exporting import build_protocol, save_graphs, write_csv, write_protocol
from simulation import LEAN_GAS, run_simulation


class ExportTests(unittest.TestCase):
    def setUp(self):
        model_values = {
            "gna": 7800.0,
            "xa": 0.5,
            "xg": 0.5,
            "gg": 1000.0,
            "xog_initial": 0.8,
            "xna_initial": 30.0,
        }
        dynamics = {
            "kind": STEP,
            "start_time": 10.0,
            "simulation_duration": 100.0,
            "effect_duration": 1.0,
            "time_constant": 10.0,
            "delay": 2.0,
        }
        self.result = run_simulation(
            LEAN_GAS,
            model_values,
            component_fraction=0.0,
            flow_fraction=0.1,
            dynamics=dynamics,
        )
        self.result["disturbance_type"] = "Ступенчатое"

    def test_csv_contains_time_targets_and_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(Path(directory) / "result.csv", self.result)
            text = path.read_text(encoding="utf-8-sig")

        header, first_row, *_rest = text.splitlines()
        self.assertIn("Время, с", header)
        self.assertIn("Отклик: Совместное воздействие", header)
        self.assertTrue(first_row.startswith("0.0;"))

    def test_protocol_is_built_and_written(self):
        protocol = build_protocol(self.result, "Учебный сценарий")

        with tempfile.TemporaryDirectory() as directory:
            path = write_protocol(Path(directory) / "protocol.txt", protocol)
            saved = path.read_text(encoding="utf-8")

        self.assertIn("Учебный сценарий", saved)
        self.assertIn("- Запаздывание L: 2 с", saved)
        self.assertIn("- Расчётное значение: 0.88", saved)

    def test_two_graphs_are_saved_with_distinct_suffixes(self):
        signal_figure = Figure()
        signal_figure.add_subplot(111).plot([0, 1], [0, 1])
        response_figure = Figure()
        response_figure.add_subplot(111).plot([0, 1], [1, 0])

        with tempfile.TemporaryDirectory() as directory:
            signal_path, response_path = save_graphs(
                signal_figure,
                response_figure,
                Path(directory) / "result.png",
            )
            signal_exists = signal_path.exists() and signal_path.stat().st_size > 0
            response_exists = response_path.exists() and response_path.stat().st_size > 0

        self.assertEqual(signal_path.name, "result_signals.png")
        self.assertEqual(response_path.name, "result_response.png")
        self.assertTrue(signal_exists)
        self.assertTrue(response_exists)


if __name__ == "__main__":
    unittest.main()
