import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from app.session_store import read_session, write_session


def run():
    return {
        "id": "run-3",
        "name": "Опыт 3",
        "chain": "lean_gas",
        "time": np.array([0.0, 1.0]),
        "response": np.array([0.8, 0.81]),
        "time_constant": 10.0,
        "delay": 2.0,
        "controller_type": "PI",
        "maximum_deviation": 0.01,
        "relative_deviation": 1.25,
        "settling_duration": 8.0,
        "static_error": 0.0,
        "input_state": {"chain": "lean_gas", "start_time": "10"},
    }


class SessionStoreTests(unittest.TestCase):
    def test_round_trip_preserves_arrays_and_input_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lesson.absession.json"
            write_session(path, [run()], 3)
            restored, counter = read_session(path)

        self.assertEqual(counter, 3)
        self.assertEqual(restored[0]["name"], "Опыт 3")
        self.assertTrue(np.array_equal(restored[0]["response"], np.array([0.8, 0.81])))
        self.assertEqual(restored[0]["input_state"]["start_time"], "10")

    def test_rejects_invalid_signal_shape(self):
        invalid = run() | {"response": [[0.8, 0.81]]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            invalid["time"] = [0.0, 1.0]
            path.write_text(
                json.dumps({"version": 1, "comparison_counter": 1, "runs": [invalid]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "одномерным"):
                read_session(path)


if __name__ == "__main__":
    unittest.main()
