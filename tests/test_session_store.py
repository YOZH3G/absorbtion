from app.calculations import DEFAULT_MODEL_VALUES
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from app.session_store import read_laboratory_session, read_session, validate_input_state, write_session
from unittest.mock import patch


def run():
    return {
        "id": "run-3",
        "name": "Опыт 3",
        "chain": "lean_gas",
        "model_version": 2, "model_values": DEFAULT_MODEL_VALUES.copy(),
        "time": np.array([0.0, 1.0]),
        "response": np.array([0.8, 0.81]),
        "time_constant": 10.0,
        "delay": 2.0,
        "controller_type": "PI",
        "maximum_deviation": 0.01,
        "relative_deviation": 1.25,
        "settling_duration": 8.0,
        "static_error": 0.0,
        "input_state": input_state(),
    }


def input_state():
    return {"chain": "lean_gas", "model_version": 2, "model_values": DEFAULT_MODEL_VALUES.copy(),
            "component_enabled": True, "flow_enabled": False, "controller_enabled": False,
            "component_value": "0.1", "flow_value": "", "disturbance_type": "Ступенчатое",
            "start_time": "10", "simulation_duration": "100", "effect_duration": "1",
            "time_constant": "10", "delay": "2", "controller_type": "PI",
            "proportional_gain": "2", "integral_time": "20", "derivative_time": "1",
            "control_limit": "100", "setpoint": "16.6667"}


class SessionStoreTests(unittest.TestCase):
    def test_optional_percent_representation_round_trips_without_changing_model(self):
        state = input_state()
        fraction = validate_input_state(state)["component"]
        state.update(disturbance_units="percent", component_value="10,0")
        self.assertEqual(validate_input_state(state)["component"], fraction)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "units.json"
            record = run()
            record["input_state"] = state
            write_session(path, [record], 3)
            records, counter = read_session(path)
            self.assertEqual(records[0]["input_state"], state)
            self.assertEqual(counter, 3)
        state["disturbance_units"] = "unknown"
        with self.assertRaises(ValueError):
            validate_input_state(state, draft=True)

    def test_interrupted_replacement_preserves_last_correct_file_and_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            write_session(path, [run()], 3)
            first = path.read_bytes()
            changed = run() | {"name": "Новый опыт"}
            write_session(path, [changed], 3)
            second = path.read_bytes()
            self.assertEqual(path.with_suffix(".json.bak").read_bytes(), first)
            original_replace = Path.replace

            def interrupted(source, target):
                if Path(target) == path:
                    raise OSError("Interrupted write")
                return original_replace(source, target)

            with patch.object(Path, "replace", interrupted), self.assertRaises(OSError):
                write_session(path, [run()], 3)
            self.assertEqual(path.read_bytes(), second)
            self.assertEqual(path.with_suffix(".json.bak").read_bytes(), second)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_corrupt_import_and_invalid_nested_values_are_rejected(self):
        for updates in ({"time": [0, 0]}, {"response": [0.1, 1.1]}, {"delay": -1},
                        {"chain": "unknown"}, {"input_state": input_state() | {"simulation_duration": "-2"}},
                        {"model_values": dict(DEFAULT_MODEL_VALUES, gg={})}):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as directory:
                candidate = run() | updates
                candidate["time"] = np.asarray(candidate["time"]).tolist()
                candidate["response"] = np.asarray(candidate["response"]).tolist()
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps({"version": 2, "runs": [candidate]}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_session(path)

    def test_draft_can_preserve_unfinished_input_but_calculation_cannot(self):
        state = input_state() | {"component_value": "-", "time_constant": ""}
        self.assertEqual(validate_input_state(state, draft=True), state)
        with self.assertRaises(ValueError):
            validate_input_state(state)

    def test_existing_model_two_session_without_laboratory_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            write_session(path, [run()], 3)
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["laboratory"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            _runs, counter, laboratory = read_laboratory_session(path)
            self.assertEqual(counter, 3)
            self.assertIsNone(laboratory)

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
                json.dumps({"version": 2, "comparison_counter": 1, "runs": [invalid]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "одномерным"):
                read_session(path)


if __name__ == "__main__":
    unittest.main()
