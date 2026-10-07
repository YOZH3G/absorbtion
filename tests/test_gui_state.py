import copy
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from app.scenario_store import ScenarioStore
from app.simulation import RICH_ABSORBENT
from ui.scenario_editor import ScenarioEditorDialog


class GuiStateTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.directory = tempfile.TemporaryDirectory()
        self.store = ScenarioStore(Path(self.directory.name) / "scenarios.json")
        with patch.object(main, "ScenarioStore", return_value=self.store):
            self.app = main.AbsorptionApp(self.root)
        self.app._apply_scenario_data(self.app.scenarios[1])
        self.app._calculate()
        self.root.update_idletasks()

    def tearDown(self):
        if hasattr(self, "root"):
            self.root.update_idletasks()
            for callback in self.root.tk.splitlist(self.root.tk.call("after", "info")):
                self.root.after_cancel(callback)
            self.root.destroy()
        if hasattr(self, "directory"):
            self.directory.cleanup()

    def test_numeric_equivalence_and_inactive_fields_do_not_make_result_stale(self):
        self.app.component_value.set("-0,1500")
        self.app.integral_time.set("999")
        self.assertNotIn("Параметры изменены", self.app.export_summary.get())
        self.app.component_value.set("-0.2")
        self.assertIn("Параметры изменены", self.app.export_summary.get())
        self.app.component_value.set("-0.15")
        self.assertNotIn("Параметры изменены", self.app.export_summary.get())

    def test_chain_and_controller_changes_keep_the_same_calculation_snapshot(self):
        result = self.app.last_calculation
        snapshot = copy.deepcopy(result["input_state"])
        self.app._select_chain("lean_gas")
        self.assertIs(self.app.last_calculation, result)
        self.assertEqual(result["chain"], RICH_ABSORBENT)
        self.assertIn("Параметры изменены", self.app.export_summary.get())
        self.app._set_controller_mode(True)
        self.assertIs(self.app.last_calculation, result)
        self.assertEqual(result["input_state"], snapshot)
        self.app._add_current_to_comparison()
        self.assertEqual(self.app.comparison_runs[-1]["input_state"], snapshot)

    def test_teacher_quality_requirements_and_absolute_tolerance_survive_editor_round_trip(self):
        scenario = copy.deepcopy(self.app.scenarios[1])
        scenario["name"] = "Quality assignment"
        scenario["steady_absolute_tolerance"] = 2e-6
        scenario["lesson"]["controller_target"] = dict(type="PI", require_settled=True,
                                                       max_iae=0.012, max_saturation_duration=4)
        scenario = self.store.save(scenario)
        editor = ScenarioEditorDialog(self.root, self.store, lambda *_: None,
                                      lambda *_: None, lambda: None, main.BACKGROUND)
        try:
            editor._load_scenario(scenario)
            restored = editor._collect_scenario()
            self.assertEqual(restored["steady_absolute_tolerance"], 2e-6)
            self.assertEqual(restored["lesson"]["controller_target"], scenario["lesson"]["controller_target"])
        finally:
            editor.destroy()

    def test_model_edits_keep_the_calculation_and_report_uses_its_graphs(self):
        result = self.app.last_calculation
        self.app._open_model_parameters()
        self.app.model_dialog._variables["eta"].set("70")
        self.app.model_dialog._apply_values()
        self.assertIs(self.app.last_calculation, result)
        self.assertIn("Параметры изменены", self.app.export_summary.get())
        self.app._add_current_to_comparison()
        self.assertIn("Теория", self.app.comparison_details.get())
        captured = []
        self.app.last_prediction = {"steady": 0.9}

        def writer(path, exported, title, lesson, evaluation, *args, **kwargs):
            captured.append(exported)
            self.assertEqual(title, result["title"])
            self.assertEqual(lesson, result["lesson"])
            self.assertEqual(kwargs["prediction"], result["prediction"])
            self.assertEqual(self.app.response_axis.get_ylabel(), "Концентрация, %")
            return Path(path)

        with patch.object(main.filedialog, "asksaveasfilename", return_value=str(Path(self.directory.name) / "report.html")):
            self.app._save_lab_report("html", writer)
        self.assertEqual(captured, [result])


if __name__ == "__main__":
    unittest.main()
