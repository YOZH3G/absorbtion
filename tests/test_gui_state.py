import copy
import tempfile
import tkinter as tk
import unittest
import time
from pathlib import Path
from unittest.mock import patch

import main
from app.scenario_store import ScenarioStore
from app.session_store import read_laboratory_session, restore_calculation, write_session
from app.exploration import controller_setting_map
from app.simulation import RICH_ABSORBENT
from ui.scenario_editor import ScenarioEditorDialog


class GuiStateTests(unittest.TestCase):
    def test_foreign_scenario_and_optional_prediction_restore_without_replacing_signal(self):
        import json
        import numpy as np
        app = self.app
        original = app.last_calculation["final_response"].copy()
        laboratory = app._capture_laboratory()
        laboratory["scenario"]["name"] = "Сценарий другого компьютера"
        laboratory["scenario_name"] = laboratory["scenario"]["name"]
        del laboratory["last_calculation"]["prediction"]
        laboratory["last_calculation"]["final_response"] = "invalid"
        path = Path(self.directory.name) / "foreign.json"
        path.write_text(json.dumps({"version": 2, "comparison_counter": 0, "runs": [],
                                    "laboratory": laboratory}), encoding="utf-8")
        runs, counter, restored = read_laboratory_session(path)
        result = restore_calculation(restored["last_calculation"])
        app.comparison_runs, app.comparison_counter = runs, counter
        app._restore_laboratory(restored, result)
        self.assertIn("Сценарий другого компьютера", app.scenarios_by_name)
        self.assertEqual(app.selected_scenario.get(), "Сценарий другого компьютера")
        self.assertIsNone(app.last_prediction)
        np.testing.assert_array_equal(app.last_calculation["final_response"], original)

    def test_new_window_automatically_restores_saved_laboratory(self):
        import numpy as np
        original = self.app.last_calculation["final_response"].copy()
        self.app.student_conclusion.set("Восстановлено после запуска")
        self.app.component_value.set("-")
        self.app._close_application()
        self.root = tk.Tk()
        self.root.withdraw()
        with patch.object(main, "ScenarioStore", return_value=self.store):
            self.app = main.AbsorptionApp(self.root)
        self.assertEqual(self.app.student_conclusion.get(), "Восстановлено после запуска")
        self.assertEqual(self.app.component_value.get(), "-")
        np.testing.assert_array_equal(self.app.last_calculation["final_response"], original)
        self.assertIn("автосохранения", self.app.status_text.get())

    def test_full_laboratory_restores_draft_result_and_assignment_without_using_attempt(self):
        import numpy as np
        app = self.app
        app.current_lesson["attempt_limit"] = 3
        app.assignment_enabled.set(True)
        app.predicted_direction.set("Уменьшится")
        app.predicted_fastest.set("Без сравнения")
        app.predicted_correction.set("Регулятор выключен")
        app.predicted_steady.set("26.27")
        app._calculate()
        self.wait_for_task()
        original = app.last_calculation
        app._add_current_to_comparison()
        app.learning_mode.set(True)
        app.learning_step = 4
        app.student_name.set("Студент")
        app.student_conclusion.set("Проверен баланс")
        app.component_value.set("-")
        app.predicted_steady.set("26,")
        path = Path(self.directory.name) / "laboratory.json"
        write_session(path, app.comparison_runs, app.comparison_counter, app._capture_laboratory())
        runs, counter, laboratory = read_laboratory_session(path)
        result = restore_calculation(laboratory["last_calculation"])
        app._reset()
        app.student_name.set("")
        app.assignment_attempts = 0
        app.comparison_runs, app.comparison_counter = runs, counter
        app._restore_laboratory(laboratory, result)
        np.testing.assert_array_equal(app.last_calculation["final_response"], original["final_response"])
        np.testing.assert_array_equal(app.last_calculation["time"], original["time"])
        self.assertEqual(app.component_value.get(), "-")
        self.assertEqual(app.predicted_steady.get(), "26,")
        self.assertEqual(app.assignment_attempts, 1)
        self.assertEqual(app.assignment_evaluation, original["evaluation"])
        self.assertEqual(app.learning_step, 4)
        self.assertEqual(app.student_name.get(), "Студент")
        self.assertEqual(app.student_conclusion.get(), "Проверен баланс")

    def test_broken_nested_import_does_not_change_current_work(self):
        import json
        app = self.app
        path = Path(self.directory.name) / "broken.json"
        write_session(path, [], 0, app._capture_laboratory())
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["laboratory"]["last_calculation"]["input_state"]["delay"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        result, form = app.last_calculation, app._capture_input_state()
        with patch.object(main.filedialog, "askopenfilename", return_value=str(path)):
            app._open_comparison_session()
        self.assertIs(app.last_calculation, result)
        self.assertEqual(app._capture_input_state(), form)

    def test_autosave_recovers_backup_and_preserves_corrupt_copy(self):
        app = self.app
        app.student_conclusion.set("Первая корректная копия")
        app._save_autosave()
        app.student_conclusion.set("Вторая копия")
        app._save_autosave()
        app.autosave_path.write_text("broken", encoding="utf-8")
        app.student_conclusion.set("")
        app._restore_autosave()
        self.assertEqual(app.student_conclusion.get(), "Первая корректная копия")
        self.assertEqual(app.autosave_path.with_suffix(".json.corrupt").read_text(encoding="utf-8"), "broken")
        self.assertIn("резервного", app.status_text.get())
        app.autosave_path.unlink()
        app.student_conclusion.set("")
        app._restore_autosave()
        self.assertEqual(app.student_conclusion.get(), "Первая корректная копия")

    def test_map_selection_explains_independent_observations(self):
        from types import SimpleNamespace
        result = self.app.last_calculation
        self.app.map_data = controller_setting_map(
            result["chain"], result["model_values"], result["component_fraction"],
            result["flow_fraction"], result["dynamics"], "PI", (6,), (2,), 1, 0.8,
        )
        self.app._show_page("tuning_map")
        self.app._select_map_cell(SimpleNamespace(inaxes=self.app.disturbance_axis, xdata=0, ydata=0))
        description = self.app.map_selection_summary.get()
        self.assertIn("Насыщение η", description)
        self.assertIn("Вывод относится только к длительности этого опыта", description)
        self.assertEqual(self.app.map_selection, (0, 0))

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
        self.wait_for_task()
        self.root.update_idletasks()

    def wait_for_task(self):
        deadline = time.monotonic() + 10
        while self.app._task is not None:
            self.assertLess(time.monotonic(), deadline, "Background task timed out")
            self.root.update()
            time.sleep(0.005)

    def test_background_map_keeps_tk_timer_responsive_and_can_be_cancelled(self):
        app = self.app
        app._show_page("tuning_map")
        original = app.last_calculation
        ticks = []
        self.root.after(0, lambda: ticks.append(True))
        app.map_grid_size.set("15")
        app._calculate_tuning_map()
        task = app._task
        self.root.update()
        self.assertTrue(ticks)
        self.assertIsNotNone(task)
        app._cancel_task()
        task.thread.join(2)
        self.assertFalse(task.thread.is_alive())
        self.assertIsNone(app.map_data)
        self.assertIs(app.last_calculation, original)

    def test_changed_form_cancels_work_but_equivalent_number_does_not(self):
        from threading import Event
        release = Event()
        original = self.app.last_calculation
        def slow_result(cancel, progress):
            release.wait(2)
            return original
        self.app._start_task("Проверка", slow_result, lambda result: self.fail("Stale result applied"))
        task = self.app._task
        self.app.component_value.set(self.app.component_value.get().replace(".", ",") + "00")
        self.assertIs(self.app._task, task)
        self.app.component_value.set("0.2")
        self.assertIsNone(self.app._task)
        self.assertTrue(task.cancel.is_set())
        release.set()
        task.thread.join(2)
        self.assertIs(self.app.last_calculation, original)

    def test_cancelled_prediction_does_not_consume_attempt_and_clear_is_undoable(self):
        app = self.app
        app.current_lesson["attempt_limit"] = 3
        app.assignment_enabled.set(True)
        app.predicted_direction.set("Уменьшится")
        app.predicted_fastest.set("Без сравнения")
        app.predicted_correction.set("Регулятор выключен")
        app.predicted_steady.set("26.27")
        original = app.last_calculation
        app._calculate()
        app._cancel_task()
        self.assertEqual(app.assignment_attempts, 0)
        self.assertIs(app.last_calculation, original)
        app._reset()
        self.assertIsNone(app.last_calculation)
        app.comparison_counter = 7
        app._undo_clear_experiment()
        self.assertIs(app.last_calculation, original)
        self.assertEqual(app.assignment_attempts, 0)
        self.assertEqual(app.comparison_counter, 7)

    def test_defaults_restore_disabled_controller_and_preserve_variant_title(self):
        app = self.app
        caption = "Применён: Снижение состава на 15% · вариант 5."
        app.active_scenario.set(caption)
        app.controller_type.set("PID")
        app.proportional_gain.set("19")
        app.integral_time.set("3")
        app._restore_experiment_defaults()
        self.assertEqual(app.controller_type.get(), "PI")
        self.assertEqual(app.proportional_gain.get(), "2")
        self.assertEqual(app.integral_time.get(), "20")
        self.assertEqual(app.active_scenario.get(), caption)

    def test_new_task_wins_even_when_old_worker_finishes_later(self):
        from threading import Event
        release = Event()
        applied = []
        self.app._start_task("Старый", lambda cancel, progress: release.wait(2), applied.append)
        old_task = self.app._task
        self.app._start_task("Новый", lambda cancel, progress: "new", applied.append)
        self.wait_for_task()
        release.set()
        old_task.thread.join(2)
        self.root.update()
        self.assertEqual(applied, ["new"])

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
