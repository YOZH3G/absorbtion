import json
import tempfile
import unittest
from pathlib import Path

from app.laboratory import SCENARIOS
from app.scenario_store import FORMAT_VERSION, ScenarioStore, normalize_scenario


def custom_scenario(name="Пользовательский сценарий"):
    scenario = dict(SCENARIOS[0])
    scenario["name"] = name
    scenario["description"] = "Проверка пользовательского сценария."
    scenario["steady_tolerance_percent"] = 7.5
    return scenario


class ScenarioValidationTests(unittest.TestCase):
    def test_normalization_adds_optional_defaults(self):
        normalized = normalize_scenario(custom_scenario())

        self.assertEqual(normalized["steady_tolerance_percent"], 7.5)
        self.assertIsNone(normalized["controller"])
        self.assertEqual(normalized["lesson"]["attempt_limit"], 1)

    def test_normalization_keeps_teacher_lesson_data(self):
        scenario = custom_scenario()
        scenario["lesson"] = {
            "task": "Подберите PI-регулятор.",
            "guidance": "Начните с малой пропорциональной части.",
            "questions": ["Почему возникает перерегулирование?"],
            "attempt_limit": 3,
            "hidden_answers": {"direction": "Увеличится"},
            "controller_target": {"type": "PI", "gain_min": 1, "gain_max": 3},
        }

        normalized = normalize_scenario(scenario)

        self.assertEqual(normalized["lesson"]["attempt_limit"], 3)
        self.assertEqual(normalized["lesson"]["questions"][0], "Почему возникает перерегулирование?")

    def test_requires_at_least_one_disturbance(self):
        scenario = custom_scenario()
        scenario["component"] = None
        scenario["flow"] = None

        with self.assertRaisesRegex(ValueError, "хотя бы одно возмущение"):
            normalize_scenario(scenario)

    def test_rejects_invalid_time_range(self):
        scenario = custom_scenario()
        scenario["start_time"] = scenario["simulation_duration"]

        with self.assertRaisesRegex(ValueError, "раньше окончания"):
            normalize_scenario(scenario)


class ScenarioStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.path = Path(self.temp_directory.name) / "scenarios.json"

    def test_save_and_reload_user_scenario(self):
        store = ScenarioStore(self.path)
        saved = store.save(custom_scenario())
        reloaded = ScenarioStore(self.path)

        self.assertEqual(saved["name"], "Пользовательский сценарий")
        self.assertEqual(len(reloaded.user_scenarios), 1)
        self.assertEqual(reloaded.user_scenarios[0]["steady_tolerance_percent"], 7.5)

    def test_builtin_scenarios_cannot_be_changed_or_deleted(self):
        store = ScenarioStore(self.path)
        builtin_name = SCENARIOS[0]["name"]

        with self.assertRaisesRegex(ValueError, "защищено"):
            store.save(custom_scenario(builtin_name))
        with self.assertRaisesRegex(ValueError, "нельзя удалить"):
            store.delete(builtin_name)

    def test_broken_file_falls_back_to_builtins(self):
        self.path.write_text("{broken", encoding="utf-8")

        store = ScenarioStore(self.path)

        self.assertEqual(store.user_scenarios, [])
        self.assertIsNotNone(store.warning)
        self.assertEqual(len(store.scenarios), len(SCENARIOS))

    def test_backup_can_be_loaded_and_restored_after_corruption(self):
        store = ScenarioStore(self.path)
        store.save(custom_scenario() | {"description": "Первая версия."})
        store.save(
            custom_scenario() | {"description": "Вторая версия."},
            original_name="Пользовательский сценарий",
        )
        self.path.write_text("{broken", encoding="utf-8")

        recovered = ScenarioStore(self.path)

        self.assertTrue(recovered.recovery_available)
        self.assertEqual(recovered.user_scenarios[0]["description"], "Первая версия.")
        self.assertEqual(recovered.restore_backup(), 1)
        self.assertFalse(recovered.recovery_available)
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8"))["scenarios"][0]["description"],
            "Первая версия.",
        )

    def test_legacy_file_is_migrated_to_new_location(self):
        legacy_path = self.path.with_name("legacy.json")
        payload = {
            "version": FORMAT_VERSION,
            "scenarios": [custom_scenario()],
        }
        legacy_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        migrated_path = self.path.with_name("profile") / "scenarios.json"

        store = ScenarioStore(migrated_path, legacy_path=legacy_path)

        self.assertTrue(migrated_path.exists())
        self.assertEqual(len(store.user_scenarios), 1)

    def test_import_merges_and_replaces_user_scenarios_by_name(self):
        store = ScenarioStore(self.path)
        store.save(custom_scenario("Сценарий А"))
        import_path = self.path.with_name("import.json")
        payload = {
            "version": FORMAT_VERSION,
            "scenarios": [
                custom_scenario("Сценарий А") | {"description": "Обновлён."},
                custom_scenario("Сценарий Б"),
            ],
        }
        import_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        count = store.import_bundle(import_path)

        self.assertEqual(count, 2)
        self.assertEqual(len(store.user_scenarios), 2)
        self.assertEqual(store.user_scenarios[0]["description"], "Обновлён.")

    def test_export_writes_versioned_bundle(self):
        store = ScenarioStore(self.path)
        store.save(custom_scenario())
        export_path = self.path.with_name("export.json")

        store.export_bundle(export_path)
        payload = json.loads(export_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["version"], FORMAT_VERSION)
        self.assertEqual(payload["scenarios"][0]["name"], "Пользовательский сценарий")

    def test_moves_only_user_scenarios_and_preserves_order(self):
        store = ScenarioStore(self.path)
        store.save(custom_scenario("Сценарий А"))
        store.save(custom_scenario("Сценарий Б"))

        self.assertTrue(store.move("Сценарий Б", -1))
        self.assertEqual([item["name"] for item in store.user_scenarios], ["Сценарий Б", "Сценарий А"])
        self.assertFalse(store.move("Сценарий Б", -1))
        with self.assertRaisesRegex(ValueError, "Встроенный"):
            store.move(SCENARIOS[0]["name"], 1)


if __name__ == "__main__":
    unittest.main()
