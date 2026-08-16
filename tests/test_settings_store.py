import json
import tempfile
import unittest
from pathlib import Path

from app.settings_store import DEFAULT_SETTINGS, SettingsStore


class SettingsStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.path = Path(self.temp_directory.name) / "settings.json"

    def test_missing_or_broken_file_uses_defaults(self):
        store = SettingsStore(self.path)
        self.assertEqual(store.load(), DEFAULT_SETTINGS)

        self.path.write_text("{broken", encoding="utf-8")
        self.assertEqual(store.load(), DEFAULT_SETTINGS)

    def test_settings_round_trip(self):
        store = SettingsStore(self.path)
        expected = {
            "geometry": "1280x720+20+30",
            "last_page": "comparison",
            "sidebar_collapsed": True,
        }

        store.save(expected)

        self.assertEqual(store.load(), expected)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["version"], 1)

    def test_invalid_values_are_replaced_with_defaults(self):
        self.path.write_text(
            json.dumps({
                "version": 1,
                "geometry": "fullscreen",
                "last_page": 42,
                "sidebar_collapsed": "yes",
            }),
            encoding="utf-8",
        )

        settings = SettingsStore(self.path).load()

        self.assertEqual(settings, DEFAULT_SETTINGS)


if __name__ == "__main__":
    unittest.main()
