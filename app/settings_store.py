import json
import re
from pathlib import Path


FORMAT_VERSION = 1
DEFAULT_SETTINGS = {
    "geometry": "1600x1000",
    "last_page": "disturbances",
    "sidebar_collapsed": False,
    "chart_mode": "both",
    "inspector_width": 410,
    "chart_split": 0.5,
}
GEOMETRY_PATTERN = re.compile(r"^\d+x\d+(?:[+-]\d+[+-]\d+)?$")


class SettingsStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return DEFAULT_SETTINGS.copy()
        if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
            return DEFAULT_SETTINGS.copy()
        settings = DEFAULT_SETTINGS.copy()
        geometry = payload.get("geometry")
        if isinstance(geometry, str) and GEOMETRY_PATTERN.fullmatch(geometry):
            settings["geometry"] = geometry
        last_page = payload.get("last_page")
        if isinstance(last_page, str):
            settings["last_page"] = last_page
        collapsed = payload.get("sidebar_collapsed")
        if isinstance(collapsed, bool):
            settings["sidebar_collapsed"] = collapsed
        if payload.get("chart_mode") in ("both", "input", "response"):
            settings["chart_mode"] = payload["chart_mode"]
        width = payload.get("inspector_width")
        if isinstance(width, int) and not isinstance(width, bool) and 300 <= width <= 1000:
            settings["inspector_width"] = width
        split = payload.get("chart_split")
        if isinstance(split, (int, float)) and not isinstance(split, bool) and 0.15 <= split <= 0.85:
            settings["chart_split"] = split
        return settings

    def save(self, settings):
        payload = {
            "version": FORMAT_VERSION,
            "geometry": settings["geometry"],
            "last_page": settings["last_page"],
            "sidebar_collapsed": bool(settings["sidebar_collapsed"]),
            "chart_mode": settings.get("chart_mode", "both"),
            "inspector_width": settings.get("inspector_width", 410),
            "chart_split": settings.get("chart_split", 0.5),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
