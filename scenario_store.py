import copy
import json
import math
import os
import shutil
from pathlib import Path

from calculations import CONTROLLER_TYPES
from laboratory import SCENARIOS, normalize_lesson
from validation import MAX_FRACTION, MIN_FRACTION


FORMAT_VERSION = 1
DISTURBANCE_TYPES = (
    "Ступенчатое",
    "Импульсное",
    "Временное прямоугольное",
    "Плавно нарастающее",
)
CHAINS = ("lean_gas", "rich_absorbent")
APP_DIRECTORY_NAME = "AbsorptionTrainer"
USER_FILE_NAME = "scenarios.json"
LEGACY_USER_FILE = Path(__file__).with_name(USER_FILE_NAME)


class ScenarioStore:
    def __init__(self, path=None, legacy_path=None):
        self.path = Path(path) if path is not None else default_user_file()
        self.backup_path = self.path.with_suffix(self.path.suffix + ".bak")
        self.user_scenarios = []
        self.warning = None
        self.recovery_available = False
        if path is None or legacy_path is not None:
            self._migrate_legacy_file(
                LEGACY_USER_FILE if legacy_path is None else Path(legacy_path)
            )
        self.reload()

    @property
    def scenarios(self):
        builtins = tuple(normalize_scenario(scenario) for scenario in SCENARIOS)
        return builtins + tuple(copy.deepcopy(self.user_scenarios))

    @property
    def builtin_names(self):
        return frozenset(scenario["name"] for scenario in SCENARIOS)

    def is_builtin(self, name):
        return name in self.builtin_names

    def reload(self):
        self.user_scenarios = []
        self.warning = None
        self.recovery_available = False
        if not self.path.exists() and not self.backup_path.exists():
            return
        try:
            self.user_scenarios = _read_bundle(self.path)
            _ensure_unique_names(self.user_scenarios)
            _ensure_no_builtin_names(self.user_scenarios, self.builtin_names)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            try:
                recovered = _read_bundle(self.backup_path)
                _ensure_unique_names(recovered)
                _ensure_no_builtin_names(recovered, self.builtin_names)
            except (OSError, ValueError, json.JSONDecodeError):
                self.user_scenarios = []
                self.warning = f"Пользовательские сценарии не загружены: {error}"
            else:
                self.user_scenarios = recovered
                self.recovery_available = True
                self.warning = (
                    "Основной файл сценариев повреждён. Загружена резервная копия; "
                    "её можно восстановить в режиме преподавателя."
                )

    def restore_backup(self):
        if not self.recovery_available:
            raise ValueError("Корректная резервная копия не найдена.")
        recovered = _read_bundle(self.backup_path)
        self._write(recovered)
        return len(recovered)

    def save(self, scenario, original_name=None):
        normalized = normalize_scenario(scenario)
        if normalized["name"] in self.builtin_names:
            raise ValueError("Имя встроенного сценария защищено. Выберите другое имя.")
        if original_name in self.builtin_names:
            raise ValueError("Встроенный сценарий нельзя изменить. Сначала создайте копию.")

        updated = copy.deepcopy(self.user_scenarios)
        replacement_index = next(
            (index for index, item in enumerate(updated) if item["name"] == original_name),
            None,
        )
        duplicate = next(
            (
                item
                for item in updated
                if item["name"] == normalized["name"] and item["name"] != original_name
            ),
            None,
        )
        if duplicate is not None:
            raise ValueError("Пользовательский сценарий с таким именем уже существует.")

        if replacement_index is None:
            updated.append(normalized)
        else:
            updated[replacement_index] = normalized
        self._write(updated)
        return copy.deepcopy(normalized)

    def delete(self, name):
        if name in self.builtin_names:
            raise ValueError("Встроенный сценарий нельзя удалить.")
        updated = [item for item in self.user_scenarios if item["name"] != name]
        if len(updated) == len(self.user_scenarios):
            raise ValueError("Пользовательский сценарий не найден.")
        self._write(updated)

    def move(self, name, direction):
        """Move a user scenario one position in its saved order."""
        if name in self.builtin_names:
            raise ValueError("Встроенный сценарий нельзя перемещать.")
        if direction not in (-1, 1):
            raise ValueError("Направление перемещения должно быть -1 или 1.")
        index = next(
            (position for position, item in enumerate(self.user_scenarios) if item["name"] == name),
            None,
        )
        if index is None:
            raise ValueError("Пользовательский сценарий не найден.")
        target_index = index + direction
        if not 0 <= target_index < len(self.user_scenarios):
            return False
        updated = copy.deepcopy(self.user_scenarios)
        updated[index], updated[target_index] = updated[target_index], updated[index]
        self._write(updated)
        return True

    def import_bundle(self, path):
        imported = _read_bundle(Path(path))
        _ensure_unique_names(imported)
        _ensure_no_builtin_names(imported, self.builtin_names)
        by_name = {scenario["name"]: scenario for scenario in self.user_scenarios}
        by_name.update({scenario["name"]: scenario for scenario in imported})
        self._write(list(by_name.values()))
        return len(imported)

    def export_bundle(self, path):
        destination = Path(path)
        _write_bundle(destination, self.user_scenarios)
        return destination

    def _write(self, scenarios):
        _ensure_unique_names(scenarios)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        _write_bundle(temporary_path, scenarios)
        if self.path.exists():
            try:
                _read_bundle(self.path)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            else:
                shutil.copy2(self.path, self.backup_path)
        temporary_path.replace(self.path)
        self.user_scenarios = copy.deepcopy(scenarios)
        self.warning = None
        self.recovery_available = False

    def _migrate_legacy_file(self, legacy_path):
        if self.path.exists() or not legacy_path.exists() or legacy_path == self.path:
            return
        try:
            scenarios = _read_bundle(legacy_path)
            _ensure_unique_names(scenarios)
            _ensure_no_builtin_names(scenarios, self.builtin_names)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _write_bundle(self.path, scenarios)
        except (OSError, ValueError, json.JSONDecodeError):
            return


def user_data_directory():
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_DIRECTORY_NAME
    if os.name == "posix" and os.uname().sysname == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIRECTORY_NAME
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / APP_DIRECTORY_NAME
    return Path.home() / ".config" / APP_DIRECTORY_NAME


def default_user_file():
    return user_data_directory() / USER_FILE_NAME


def normalize_scenario(scenario):
    if not isinstance(scenario, dict):
        raise ValueError("Сценарий должен быть объектом JSON.")

    name = _required_text(scenario.get("name"), "Название")
    description = _required_text(scenario.get("description"), "Описание")
    chain = scenario.get("chain")
    if chain not in CHAINS:
        raise ValueError("Выберите допустимую цепь управления.")

    component = _optional_fraction(scenario.get("component"), "Возмущение состава")
    flow = _optional_fraction(scenario.get("flow"), "Возмущение расхода")
    if component is None and flow is None:
        raise ValueError("Включите хотя бы одно возмущение: состав или расход.")

    disturbance_type = scenario.get("disturbance_type")
    if disturbance_type not in DISTURBANCE_TYPES:
        raise ValueError("Выберите допустимый вид воздействия.")

    start_time = _number(scenario.get("start_time"), "Начало воздействия", minimum=0.0)
    simulation_duration = _number(
        scenario.get("simulation_duration"),
        "Длительность моделирования",
        minimum=0.0,
        strict=True,
    )
    if start_time >= simulation_duration:
        raise ValueError("Начало воздействия должно быть раньше окончания моделирования.")
    effect_duration = _number(
        scenario.get("effect_duration"),
        "Длительность воздействия",
        minimum=0.0,
        strict=True,
    )
    time_constant = _number(
        scenario.get("time_constant"),
        "Постоянная времени T",
        minimum=0.0,
        strict=True,
    )
    delay = _number(scenario.get("delay"), "Запаздывание L", minimum=0.0)
    tolerance = _number(
        scenario.get("steady_tolerance_percent", 5.0),
        "Допуск прогноза",
        minimum=0.0,
        maximum=100.0,
        strict=True,
    )

    controller = _normalize_controller(scenario.get("controller"))
    lesson = normalize_lesson(scenario.get("lesson"))
    return {
        "name": name,
        "description": description,
        "chain": chain,
        "component": component,
        "flow": flow,
        "disturbance_type": disturbance_type,
        "start_time": start_time,
        "simulation_duration": simulation_duration,
        "effect_duration": effect_duration,
        "time_constant": time_constant,
        "delay": delay,
        "controller": controller,
        "steady_tolerance_percent": tolerance,
        "lesson": lesson,
    }


def _normalize_controller(controller):
    if controller is None:
        return None
    if not isinstance(controller, dict):
        raise ValueError("Параметры регулятора должны быть объектом JSON.")
    controller_type = controller.get("type")
    if controller_type not in CONTROLLER_TYPES:
        raise ValueError("Выберите допустимый тип регулятора.")
    setpoint = controller.get("setpoint")
    if setpoint is not None:
        setpoint = _number(setpoint, "Заданное значение", minimum=0.0, strict=True)
    return {
        "type": controller_type,
        "gain": _number(controller.get("gain"), "Коэффициент K", minimum=0.0),
        "integral_time": _number(
            controller.get("integral_time", 1.0),
            "Время интегрирования Ti",
            minimum=0.0,
            strict=True,
        ),
        "derivative_time": _number(
            controller.get("derivative_time", 0.0),
            "Время дифференцирования Td",
            minimum=0.0,
        ),
        "control_limit": _number(
            controller.get("control_limit"),
            "Ограничение управляющего воздействия",
            minimum=0.0,
            strict=True,
        ),
        "setpoint": setpoint,
    }


def _read_bundle(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"Поддерживается формат сценариев версии {FORMAT_VERSION}.")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Поле scenarios должно содержать список.")
    return [normalize_scenario(scenario) for scenario in scenarios]


def _write_bundle(path, scenarios):
    payload = {
        "version": FORMAT_VERSION,
        "scenarios": [normalize_scenario(scenario) for scenario in scenarios],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _ensure_unique_names(scenarios):
    names = [scenario["name"] for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("Названия сценариев в комплекте должны быть уникальными.")


def _ensure_no_builtin_names(scenarios, builtin_names):
    conflict = next(
        (scenario["name"] for scenario in scenarios if scenario["name"] in builtin_names),
        None,
    )
    if conflict is not None:
        raise ValueError(f"Имя «{conflict}» занято встроенным сценарием.")


def _required_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: заполните поле.")
    return value.strip()


def _optional_fraction(value, label):
    if value is None:
        return None
    return _number(value, label, minimum=MIN_FRACTION, maximum=MAX_FRACTION)


def _number(value, label, minimum=None, maximum=None, strict=False):
    if isinstance(value, bool):
        raise ValueError(f"{label}: введите число.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: введите число.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label}: значение должно быть конечным числом.")
    if minimum is not None and (number <= minimum if strict else number < minimum):
        relation = "больше" if strict else "не меньше"
        raise ValueError(f"{label}: значение должно быть {relation} {minimum:g}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label}: значение должно быть не больше {maximum:g}.")
    return number
