import copy
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from app.calculations import CONTROLLER_TYPES
from app.scenario_store import CHAINS, DISTURBANCE_TYPES, normalize_scenario


CHAIN_LABELS = {
    "Обеднённый газ": "lean_gas",
    "Насыщенный абсорбент": "rich_absorbent",
}
CHAIN_NAMES = {value: key for key, value in CHAIN_LABELS.items()}


class ScenarioEditorDialog(tk.Toplevel):
    def __init__(self, parent, store, on_change, on_preview, on_close, background):
        super().__init__(parent)
        self.store = store
        self._on_change = on_change
        self._on_preview = on_preview
        self._on_close = on_close
        self._original_name = None
        self._editable = False
        self._dirty = False
        self._loading_form = False
        self._current_list_index = None
        self._scenario_names = []
        self._normal_widgets = []
        self._readonly_widgets = []

        self._variables = {
            "name": tk.StringVar(),
            "description": tk.StringVar(),
            "chain": tk.StringVar(),
            "component_enabled": tk.BooleanVar(),
            "component": tk.StringVar(),
            "flow_enabled": tk.BooleanVar(),
            "flow": tk.StringVar(),
            "disturbance_type": tk.StringVar(),
            "start_time": tk.StringVar(),
            "simulation_duration": tk.StringVar(),
            "effect_duration": tk.StringVar(),
            "time_constant": tk.StringVar(),
            "delay": tk.StringVar(),
            "controller_enabled": tk.BooleanVar(),
            "controller_type": tk.StringVar(),
            "gain": tk.StringVar(),
            "integral_time": tk.StringVar(),
            "derivative_time": tk.StringVar(),
            "control_limit": tk.StringVar(),
            "setpoint": tk.StringVar(),
            "steady_tolerance_percent": tk.StringVar(),
            "attempt_limit": tk.StringVar(),
            "target_enabled": tk.BooleanVar(),
            "target_type": tk.StringVar(),
            "target_gain_min": tk.StringVar(),
            "target_gain_max": tk.StringVar(),
            "target_integral_min": tk.StringVar(),
            "target_integral_max": tk.StringVar(),
            "target_derivative_min": tk.StringVar(),
            "target_derivative_max": tk.StringVar(),
            "answer_direction": tk.StringVar(),
            "answer_steady": tk.StringVar(),
            "answer_fastest": tk.StringVar(),
            "answer_correction": tk.StringVar(),
        }
        self._status = tk.StringVar()
        self._source = tk.StringVar()
        self._search = tk.StringVar()
        self._filter = tk.StringVar(value="Все")
        self._sort = tk.StringVar(value="Как задано")
        self._loaded_values = None
        self._field_widgets = {}
        self._field_errors = {}
        self._lesson_texts = {}
        self._loaded_lesson_texts = {}

        self.title("Редактор лабораторных сценариев")
        self.geometry("1180x780")
        self.minsize(1040, 700)
        self.configure(background=background)
        self.transient(parent)
        self._build_content()
        for variable in self._variables.values():
            variable.trace_add("write", self._mark_dirty)
        self._search.trace_add("write", self._apply_list_filter)
        self._sort.trace_add("write", self._apply_list_filter)
        self._refresh_list()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        self._center(parent)

    def _build_content(self):
        content = ttk.Frame(self, style="App.TFrame", padding=18)
        content.pack(fill="both", expand=True)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(1, weight=1)

        ttk.Label(content, text="Редактор лабораторных сценариев", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            content,
            text="Встроенные сценарии защищены. Чтобы изменить такой сценарий, сначала создайте его копию.",
            style="Status.TLabel",
        ).grid(row=0, column=1, sticky="e")

        list_card = ttk.Frame(content, style="Card.TFrame", padding=(12, 12))
        list_card.grid(row=1, column=0, sticky="nsew", pady=(14, 0), padx=(0, 12))
        list_card.rowconfigure(4, weight=1)
        list_card.columnconfigure(0, weight=1)
        ttk.Label(list_card, text="Сценарии", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        search_entry = ttk.Entry(list_card, textvariable=self._search)
        search_entry.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        filter_box = ttk.Combobox(
            list_card,
            textvariable=self._filter,
            values=("Все", "Встроенные", "Пользовательские"),
            state="readonly",
        )
        filter_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        filter_box.bind("<<ComboboxSelected>>", self._apply_list_filter)
        sort_box = ttk.Combobox(
            list_card,
            textvariable=self._sort,
            values=("Как задано", "По названию"),
            state="readonly",
        )
        sort_box.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        sort_box.bind("<<ComboboxSelected>>", self._apply_list_filter)
        list_host = ttk.Frame(list_card, style="CardBody.TFrame")
        list_host.grid(row=4, column=0, sticky="nsew")
        list_host.rowconfigure(0, weight=1)
        list_host.columnconfigure(0, weight=1)
        self._listbox = tk.Listbox(
            list_host,
            width=32,
            activestyle="none",
            exportselection=False,
            font=("Segoe UI", 10),
            borderwidth=0,
            highlightthickness=1,
            highlightcolor="#2563EB",
            highlightbackground="#D7DCE2",
        )
        self._listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_host, orient="vertical", command=self._listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar = ttk.Scrollbar(
            list_host,
            orient="horizontal",
            command=self._listbox.xview,
        )
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        self._listbox.configure(
            yscrollcommand=scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )
        self._listbox.bind("<<ListboxSelect>>", self._select_from_list)

        list_actions = ttk.Frame(list_card, style="CardBody.TFrame")
        list_actions.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        list_actions.columnconfigure((0, 1), weight=1)
        ttk.Button(
            list_actions,
            text="Создать",
            command=self._new_scenario,
            style="Primary.TButton",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            list_actions,
            text="Дублировать",
            command=self._duplicate_scenario,
            style="Secondary.TButton",
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(
            list_actions,
            text="Импорт JSON",
            command=self._import_bundle,
            style="Secondary.TButton",
        ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(
            list_actions,
            text="Экспорт JSON",
            command=self._export_bundle,
            style="Secondary.TButton",
        ).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(8, 0))
        ttk.Button(
            list_actions,
            text="↑ Выше",
            command=lambda: self._move_selected_scenario(-1),
            style="Secondary.TButton",
        ).grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(
            list_actions,
            text="↓ Ниже",
            command=lambda: self._move_selected_scenario(1),
            style="Secondary.TButton",
        ).grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=(8, 0))
        ttk.Label(
            list_card,
            text=f"Формат JSON v1\n{self.store.path}",
            style="Muted.TLabel",
            wraplength=245,
        ).grid(row=6, column=0, sticky="w", pady=(10, 0))

        form = ttk.Frame(content, style="Card.TFrame", padding=(16, 14))
        form.grid(row=1, column=1, sticky="nsew", pady=(14, 0))
        form.columnconfigure((0, 1), weight=1)
        form.rowconfigure(1, weight=1)
        ttk.Label(form, text="Параметры сценария", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        ttk.Label(form, textvariable=self._source, style="Muted.TLabel").grid(
            row=0, column=1, sticky="e", pady=(0, 10)
        )

        notebook = ttk.Notebook(form)
        notebook.grid(row=1, column=0, columnspan=2, sticky="nsew")
        general = ttk.Frame(notebook, style="CardBody.TFrame", padding=16)
        dynamics = ttk.Frame(notebook, style="CardBody.TFrame", padding=16)
        controller = ttk.Frame(notebook, style="CardBody.TFrame", padding=16)
        lesson = ttk.Frame(notebook, style="CardBody.TFrame", padding=16)
        notebook.add(general, text="Сценарий")
        notebook.add(dynamics, text="Динамика")
        notebook.add(controller, text="Регулятор и задание")
        notebook.add(lesson, text="Учебное задание")
        self._build_general_tab(general)
        self._build_dynamics_tab(dynamics)
        self._build_controller_tab(controller)
        self._build_lesson_tab(lesson)

        ttk.Label(
            form,
            textvariable=self._status,
            style="Muted.TLabel",
            wraplength=680,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        actions = ttk.Frame(form, style="CardBody.TFrame")
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        actions.columnconfigure((0, 1, 2, 3), weight=1)
        self._delete_button = ttk.Button(
            actions,
            text="Удалить",
            command=self._delete_scenario,
            style="Secondary.TButton",
        )
        self._delete_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._validate_button = ttk.Button(
            actions,
            text="Проверить поля",
            command=self._validate_scenario,
            style="Secondary.TButton",
        )
        self._validate_button.grid(row=0, column=1, sticky="ew", padx=4)
        self._preview_button = ttk.Button(
            actions,
            text="Открыть в расчёте",
            command=self._preview_scenario,
            style="Secondary.TButton",
        )
        self._preview_button.grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(
            actions,
            text="Отменить изменения",
            command=self._revert_changes,
            style="Secondary.TButton",
        ).grid(row=0, column=3, sticky="ew", padx=(4, 0))
        ttk.Button(
            actions,
            text="Закрыть",
            command=self._close,
            style="Secondary.TButton",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0), padx=(0, 4))
        self._save_as_button = ttk.Button(
            actions,
            text="Сохранить как новый",
            command=self._save_as_new,
            style="Secondary.TButton",
        )
        self._save_as_button.grid(row=1, column=2, sticky="ew", pady=(8, 0), padx=4)
        self._save_button = ttk.Button(
            actions,
            text="Сохранить",
            command=self._save_scenario,
            style="Primary.TButton",
        )
        self._save_button.grid(row=1, column=3, sticky="ew", pady=(8, 0), padx=(4, 0))

    def _build_general_tab(self, tab):
        tab.columnconfigure(1, weight=1)
        self._entry(tab, 0, "Название", "name")
        self._entry(tab, 1, "Описание", "description")
        self._combobox(tab, 2, "Цепь управления", "chain", tuple(CHAIN_LABELS))

        ttk.Label(tab, text="Возмущения", style="CardTitle.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(18, 8)
        )
        self._component_check = ttk.Checkbutton(
            tab,
            text="Состав",
            variable=self._variables["component_enabled"],
            command=self._update_dependent_states,
        )
        self._component_check.grid(row=4, column=0, sticky="w", pady=5)
        self._normal_widgets.append(self._component_check)
        self._component_entry = ttk.Entry(tab, textvariable=self._variables["component"])
        self._component_entry.grid(row=4, column=1, sticky="ew", pady=5)

        self._flow_check = ttk.Checkbutton(
            tab,
            text="Расход",
            variable=self._variables["flow_enabled"],
            command=self._update_dependent_states,
        )
        self._flow_check.grid(row=5, column=0, sticky="w", pady=5)
        self._normal_widgets.append(self._flow_check)
        self._flow_entry = ttk.Entry(tab, textvariable=self._variables["flow"])
        self._flow_entry.grid(row=5, column=1, sticky="ew", pady=5)
        ttk.Label(
            tab,
            text="Доля: −0,99…9,99. Отключённое воздействие в расчёте не используется.",
            style="Muted.TLabel",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_dynamics_tab(self, tab):
        tab.columnconfigure(1, weight=1)
        self._combobox(tab, 0, "Вид воздействия", "disturbance_type", DISTURBANCE_TYPES)
        fields = (
            ("Начало воздействия, с", "start_time"),
            ("Длительность моделирования, с", "simulation_duration"),
            ("Длительность воздействия, с", "effect_duration"),
            ("Постоянная времени T, с", "time_constant"),
            ("Запаздывание L, с", "delay"),
        )
        for row, (label, key) in enumerate(fields, start=1):
            self._entry(tab, row, label, key)

    def _build_controller_tab(self, tab):
        tab.columnconfigure(1, weight=1)
        self._controller_check = ttk.Checkbutton(
            tab,
            text="Использовать регулятор",
            variable=self._variables["controller_enabled"],
            command=self._update_dependent_states,
        )
        self._controller_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self._normal_widgets.append(self._controller_check)
        self._controller_type_box = self._combobox(
            tab,
            1,
            "Тип регулятора",
            "controller_type",
            CONTROLLER_TYPES,
        )
        self._controller_type_box.bind("<<ComboboxSelected>>", self._update_dependent_states)
        self._gain_entry = self._entry(tab, 2, "Коэффициент K", "gain")
        self._integral_entry = self._entry(tab, 3, "Время интегрирования Ti, с", "integral_time")
        self._derivative_entry = self._entry(tab, 4, "Время дифференцирования Td, с", "derivative_time")
        self._limit_entry = self._entry(tab, 5, "Ограничение |u|", "control_limit")
        self._setpoint_entry = self._entry(tab, 6, "Задание (пусто = базовое)", "setpoint")
        ttk.Separator(tab).grid(row=7, column=0, columnspan=2, sticky="ew", pady=14)
        self._entry(tab, 8, "Допуск прогноза установившегося значения, %", "steady_tolerance_percent")

    def _build_lesson_tab(self, tab):
        tab.columnconfigure(1, weight=1)
        self._text_field(tab, 0, "Формулировка задания", "task", height=3)
        self._text_field(tab, 1, "Методические указания", "guidance", height=3)
        self._text_field(tab, 2, "Вопросы студенту\n(один на строку)", "questions", height=3)
        self._entry(tab, 3, "Количество попыток", "attempt_limit")
        ttk.Separator(tab).grid(row=4, column=0, columnspan=3, sticky="ew", pady=12)
        self._target_check = ttk.Checkbutton(
            tab,
            text="Оценивать выбранные настройки регулятора",
            variable=self._variables["target_enabled"],
            command=self._update_lesson_states,
        )
        self._target_check.grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 6))
        self._normal_widgets.append(self._target_check)
        self._target_type_box = self._combobox(
            tab, 6, "Целевой тип регулятора", "target_type", CONTROLLER_TYPES
        )
        self._target_gain_entries = self._range_entries(tab, 7, "K: минимум / максимум", "target_gain_min", "target_gain_max")
        self._target_integral_entries = self._range_entries(tab, 8, "Ti: минимум / максимум", "target_integral_min", "target_integral_max")
        self._target_derivative_entries = self._range_entries(tab, 9, "Td: минимум / максимум", "target_derivative_min", "target_derivative_max")
        ttk.Separator(tab).grid(row=10, column=0, columnspan=3, sticky="ew", pady=12)
        ttk.Label(tab, text="Скрытые правильные ответы (необязательно)", style="CardTitle.TLabel").grid(
            row=11, column=0, columnspan=3, sticky="w", pady=(0, 6)
        )
        self._combobox(tab, 12, "Направление", "answer_direction", ("", "Увеличится", "Уменьшится", "Не изменится"))
        self._entry(tab, 13, "Установившееся значение", "answer_steady")
        self._combobox(tab, 14, "Скорость реакции", "answer_fastest", ("", "Без регулятора", "С регулятором", "Одинаково", "Без сравнения"))
        self._combobox(tab, 15, "Действие регулятора", "answer_correction", ("", "Да", "Нет", "Регулятор выключен"))

    def _text_field(self, parent, row, label, key, height):
        ttk.Label(parent, text=label, style="Body.TLabel").grid(
            row=row, column=0, sticky="nw", pady=6, padx=(0, 14)
        )
        text = tk.Text(parent, height=height, wrap="word", font=("Segoe UI", 10), relief="solid", borderwidth=1)
        text.grid(row=row, column=1, columnspan=2, sticky="ew", pady=6)
        text.bind("<KeyRelease>", self._mark_dirty, add="+")
        text.bind("<<Paste>>", lambda _event: self.after_idle(self._mark_dirty), add="+")
        self._normal_widgets.append(text)
        self._lesson_texts[key] = text

    def _range_entries(self, parent, row, label, minimum_key, maximum_key):
        ttk.Label(parent, text=label, style="Body.TLabel").grid(
            row=row, column=0, sticky="w", pady=6, padx=(0, 14)
        )
        host = ttk.Frame(parent, style="CardBody.TFrame")
        host.grid(row=row, column=1, columnspan=2, sticky="ew", pady=6)
        host.columnconfigure((0, 1), weight=1)
        minimum = ttk.Entry(host, textvariable=self._variables[minimum_key])
        minimum.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        maximum = ttk.Entry(host, textvariable=self._variables[maximum_key])
        maximum.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self._normal_widgets.extend((minimum, maximum))
        self._field_widgets[minimum_key] = minimum
        self._field_widgets[maximum_key] = maximum
        return minimum, maximum

    def _entry(self, parent, row, label, key):
        ttk.Label(parent, text=label, style="Body.TLabel").grid(
            row=row, column=0, sticky="w", pady=6, padx=(0, 14)
        )
        entry = ttk.Entry(parent, textvariable=self._variables[key])
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        self._normal_widgets.append(entry)
        self._field_widgets[key] = entry
        error = ttk.Label(parent, text="", style="Error.TLabel", wraplength=180)
        error.grid(row=row, column=2, sticky="w", pady=6, padx=(8, 0))
        self._field_errors[key] = error
        return entry

    def _combobox(self, parent, row, label, key, values):
        ttk.Label(parent, text=label, style="Body.TLabel").grid(
            row=row, column=0, sticky="w", pady=6, padx=(0, 14)
        )
        widget = ttk.Combobox(
            parent,
            textvariable=self._variables[key],
            values=values,
            state="readonly",
        )
        widget.grid(row=row, column=1, sticky="ew", pady=6)
        self._readonly_widgets.append(widget)
        self._field_widgets[key] = widget
        error = ttk.Label(parent, text="", style="Error.TLabel", wraplength=180)
        error.grid(row=row, column=2, sticky="w", pady=6, padx=(8, 0))
        self._field_errors[key] = error
        return widget

    def _refresh_list(self, selected_name=None, load_selection=True):
        scenarios = self._filtered_scenarios()
        self._scenario_names = [scenario["name"] for scenario in scenarios]
        self._listbox.delete(0, tk.END)
        for scenario in scenarios:
            marker = "[встроенный] " if self.store.is_builtin(scenario["name"]) else "[мой] "
            self._listbox.insert(tk.END, marker + scenario["name"])
        if not scenarios:
            self._current_list_index = None
            return
        if selected_name not in self._scenario_names:
            selected_name = self._scenario_names[0]
        index = self._scenario_names.index(selected_name)
        self._listbox.selection_set(index)
        self._listbox.see(index)
        self._current_list_index = index
        if load_selection:
            self._load_scenario(scenarios[index])

    def _filtered_scenarios(self):
        query = self._search.get().strip().casefold()
        source_filter = self._filter.get()
        scenarios = []
        for scenario in self.store.scenarios:
            builtin = self.store.is_builtin(scenario["name"])
            if source_filter == "Встроенные" and not builtin:
                continue
            if source_filter == "Пользовательские" and builtin:
                continue
            searchable = f"{scenario['name']} {scenario['description']}".casefold()
            if query and query not in searchable:
                continue
            scenarios.append(scenario)
        if self._sort.get() == "По названию":
            scenarios.sort(key=lambda scenario: scenario["name"].casefold())
        return scenarios

    def _apply_list_filter(self, _event=None, *_):
        selected_name = self._original_name
        self._refresh_list(selected_name, load_selection=not self._dirty)

    def _select_from_list(self, _event=None):
        selection = self._listbox.curselection()
        if not selection:
            return
        index = selection[0]
        if index == self._current_list_index:
            return
        if not self._confirm_discard():
            self._listbox.selection_clear(0, tk.END)
            if self._current_list_index is not None:
                self._listbox.selection_set(self._current_list_index)
            return
        name = self._scenario_names[index]
        scenario = next(item for item in self.store.scenarios if item["name"] == name)
        self._current_list_index = index
        self._load_scenario(scenario)

    def _load_scenario(self, scenario, original_name=None):
        scenario = normalize_scenario(scenario)
        self._loading_form = True
        controller = scenario["controller"]
        lesson = scenario["lesson"]
        target = lesson["controller_target"] or {}
        answers = lesson["hidden_answers"]
        values = {
            "name": scenario["name"],
            "description": scenario["description"],
            "chain": CHAIN_NAMES[scenario["chain"]],
            "component_enabled": scenario["component"] is not None,
            "component": self._format_optional(scenario["component"]),
            "flow_enabled": scenario["flow"] is not None,
            "flow": self._format_optional(scenario["flow"]),
            "disturbance_type": scenario["disturbance_type"],
            "start_time": self._format_number(scenario["start_time"]),
            "simulation_duration": self._format_number(scenario["simulation_duration"]),
            "effect_duration": self._format_number(scenario["effect_duration"]),
            "time_constant": self._format_number(scenario["time_constant"]),
            "delay": self._format_number(scenario["delay"]),
            "controller_enabled": controller is not None,
            "controller_type": "PI" if controller is None else controller["type"],
            "gain": "2" if controller is None else self._format_number(controller["gain"]),
            "integral_time": "20" if controller is None else self._format_number(controller["integral_time"]),
            "derivative_time": "1" if controller is None else self._format_number(controller["derivative_time"]),
            "control_limit": "100" if controller is None else self._format_number(controller["control_limit"]),
            "setpoint": "" if controller is None else self._format_optional(controller.get("setpoint")),
            "steady_tolerance_percent": self._format_number(scenario.get("steady_tolerance_percent", 5.0)),
            "attempt_limit": str(lesson["attempt_limit"]),
            "target_enabled": bool(target),
            "target_type": target.get("type", "PI"),
            "target_gain_min": self._format_optional(target.get("gain_min")),
            "target_gain_max": self._format_optional(target.get("gain_max")),
            "target_integral_min": self._format_optional(target.get("integral_time_min")),
            "target_integral_max": self._format_optional(target.get("integral_time_max")),
            "target_derivative_min": self._format_optional(target.get("derivative_time_min")),
            "target_derivative_max": self._format_optional(target.get("derivative_time_max")),
            "answer_direction": answers.get("direction", ""),
            "answer_steady": self._format_optional(answers.get("steady")),
            "answer_fastest": answers.get("fastest", ""),
            "answer_correction": answers.get("correction", ""),
        }
        for key, value in values.items():
            self._variables[key].set(value)
        self._loading_form = False
        self._original_name = scenario["name"] if original_name is None else original_name
        builtin = self.store.is_builtin(self._original_name)
        self._set_editable(not builtin)
        self._dirty = False
        self._loaded_values = values.copy()
        lesson_values = {
            "task": lesson["task"],
            "guidance": lesson["guidance"],
            "questions": "\n".join(lesson["questions"]),
        }
        for key, value in lesson_values.items():
            text = self._lesson_texts[key]
            text.delete("1.0", tk.END)
            text.insert("1.0", value)
        self._loaded_lesson_texts = lesson_values.copy()
        self._clear_field_errors()
        self._update_field_styles()
        self._source.set("Встроенный · только просмотр" if builtin else "Пользовательский · можно редактировать")
        self._status.set(
            "Создайте копию, чтобы изменить встроенный сценарий."
            if builtin
            else "Изменения ещё не сохранены."
        )

    def _mark_dirty(self, *_):
        if self._loading_form or not self._editable:
            return
        current_values = {key: variable.get() for key, variable in self._variables.items()}
        lesson_values = {key: text.get("1.0", "end-1c") for key, text in self._lesson_texts.items()}
        self._dirty = (
            self._loaded_values is None
            or current_values != self._loaded_values
            or lesson_values != self._loaded_lesson_texts
        )
        self._update_field_styles()
        if not self._dirty:
            self._source.set("Пользовательский · можно редактировать")
            self._status.set("Изменения совпадают с сохранённой версией.")
            return
        if self._original_name is None:
            self._source.set("● Новый пользовательский сценарий")
        else:
            self._source.set("● Пользовательский · есть несохранённые изменения")
        self._status.set("Есть несохранённые изменения.")

    def _update_field_styles(self):
        if self._loaded_values is None:
            return
        for key, widget in self._field_widgets.items():
            changed = self._variables[key].get() != self._loaded_values.get(key)
            if isinstance(widget, ttk.Combobox):
                widget.configure(style="Dirty.TCombobox" if changed else "TCombobox")
            else:
                widget.configure(style="Dirty.TEntry" if changed else "TEntry")

    def _clear_field_errors(self):
        for label in self._field_errors.values():
            label.configure(text="")

    def _show_validation_error(self, error):
        self._clear_field_errors()
        message = str(error)
        labels = {
            "Название": "name", "Описание": "description", "Состав": "component",
            "Расход": "flow", "Цепь управления": "chain",
            "Начало воздействия": "start_time",
            "Длительность моделирования": "simulation_duration",
            "Длительность воздействия": "effect_duration",
            "Постоянная времени T": "time_constant", "Запаздывание L": "delay",
            "Допуск прогноза": "steady_tolerance_percent",
            "Коэффициент K": "gain", "Время интегрирования Ti": "integral_time",
            "Время дифференцирования Td": "derivative_time",
            "Ограничение управляющего воздействия": "control_limit",
            "Заданное значение": "setpoint",
        }
        key = labels.get(message.partition(":")[0])
        if key is None:
            key = next(
                (field_key for label, field_key in labels.items() if message.startswith(label)),
                None,
            )
        if key in self._field_errors:
            self._field_errors[key].configure(text=message)
        self._status.set(message)

    def _validate_scenario(self):
        try:
            self._collect_scenario()
        except ValueError as error:
            self._show_validation_error(error)
            return False
        self._clear_field_errors()
        self._status.set("Параметры корректны. Сценарий можно сохранить или открыть в расчёте.")
        return True

    def _confirm_discard(self):
        if not self._dirty:
            return True
        return messagebox.askyesno(
            "Несохранённые изменения",
            "Отменить несохранённые изменения сценария?",
            parent=self,
        )

    def _set_editable(self, editable):
        self._editable = editable
        for widget in self._normal_widgets:
            widget.configure(state="normal" if editable else "disabled")
        for widget in self._readonly_widgets:
            widget.configure(state="readonly" if editable else "disabled")
        self._save_button.configure(state="normal" if editable else "disabled")
        self._delete_button.configure(
            state="normal" if editable and self._original_name is not None else "disabled"
        )
        self._update_dependent_states()

    def _update_dependent_states(self, _event=None):
        if not self._editable:
            self._component_entry.configure(state="disabled")
            self._flow_entry.configure(state="disabled")
            for widget in (
                self._controller_type_box,
                self._gain_entry,
                self._integral_entry,
                self._derivative_entry,
                self._limit_entry,
                self._setpoint_entry,
            ):
                widget.configure(state="disabled")
            return

        self._component_entry.configure(
            state="normal" if self._variables["component_enabled"].get() else "disabled"
        )
        self._flow_entry.configure(
            state="normal" if self._variables["flow_enabled"].get() else "disabled"
        )
        controller_enabled = self._variables["controller_enabled"].get()
        self._controller_type_box.configure(state="readonly" if controller_enabled else "disabled")
        self._gain_entry.configure(state="normal" if controller_enabled else "disabled")
        self._limit_entry.configure(state="normal" if controller_enabled else "disabled")
        self._setpoint_entry.configure(state="normal" if controller_enabled else "disabled")
        controller_type = self._variables["controller_type"].get()
        self._integral_entry.configure(
            state="normal" if controller_enabled and "I" in controller_type else "disabled"
        )
        self._derivative_entry.configure(
            state="normal" if controller_enabled and "D" in controller_type else "disabled"
        )
        self._update_lesson_states()

    def _update_lesson_states(self):
        if not self._editable:
            state = "disabled"
        else:
            state = "normal" if self._variables["target_enabled"].get() else "disabled"
        self._target_type_box.configure(state="readonly" if state == "normal" else "disabled")
        for entry in (*self._target_gain_entries, *self._target_integral_entries, *self._target_derivative_entries):
            entry.configure(state=state)

    def _new_scenario(self):
        if not self._confirm_discard():
            return
        name = self._unique_name("Новый сценарий")
        scenario = {
            "name": name,
            "description": "Описание учебной ситуации.",
            "chain": CHAINS[0],
            "component": 0.1,
            "flow": None,
            "disturbance_type": DISTURBANCE_TYPES[0],
            "start_time": 10.0,
            "simulation_duration": 100.0,
            "effect_duration": 10.0,
            "time_constant": 10.0,
            "delay": 2.0,
            "controller": None,
            "steady_tolerance_percent": 5.0,
            "lesson": None,
        }
        self._listbox.selection_clear(0, tk.END)
        self._current_list_index = None
        self._load_scenario(scenario, original_name=None)
        self._dirty = True
        self._source.set("● Новый пользовательский сценарий")
        self._status.set("Заполните параметры и сохраните сценарий.")

    def _duplicate_scenario(self):
        try:
            scenario = self._selected_or_form_scenario()
        except ValueError as error:
            self._status.set(str(error))
            return
        duplicate = copy.deepcopy(scenario)
        duplicate["name"] = self._unique_name(f"Копия — {scenario['name']}")
        self._listbox.selection_clear(0, tk.END)
        self._current_list_index = None
        self._load_scenario(duplicate, original_name=None)
        self._dirty = True
        self._source.set("● Копия · новый пользовательский сценарий")
        self._status.set("Измените параметры при необходимости и нажмите «Сохранить».")

    def _save_scenario(self):
        try:
            scenario = self._collect_scenario()
            saved = self.store.save(scenario, self._original_name)
        except (OSError, ValueError) as error:
            self._show_validation_error(error)
            messagebox.showerror("Сценарий не сохранён", str(error), parent=self)
            return
        self._original_name = saved["name"]
        self._dirty = False
        self._search.set("")
        self._filter.set("Все")
        self._refresh_list(saved["name"])
        self._status.set("Сценарий сохранён.")
        self._on_change(saved["name"], "Пользовательский сценарий сохранён")

    def _save_as_new(self):
        try:
            scenario = self._collect_scenario()
            existing_names = {item["name"] for item in self.store.scenarios}
            if scenario["name"] in existing_names:
                scenario["name"] = self._unique_name(f"Копия — {scenario['name']}")
            saved = self.store.save(scenario)
        except (OSError, ValueError) as error:
            self._show_validation_error(error)
            messagebox.showerror("Сценарий не сохранён", str(error), parent=self)
            return
        self._dirty = False
        self._search.set("")
        self._filter.set("Все")
        self._refresh_list(saved["name"])
        self._status.set("Создан новый пользовательский сценарий.")
        self._on_change(saved["name"], "Создан новый пользовательский сценарий")

    def _revert_changes(self):
        if not self._dirty:
            self._status.set("Несохранённых изменений нет.")
            return
        if self._original_name is None:
            self._dirty = False
            self._new_scenario()
            return
        scenario = next(
            (item for item in self.store.scenarios if item["name"] == self._original_name),
            None,
        )
        if scenario is None:
            self._status.set("Исходный сценарий больше не существует.")
            return
        self._load_scenario(scenario)
        self._status.set("Несохранённые изменения отменены.")

    def _delete_scenario(self):
        if self._original_name is None:
            return
        if not messagebox.askyesno(
            "Удалить сценарий",
            f"Удалить пользовательский сценарий «{self._original_name}»?",
            parent=self,
        ):
            return
        try:
            self.store.delete(self._original_name)
        except (OSError, ValueError) as error:
            self._status.set(str(error))
            messagebox.showerror("Сценарий не удалён", str(error), parent=self)
            return
        self._refresh_list()
        self._on_change(self._variables["name"].get(), "Пользовательский сценарий удалён")

    def _move_selected_scenario(self, direction):
        if self._sort.get() != "Как задано":
            self._status.set("Для изменения порядка выберите сортировку «Как задано».")
            return
        selection = self._listbox.curselection()
        if not selection:
            self._status.set("Выберите пользовательский сценарий.")
            return
        name = self._scenario_names[selection[0]]
        try:
            moved = self.store.move(name, direction)
        except (OSError, ValueError) as error:
            self._status.set(str(error))
            return
        self._refresh_list(name)
        self._on_change(name, "Порядок пользовательских сценариев изменён" if moved else "Сценарий уже на границе списка")

    def _import_bundle(self):
        if not self._confirm_discard():
            return
        selected = filedialog.askopenfilename(
            parent=self,
            title="Импортировать комплект сценариев",
            filetypes=(("Комплект сценариев JSON", "*.json"),),
        )
        if not selected:
            return
        try:
            count = self.store.import_bundle(selected)
        except (OSError, ValueError) as error:
            self._status.set(str(error))
            messagebox.showerror("Импорт не выполнен", str(error), parent=self)
            return
        self._dirty = False
        self._search.set("")
        self._filter.set("Все")
        self._refresh_list()
        message = f"Импортировано сценариев: {count}. Совпадающие пользовательские записи обновлены."
        self._status.set(message)
        self._on_change(self._variables["name"].get(), message)

    def _export_bundle(self):
        selected = filedialog.asksaveasfilename(
            parent=self,
            title="Экспортировать комплект сценариев",
            defaultextension=".json",
            filetypes=(("Комплект сценариев JSON", "*.json"),),
            initialfile="absorption_scenarios.json",
        )
        if not selected:
            return
        try:
            destination = self.store.export_bundle(selected)
        except (OSError, ValueError) as error:
            self._status.set(str(error))
            messagebox.showerror("Экспорт не выполнен", str(error), parent=self)
            return
        self._status.set(f"Комплект сохранён: {destination.name}.")

    def _preview_scenario(self):
        if not self._validate_scenario():
            return
        scenario = self._collect_scenario()
        self._close(force=True)
        self._on_preview(scenario)

    def _selected_or_form_scenario(self):
        selection = self._listbox.curselection()
        if selection:
            name = self._scenario_names[selection[0]]
            return next(item for item in self.store.scenarios if item["name"] == name)
        return self._collect_scenario()

    def _collect_scenario(self):
        component = self._variables["component"].get() if self._variables["component_enabled"].get() else None
        flow = self._variables["flow"].get() if self._variables["flow_enabled"].get() else None
        controller = None
        if self._variables["controller_enabled"].get():
            setpoint_text = self._variables["setpoint"].get().strip()
            controller = {
                "type": self._variables["controller_type"].get(),
                "gain": self._variables["gain"].get(),
                "integral_time": self._variables["integral_time"].get(),
                "derivative_time": self._variables["derivative_time"].get(),
                "control_limit": self._variables["control_limit"].get(),
                "setpoint": None if not setpoint_text else setpoint_text,
            }
        return normalize_scenario({
            "name": self._variables["name"].get(),
            "description": self._variables["description"].get(),
            "chain": CHAIN_LABELS.get(self._variables["chain"].get()),
            "component": component,
            "flow": flow,
            "disturbance_type": self._variables["disturbance_type"].get(),
            "start_time": self._variables["start_time"].get(),
            "simulation_duration": self._variables["simulation_duration"].get(),
            "effect_duration": self._variables["effect_duration"].get(),
            "time_constant": self._variables["time_constant"].get(),
            "delay": self._variables["delay"].get(),
            "controller": controller,
            "steady_tolerance_percent": self._variables["steady_tolerance_percent"].get(),
            "lesson": self._collect_lesson(),
        })

    def _collect_lesson(self):
        answers = {}
        answer_keys = {
            "direction": "answer_direction",
            "steady": "answer_steady",
            "fastest": "answer_fastest",
            "correction": "answer_correction",
        }
        for lesson_key, variable_key in answer_keys.items():
            value = self._variables[variable_key].get().strip()
            if value:
                answers[lesson_key] = value
        target = None
        if self._variables["target_enabled"].get():
            target = {"type": self._variables["target_type"].get()}
            for source, destination in (
                ("target_gain_min", "gain_min"), ("target_gain_max", "gain_max"),
                ("target_integral_min", "integral_time_min"), ("target_integral_max", "integral_time_max"),
                ("target_derivative_min", "derivative_time_min"), ("target_derivative_max", "derivative_time_max"),
            ):
                value = self._variables[source].get().strip()
                if value:
                    target[destination] = value
        return {
            "task": self._lesson_texts["task"].get("1.0", "end-1c"),
            "guidance": self._lesson_texts["guidance"].get("1.0", "end-1c"),
            "questions": [
                question.strip()
                for question in self._lesson_texts["questions"].get("1.0", "end-1c").splitlines()
                if question.strip()
            ],
            "attempt_limit": self._variables["attempt_limit"].get(),
            "hidden_answers": answers,
            "controller_target": target,
        }

    def _unique_name(self, base):
        existing = {scenario["name"] for scenario in self.store.scenarios}
        if base not in existing:
            return base
        index = 2
        while f"{base} {index}" in existing:
            index += 1
        return f"{base} {index}"

    def request_close(self):
        return self._close()

    def _close(self, force=False):
        if not force and not self._confirm_discard():
            return False
        self._on_close()
        self.destroy()
        return True

    def _center(self, parent):
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        self.lift()
        self.focus_force()

    @staticmethod
    def _format_number(value):
        return f"{value:.4f}".rstrip("0").rstrip(".")

    @classmethod
    def _format_optional(cls, value):
        return "" if value is None else cls._format_number(value)
