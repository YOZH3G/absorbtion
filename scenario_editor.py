import copy
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from calculations import CONTROLLER_TYPES
from scenario_store import CHAINS, DISTURBANCE_TYPES, normalize_scenario


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
        }
        self._status = tk.StringVar()
        self._source = tk.StringVar()

        self.title("Редактор лабораторных сценариев")
        self.geometry("1180x780")
        self.minsize(1040, 700)
        self.configure(background=background)
        self.transient(parent)
        self._build_content()
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
        list_card.rowconfigure(1, weight=1)
        list_card.columnconfigure(0, weight=1)
        ttk.Label(list_card, text="Сценарии", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        list_host = ttk.Frame(list_card, style="CardBody.TFrame")
        list_host.grid(row=1, column=0, sticky="nsew")
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
        list_actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
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
        notebook.add(general, text="Сценарий")
        notebook.add(dynamics, text="Динамика")
        notebook.add(controller, text="Регулятор и задание")
        self._build_general_tab(general)
        self._build_dynamics_tab(dynamics)
        self._build_controller_tab(controller)

        ttk.Label(
            form,
            textvariable=self._status,
            style="Muted.TLabel",
            wraplength=680,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        actions = ttk.Frame(form, style="CardBody.TFrame")
        actions.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        actions.columnconfigure(2, weight=1)
        self._delete_button = ttk.Button(
            actions,
            text="Удалить",
            command=self._delete_scenario,
            style="Secondary.TButton",
        )
        self._delete_button.grid(row=0, column=0, padx=(0, 8))
        self._preview_button = ttk.Button(
            actions,
            text="Проверить в расчёте",
            command=self._preview_scenario,
            style="Secondary.TButton",
        )
        self._preview_button.grid(row=0, column=1)
        ttk.Button(
            actions,
            text="Закрыть",
            command=self._close,
            style="Secondary.TButton",
        ).grid(row=0, column=3, padx=(8, 8))
        self._save_button = ttk.Button(
            actions,
            text="Сохранить",
            command=self._save_scenario,
            style="Primary.TButton",
        )
        self._save_button.grid(row=0, column=4)

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

    def _entry(self, parent, row, label, key):
        ttk.Label(parent, text=label, style="Body.TLabel").grid(
            row=row, column=0, sticky="w", pady=6, padx=(0, 14)
        )
        entry = ttk.Entry(parent, textvariable=self._variables[key])
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        self._normal_widgets.append(entry)
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
        return widget

    def _refresh_list(self, selected_name=None):
        scenarios = self.store.scenarios
        self._scenario_names = [scenario["name"] for scenario in scenarios]
        self._listbox.delete(0, tk.END)
        for scenario in scenarios:
            marker = "[встроенный] " if self.store.is_builtin(scenario["name"]) else "[мой] "
            self._listbox.insert(tk.END, marker + scenario["name"])
        if not scenarios:
            self._new_scenario()
            return
        if selected_name not in self._scenario_names:
            selected_name = self._scenario_names[0]
        index = self._scenario_names.index(selected_name)
        self._listbox.selection_set(index)
        self._listbox.see(index)
        self._load_scenario(scenarios[index])

    def _select_from_list(self, _event=None):
        selection = self._listbox.curselection()
        if not selection:
            return
        name = self._scenario_names[selection[0]]
        scenario = next(item for item in self.store.scenarios if item["name"] == name)
        self._load_scenario(scenario)

    def _load_scenario(self, scenario, original_name=None):
        controller = scenario["controller"]
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
        }
        for key, value in values.items():
            self._variables[key].set(value)
        self._original_name = scenario["name"] if original_name is None else original_name
        builtin = self.store.is_builtin(self._original_name)
        self._set_editable(not builtin)
        self._source.set("Встроенный · только просмотр" if builtin else "Пользовательский · можно редактировать")
        self._status.set(
            "Создайте копию, чтобы изменить встроенный сценарий."
            if builtin
            else "Изменения ещё не сохранены."
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

    def _new_scenario(self):
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
        }
        self._listbox.selection_clear(0, tk.END)
        self._load_scenario(scenario, original_name=None)
        self._source.set("Новый пользовательский сценарий")
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
        self._load_scenario(duplicate, original_name=None)
        self._source.set("Копия · новый пользовательский сценарий")
        self._status.set("Измените параметры при необходимости и нажмите «Сохранить».")

    def _save_scenario(self):
        try:
            scenario = self._collect_scenario()
            saved = self.store.save(scenario, self._original_name)
        except (OSError, ValueError) as error:
            self._status.set(str(error))
            messagebox.showerror("Сценарий не сохранён", str(error), parent=self)
            return
        self._original_name = saved["name"]
        self._refresh_list(saved["name"])
        self._status.set("Сценарий сохранён.")
        self._on_change(saved["name"], "Пользовательский сценарий сохранён")

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

    def _import_bundle(self):
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
        try:
            scenario = self._collect_scenario()
        except ValueError as error:
            self._status.set(str(error))
            messagebox.showerror("Предварительный расчёт невозможен", str(error), parent=self)
            return
        self._close()
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
        })

    def _unique_name(self, base):
        existing = {scenario["name"] for scenario in self.store.scenarios}
        if base not in existing:
            return base
        index = 2
        while f"{base} {index}" in existing:
            index += 1
        return f"{base} {index}"

    def _close(self):
        self._on_close()
        self.destroy()

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
