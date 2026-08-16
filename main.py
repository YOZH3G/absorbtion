import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from calculations import (
    CONTROLLER_TYPES,
    IMPULSE,
    RAMP,
    RECTANGLE,
    STEP,
    calculate_xna,
    calculate_xog,
    combine_fractions,
    controller_response,
    controller_steady_state,
    disturbance_profile,
    first_order_response,
    transition_metrics,
    tune_controller_parameters,
)
from validation import parse_fraction, parse_nonnegative_number, parse_positive_number


BACKGROUND = "#F4F6F8"
CARD_BACKGROUND = "#FFFFFF"
BORDER = "#D7DCE2"
TEXT = "#1F2937"
MUTED = "#667085"
ACCENT = "#2563EB"
ACCENT_ACTIVE = "#1D4ED8"
ERROR = "#B42318"
SUCCESS = "#16A34A"
SIDEBAR = "#10243E"
SIDEBAR_MUTED = "#9AAAC0"

LEAN_GAS = "lean_gas"
RICH_ABSORBENT = "rich_absorbent"

DISTURBANCE_TYPES = {
    "Ступенчатое": STEP,
    "Импульсное": IMPULSE,
    "Временное прямоугольное": RECTANGLE,
    "Плавно нарастающее": RAMP,
}

CURVE_STYLES = {
    "Исходный режим": ("#667085", "--"),
    "Только состав": ("#F59E0B", "-"),
    "Только расход": ("#16A34A", "-"),
    "Совместное воздействие": (ACCENT, "-"),
}

DEFAULT_MODEL_VALUES = {
    "gna": 7800.0,
    "xa": 0.5,
    "xg": 0.5,
    "gg": 1000.0,
    "xog_initial": 0.8,
    "xna_initial": 30.0,
}


class AbsorptionApp(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, style="App.TFrame")
        self.root = root
        self.chain = LEAN_GAS
        self.model_values = DEFAULT_MODEL_VALUES.copy()
        self.model_dialog = None
        self.pages = {}
        self.nav_buttons = {}

        self.component_enabled = tk.BooleanVar(value=False)
        self.flow_enabled = tk.BooleanVar(value=False)
        self.component_value = tk.StringVar()
        self.flow_value = tk.StringVar()
        self.disturbance_type = tk.StringVar(value="Ступенчатое")
        self.dynamics_summary = tk.StringVar()
        self.page_title = tk.StringVar(value="Возмущения")
        self.start_time = tk.StringVar(value="10")
        self.simulation_duration = tk.StringVar(value="100")
        self.effect_duration = tk.StringVar(value="10")
        self.time_constant = tk.StringVar(value="10")
        self.delay = tk.StringVar(value="2")
        self.controller_enabled = tk.BooleanVar(value=False)
        self.controller_type = tk.StringVar(value="PI")
        self.proportional_gain = tk.StringVar(value="2")
        self.integral_time = tk.StringVar(value="20")
        self.derivative_time = tk.StringVar(value="1")
        self.control_limit = tk.StringVar(value="100")
        self.setpoint = tk.StringVar(value=self._format_number(DEFAULT_MODEL_VALUES["xog_initial"]))
        self.component_error = tk.StringVar()
        self.flow_error = tk.StringVar()
        self.dynamics_error = tk.StringVar()
        self.controller_error = tk.StringVar()
        self.tuning_summary = tk.StringVar(value="Автоподбор ещё не выполнялся.")
        self.controller_settings_title = tk.StringVar(value="Настройки PI-регулятора")
        self.controller_formula = tk.StringVar()
        self.controller_on_text = tk.StringVar(value="С PI-регулятором")
        self.component_label = tk.StringVar()
        self.component_symbol = tk.StringVar()
        self.flow_label = tk.StringVar()
        self.flow_symbol = tk.StringVar()
        self.baseline_result = tk.StringVar(value="—")
        self.disturbance_result = tk.StringVar(value="—")
        self.calculated_result = tk.StringVar(value="—")
        self.final_result = tk.StringVar(value="—")
        self.result_mode = tk.StringVar(value="Без регулятора")
        self.settling_time_label = tk.StringVar(value="Длительность установления (±5%)")
        self.calculation_steps = tk.StringVar(value="Выполните расчёт, чтобы увидеть происхождение результата.")
        self.transition_values = {
            key: tk.StringVar(value="—")
            for key in (
                "initial",
                "steady",
                "maximum_deviation",
                "relative_deviation",
                "time_constant",
                "settling_time",
                "settling_moment",
                "static_error",
            )
        }
        self.response_subtitle = tk.StringVar()
        self.primary_chart_title = tk.StringVar(value="Возмущающее воздействие")
        self.primary_chart_subtitle = tk.StringVar(value="Изменение относительно базового уровня")
        self.response_chart_title = tk.StringVar(value="Кривая разгона")
        self.status_text = tk.StringVar(value="Готово к расчёту")
        self.controller_signal_axis = None

        self._configure_window()
        self._configure_styles()
        self._build_layout()
        self._select_chain(LEAN_GAS)

    def _configure_window(self):
        self.root.title("Анализ процесса абсорбции")
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 780)
        self.root.configure(background=BACKGROUND)
        self.pack(fill="both", expand=True)

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("App.TFrame", background=BACKGROUND)
        style.configure("Card.TFrame", background=CARD_BACKGROUND, relief="solid", borderwidth=1)
        style.configure("CardBody.TFrame", background=CARD_BACKGROUND)
        style.configure("Header.TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 17, "bold"))
        style.configure("CardTitle.TLabel", background=CARD_BACKGROUND, foreground=TEXT, font=("Segoe UI", 12, "bold"))
        style.configure("Body.TLabel", background=CARD_BACKGROUND, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=CARD_BACKGROUND, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Error.TLabel", background=CARD_BACKGROUND, foreground=ERROR, font=("Segoe UI", 8))
        style.configure("ResultValue.TLabel", background=CARD_BACKGROUND, foreground=TEXT, font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("StatusDot.TLabel", background=BACKGROUND, foreground=SUCCESS, font=("Segoe UI", 14))
        style.configure("Topbar.TFrame", background="#FFFFFF")
        style.configure("TopbarTitle.TLabel", background="#FFFFFF", foreground=TEXT, font=("Segoe UI", 16, "bold"))
        style.configure("TopbarMeta.TLabel", background="#FFFFFF", foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Sidebar.TFrame", background=SIDEBAR)
        style.configure("SidebarTitle.TLabel", background=SIDEBAR, foreground="#FFFFFF", font=("Segoe UI", 14, "bold"))
        style.configure("SidebarMeta.TLabel", background=SIDEBAR, foreground=SIDEBAR_MUTED, font=("Segoe UI", 9))
        style.configure("SidebarNav.TButton", background=SIDEBAR, foreground="#E5ECF5", borderwidth=0, padding=(18, 13), font=("Segoe UI", 10), anchor="w")
        style.map("SidebarNav.TButton", background=[("active", "#193553")])
        style.configure("SelectedSidebarNav.TButton", background=ACCENT, foreground="#FFFFFF", borderwidth=0, padding=(18, 13), font=("Segoe UI", 10, "bold"), anchor="w")
        style.map("SelectedSidebarNav.TButton", background=[("active", ACCENT_ACTIVE)])
        style.configure("FutureSidebarNav.TButton", background=SIDEBAR, foreground="#687B94", borderwidth=0, padding=(18, 13), font=("Segoe UI", 10), anchor="w")
        style.map("FutureSidebarNav.TButton", background=[("disabled", SIDEBAR)], foreground=[("disabled", "#687B94")])
        style.configure("SectionHeader.TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 15, "bold"))
        style.configure("MetricTitle.TLabel", background=CARD_BACKGROUND, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("MetricValue.TLabel", background=CARD_BACKGROUND, foreground=TEXT, font=("Segoe UI", 18, "bold"))

        style.configure("TEntry", padding=7, fieldbackground="#FFFFFF", bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        style.configure("Error.TEntry", padding=7, fieldbackground="#FFF6F5", bordercolor=ERROR, lightcolor=ERROR, darkcolor=ERROR)
        style.configure("TCheckbutton", background=CARD_BACKGROUND, foreground=TEXT, font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", CARD_BACKGROUND)])

        style.configure("Primary.TButton", background=ACCENT, foreground="#FFFFFF", borderwidth=0, padding=(14, 10), font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", background=[("active", ACCENT_ACTIVE), ("disabled", "#AEBBD0")], foreground=[("disabled", "#F2F4F7")])
        style.configure("Secondary.TButton", background="#FFFFFF", foreground=TEXT, bordercolor=BORDER, padding=(12, 9), font=("Segoe UI", 10))
        style.map("Secondary.TButton", background=[("active", "#EEF2F6")])
        style.configure("Segment.TButton", background="#FFFFFF", foreground=TEXT, bordercolor=BORDER, padding=(12, 10), font=("Segoe UI", 10))
        style.configure("SelectedSegment.TButton", background=ACCENT, foreground="#FFFFFF", bordercolor=ACCENT, padding=(12, 10), font=("Segoe UI", 10))
        style.map("SelectedSegment.TButton", background=[("active", ACCENT_ACTIVE)])
        style.configure("Toolbar.TButton", background="#FFFFFF", foreground=TEXT, bordercolor=BORDER, padding=(6, 4), font=("Segoe UI", 8))
        style.map("Toolbar.TButton", background=[("active", "#EEF2F6")])

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        topbar = ttk.Frame(self, style="Topbar.TFrame", padding=(22, 12))
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.columnconfigure(0, weight=1)
        ttk.Label(topbar, text="Анализ процесса абсорбции", style="TopbarTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(topbar, text="Учебный стенд АТПП", style="TopbarMeta.TLabel").grid(row=0, column=1, sticky="e")

        body = ttk.Frame(self, style="App.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, minsize=220)
        body.columnconfigure(1, minsize=390)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(body, style="Sidebar.TFrame", padding=(12, 20))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)
        ttk.Label(sidebar, text="АБСОРБЦИЯ", style="SidebarTitle.TLabel").grid(row=0, column=0, sticky="w", padx=8, pady=(0, 4))
        ttk.Label(sidebar, text="Моделирование контуров", style="SidebarMeta.TLabel").grid(row=1, column=0, sticky="w", padx=8, pady=(0, 24))

        active_navigation = (
            ("disturbances", "Возмущения"),
            ("dynamics", "Динамика"),
            ("results", "Результаты"),
            ("controller", "Регулятор"),
        )
        for row, (key, label) in enumerate(active_navigation, start=2):
            button = ttk.Button(sidebar, text=label, command=lambda page=key: self._show_page(page), style="SidebarNav.TButton")
            button.grid(row=row, column=0, sticky="ew", pady=2)
            self.nav_buttons[key] = button

        ttk.Separator(sidebar).grid(row=6, column=0, sticky="ew", padx=8, pady=16)
        for row, label in enumerate(("Сценарии · скоро", "Экспорт · скоро"), start=7):
            ttk.Button(sidebar, text=label, state="disabled", style="FutureSidebarNav.TButton").grid(row=row, column=0, sticky="ew", pady=2)

        inspector = ttk.Frame(body, style="App.TFrame", padding=(16, 16, 12, 12))
        inspector.grid(row=0, column=1, sticky="nsew")
        inspector.columnconfigure(0, weight=1)
        inspector.rowconfigure(1, weight=1)
        ttk.Label(inspector, textvariable=self.page_title, style="SectionHeader.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 12))

        page_host = ttk.Frame(inspector, style="App.TFrame")
        page_host.grid(row=1, column=0, sticky="nsew")
        page_host.columnconfigure(0, weight=1)
        page_host.rowconfigure(0, weight=1)
        for key in ("disturbances", "dynamics", "results", "controller"):
            page = ttk.Frame(page_host, style="App.TFrame")
            page.grid(row=0, column=0, sticky="nsew")
            page.columnconfigure(0, weight=1)
            self.pages[key] = page

        self._build_chain_card(self.pages["disturbances"])
        self._build_disturbance_card(self.pages["disturbances"])
        self._build_control_diagram(self.pages["disturbances"])
        self._build_dynamics_card(self.pages["dynamics"])
        self._build_result_card(self.pages["results"])
        self._build_controller_card(self.pages["controller"])

        actions = ttk.Frame(inspector, style="App.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure((0, 1), weight=1)
        self.calculate_button = ttk.Button(actions, text="Рассчитать", command=self._calculate, style="Primary.TButton", state="disabled")
        self.calculate_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(actions, text="Сбросить", command=self._reset, style="Secondary.TButton").grid(row=0, column=1, sticky="ew", padx=(5, 0))

        workspace = ttk.Frame(body, style="App.TFrame", padding=(4, 16, 16, 12))
        workspace.grid(row=0, column=2, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(1, weight=1)
        workspace.rowconfigure(2, weight=1)
        self._build_metric_strip(workspace)

        self.disturbance_axis, self.disturbance_canvas, self.disturbance_toolbar = self._build_chart_card(
            workspace,
            row=1,
            title_variable=self.primary_chart_title,
            subtitle_variable=self.primary_chart_subtitle,
        )
        self.response_axis, self.response_canvas, self.response_toolbar = self._build_chart_card(
            workspace,
            row=2,
            title_variable=self.response_chart_title,
            subtitle_variable=self.response_subtitle,
        )

        status = ttk.Frame(self, style="App.TFrame", padding=(18, 6, 18, 8))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(1, weight=1)
        self.status_dot = ttk.Label(status, text="●", style="StatusDot.TLabel")
        self.status_dot.grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.status_text, style="Status.TLabel").grid(row=0, column=1, sticky="w", padx=(5, 0))
        ttk.Label(status, text="Python 3.14", style="Status.TLabel").grid(row=0, column=2, sticky="e")
        self._show_page("disturbances")

    def _build_metric_strip(self, parent):
        metrics = ttk.Frame(parent, style="App.TFrame")
        metrics.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        metrics.columnconfigure((0, 1, 2), weight=1)
        for column, (label, variable) in enumerate((
            ("Базовое значение", self.baseline_result),
            ("Суммарная доля", self.disturbance_result),
            ("Расчётное значение", self.calculated_result),
        )):
            card = ttk.Frame(metrics, style="Card.TFrame", padding=(16, 12))
            card.grid(row=0, column=column, sticky="ew", padx=(0, 6) if column < 2 else 0)
            ttk.Label(card, text=label, style="MetricTitle.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(card, textvariable=variable, style="MetricValue.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))

    def _show_page(self, page):
        titles = {
            "disturbances": "Возмущения",
            "dynamics": "Динамика объекта",
            "results": "Результаты расчёта",
            "controller": "Регулятор",
        }
        self.pages[page].tkraise()
        self.page_title.set(titles[page])
        for key, button in self.nav_buttons.items():
            button.configure(style="SelectedSidebarNav.TButton" if key == page else "SidebarNav.TButton")

    def _card(self, parent, row, padding=(18, 16)):
        card = ttk.Frame(parent, style="Card.TFrame", padding=padding)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 14))
        card.columnconfigure(0, weight=1)
        return card

    def _build_chain_card(self, parent):
        card = self._card(parent, 0)
        ttk.Label(card, text="Цепь управления", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 14))
        ttk.Button(
            card,
            text="Параметры модели",
            command=self._open_model_parameters,
            style="Toolbar.TButton",
        ).grid(row=0, column=1, sticky="e", pady=(0, 14))

        self.lean_button = ttk.Button(card, text="Обеднённый газ", command=lambda: self._select_chain(LEAN_GAS), style="Segment.TButton")
        self.lean_button.grid(row=1, column=0, sticky="ew")
        self.rich_button = ttk.Button(card, text="Насыщенный абсорбент", command=lambda: self._select_chain(RICH_ABSORBENT), style="Segment.TButton")
        self.rich_button.grid(row=1, column=1, sticky="ew", padx=(4, 0))
        card.columnconfigure((0, 1), weight=1)

    def _open_model_parameters(self):
        if self.model_dialog is not None and self.model_dialog.winfo_exists():
            self.model_dialog.lift()
            self.model_dialog.focus_force()
            return

        dialog = tk.Toplevel(self.root)
        self.model_dialog = dialog
        dialog.title("Параметры математической модели")
        dialog.configure(background=BACKGROUND)
        dialog.resizable(False, False)
        dialog.transient(self.root)

        content = ttk.Frame(dialog, style="App.TFrame", padding=20)
        content.pack(fill="both", expand=True)
        content.columnconfigure((0, 1), weight=1)

        ttk.Label(content, text="Параметры математической модели", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            content,
            text="Все изменения применяются одновременно. Допустимы конечные числа больше нуля.",
            style="Status.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 16))

        variables = {
            key: tk.StringVar(value=self._format_number(value))
            for key, value in self.model_values.items()
        }
        errors = {key: tk.StringVar() for key in variables}
        entries = {}

        gas_specs = (
            ("gg", "Расход газовой смеси", "Gг"),
            ("xg", "Доля компонента в исходном газе", "Xг"),
            ("xog_initial", "Начальная концентрация обеднённого газа", "Xог₀"),
        )
        absorbent_specs = (
            ("gna", "Расход насыщенного абсорбента", "Gна"),
            ("xa", "Состав исходного абсорбента", "Xа"),
            ("xna_initial", "Начальная концентрация насыщенного абсорбента", "Xна₀"),
        )

        def build_parameter_group(column, title, specs):
            group = ttk.Frame(content, style="Card.TFrame", padding=(16, 14))
            group.grid(row=2, column=column, sticky="nsew", padx=(0, 7) if column == 0 else (7, 0))
            group.columnconfigure(0, weight=1)
            ttk.Label(group, text=title, style="CardTitle.TLabel").grid(
                row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
            )
            for index, (key, label, symbol) in enumerate(specs):
                row = 1 + index * 2
                ttk.Label(group, text=label, style="Body.TLabel", wraplength=220).grid(row=row, column=0, sticky="w")
                ttk.Label(group, text=symbol, style="Body.TLabel").grid(row=row, column=1, padx=(10, 8))
                entry = ttk.Entry(group, textvariable=variables[key], width=11)
                entry.grid(row=row, column=2, sticky="e")
                entries[key] = entry
                ttk.Label(group, textvariable=errors[key], style="Error.TLabel").grid(
                    row=row + 1, column=0, columnspan=3, sticky="w", pady=(2, 7)
                )

        build_parameter_group(0, "Газовая часть", gas_specs)
        build_parameter_group(1, "Абсорбент", absorbent_specs)

        derived = ttk.Frame(content, style="Card.TFrame", padding=(16, 12))
        derived.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        derived.columnconfigure((1, 3), weight=1)
        ttk.Label(derived, text="Расчётные значения", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8)
        )
        gog_value = tk.StringVar(value="—")
        ga_value = tk.StringVar(value="—")
        ttk.Label(derived, text="Gог — расход обеднённого газа", style="Body.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(derived, textvariable=gog_value, style="ResultValue.TLabel").grid(row=1, column=1, sticky="w", padx=(8, 28))
        ttk.Label(derived, text="Gа — расход абсорбента", style="Body.TLabel").grid(row=1, column=2, sticky="w")
        ttk.Label(derived, textvariable=ga_value, style="ResultValue.TLabel").grid(row=1, column=3, sticky="w", padx=(8, 0))

        actions = ttk.Frame(content, style="App.TFrame")
        actions.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        actions.columnconfigure(1, weight=1)

        parsed_values = None

        def validate_values(*_):
            nonlocal parsed_values
            parsed = {}
            valid = True
            for key, variable in variables.items():
                try:
                    parsed[key] = parse_positive_number(variable.get())
                    errors[key].set("")
                    entries[key].configure(style="TEntry")
                except ValueError as error:
                    errors[key].set(str(error))
                    entries[key].configure(style="Error.TEntry")
                    valid = False

            if valid:
                gog = (parsed["gg"] * parsed["xg"]) / parsed["xog_initial"]
                ga = (parsed["gna"] * parsed["xna_initial"]) / parsed["xa"]
                gog_value.set(self._format_number(gog))
                ga_value.set(self._format_number(ga))
                parsed_values = parsed
                apply_button.configure(state="normal")
            else:
                gog_value.set("—")
                ga_value.set("—")
                parsed_values = None
                apply_button.configure(state="disabled")

        def restore_defaults():
            for key, value in DEFAULT_MODEL_VALUES.items():
                variables[key].set(self._format_number(value))

        def close_dialog():
            if dialog.grab_current() is dialog:
                dialog.grab_release()
            self.model_dialog = None
            dialog.destroy()

        def apply_values():
            validate_values()
            if parsed_values is None:
                return
            self.model_values = parsed_values.copy()
            self.setpoint.set(self._format_number(self._current_baseline()))
            close_dialog()
            self._reset()
            self._set_status("Параметры модели обновлены")

        ttk.Button(actions, text="По умолчанию", command=restore_defaults, style="Secondary.TButton").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(actions, text="Отмена", command=close_dialog, style="Secondary.TButton").grid(
            row=0, column=2, padx=(8, 8)
        )
        apply_button = ttk.Button(actions, text="Применить", command=apply_values, style="Primary.TButton")
        apply_button.grid(row=0, column=3)

        for variable in variables.values():
            variable.trace_add("write", validate_values)
        validate_values()

        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.bind("<Escape>", lambda _event: close_dialog())
        dialog.bind("<Return>", lambda _event: apply_values())
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - dialog.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dialog.winfo_reqheight()) // 2
        dialog.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        dialog.grab_set()
        dialog.lift()
        dialog.focus_force()
        entries["gg"].focus_set()

    def _build_disturbance_card(self, parent):
        card = self._card(parent, 1)
        ttk.Label(card, text="Возмущающие воздействия", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 14))

        self.component_check = ttk.Checkbutton(
            card,
            textvariable=self.component_label,
            variable=self.component_enabled,
            command=self._update_input_states,
        )
        self.component_check.grid(row=1, column=0, sticky="w")
        ttk.Label(card, textvariable=self.component_symbol, style="Body.TLabel").grid(row=1, column=1, padx=(8, 8))
        self.component_entry = ttk.Entry(card, textvariable=self.component_value, width=8, state="disabled")
        self.component_entry.grid(row=1, column=2, sticky="ew")
        ttk.Label(card, textvariable=self.component_error, style="Error.TLabel", wraplength=300).grid(row=2, column=0, columnspan=3, sticky="w", pady=(3, 8))

        self.flow_check = ttk.Checkbutton(
            card,
            textvariable=self.flow_label,
            variable=self.flow_enabled,
            command=self._update_input_states,
        )
        self.flow_check.grid(row=3, column=0, sticky="w")
        ttk.Label(card, textvariable=self.flow_symbol, style="Body.TLabel").grid(row=3, column=1, padx=(8, 8))
        self.flow_entry = ttk.Entry(card, textvariable=self.flow_value, width=8, state="disabled")
        self.flow_entry.grid(row=3, column=2, sticky="ew")
        ttk.Label(card, textvariable=self.flow_error, style="Error.TLabel", wraplength=300).grid(row=4, column=0, columnspan=3, sticky="w", pady=(3, 8))

        ttk.Label(card, text="Доля: +0.10 = увеличение на 10%, −0.10 = уменьшение на 10%. Диапазон: −0.99…9.99", style="Muted.TLabel", wraplength=320).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(2, 14)
        )

        ttk.Separator(card).grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        ttk.Button(
            card,
            text="Параметры динамики  →",
            command=lambda: self._show_page("dynamics"),
            style="Secondary.TButton",
        ).grid(row=7, column=0, columnspan=3, sticky="ew")
        ttk.Label(card, textvariable=self.dynamics_summary, style="Muted.TLabel", wraplength=320).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )
        card.columnconfigure(0, weight=1)

    def _build_control_diagram(self, parent):
        card = self._card(parent, 2, padding=(14, 12))
        ttk.Label(card, text="Структурная схема", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(
            card,
            text="Включённые возмущения подсвечиваются синим.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(0, 6))
        self.control_diagram = tk.Canvas(
            card,
            width=340,
            height=180,
            background=CARD_BACKGROUND,
            borderwidth=0,
            highlightthickness=0,
        )
        self.control_diagram.grid(row=2, column=0, sticky="ew")
        self.component_value.trace_add("write", lambda *_: self._draw_control_diagram())
        self.flow_value.trace_add("write", lambda *_: self._draw_control_diagram())
        self._draw_control_diagram()

    def _build_dynamics_card(self, parent):
        card = self._card(parent, 0)
        ttk.Label(card, text="Параметры времени и формы", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 14)
        )

        ttk.Label(card, text="Вид воздействия", style="Body.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(card, text="Начало, с", style="Body.TLabel").grid(row=1, column=1, sticky="w", padx=(10, 0))
        self.disturbance_type_box = ttk.Combobox(
            card,
            textvariable=self.disturbance_type,
            values=tuple(DISTURBANCE_TYPES),
            state="readonly",
            width=23,
        )
        self.disturbance_type_box.grid(row=2, column=0, sticky="ew", pady=(3, 12))
        self.disturbance_type_box.bind("<<ComboboxSelected>>", self._update_disturbance_type)
        self.start_time_entry = ttk.Entry(card, textvariable=self.start_time)
        self.start_time_entry.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=(3, 12))

        ttk.Label(card, text="Длительность моделирования, с", style="Body.TLabel", wraplength=170).grid(row=3, column=0, sticky="w")
        ttk.Label(card, text="Длительность воздействия, с", style="Body.TLabel", wraplength=170).grid(row=3, column=1, sticky="w", padx=(10, 0))
        self.simulation_duration_entry = ttk.Entry(card, textvariable=self.simulation_duration)
        self.simulation_duration_entry.grid(row=4, column=0, sticky="ew", pady=(3, 12))
        self.effect_duration_entry = ttk.Entry(card, textvariable=self.effect_duration)
        self.effect_duration_entry.grid(row=4, column=1, sticky="ew", padx=(10, 0), pady=(3, 12))

        ttk.Label(card, text="Постоянная времени T, с", style="Body.TLabel").grid(row=5, column=0, sticky="w")
        ttk.Label(card, text="Запаздывание L, с", style="Body.TLabel").grid(row=5, column=1, sticky="w", padx=(10, 0))
        self.time_constant_entry = ttk.Entry(card, textvariable=self.time_constant)
        self.time_constant_entry.grid(row=6, column=0, sticky="ew", pady=(3, 0))
        self.delay_entry = ttk.Entry(card, textvariable=self.delay)
        self.delay_entry.grid(row=6, column=1, sticky="ew", padx=(10, 0), pady=(3, 0))

        self.dynamics_error_label = ttk.Label(
            card,
            textvariable=self.dynamics_error,
            style="Error.TLabel",
            wraplength=330,
        )
        self.dynamics_error_label.grid(row=7, column=0, columnspan=2, sticky="w", pady=(6, 0))
        card.columnconfigure((0, 1), weight=1)
        for variable in (
            self.disturbance_type,
            self.start_time,
            self.simulation_duration,
            self.time_constant,
            self.delay,
        ):
            variable.trace_add("write", self._update_dynamics_summary)
        self._update_disturbance_type()

    def _build_controller_card(self, parent):
        card = self._card(parent, 0, padding=(14, 10))
        ttk.Label(card, text="Режим управления", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )
        self.controller_off_button = ttk.Button(
            card,
            text="Без регулятора",
            command=lambda: self._set_controller_mode(False),
            style="SelectedSegment.TButton",
        )
        self.controller_off_button.grid(row=1, column=0, sticky="ew")
        self.controller_on_button = ttk.Button(
            card,
            textvariable=self.controller_on_text,
            command=lambda: self._set_controller_mode(True),
            style="Segment.TButton",
        )
        self.controller_on_button.grid(row=1, column=1, sticky="ew", padx=(4, 0))
        card.columnconfigure((0, 1), weight=1)

        parameters = self._card(parent, 1, padding=(14, 10))
        ttk.Label(parameters, textvariable=self.controller_settings_title, style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        ttk.Label(parameters, text="Тип регулятора", style="Body.TLabel").grid(
            row=1, column=0, sticky="w", pady=3
        )
        self.controller_type_box = ttk.Combobox(
            parameters,
            textvariable=self.controller_type,
            values=CONTROLLER_TYPES,
            state="readonly",
            width=10,
        )
        self.controller_type_box.grid(row=1, column=1, sticky="e", pady=3, padx=(10, 0))
        self.controller_type_box.bind("<<ComboboxSelected>>", self._update_controller_type)
        fields = (
            ("Коэффициент регулятора K", self.proportional_gain, "proportional_gain_entry"),
            ("Время интегрирования Ti, с", self.integral_time, "integral_time_entry"),
            ("Время дифференцирования Td, с", self.derivative_time, "derivative_time_entry"),
            ("Ограничение |u|", self.control_limit, "control_limit_entry"),
            ("Заданное значение", self.setpoint, "setpoint_entry"),
        )
        self.controller_entries = []
        for row, (label, variable, attribute) in enumerate(fields, start=2):
            ttk.Label(parameters, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(parameters, textvariable=variable, width=12, state="disabled")
            entry.grid(row=row, column=1, sticky="e", pady=3, padx=(10, 0))
            setattr(self, attribute, entry)
            self.controller_entries.append(entry)
        self.auto_tune_button = ttk.Button(
            parameters,
            text="Подобрать автоматически",
            command=self._auto_tune_controller,
            style="Secondary.TButton",
        )
        self.auto_tune_button.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        ttk.Label(
            parameters,
            textvariable=self.tuning_summary,
            style="Muted.TLabel",
            wraplength=330,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(3, 0))
        ttk.Label(
            parameters,
            textvariable=self.controller_error,
            style="Error.TLabel",
            wraplength=330,
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(5, 0))
        parameters.columnconfigure(0, weight=1)

        explanation = self._card(parent, 2, padding=(14, 10))
        ttk.Label(explanation, text="Учебная модель", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(
            explanation,
            textvariable=self.controller_formula,
            style="Body.TLabel",
            justify="left",
            wraplength=330,
        ).grid(row=1, column=0, sticky="w")
        self._update_controller_type()

    def _build_result_card(self, parent):
        card = self._card(parent, 0)
        ttk.Label(card, text="Результат", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        rows = (
            ("Базовое значение", self.baseline_result),
            ("Суммарная доля", self.disturbance_result),
            ("Расчётное значение", self.calculated_result),
            ("Режим", self.result_mode),
            ("В конце моделирования", self.final_result),
        )
        for row, (label, variable) in enumerate(rows, start=1):
            ttk.Label(card, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=6)
            ttk.Label(card, textvariable=variable, style="ResultValue.TLabel").grid(row=row, column=1, sticky="e", pady=6)
        card.columnconfigure(0, weight=1)

        calculation = self._card(parent, 1)
        ttk.Label(calculation, text="Ход расчёта", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        ttk.Label(
            calculation,
            textvariable=self.calculation_steps,
            style="Body.TLabel",
            justify="left",
            wraplength=330,
        ).grid(row=1, column=0, sticky="w")

        metrics = self._card(parent, 2)
        ttk.Label(metrics, text="Показатели переходного процесса", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        metric_rows = (
            ("Начальное значение", "initial"),
            ("Установившееся значение", "steady"),
            ("Максимальное отклонение", "maximum_deviation"),
            ("Относительное отклонение", "relative_deviation"),
            ("Постоянная времени T", "time_constant"),
            (self.settling_time_label, "settling_time"),
            ("Момент установления на графике", "settling_moment"),
            ("Статическая ошибка eуст", "static_error"),
        )
        for row, (label, key) in enumerate(metric_rows, start=1):
            label_options = {"textvariable": label} if isinstance(label, tk.StringVar) else {"text": label}
            ttk.Label(metrics, style="Body.TLabel", **label_options).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Label(metrics, textvariable=self.transition_values[key], style="ResultValue.TLabel").grid(
                row=row, column=1, sticky="e", pady=3, padx=(8, 0)
            )
        metrics.columnconfigure(0, weight=1)

    def _build_chart_card(
        self,
        parent,
        row,
        title=None,
        subtitle=None,
        title_variable=None,
        subtitle_variable=None,
    ):
        card = ttk.Frame(parent, style="Card.TFrame", padding=(16, 12))
        card.grid(row=row, column=0, sticky="nsew", pady=(0, 8) if row == 0 else (8, 0))
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        header = ttk.Frame(card, style="CardBody.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header.columnconfigure(0, weight=1)
        if title_variable is not None:
            ttk.Label(header, textvariable=title_variable, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        else:
            ttk.Label(header, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        if subtitle_variable is not None:
            ttk.Label(header, textvariable=subtitle_variable, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        else:
            ttk.Label(header, text=subtitle, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))

        plot_host = ttk.Frame(card, style="CardBody.TFrame")
        plot_host.grid(row=1, column=0, sticky="nsew")
        plot_host.columnconfigure(0, weight=1)
        plot_host.rowconfigure(0, weight=1)

        figure = Figure(figsize=(7, 3), dpi=100, facecolor=CARD_BACKGROUND, constrained_layout=True)
        axis = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=plot_host)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        hidden_toolbar_host = ttk.Frame(card, style="CardBody.TFrame")
        toolbar = NavigationToolbar2Tk(canvas, hidden_toolbar_host, pack_toolbar=False)
        toolbar.update()

        tools = ttk.Frame(header, style="CardBody.TFrame")
        tools.grid(row=0, column=1, rowspan=2, sticky="e")
        for column, (label, command) in enumerate((
            ("Сброс", toolbar.home),
            ("Zoom", toolbar.zoom),
            ("Pan", toolbar.pan),
            ("PNG", toolbar.save_figure),
        )):
            ttk.Button(tools, text=label, command=command, style="Toolbar.TButton", width=7).grid(row=0, column=column, padx=(4, 0))

        return axis, canvas, toolbar

    def _select_chain(self, chain):
        self.chain = chain
        self.lean_button.configure(style="SelectedSegment.TButton" if chain == LEAN_GAS else "Segment.TButton")
        self.rich_button.configure(style="SelectedSegment.TButton" if chain == RICH_ABSORBENT else "Segment.TButton")

        if chain == LEAN_GAS:
            self.component_label.set("Состав исходного газа")
            self.component_symbol.set("Xг")
            self.flow_label.set("Расход газовой смеси")
            self.flow_symbol.set("Gг")
            self.response_subtitle.set("Концентрация обеднённого газа")
        else:
            self.component_label.set("Исходный состав абсорбента")
            self.component_symbol.set("Xа")
            self.flow_label.set("Расход абсорбента")
            self.flow_symbol.set("Gа")
            self.response_subtitle.set("Концентрация насыщенного абсорбента")

        self.setpoint.set(self._format_number(self._current_baseline()))
        self._reset()

    def _current_baseline(self):
        return (
            self.model_values["xog_initial"]
            if self.chain == LEAN_GAS
            else self.model_values["xna_initial"]
        )

    def _update_controller_type(self, _event=None):
        controller_type = self.controller_type.get()
        self.controller_settings_title.set(f"Настройки {controller_type}-регулятора")
        self.controller_on_text.set(f"С {controller_type}-регулятором")
        formulas = {
            "P": "u(t) = K · e(t)",
            "PI": "u(t) = K · (e(t) + 1/Ti · ∫e(t)dt)",
            "PID": "u(t) = K · (e(t) + 1/Ti · ∫e(t)dt − Td · dy(t)/dt)",
            "PD": "u(t) = K · (e(t) − Td · dy(t)/dt)",
        }
        details = []
        if "I" in controller_type:
            details.append("Интегратор защищён от насыщения.")
        if "D" in controller_type:
            details.append("Производная берётся по выходу без скачка от задания.")
        details.append("|u| задаёт предел воздействия.")
        self.controller_formula.set(
            f"{formulas[controller_type]}\n\n"
            f"{' '.join(details)}\n\n"
            "Подбор: λ = max(0.5T, L); для PID/PD при L > 0 — λ/2; "
            "K = T/(λ + L); Ti = min(T, 4(λ + L)); Td = L/3."
        )
        self.auto_tune_button.configure(state="normal")
        self.tuning_summary.set(f"Автоподбор {controller_type} готов к запуску.")
        self._update_controller_entry_states()
        if hasattr(self, "disturbance_axis") and _event is not None:
            self._clear_result_values()
            self._draw_static_charts()
            self._set_status("Тип регулятора изменён — выполните расчёт")

    def _update_controller_entry_states(self):
        enabled = self.controller_enabled.get()
        controller_type = self.controller_type.get()
        states = {
            self.proportional_gain_entry: enabled,
            self.integral_time_entry: enabled and "I" in controller_type,
            self.derivative_time_entry: enabled and "D" in controller_type,
            self.control_limit_entry: enabled,
            self.setpoint_entry: enabled,
        }
        for entry, active in states.items():
            entry.configure(state="normal" if active else "disabled")

    def _set_controller_mode(self, enabled):
        self.controller_enabled.set(enabled)
        self.settling_time_label.set(
            "Длительность регулирования (±5%)"
            if enabled
            else "Длительность установления (±5%)"
        )
        self.controller_off_button.configure(
            style="Segment.TButton" if enabled else "SelectedSegment.TButton"
        )
        self.controller_on_button.configure(
            style="SelectedSegment.TButton" if enabled else "Segment.TButton"
        )
        self._update_controller_entry_states()
        self.controller_error.set("")
        if hasattr(self, "disturbance_axis"):
            self._clear_result_values()
            self._draw_static_charts()
            self._set_status("Режим управления изменён — выполните расчёт")

    def _auto_tune_controller(self):
        fields = (
            ("Постоянная времени T", self.time_constant, self.time_constant_entry, parse_positive_number),
            ("Запаздывание L", self.delay, self.delay_entry, parse_nonnegative_number),
        )
        parsed = {}
        self.dynamics_error.set("")
        for _label, _variable, entry, _parser in fields:
            entry.configure(style="TEntry")

        for label, variable, entry, parser in fields:
            try:
                parsed[label] = parser(variable.get())
            except ValueError as error:
                entry.configure(style="Error.TEntry")
                self.dynamics_error.set(f"{label}: {error}")
                self._show_page("dynamics")
                self._set_status("Исправьте параметры динамики", error=True)
                return

        controller_type = self.controller_type.get()
        tuning = tune_controller_parameters(
            controller_type,
            parsed["Постоянная времени T"],
            parsed["Запаздывание L"],
        )
        self._set_controller_mode(True)
        self.proportional_gain.set(self._format_number(tuning["proportional_gain"]))
        if tuning["integral_time"] is not None:
            self.integral_time.set(self._format_number(tuning["integral_time"]))
        if tuning["derivative_time"] is not None:
            self.derivative_time.set(self._format_number(tuning["derivative_time"]))
        tuned_parameters = [
            f"K={self._format_number(tuning['proportional_gain'])}",
        ]
        if tuning["integral_time"] is not None:
            tuned_parameters.append(f"Ti={self._format_number(tuning['integral_time'])} с")
        if tuning["derivative_time"] is not None:
            tuned_parameters.append(f"Td={self._format_number(tuning['derivative_time'])} с")
        self.tuning_summary.set(
            f"Для T={self._format_number(parsed['Постоянная времени T'])} с, "
            f"L={self._format_number(parsed['Запаздывание L'])} с: "
            f"λ={self._format_number(tuning['closed_loop_time'])} с, "
            f"{', '.join(tuned_parameters)}."
        )
        self._show_page("controller")
        self._set_status(f"Настройки {controller_type}-регулятора подобраны — выполните расчёт")

    def _draw_control_diagram(self):
        if not hasattr(self, "control_diagram"):
            return

        canvas = self.control_diagram
        canvas.delete("all")
        active_fill = "#DBEAFE"
        inactive_fill = "#F3F4F6"
        component_symbol = "Xг" if self.chain == LEAN_GAS else "Xа"
        flow_symbol = "Gг" if self.chain == LEAN_GAS else "Gа"
        output_symbol = "Xог" if self.chain == LEAN_GAS else "Xна"

        def disturbance_box(y, symbol, enabled, value):
            fill = active_fill if enabled else inactive_fill
            outline = ACCENT if enabled else BORDER
            canvas.create_rectangle(8, y, 112, y + 40, fill=fill, outline=outline, width=2)
            canvas.create_text(
                60,
                y + 20,
                text=self._diagram_disturbance_text(symbol, enabled, value),
                fill=ACCENT_ACTIVE if enabled else MUTED,
                font=("Segoe UI", 9, "bold" if enabled else "normal"),
                width=94,
            )

        disturbance_box(10, component_symbol, self.component_enabled.get(), self.component_value.get())
        disturbance_box(106, flow_symbol, self.flow_enabled.get(), self.flow_value.get())
        canvas.create_line(112, 30, 148, 66, fill=MUTED, width=2, arrow=tk.LAST)
        canvas.create_line(112, 126, 148, 90, fill=MUTED, width=2, arrow=tk.LAST)
        canvas.create_rectangle(148, 48, 282, 106, fill="#EAF2FF", outline=ACCENT, width=2)
        canvas.create_text(215, 77, text="Объект управления", fill=TEXT, font=("Segoe UI", 10, "bold"), width=120)
        canvas.create_line(215, 106, 215, 134, fill=MUTED, width=2, arrow=tk.LAST)
        canvas.create_rectangle(148, 136, 282, 174, fill="#ECFDF3", outline=SUCCESS, width=2)
        canvas.create_text(215, 155, text=f"Выходная концентрация {output_symbol}", fill=TEXT, font=("Segoe UI", 9, "bold"), width=124)

    @staticmethod
    def _diagram_disturbance_text(symbol, enabled, value):
        if not enabled:
            return f"{symbol}  выключено"
        try:
            return f"{symbol}  {float(value) * 100:+.1f}%"
        except ValueError:
            return f"{symbol}  активно"

    def _update_input_states(self):
        self._set_entry_state(self.component_entry, self.component_enabled.get(), self.component_value)
        self._set_entry_state(self.flow_entry, self.flow_enabled.get(), self.flow_value)
        enabled = self.component_enabled.get() or self.flow_enabled.get()
        self.calculate_button.configure(state="normal" if enabled else "disabled")
        self._clear_errors()
        self._draw_control_diagram()

    def _update_dynamics_summary(self, *_):
        self.dynamics_summary.set(
            f"{self.disturbance_type.get()} · "
            f"t₀={self.start_time.get() or '—'} с · "
            f"T={self.time_constant.get() or '—'} с · "
            f"L={self.delay.get() or '—'} с"
        )

    def _update_disturbance_type(self, _event=None):
        state = "disabled" if DISTURBANCE_TYPES[self.disturbance_type.get()] == STEP else "normal"
        self.effect_duration_entry.configure(state=state)
        self._update_dynamics_summary()

    @staticmethod
    def _set_entry_state(entry, enabled, value):
        if enabled:
            entry.configure(state="normal")
        else:
            value.set("")
            entry.configure(state="disabled")

    def _clear_errors(self):
        self.component_error.set("")
        self.flow_error.set("")
        self.dynamics_error.set("")
        self.controller_error.set("")
        self.component_entry.configure(style="TEntry")
        self.flow_entry.configure(style="TEntry")
        for entry in (
            self.start_time_entry,
            self.simulation_duration_entry,
            self.effect_duration_entry,
            self.time_constant_entry,
            self.delay_entry,
        ):
            entry.configure(style="TEntry")
        for entry in self.controller_entries:
            entry.configure(style="TEntry")

    def _read_fraction(self, enabled, value, error_variable, entry):
        if not enabled:
            return 0.0
        try:
            return parse_fraction(value.get())
        except ValueError as error:
            error_variable.set(str(error))
            entry.configure(style="Error.TEntry")
            self._show_page("disturbances")
            raise

    def _read_dynamic_parameters(self):
        fields = (
            ("Начало воздействия", self.start_time, self.start_time_entry, parse_nonnegative_number),
            ("Длительность моделирования", self.simulation_duration, self.simulation_duration_entry, parse_positive_number),
            ("Постоянная времени T", self.time_constant, self.time_constant_entry, parse_positive_number),
            ("Запаздывание L", self.delay, self.delay_entry, parse_nonnegative_number),
        )
        parsed = {}
        for label, variable, entry, parser in fields:
            try:
                parsed[label] = parser(variable.get())
            except ValueError as error:
                entry.configure(style="Error.TEntry")
                self.dynamics_error.set(f"{label}: {error}")
                self._show_page("dynamics")
                raise

        kind = DISTURBANCE_TYPES[self.disturbance_type.get()]
        if kind == STEP:
            effect_duration = 1.0
        else:
            try:
                effect_duration = parse_positive_number(self.effect_duration.get())
            except ValueError as error:
                self.effect_duration_entry.configure(style="Error.TEntry")
                self.dynamics_error.set(f"Длительность воздействия: {error}")
                self._show_page("dynamics")
                raise

        start_time = parsed["Начало воздействия"]
        simulation_duration = parsed["Длительность моделирования"]
        if start_time >= simulation_duration:
            self.start_time_entry.configure(style="Error.TEntry")
            self.simulation_duration_entry.configure(style="Error.TEntry")
            self.dynamics_error.set("Начало воздействия должно быть раньше окончания моделирования.")
            self._show_page("dynamics")
            raise ValueError(self.dynamics_error.get())

        return {
            "kind": kind,
            "start_time": start_time,
            "simulation_duration": simulation_duration,
            "effect_duration": effect_duration,
            "time_constant": parsed["Постоянная времени T"],
            "delay": parsed["Запаздывание L"],
        }

    def _read_controller_parameters(self):
        if not self.controller_enabled.get():
            return None

        controller_type = self.controller_type.get()
        fields = [
            ("Коэффициент K", self.proportional_gain, self.proportional_gain_entry, parse_nonnegative_number),
            ("Ограничение |u|", self.control_limit, self.control_limit_entry, parse_positive_number),
            ("Заданное значение", self.setpoint, self.setpoint_entry, parse_positive_number),
        ]
        if "I" in controller_type:
            fields.append(("Время интегрирования Ti", self.integral_time, self.integral_time_entry, parse_positive_number))
        if "D" in controller_type:
            fields.append(("Время дифференцирования Td", self.derivative_time, self.derivative_time_entry, parse_nonnegative_number))
        parsed = {}
        for label, variable, entry, parser in fields:
            try:
                parsed[label] = parser(variable.get())
            except ValueError as error:
                entry.configure(style="Error.TEntry")
                self.controller_error.set(f"{label}: {error}")
                self._show_page("controller")
                raise

        return {
            "controller_type": controller_type,
            "controller_gain": parsed["Коэффициент K"],
            "integral_time": parsed.get("Время интегрирования Ti", 1.0),
            "derivative_time": parsed.get("Время дифференцирования Td", 0.0),
            "control_limit": parsed["Ограничение |u|"],
            "setpoint": parsed["Заданное значение"],
        }

    def _calculate(self):
        self._clear_errors()
        try:
            component_fraction = self._read_fraction(
                self.component_enabled.get(), self.component_value, self.component_error, self.component_entry
            )
            flow_fraction = self._read_fraction(
                self.flow_enabled.get(), self.flow_value, self.flow_error, self.flow_entry
            )
            dynamics = self._read_dynamic_parameters()
            controller = self._read_controller_parameters()
        except ValueError:
            self._set_status("Исправьте значение", error=True)
            return

        combined_fraction = combine_fractions(component_fraction, flow_fraction)
        if self.chain == LEAN_GAS:
            baseline = self.model_values["xog_initial"]
            gog = (
                self.model_values["gg"]
                * self.model_values["xg"]
                / self.model_values["xog_initial"]
            )
            calculator = lambda component, flow: calculate_xog(
                self.model_values["gg"], self.model_values["xg"], gog, component, flow
            )
        else:
            baseline = self.model_values["xna_initial"]
            ga = (
                self.model_values["gna"]
                * self.model_values["xna_initial"]
                / self.model_values["xa"]
            )
            calculator = lambda component, flow: calculate_xna(
                ga, self.model_values["gna"], self.model_values["xa"], component, flow
            )

        calculated = calculator(component_fraction, flow_fraction)
        time = np.linspace(0.0, dynamics["simulation_duration"], 501)
        profile = disturbance_profile(
            time,
            dynamics["kind"],
            dynamics["start_time"],
            dynamics["effect_duration"],
        )
        targets = {
            "Исходный режим": np.full_like(time, baseline),
            "Только состав": calculator(component_fraction * profile, 0.0),
            "Только расход": calculator(0.0, flow_fraction * profile),
            "Совместное воздействие": calculator(
                component_fraction * profile,
                flow_fraction * profile,
            ),
        }
        responses = {
            label: first_order_response(
                time,
                baseline,
                target,
                dynamics["time_constant"],
                dynamics["delay"],
            )
            for label, target in targets.items()
        }

        self.baseline_result.set(self._format_number(baseline))
        self.disturbance_result.set(f"{self._format_number(combined_fraction)} ({combined_fraction * 100:+.1f}%)")
        self.calculated_result.set(self._format_number(calculated))
        self._update_calculation_steps(
            component_fraction,
            flow_fraction,
            combined_fraction,
            baseline,
            calculated,
        )
        if controller is None:
            final_response = responses["Совместное воздействие"]
            metrics = transition_metrics(
                time,
                final_response,
                targets["Совместное воздействие"],
                baseline,
            )
            self.result_mode.set("Без регулятора")
            metric_response_start = dynamics["start_time"] + dynamics["delay"]
            self._draw_disturbance(time, profile, component_fraction, flow_fraction)
            self._draw_response(time, responses)
        else:
            controlled_response, error, control = controller_response(
                time,
                baseline,
                targets["Совместное воздействие"],
                dynamics["time_constant"],
                controller["controller_type"],
                controller["controller_gain"],
                controller["integral_time"],
                controller["derivative_time"],
                controller["control_limit"],
                controller["setpoint"],
                dynamics["delay"],
            )
            final_response = controlled_response
            steady_state = controller_steady_state(
                targets["Совместное воздействие"][-1],
                controller["setpoint"],
                controller["controller_type"],
                controller["controller_gain"],
                controller["control_limit"],
            )
            setpoint_target = np.full_like(time, steady_state)
            metrics = transition_metrics(
                time,
                controlled_response,
                setpoint_target,
                baseline,
            )
            metrics["static_error"] = controller["setpoint"] - steady_state
            self.result_mode.set(f"С {controller['controller_type']}-регулятором")
            metric_response_start = (
                dynamics["delay"]
                if controller["setpoint"] != baseline
                else dynamics["start_time"] + dynamics["delay"]
            )
            self._draw_controller_signals(time, error, control)
            self._draw_control_comparison(
                time,
                responses["Совместное воздействие"],
                controlled_response,
                controller["setpoint"],
                controller["controller_type"],
            )

        self.final_result.set(self._format_number(final_response[-1]))
        self._update_transition_metrics(metrics, dynamics, metric_response_start)
        self._show_page("results")
        self._set_status("Расчёт выполнен")

    def _update_calculation_steps(
        self,
        component_fraction,
        flow_fraction,
        combined_fraction,
        baseline,
        calculated,
    ):
        component_symbol = "Xг" if self.chain == LEAN_GAS else "Xа"
        flow_symbol = "Gг" if self.chain == LEAN_GAS else "Gа"
        self.calculation_steps.set(
            f"Возмущение состава {component_symbol}: {component_fraction * 100:+.1f}%\n"
            f"Возмущение расхода {flow_symbol}: {flow_fraction * 100:+.1f}%\n\n"
            "Совместная доля:\n"
            f"({self._factor_expression(component_fraction)}) · "
            f"({self._factor_expression(flow_fraction)}) − 1 = "
            f"{self._format_signed_number(combined_fraction)}\n\n"
            "Новое значение:\n"
            f"{self._format_number(baseline)} · "
            f"{self._format_number(1 + combined_fraction)} = "
            f"{self._format_number(calculated)}"
        )

    def _update_transition_metrics(self, metrics, dynamics, response_start):
        settling_time = metrics["settling_time"]
        if settling_time is None:
            settling_text = "не достигнуто"
            settling_moment_text = "не достигнут"
        else:
            settling_text = f"{max(0.0, settling_time - response_start):.1f} с"
            settling_moment_text = f"t = {settling_time:.1f} с"

        relative_deviation = metrics["relative_deviation"]
        self.transition_values["initial"].set(self._format_number(metrics["initial_value"]))
        self.transition_values["steady"].set(self._format_number(metrics["steady_state"]))
        self.transition_values["maximum_deviation"].set(self._format_number(metrics["maximum_deviation"]))
        self.transition_values["relative_deviation"].set(
            f"{relative_deviation:.2f}%" if relative_deviation is not None else "не определено"
        )
        self.transition_values["time_constant"].set(f"{self._format_number(dynamics['time_constant'])} с")
        self.transition_values["settling_time"].set(settling_text)
        self.transition_values["settling_moment"].set(settling_moment_text)
        self.transition_values["static_error"].set(self._format_signed_number(metrics["static_error"]))

    def _reset(self):
        self.component_enabled.set(False)
        self.flow_enabled.set(False)
        self.component_value.set("")
        self.flow_value.set("")
        self.component_entry.configure(state="disabled")
        self.flow_entry.configure(state="disabled")
        self.calculate_button.configure(state="disabled")
        self._clear_errors()
        self._clear_result_values()
        self._draw_control_diagram()
        self._draw_static_charts()
        self._set_status("Готово к расчёту")

    def _clear_result_values(self):
        self.baseline_result.set("—")
        self.disturbance_result.set("—")
        self.calculated_result.set("—")
        self.final_result.set("—")
        self.result_mode.set(
            f"С {self.controller_type.get()}-регулятором"
            if self.controller_enabled.get()
            else "Без регулятора"
        )
        self.calculation_steps.set("Выполните расчёт, чтобы увидеть происхождение результата.")
        for variable in self.transition_values.values():
            variable.set("—")

    def _draw_static_charts(self):
        baseline = self._current_baseline()
        simulation_duration = self._safe_simulation_duration()
        time = np.linspace(0.0, simulation_duration, 501)
        self._draw_disturbance(time, np.zeros_like(time), 0.0, 0.0)
        self.response_chart_title.set("Кривая разгона")
        self._style_axis(self.response_axis, "Время", "Концентрация")
        self.response_axis.plot(
            [0, simulation_duration],
            [baseline, baseline],
            color=CURVE_STYLES["Исходный режим"][0],
            linestyle=CURVE_STYLES["Исходный режим"][1],
            linewidth=2,
            label="Исходный режим",
        )
        self.response_axis.legend(loc="best", frameon=False, fontsize=8)
        self.response_canvas.draw_idle()

    def _draw_disturbance(self, time, profile, component_fraction, flow_fraction):
        self._remove_controller_axis()
        self.primary_chart_title.set("Возмущающее воздействие")
        self.primary_chart_subtitle.set("Изменение относительно базового уровня")
        component_signal = component_fraction * profile
        flow_signal = flow_fraction * profile
        combined_signal = (1 + component_signal) * (1 + flow_signal) - 1

        self._style_axis(self.disturbance_axis, "Время, с", "Относительное изменение")
        self.disturbance_axis.axhline(0.0, color=CURVE_STYLES["Исходный режим"][0], linestyle="--", linewidth=1.5, label="Исходный режим")
        self.disturbance_axis.plot(time, component_signal, color=CURVE_STYLES["Только состав"][0], linewidth=2, label="Состав")
        self.disturbance_axis.plot(time, flow_signal, color=CURVE_STYLES["Только расход"][0], linewidth=2, label="Расход")
        self.disturbance_axis.plot(time, combined_signal, color=CURVE_STYLES["Совместное воздействие"][0], linewidth=2.4, label="Совместно")
        self.disturbance_axis.margins(x=0.02, y=0.15)
        self.disturbance_axis.legend(loc="best", frameon=False, fontsize=8, ncol=2)
        self.disturbance_canvas.draw_idle()

    def _draw_response(self, time, responses):
        self.response_chart_title.set("Кривая разгона")
        self._style_axis(self.response_axis, "Время, с", "Концентрация")
        for label, response in responses.items():
            color, linestyle = CURVE_STYLES[label]
            self.response_axis.plot(
                time,
                response,
                color=color,
                linestyle=linestyle,
                linewidth=2.4 if label == "Совместное воздействие" else 1.8,
                label=label,
            )
        self.response_axis.margins(x=0.02, y=0.12)
        self.response_axis.legend(loc="best", frameon=False, fontsize=8, ncol=2)
        self.response_canvas.draw_idle()

    def _draw_controller_signals(self, time, error, control):
        self._remove_controller_axis()
        self.primary_chart_title.set("Ошибка и управляющее воздействие")
        self.primary_chart_subtitle.set("Сигналы замкнутой системы")
        self._style_axis(self.disturbance_axis, "Время, с", "Ошибка e(t)")
        error_line = self.disturbance_axis.plot(
            time,
            error,
            color="#F59E0B",
            linewidth=2,
            label="Ошибка e(t)",
        )[0]
        self.disturbance_axis.axhline(0.0, color=MUTED, linestyle="--", linewidth=1)

        self.controller_signal_axis = self.disturbance_axis.twinx()
        control_line = self.controller_signal_axis.plot(
            time,
            control,
            color="#16A34A",
            linewidth=2,
            label="Воздействие u(t)",
        )[0]
        self.controller_signal_axis.set_ylabel("Управляющее воздействие u(t)", color=TEXT)
        self.controller_signal_axis.tick_params(colors=TEXT)
        self.controller_signal_axis.spines["top"].set_visible(False)
        self.controller_signal_axis.spines["right"].set_color(BORDER)
        self.disturbance_axis.margins(x=0.02, y=0.15)
        self.disturbance_axis.legend(
            [error_line, control_line],
            [error_line.get_label(), control_line.get_label()],
            loc="best",
            frameon=False,
            fontsize=8,
        )
        self.disturbance_canvas.draw_idle()

    def _draw_control_comparison(
        self,
        time,
        open_response,
        controlled_response,
        setpoint,
        controller_type,
    ):
        self.response_chart_title.set(f"Без регулятора / {controller_type}-регулятор")
        self._style_axis(self.response_axis, "Время, с", "Концентрация")
        self.response_axis.axhline(
            setpoint,
            color=MUTED,
            linestyle="--",
            linewidth=1.5,
            label="Задание",
        )
        self.response_axis.plot(
            time,
            open_response,
            color="#F59E0B",
            linewidth=2,
            label="Без регулятора",
        )
        self.response_axis.plot(
            time,
            controlled_response,
            color=ACCENT,
            linewidth=2.4,
            label=f"{controller_type}-регулятор",
        )
        self.response_axis.margins(x=0.02, y=0.12)
        self.response_axis.legend(loc="best", frameon=False, fontsize=8, ncol=3)
        self.response_canvas.draw_idle()

    def _remove_controller_axis(self):
        if self.controller_signal_axis is not None:
            self.controller_signal_axis.remove()
            self.controller_signal_axis = None

    def _safe_simulation_duration(self):
        try:
            return parse_positive_number(self.simulation_duration.get())
        except ValueError:
            return 100.0

    @staticmethod
    def _style_axis(axis, x_label, y_label):
        axis.clear()
        axis.set_facecolor(CARD_BACKGROUND)
        axis.set_xlabel(x_label, color=TEXT)
        axis.set_ylabel(y_label, color=TEXT)
        axis.grid(True, color="#DDE2E8", linewidth=0.8)
        axis.tick_params(colors=TEXT)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(BORDER)
        axis.spines["bottom"].set_color(BORDER)

    def _set_status(self, message, error=False):
        self.status_text.set(message)
        self.status_dot.configure(foreground=ERROR if error else SUCCESS)

    @staticmethod
    def _format_number(value):
        return f"{value:.4f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_signed_number(value):
        if abs(value) < 0.00005:
            return "0"
        return f"{value:+.4f}".rstrip("0").rstrip(".").replace("-", "−")

    @staticmethod
    def _factor_expression(fraction):
        sign = "+" if fraction >= 0 else "−"
        return f"1 {sign} {abs(fraction):.2f}"


def main():
    root = tk.Tk()
    AbsorptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
