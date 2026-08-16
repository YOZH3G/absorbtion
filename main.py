import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from app_info import APP_NAME, APP_VERSION
from calculations import (
    CONTROLLER_TYPES,
    IMPULSE,
    RAMP,
    RECTANGLE,
    STEP,
    tune_controller_parameters,
)
from comparison import MAX_COMPARISON_RUNS, build_comparison_run, write_comparison_csv
from exporting import (
    build_protocol,
    save_graphs,
    write_comparison_html_report,
    write_comparison_pdf_report,
    write_csv,
    write_html_report,
    write_pdf_report,
    write_protocol,
)
from laboratory import (
    CORRECTION_OPTIONS,
    DIRECTION_OPTIONS,
    FASTEST_OPTIONS,
    evaluate_prediction,
)
from model_dialog import ModelParametersDialog
from scenario_editor import ScenarioEditorDialog
from scenario_store import ScenarioStore
from session_store import read_session, write_session
from settings_store import SettingsStore
from simulation import LEAN_GAS, RICH_ABSORBENT, run_simulation
from ui_helpers import (
    DisturbanceTooltip,
    FormulaPanel,
    ScrollablePage,
    TextTooltip,
    create_icon,
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
        self.scenario_editor = None
        self.scenario_store = ScenarioStore()
        self.settings_store = SettingsStore(
            self.scenario_store.path.with_name("settings.json")
        )
        self.settings = self.settings_store.load()
        self.sidebar_collapsed = self.settings["sidebar_collapsed"]
        self.scenarios = self.scenario_store.scenarios
        self.scenarios_by_name = {scenario["name"]: scenario for scenario in self.scenarios}
        self.pages = {}
        self.nav_buttons = {}

        self.component_enabled = tk.BooleanVar(value=False)
        self.flow_enabled = tk.BooleanVar(value=False)
        self.component_value = tk.StringVar()
        self.flow_value = tk.StringVar()
        self.disturbance_type = tk.StringVar(value="Ступенчатое")
        self.dynamics_summary = tk.StringVar()
        self.page_title = tk.StringVar(value="Возмущения")
        self.topbar_context = tk.StringVar(value="Свободный расчёт · без регулятора")
        self.calculate_button_text = tk.StringVar(value="Рассчитать")
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
        self.selected_scenario = tk.StringVar(value=self.scenarios[0]["name"])
        self.scenario_description = tk.StringVar(value=self.scenarios[0]["description"])
        self.active_scenario = tk.StringVar(value="Сценарий ещё не применён.")
        self.teacher_mode = tk.BooleanVar(value=False)
        self.scenario_storage_status = tk.StringVar(
            value=self._scenario_storage_text()
        )
        self.assignment_tolerance_percent = 5.0
        self.current_lesson = self.scenarios[0]["lesson"]
        self.assignment_attempts = 0
        self.assignment_evaluation = None
        self.learning_mode = tk.BooleanVar(value=False)
        self.learning_step = 1
        self.lesson_summary = tk.StringVar()
        self.learning_route = tk.StringVar()
        self.student_name = tk.StringVar()
        self.student_conclusion = tk.StringVar()
        self.assignment_enabled = tk.BooleanVar(value=False)
        self.predicted_direction = tk.StringVar()
        self.predicted_steady = tk.StringVar()
        self.predicted_fastest = tk.StringVar()
        self.predicted_correction = tk.StringVar()
        self.assignment_feedback = tk.StringVar(
            value="Включите режим задания и заполните прогноз до расчёта."
        )
        self.export_summary = tk.StringVar(value="Сначала выполните расчёт.")
        self.export_buttons = []
        self.last_calculation = None
        self.last_prediction = None
        self.controller_signal_axis = None
        self.comparison_runs = []
        self.comparison_counter = 0
        self.comparison_summary = tk.StringVar(
            value="Закрепите результаты нескольких расчётов для сравнения."
        )
        self.current_page = self.settings["last_page"]
        self._charts_show_comparison = False
        self.text_tooltips = []

        self._configure_window()
        self._configure_styles()
        self._build_layout()
        self._update_lesson_summary()
        self._update_learning_route()
        self._select_chain(LEAN_GAS)
        if self.scenario_store.warning:
            self._set_status(self.scenario_store.warning, error=True)

    def _configure_window(self):
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry(self.settings["geometry"])
        self.root.minsize(1100, 680)
        self.root.configure(background=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self._close_application)
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
        style.configure("Dirty.TEntry", fieldbackground="#FFFAEB", bordercolor="#F79009")
        style.configure("Dirty.TCombobox", fieldbackground="#FFFAEB", bordercolor="#F79009")
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
        style.configure("SidebarToggle.TButton", background=SIDEBAR, foreground="#E5ECF5", borderwidth=0, padding=(12, 10), font=("Segoe UI", 10), anchor="center")
        style.map("SidebarToggle.TButton", background=[("active", "#193553")])
        style.configure("SelectedSidebarNav.TButton", background=ACCENT, foreground="#FFFFFF", borderwidth=0, padding=(18, 13), font=("Segoe UI", 10, "bold"), anchor="w")
        style.map("SelectedSidebarNav.TButton", background=[("active", ACCENT_ACTIVE)])
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
        ttk.Label(topbar, text=APP_NAME, style="TopbarTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(topbar, text="Учебный стенд АТПП", style="TopbarMeta.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Label(
            topbar,
            textvariable=self.topbar_context,
            style="TopbarMeta.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))

        self.body = ttk.Frame(self, style="App.TFrame")
        self.body.grid(row=1, column=0, sticky="nsew")
        self.body.columnconfigure(0, minsize=220)
        self.body.columnconfigure(1, minsize=390)
        self.body.columnconfigure(2, weight=1)
        self.body.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(self.body, style="Sidebar.TFrame", padding=(12, 20))
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.columnconfigure(0, weight=1)
        self.sidebar_title = ttk.Label(
            self.sidebar,
            text="АБСОРБЦИЯ",
            style="SidebarTitle.TLabel",
        )
        self.sidebar_title.grid(row=0, column=0, sticky="w", padx=8, pady=(0, 4))
        self.sidebar_meta = ttk.Label(
            self.sidebar,
            text="Моделирование контуров",
            style="SidebarMeta.TLabel",
        )
        self.sidebar_meta.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 24))

        active_navigation = (
            ("disturbances", "Возмущения"),
            ("dynamics", "Динамика"),
            ("results", "Результаты"),
            ("comparison", "Сравнение"),
            ("controller", "Регулятор"),
            ("scenarios", "Сценарии"),
            ("export", "Экспорт"),
        )
        self.nav_icons = {
            key: create_icon(self.root, key)
            for key, _label in active_navigation
        }
        self.nav_labels = dict(active_navigation)
        for row, (key, label) in enumerate(active_navigation, start=2):
            button = ttk.Button(
                self.sidebar,
                text=label,
                image=self.nav_icons[key],
                compound="left",
                command=lambda page=key: self._show_page(page),
                style="SidebarNav.TButton",
            )
            button.grid(row=row, column=0, sticky="ew", pady=2)
            self.nav_buttons[key] = button

        separator_row = 2 + len(active_navigation)
        ttk.Separator(self.sidebar).grid(
            row=separator_row,
            column=0,
            sticky="ew",
            padx=8,
            pady=16,
        )
        self.sidebar_toggle = ttk.Button(
            self.sidebar,
            command=self._toggle_sidebar,
            style="SidebarToggle.TButton",
        )
        self.sidebar_toggle.grid(row=separator_row + 1, column=0, sticky="ew", pady=2)

        inspector = ttk.Frame(self.body, style="App.TFrame", padding=(16, 16, 12, 12))
        inspector.grid(row=0, column=1, sticky="nsew")
        inspector.columnconfigure(0, weight=1)
        inspector.rowconfigure(1, weight=1)
        ttk.Label(inspector, textvariable=self.page_title, style="SectionHeader.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 12))

        page_host = ttk.Frame(inspector, style="App.TFrame")
        page_host.grid(row=1, column=0, sticky="nsew")
        page_host.columnconfigure(0, weight=1)
        page_host.rowconfigure(0, weight=1)
        self.page_contents = {}
        for key in (
            "disturbances",
            "dynamics",
            "results",
            "comparison",
            "controller",
            "scenarios",
            "export",
        ):
            page = ScrollablePage(page_host, BACKGROUND)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page
            self.page_contents[key] = page.content

        self._build_chain_card(self.page_contents["disturbances"])
        self._build_disturbance_card(self.page_contents["disturbances"])
        self._build_control_diagram(self.page_contents["disturbances"])
        self._build_dynamics_card(self.page_contents["dynamics"])
        self._build_result_card(self.page_contents["results"])
        self._build_comparison_card(self.page_contents["comparison"])
        self._build_controller_card(self.page_contents["controller"])
        self._build_scenarios_card(self.page_contents["scenarios"])
        self._build_export_card(self.page_contents["export"])
        for page in self.pages.values():
            page.bind_mousewheel()

        actions = ttk.Frame(inspector, style="App.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure((0, 1), weight=1)
        self.calculate_button = ttk.Button(
            actions,
            textvariable=self.calculate_button_text,
            command=self._calculate,
            style="Primary.TButton",
            state="disabled",
        )
        self.calculate_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(actions, text="Сбросить", command=self._reset, style="Secondary.TButton").grid(row=0, column=1, sticky="ew", padx=(5, 0))

        workspace = ttk.Frame(self.body, style="App.TFrame", padding=(4, 16, 16, 12))
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
        ttk.Label(status, text=f"v{APP_VERSION} · Python 3.14", style="Status.TLabel").grid(row=0, column=2, sticky="e")
        self._apply_sidebar_state()
        initial_page = self.current_page if self.current_page in self.pages else "disturbances"
        self._show_page(initial_page)

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
            "comparison": "Сравнение опытов",
            "controller": "Регулятор",
            "scenarios": "Лабораторные сценарии",
            "export": "Экспорт результатов",
        }
        previous_page = self.current_page
        self.current_page = page
        self.pages[page].tkraise()
        self.pages[page].scroll_to_top()
        self.page_title.set(titles[page])
        for key, button in self.nav_buttons.items():
            button.configure(style="SelectedSidebarNav.TButton" if key == page else "SidebarNav.TButton")
        if page == "comparison":
            self._draw_comparison()
        elif previous_page == "comparison" and self._charts_show_comparison:
            self._draw_last_calculation()

    def _toggle_sidebar(self):
        self.sidebar_collapsed = not self.sidebar_collapsed
        self._apply_sidebar_state()

    def _apply_sidebar_state(self):
        if self.sidebar_collapsed:
            self.sidebar_title.grid_remove()
            self.sidebar_meta.grid_remove()
            self.sidebar.configure(padding=(6, 20))
            self.body.columnconfigure(0, minsize=64)
        else:
            self.sidebar_title.grid()
            self.sidebar_meta.grid()
            self.sidebar.configure(padding=(12, 20))
            self.body.columnconfigure(0, minsize=220)
        for key, button in self.nav_buttons.items():
            button.configure(
                text="" if self.sidebar_collapsed else self.nav_labels[key],
                compound="center" if self.sidebar_collapsed else "left",
                width=3 if self.sidebar_collapsed else 18,
            )
        self.sidebar_toggle.configure(
            text="»" if self.sidebar_collapsed else "«  Свернуть",
        )

    def _card(self, parent, row, padding=(18, 16)):
        card = ttk.Frame(parent, style="Card.TFrame", padding=padding)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 14))
        card.columnconfigure(0, weight=1)
        return card

    def _attach_tooltip(self, widget, text):
        self.text_tooltips.append(TextTooltip(widget, text))

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

        def apply_values(values):
            self.model_values = values
            self.setpoint.set(self._format_number(self._current_baseline()))
            self._reset()
            self._set_status("Параметры модели обновлены")

        self.model_dialog = ModelParametersDialog(
            self.root,
            self.model_values,
            DEFAULT_MODEL_VALUES,
            apply_values,
            lambda: setattr(self, "model_dialog", None),
            self._format_number,
            BACKGROUND,
        )

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
        self.disturbance_tooltip = DisturbanceTooltip(
            self.disturbance_type_box,
            self.disturbance_type.get,
        )
        self.start_time_entry = ttk.Entry(card, textvariable=self.start_time)
        self.start_time_entry.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=(3, 12))

        ttk.Label(card, text="Длительность моделирования, с", style="Body.TLabel", wraplength=170).grid(row=3, column=0, sticky="w")
        ttk.Label(card, text="Длительность воздействия, с", style="Body.TLabel", wraplength=170).grid(row=3, column=1, sticky="w", padx=(10, 0))
        self.simulation_duration_entry = ttk.Entry(card, textvariable=self.simulation_duration)
        self.simulation_duration_entry.grid(row=4, column=0, sticky="ew", pady=(3, 12))
        self.effect_duration_entry = ttk.Entry(card, textvariable=self.effect_duration)
        self.effect_duration_entry.grid(row=4, column=1, sticky="ew", padx=(10, 0), pady=(3, 12))

        time_constant_label = ttk.Label(
            card,
            text="Постоянная времени T, с",
            style="Body.TLabel",
        )
        time_constant_label.grid(row=5, column=0, sticky="w")
        delay_label = ttk.Label(card, text="Запаздывание L, с", style="Body.TLabel")
        delay_label.grid(row=5, column=1, sticky="w", padx=(10, 0))
        self._attach_tooltip(
            time_constant_label,
            "T характеризует инерционность объекта: примерно за T секунд отклик проходит 63% изменения.",
        )
        self._attach_tooltip(
            delay_label,
            "L — чистое запаздывание между воздействием и началом реакции объекта.",
        )
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
        controller_help = {
            "Коэффициент регулятора K": "K определяет силу реакции регулятора на текущую ошибку.",
            "Время интегрирования Ti, с": "Ti задаёт скорость накопления интегральной составляющей: меньше Ti — сильнее интегральное действие.",
            "Время дифференцирования Td, с": "Td определяет влияние скорости изменения выхода; большое Td повышает чувствительность к шуму.",
            "Ограничение |u|": "Максимальный модуль управляющего воздействия, доступный исполнительному механизму.",
            "Заданное значение": "Значение выхода, к которому регулятор должен вернуть объект.",
        }
        for row, (label, variable, attribute) in enumerate(fields, start=2):
            field_label = ttk.Label(parameters, text=label, style="Body.TLabel")
            field_label.grid(row=row, column=0, sticky="w", pady=3)
            self._attach_tooltip(field_label, controller_help[label])
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
        self.formula_panel = FormulaPanel(explanation, CARD_BACKGROUND)
        self.formula_panel.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            explanation,
            textvariable=self.controller_formula,
            style="Body.TLabel",
            justify="left",
            wraplength=330,
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))
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
        metric_help = {
            "maximum_deviation": "Наибольшее расстояние выхода от начального значения за время моделирования.",
            "relative_deviation": "Максимальное отклонение, выраженное в процентах от начального значения.",
            "time_constant": "Параметр инерционности объекта, заданный перед расчётом.",
            "settling_time": "Время после начала реакции, когда выход окончательно входит в полосу ±5%.",
            "settling_moment": "Абсолютная координата момента установления на оси времени графика.",
            "static_error": "Разность между заданным и установившимся значениями выхода.",
        }
        for row, (label, key) in enumerate(metric_rows, start=1):
            label_options = {"textvariable": label} if isinstance(label, tk.StringVar) else {"text": label}
            metric_label = ttk.Label(metrics, style="Body.TLabel", **label_options)
            metric_label.grid(row=row, column=0, sticky="w", pady=3)
            if key in metric_help:
                self._attach_tooltip(metric_label, metric_help[key])
            ttk.Label(metrics, textvariable=self.transition_values[key], style="ResultValue.TLabel").grid(
                row=row, column=1, sticky="e", pady=3, padx=(8, 0)
            )
        metrics.columnconfigure(0, weight=1)

        comparison_action = self._card(parent, 3, padding=(14, 12))
        ttk.Label(
            comparison_action,
            text="Сравнение опытов",
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(
            comparison_action,
            text="Закрепите текущий расчёт, затем измените параметры и повторите опыт.",
            style="Muted.TLabel",
            wraplength=330,
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.add_comparison_button = ttk.Button(
            comparison_action,
            text="Добавить текущий расчёт",
            command=self._add_current_to_comparison,
            style="Primary.TButton",
            state="disabled",
        )
        self.add_comparison_button.grid(row=2, column=0, sticky="ew")

    def _build_comparison_card(self, parent):
        card = self._card(parent, 0)
        ttk.Label(card, text="Закреплённые опыты", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(
            card,
            textvariable=self.comparison_summary,
            style="Muted.TLabel",
            wraplength=330,
        ).grid(row=1, column=0, sticky="w", pady=(0, 10))

        table_host = ttk.Frame(card, style="CardBody.TFrame")
        table_host.grid(row=2, column=0, sticky="nsew")
        table_host.columnconfigure(0, weight=1)
        table_host.rowconfigure(0, weight=1)
        columns = (
            "name",
            "T",
            "L",
            "controller",
            "deviation",
            "settling",
            "error",
        )
        self.comparison_table = ttk.Treeview(
            table_host,
            columns=columns,
            show="headings",
            selectmode="extended",
            height=10,
        )
        headings = {
            "name": "Опыт",
            "T": "T",
            "L": "L",
            "controller": "Рег.",
            "deviation": "Δmax",
            "settling": "tуст",
            "error": "eуст",
        }
        widths = {
            "name": 190,
            "T": 45,
            "L": 45,
            "controller": 55,
            "deviation": 70,
            "settling": 70,
            "error": 70,
        }
        for column in columns:
            self.comparison_table.heading(column, text=headings[column])
            self.comparison_table.column(
                column,
                width=widths[column],
                minwidth=widths[column],
                anchor="w" if column == "name" else "center",
                stretch=column == "name",
            )
        self.comparison_table.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(
            table_host,
            orient="vertical",
            command=self.comparison_table.yview,
        )
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(
            table_host,
            orient="horizontal",
            command=self.comparison_table.xview,
        )
        horizontal.grid(row=1, column=0, sticky="ew")
        self.comparison_table.configure(
            yscrollcommand=vertical.set,
            xscrollcommand=horizontal.set,
        )
        self.comparison_table.bind("<<TreeviewSelect>>", lambda _event: self._draw_comparison())

        actions = ttk.Frame(card, style="CardBody.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        actions.columnconfigure((0, 1), weight=1)
        ttk.Button(
            actions,
            text="Показать все",
            command=self._select_all_comparison_runs,
            style="Secondary.TButton",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(
            actions,
            text="Удалить выбранные",
            command=self._remove_comparison_runs,
            style="Secondary.TButton",
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Button(
            actions,
            text="Очистить",
            command=self._clear_comparison_runs,
            style="Secondary.TButton",
        ).grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(
            actions,
            text="Экспорт CSV",
            command=self._export_comparison_csv,
            style="Primary.TButton",
        ).grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(8, 0))
        ttk.Button(
            actions,
            text="Вернуть параметры",
            command=self._restore_selected_comparison_run,
            style="Secondary.TButton",
        ).grid(row=2, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(
            actions,
            text="Переименовать",
            command=self._rename_comparison_run,
            style="Secondary.TButton",
        ).grid(row=2, column=1, sticky="ew", padx=(4, 0), pady=(8, 0))
        ttk.Button(
            actions,
            text="Экспорт PNG",
            command=self._export_comparison_graphs,
            style="Secondary.TButton",
        ).grid(row=3, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(
            actions,
            text="Сохранить сеанс",
            command=self._save_comparison_session,
            style="Secondary.TButton",
        ).grid(row=3, column=1, sticky="ew", padx=(4, 0), pady=(8, 0))
        ttk.Button(
            actions,
            text="Открыть сеанс",
            command=self._open_comparison_session,
            style="Secondary.TButton",
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(
            actions,
            text="Отчёт HTML (все)",
            command=self._save_comparison_html_report,
            style="Secondary.TButton",
        ).grid(row=5, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(
            actions,
            text="Отчёт PDF (все)",
            command=self._save_comparison_pdf_report,
            style="Secondary.TButton",
        ).grid(row=5, column=1, sticky="ew", padx=(4, 0), pady=(8, 0))

        note = self._card(parent, 1)
        ttk.Label(note, text="Как сравнивать", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Label(
            note,
            text=(
                "Выберите один или несколько опытов в таблице. Верхний график показывает "
                "переходные процессы, нижний — длительность установления. Максимум — шесть опытов."
            ),
            style="Body.TLabel",
            wraplength=330,
            justify="left",
        ).grid(row=1, column=0, sticky="w")

    def _build_scenarios_card(self, parent):
        scenario_card = self._card(parent, 0)
        ttk.Label(scenario_card, text="Готовый сценарий", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        self.scenario_box = ttk.Combobox(
            scenario_card,
            textvariable=self.selected_scenario,
            values=tuple(scenario["name"] for scenario in self.scenarios),
            state="readonly",
        )
        self.scenario_box.grid(row=1, column=0, sticky="ew")
        self.scenario_box.bind("<<ComboboxSelected>>", self._update_scenario_description)
        ttk.Label(
            scenario_card,
            textvariable=self.scenario_description,
            style="Muted.TLabel",
            wraplength=330,
        ).grid(row=2, column=0, sticky="w", pady=(8, 12))
        ttk.Button(
            scenario_card,
            text="Применить сценарий",
            command=self._apply_scenario,
            style="Primary.TButton",
        ).grid(row=3, column=0, sticky="ew")
        ttk.Label(
            scenario_card,
            textvariable=self.active_scenario,
            style="Muted.TLabel",
            wraplength=330,
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))

        ttk.Separator(scenario_card).grid(row=5, column=0, sticky="ew", pady=14)
        ttk.Checkbutton(
            scenario_card,
            text="Режим преподавателя",
            variable=self.teacher_mode,
            command=self._toggle_teacher_mode,
        ).grid(row=6, column=0, sticky="w")
        self.teacher_editor_button = ttk.Button(
            scenario_card,
            text="Открыть редактор сценариев",
            command=self._open_scenario_editor,
            style="Secondary.TButton",
        )
        self.teacher_editor_button.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        self.teacher_storage_label = ttk.Label(
            scenario_card,
            textvariable=self.scenario_storage_status,
            style="Muted.TLabel",
            wraplength=330,
        )
        self.teacher_storage_label.grid(row=8, column=0, sticky="w", pady=(8, 0))
        self.restore_scenarios_button = ttk.Button(
            scenario_card,
            text="Восстановить резервную копию",
            command=self._restore_scenario_backup,
            style="Secondary.TButton",
        )
        self.restore_scenarios_button.grid(row=9, column=0, sticky="ew", pady=(8, 0))
        self.teacher_editor_button.grid_remove()
        self.teacher_storage_label.grid_remove()
        self.restore_scenarios_button.grid_remove()

        lesson_card = self._card(parent, 1, padding=(14, 12))
        ttk.Label(lesson_card, text="Учебное задание", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Label(
            lesson_card, textvariable=self.lesson_summary, style="Body.TLabel",
            justify="left", wraplength=330,
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            lesson_card, text="Пошаговый учебный режим", variable=self.learning_mode,
            command=self._update_learning_route,
        ).grid(row=2, column=0, sticky="w", pady=(10, 4))
        ttk.Label(lesson_card, textvariable=self.learning_route, style="Muted.TLabel", wraplength=330).grid(
            row=3, column=0, sticky="w"
        )
        ttk.Button(
            lesson_card, text="Следующий шаг", command=self._advance_learning_step,
            style="Secondary.TButton",
        ).grid(row=4, column=0, sticky="ew", pady=(8, 0))

        assignment = self._card(parent, 2, padding=(14, 12))
        ttk.Checkbutton(
            assignment,
            text="Режим задания: сначала сделать прогноз",
            variable=self.assignment_enabled,
            command=self._update_assignment_states,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        prediction_fields = (
            ("Направление изменения выхода", self.predicted_direction, DIRECTION_OPTIONS, "direction_prediction"),
            ("Какая реакция завершится быстрее", self.predicted_fastest, FASTEST_OPTIONS, "fastest_prediction"),
            ("Устранит ли регулятор отклонение", self.predicted_correction, CORRECTION_OPTIONS, "correction_prediction"),
        )
        self.prediction_widgets = []
        row = 1
        for label, variable, values, attribute in prediction_fields:
            ttk.Label(assignment, text=label, style="Body.TLabel", wraplength=160).grid(
                row=row, column=0, sticky="w", pady=4
            )
            widget = ttk.Combobox(
                assignment,
                textvariable=variable,
                values=values,
                state="disabled",
                width=20,
            )
            widget.grid(row=row, column=1, sticky="e", padx=(8, 0), pady=4)
            setattr(self, attribute, widget)
            self.prediction_widgets.append(widget)
            row += 1
        ttk.Label(assignment, text="Ожидаемое установившееся значение", style="Body.TLabel", wraplength=170).grid(
            row=row, column=0, sticky="w", pady=4
        )
        self.steady_prediction = ttk.Entry(
            assignment,
            textvariable=self.predicted_steady,
            state="disabled",
            width=22,
        )
        self.steady_prediction.grid(row=row, column=1, sticky="e", padx=(8, 0), pady=4)
        self.prediction_widgets.append(self.steady_prediction)
        row += 1
        ttk.Label(assignment, text="Студент", style="Body.TLabel").grid(
            row=row, column=0, sticky="w", pady=4
        )
        self.student_name_entry = ttk.Entry(assignment, textvariable=self.student_name, state="disabled", width=22)
        self.student_name_entry.grid(row=row, column=1, sticky="e", padx=(8, 0), pady=4)
        self.prediction_widgets.append(self.student_name_entry)
        row += 1
        ttk.Label(assignment, text="Краткий вывод", style="Body.TLabel").grid(
            row=row, column=0, sticky="w", pady=4
        )
        self.student_conclusion_entry = ttk.Entry(assignment, textvariable=self.student_conclusion, state="disabled", width=22)
        self.student_conclusion_entry.grid(row=row, column=1, sticky="e", padx=(8, 0), pady=4)
        self.prediction_widgets.append(self.student_conclusion_entry)
        assignment.columnconfigure(0, weight=1)

        feedback = self._card(parent, 3, padding=(14, 12))
        ttk.Label(feedback, text="Проверка прогноза", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Label(
            feedback,
            textvariable=self.assignment_feedback,
            style="Body.TLabel",
            justify="left",
            wraplength=330,
        ).grid(row=1, column=0, sticky="w")

    def _build_export_card(self, parent):
        card = self._card(parent, 0)
        ttk.Label(card, text="Материалы для отчёта", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Label(
            card,
            textvariable=self.export_summary,
            style="Muted.TLabel",
            wraplength=330,
        ).grid(row=1, column=0, sticky="w", pady=(0, 14))
        actions = (
            ("Сохранить два графика PNG", self._export_graphs_png),
            ("Экспортировать точки CSV", self._export_csv),
            ("Сохранить отчёт HTML", self._save_html_report),
            ("Сохранить отчёт PDF", self._save_pdf_report),
            ("Копировать параметры и результаты", self._copy_protocol),
            ("Сохранить протокол TXT", self._save_protocol),
        )
        for row, (label, command) in enumerate(actions, start=2):
            button = ttk.Button(
                card,
                text=label,
                command=command,
                style="Primary.TButton" if row == 2 else "Secondary.TButton",
                state="disabled",
            )
            button.grid(row=row, column=0, sticky="ew", pady=(0, 8))
            self.export_buttons.append(button)

        note = self._card(parent, 1)
        ttk.Label(note, text="Состав файлов", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        ttk.Label(
            note,
            text=(
                "PNG сохраняет оба текущих графика. CSV содержит временную сетку, воздействие, "
                "целевые значения и рассчитанные отклики. Протокол TXT включает параметры и показатели."
            ),
            style="Body.TLabel",
            wraplength=330,
            justify="left",
        ).grid(row=1, column=0, sticky="w")

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

    def _update_scenario_description(self, _event=None):
        scenario = self.scenarios_by_name[self.selected_scenario.get()]
        self.scenario_description.set(scenario["description"])

    def _apply_scenario(self):
        scenario = self.scenarios_by_name[self.selected_scenario.get()]
        self._apply_scenario_data(scenario)

    def _apply_scenario_data(self, scenario):
        self.current_lesson = scenario["lesson"]
        self.assignment_attempts = 0
        self.assignment_evaluation = None
        self.learning_step = 1
        self._update_lesson_summary()
        self._update_learning_route()
        self._select_chain(scenario["chain"])
        self.component_enabled.set(scenario["component"] is not None)
        self.flow_enabled.set(scenario["flow"] is not None)
        self.component_value.set(
            "" if scenario["component"] is None else self._format_number(scenario["component"])
        )
        self.flow_value.set(
            "" if scenario["flow"] is None else self._format_number(scenario["flow"])
        )
        self.disturbance_type.set(scenario["disturbance_type"])
        self.start_time.set(self._format_number(scenario["start_time"]))
        self.simulation_duration.set(self._format_number(scenario["simulation_duration"]))
        self.effect_duration.set(self._format_number(scenario["effect_duration"]))
        self.time_constant.set(self._format_number(scenario["time_constant"]))
        self.delay.set(self._format_number(scenario["delay"]))
        self._update_disturbance_type()

        controller = scenario["controller"]
        if controller is None:
            self._set_controller_mode(False)
        else:
            self.controller_type.set(controller["type"])
            self._update_controller_type()
            self._set_controller_mode(True)
            self.proportional_gain.set(self._format_number(controller["gain"]))
            self.integral_time.set(self._format_number(controller["integral_time"]))
            self.derivative_time.set(self._format_number(controller["derivative_time"]))
            self.control_limit.set(self._format_number(controller["control_limit"]))
            setpoint = controller.get("setpoint")
            self.setpoint.set(
                self._format_number(self._current_baseline() if setpoint is None else setpoint)
            )

        self.assignment_tolerance_percent = scenario.get("steady_tolerance_percent", 5.0)
        self._update_input_states()
        self._clear_result_values()
        self.predicted_direction.set("")
        self.predicted_steady.set("")
        self.predicted_fastest.set("")
        self.predicted_correction.set("")
        self.assignment_feedback.set("Заполните прогноз и нажмите «Рассчитать».")
        self.active_scenario.set(f"Применён: {scenario['name']}.")
        self.topbar_context.set(
            f"{scenario['name']} · "
            f"{'без регулятора' if controller is None else controller['type'] + '-регулятор'}"
        )
        self._draw_static_charts()
        self._show_page("scenarios")
        self._set_status("Сценарий применён — выполните расчёт")

    def _toggle_teacher_mode(self):
        if self.teacher_mode.get():
            self.teacher_editor_button.grid()
            self.teacher_storage_label.grid()
            if self.scenario_store.recovery_available:
                self.restore_scenarios_button.grid()
            self._set_status("Режим преподавателя включён")
        else:
            if self.scenario_editor is not None and self.scenario_editor.winfo_exists():
                if not self.scenario_editor.request_close():
                    self.teacher_mode.set(True)
                    return
            self.teacher_editor_button.grid_remove()
            self.teacher_storage_label.grid_remove()
            self.restore_scenarios_button.grid_remove()
            self._set_status("Режим преподавателя выключен")

    def _close_application(self):
        if self.scenario_editor is not None and self.scenario_editor.winfo_exists():
            if not self.scenario_editor.request_close():
                return
        try:
            self.settings_store.save({
                "geometry": self.root.geometry(),
                "last_page": self.current_page,
                "sidebar_collapsed": self.sidebar_collapsed,
            })
        except OSError:
            pass
        self.root.destroy()

    def _open_scenario_editor(self):
        if self.scenario_editor is not None and self.scenario_editor.winfo_exists():
            self.scenario_editor.lift()
            self.scenario_editor.focus_force()
            return
        self.scenario_editor = ScenarioEditorDialog(
            self.root,
            self.scenario_store,
            self._refresh_scenarios,
            self._preview_scenario,
            lambda: setattr(self, "scenario_editor", None),
            BACKGROUND,
        )

    def _refresh_scenarios(self, selected_name=None, status=None):
        self.scenarios = self.scenario_store.scenarios
        self.scenarios_by_name = {scenario["name"]: scenario for scenario in self.scenarios}
        names = tuple(self.scenarios_by_name)
        self.scenario_box.configure(values=names)
        if selected_name not in self.scenarios_by_name:
            selected_name = names[0]
        self.selected_scenario.set(selected_name)
        self._update_scenario_description()
        self.scenario_storage_status.set(
            self._scenario_storage_text()
        )
        if not self.scenario_store.recovery_available:
            self.restore_scenarios_button.grid_remove()
        if status:
            self._set_status(status)

    def _restore_scenario_backup(self):
        try:
            count = self.scenario_store.restore_backup()
        except (OSError, ValueError) as error:
            self._set_status(f"Не удалось восстановить сценарии: {error}", error=True)
            return
        self._refresh_scenarios(status=f"Восстановлено пользовательских сценариев: {count}")

    def _scenario_storage_text(self):
        prefix = f"{len(self.scenario_store.user_scenarios)} пользовательских сценариев."
        path_text = f"Файл: {self.scenario_store.path}"
        if self.scenario_store.warning:
            return f"{self.scenario_store.warning}\n{path_text}"
        return f"{prefix}\n{path_text}"

    def _preview_scenario(self, scenario):
        if scenario["name"] in self.scenarios_by_name:
            self.selected_scenario.set(scenario["name"])
            self._update_scenario_description()
        else:
            self.scenario_description.set(scenario["description"])
        self.assignment_enabled.set(False)
        self._update_assignment_states()
        self._apply_scenario_data(scenario)
        self._calculate()

    def _update_assignment_states(self):
        enabled = self.assignment_enabled.get()
        for widget in self.prediction_widgets:
            widget.configure(state="normal" if enabled else "disabled")
        for widget in (
            self.direction_prediction,
            self.fastest_prediction,
            self.correction_prediction,
        ):
            widget.configure(state="readonly" if enabled else "disabled")
        self.assignment_feedback.set(
            "Заполните прогноз и нажмите «Рассчитать»."
            if enabled
            else "Включите режим задания и заполните прогноз до расчёта."
        )
        self.calculate_button_text.set("Проверить прогноз" if enabled else "Рассчитать")

    def _update_lesson_summary(self):
        lesson = self.current_lesson
        questions = lesson["questions"]
        question_text = "" if not questions else "\nВопросы:\n" + "\n".join(f"• {item}" for item in questions)
        self.lesson_summary.set(
            f"{lesson['task']}\n\n{lesson['guidance']}"
            f"{question_text}\n\nПопыток для прогноза: {lesson['attempt_limit']}."
        )

    def _update_learning_route(self):
        steps = (
            "Выберите и примените сценарий",
            "Сформулируйте прогноз",
            "Выполните расчёт",
            "Сравните прогноз с результатом",
            "Измените настройки регулятора",
            "Сделайте вывод",
        )
        if not self.learning_mode.get():
            self.learning_route.set("Включите режим, чтобы идти по этапам лабораторной работы.")
            return
        self.learning_route.set(
            " → ".join(
                f"[{index + 1}] {text}" if index + 1 == self.learning_step else text
                for index, text in enumerate(steps)
            )
        )

    def _advance_learning_step(self):
        if not self.learning_mode.get():
            self.learning_mode.set(True)
        self.learning_step = min(6, self.learning_step + 1)
        self._update_learning_route()

    def _read_prediction(self):
        if not self.assignment_enabled.get():
            return None
        if self.assignment_attempts >= self.current_lesson["attempt_limit"]:
            self.assignment_feedback.set("Лимит попыток для этого задания исчерпан.")
            self._show_page("scenarios")
            raise ValueError("Лимит попыток исчерпан.")
        self.steady_prediction.configure(style="TEntry")
        if not all((
            self.predicted_direction.get(),
            self.predicted_fastest.get(),
            self.predicted_correction.get(),
            self.predicted_steady.get().strip(),
        )):
            self.assignment_feedback.set("Заполните все четыре поля прогноза до расчёта.")
            self._show_page("scenarios")
            raise ValueError("Прогноз заполнен не полностью.")
        try:
            steady = parse_positive_number(self.predicted_steady.get())
        except ValueError as error:
            self.steady_prediction.configure(style="Error.TEntry")
            self.assignment_feedback.set(f"Установившееся значение: {error}")
            self._show_page("scenarios")
            raise
        return {
            "direction": self.predicted_direction.get(),
            "steady": steady,
            "fastest": self.predicted_fastest.get(),
            "correction": self.predicted_correction.get(),
        }

    def _evaluate_assignment(self, prediction, outcome):
        if prediction is None:
            return
        evaluation = evaluate_prediction(
            prediction,
            outcome,
            self.assignment_tolerance_percent,
            lesson=self.current_lesson,
            controller=self.last_calculation["controller"] if self.last_calculation is not None else None,
        )
        self.assignment_attempts += 1
        self.assignment_evaluation = evaluation
        remaining = self.current_lesson["attempt_limit"] - self.assignment_attempts
        self.assignment_feedback.set(
            f"Результат: {evaluation['score']} из {evaluation['total']}.\n"
            + "\n".join(evaluation["lines"])
            + f"\nОсталось попыток: {remaining}."
        )
        if self.learning_mode.get():
            self.learning_step = max(self.learning_step, 4)
            self._update_learning_route()

    def _add_current_to_comparison(self):
        if self.last_calculation is None:
            self._set_status("Сначала выполните расчёт", error=True)
            return
        if len(self.comparison_runs) >= MAX_COMPARISON_RUNS:
            self._set_status(
                f"Можно сравнивать не больше {MAX_COMPARISON_RUNS} опытов",
                error=True,
            )
            return
        self.comparison_counter += 1
        scenario_name = (
            self.active_scenario.get().removeprefix("Применён: ").removesuffix(".")
            if self.active_scenario.get().startswith("Применён:")
            else "Свободный расчёт"
        )
        name = f"Опыт {self.comparison_counter}: {scenario_name}"
        run = build_comparison_run(
            self.last_calculation,
            name,
            self._capture_input_state(),
        )
        run["id"] = f"run-{self.comparison_counter}"
        self.comparison_runs.append(run)
        self._refresh_comparison_table(
            tuple(item["id"] for item in self.comparison_runs)
        )
        self._show_page("comparison")
        self._set_status(f"{name} добавлен к сравнению")

    def _refresh_comparison_table(self, selected_ids=()):
        self.comparison_table.delete(*self.comparison_table.get_children())
        for run in self.comparison_runs:
            settling = (
                "—"
                if run["settling_duration"] is None
                else f"{run['settling_duration']:.1f}"
            )
            self.comparison_table.insert(
                "",
                "end",
                iid=run["id"],
                values=(
                    run["name"],
                    self._format_number(run["time_constant"]),
                    self._format_number(run["delay"]),
                    run["controller_type"],
                    self._format_number(run["maximum_deviation"]),
                    settling,
                    self._format_signed_number(run["static_error"]),
                ),
            )
        for run_id in selected_ids:
            if self.comparison_table.exists(run_id):
                self.comparison_table.selection_add(run_id)
                self.comparison_table.see(run_id)
        count = len(self.comparison_runs)
        self.comparison_summary.set(
            f"Закреплено опытов: {count} из {MAX_COMPARISON_RUNS}."
            if count
            else "Закрепите результаты нескольких расчётов для сравнения."
        )

    def _selected_comparison_runs(self):
        selected_ids = set(self.comparison_table.selection())
        if not selected_ids:
            return list(self.comparison_runs)
        return [run for run in self.comparison_runs if run["id"] in selected_ids]

    def _select_all_comparison_runs(self):
        children = self.comparison_table.get_children()
        self.comparison_table.selection_set(children)
        self._draw_comparison()

    def _remove_comparison_runs(self):
        selected_ids = set(self.comparison_table.selection())
        if not selected_ids:
            self._set_status("Выберите опыты для удаления", error=True)
            return
        self.comparison_runs = [
            run for run in self.comparison_runs if run["id"] not in selected_ids
        ]
        self._refresh_comparison_table()
        self._draw_comparison()
        self._set_status("Выбранные опыты удалены из сравнения")

    def _clear_comparison_runs(self):
        if not self.comparison_runs:
            return
        if not messagebox.askyesno(
            "Очистить сравнение",
            "Удалить все закреплённые опыты из текущего сеанса?",
            parent=self.root,
        ):
            return
        self.comparison_runs.clear()
        self._refresh_comparison_table()
        self._draw_comparison()
        self._set_status("Сравнение очищено")

    def _export_comparison_csv(self):
        runs = self._selected_comparison_runs()
        if not runs:
            self._set_status("Нет опытов для экспорта", error=True)
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Экспортировать сравнение",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"),),
            initialfile="absorption_comparison.csv",
        )
        if not selected:
            return
        try:
            path = write_comparison_csv(selected, runs)
        except (OSError, ValueError) as error:
            self._set_status(f"Не удалось экспортировать сравнение: {error}", error=True)
            return
        self._set_status(f"Сравнение сохранено: {path.name}")

    def _save_comparison_html_report(self):
        self._save_comparison_report("html")

    def _save_comparison_pdf_report(self):
        self._save_comparison_report("pdf")

    def _save_comparison_report(self, format_name):
        if not self.comparison_runs:
            self._set_status("Нет закреплённых опытов для отчёта", error=True)
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить отчёт по сравнению",
            defaultextension=f".{format_name}",
            filetypes=((format_name.upper(), f"*.{format_name}"),),
            initialfile=f"absorption_comparison_report.{format_name}",
        )
        if not selected:
            return
        writer = write_comparison_html_report if format_name == "html" else write_comparison_pdf_report
        try:
            path = writer(selected, self.comparison_runs)
        except (OSError, ValueError) as error:
            self._set_status(f"Не удалось сохранить отчёт сравнения: {error}", error=True)
            return
        self._set_status(f"Отчёт по всем закреплённым опытам сохранён: {path.name}")

    def _rename_comparison_run(self):
        selected = self._selected_comparison_runs()
        if len(selected) != 1:
            self._set_status("Выберите один опыт для переименования", error=True)
            return
        run = selected[0]
        name = self._ask_comparison_run_name(run["name"])
        if name is None:
            return
        name = name.strip()
        if not name:
            self._set_status("Название опыта не должно быть пустым", error=True)
            return
        run["name"] = name
        self._refresh_comparison_table((run["id"],))

    def _ask_comparison_run_name(self, initial_name):
        dialog = tk.Toplevel(self.root)
        dialog.title("Переименовать опыт")
        dialog.configure(background=BACKGROUND)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.geometry("460x150")

        name = tk.StringVar(value=initial_name)
        result = {"value": None}
        content = ttk.Frame(dialog, style="App.TFrame", padding=20)
        content.pack(fill="both", expand=True)
        ttk.Label(content, text="Название опыта", style="CardTitle.TLabel").pack(anchor="w")
        entry = ttk.Entry(content, textvariable=name, width=48)
        entry.pack(fill="x", pady=(8, 16))

        actions = ttk.Frame(content, style="App.TFrame")
        actions.pack(fill="x")
        actions.columnconfigure(0, weight=1)

        def apply_name():
            result["value"] = name.get()
            dialog.destroy()

        ttk.Button(actions, text="Отмена", command=dialog.destroy, style="Secondary.TButton").grid(
            row=0, column=1, padx=(0, 8)
        )
        ttk.Button(actions, text="Переименовать", command=apply_name, style="Primary.TButton").grid(
            row=0, column=2
        )
        dialog.bind("<Return>", lambda _event: apply_name())
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
        entry.focus_set()
        entry.select_range(0, "end")
        self.root.wait_window(dialog)
        return result["value"]
        self._draw_comparison()
        self._set_status("Название опыта изменено")

    def _restore_selected_comparison_run(self):
        selected = self._selected_comparison_runs()
        if len(selected) != 1:
            self._set_status("Выберите один опыт для возврата параметров", error=True)
            return
        run = selected[0]
        input_state = run.get("input_state")
        if input_state is None:
            self._set_status("В этом опыте нет сохранённых параметров формы", error=True)
            return
        try:
            self._restore_input_state(input_state)
        except (KeyError, TypeError, ValueError) as error:
            self._set_status(f"Не удалось вернуть параметры: {error}", error=True)
            return
        self.active_scenario.set(f"Восстановлен: {run['name']}.")
        self.topbar_context.set(f"Параметры опыта «{run['name']}» возвращены в форму")
        self._set_status("Параметры опыта возвращены — при необходимости измените их и рассчитайте")
        self._show_page("disturbances")

    def _save_comparison_session(self):
        if not self.comparison_runs:
            self._set_status("Нет опытов для сохранения сеанса", error=True)
            return
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить учебный сеанс",
            defaultextension=".json",
            filetypes=(("Учебный сеанс JSON", "*.json"),),
            initialfile="absorption_session.json",
        )
        if not selected:
            return
        try:
            path = write_session(selected, self.comparison_runs, self.comparison_counter)
        except (OSError, ValueError, TypeError) as error:
            self._set_status(f"Не удалось сохранить сеанс: {error}", error=True)
            return
        self._set_status(f"Учебный сеанс сохранён: {path.name}")

    def _open_comparison_session(self):
        if self.comparison_runs and not messagebox.askyesno(
            "Открыть учебный сеанс",
            "Текущая история опытов будет заменена. Продолжить?",
            parent=self.root,
        ):
            return
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Открыть учебный сеанс",
            filetypes=(("Учебный сеанс JSON", "*.json"),),
        )
        if not selected:
            return
        try:
            runs, counter = read_session(selected)
        except (OSError, ValueError) as error:
            self._set_status(str(error), error=True)
            return
        if len(runs) > MAX_COMPARISON_RUNS:
            self._set_status(
                f"В сеансе {len(runs)} опытов; поддерживается не больше {MAX_COMPARISON_RUNS}",
                error=True,
            )
            return
        self.comparison_runs = runs
        self.comparison_counter = counter
        selected_ids = tuple(run["id"] for run in runs)
        self._refresh_comparison_table(selected_ids)
        self._draw_comparison()
        self._set_status(f"Открыт учебный сеанс: {len(runs)} опытов")

    def _export_comparison_graphs(self):
        self._export_graphs_png()

    def _capture_input_state(self):
        return {
            "chain": self.chain,
            "model_values": self.model_values.copy(),
            "component_enabled": self.component_enabled.get(),
            "flow_enabled": self.flow_enabled.get(),
            "component_value": self.component_value.get(),
            "flow_value": self.flow_value.get(),
            "disturbance_type": self.disturbance_type.get(),
            "start_time": self.start_time.get(),
            "simulation_duration": self.simulation_duration.get(),
            "effect_duration": self.effect_duration.get(),
            "time_constant": self.time_constant.get(),
            "delay": self.delay.get(),
            "controller_enabled": self.controller_enabled.get(),
            "controller_type": self.controller_type.get(),
            "proportional_gain": self.proportional_gain.get(),
            "integral_time": self.integral_time.get(),
            "derivative_time": self.derivative_time.get(),
            "control_limit": self.control_limit.get(),
            "setpoint": self.setpoint.get(),
        }

    def _restore_input_state(self, state):
        if state["chain"] not in (LEAN_GAS, RICH_ABSORBENT):
            raise ValueError("неизвестная цепь управления")
        model_values = state.get("model_values", {})
        if not isinstance(model_values, dict):
            raise ValueError("некорректные параметры математической модели")
        restored_model = DEFAULT_MODEL_VALUES.copy()
        for key in restored_model:
            restored_model[key] = float(model_values.get(key, restored_model[key]))
        self.model_values = restored_model
        self._select_chain(state["chain"])
        for key, variable in (
            ("component_enabled", self.component_enabled),
            ("flow_enabled", self.flow_enabled),
            ("component_value", self.component_value),
            ("flow_value", self.flow_value),
            ("disturbance_type", self.disturbance_type),
            ("start_time", self.start_time),
            ("simulation_duration", self.simulation_duration),
            ("effect_duration", self.effect_duration),
            ("time_constant", self.time_constant),
            ("delay", self.delay),
            ("controller_type", self.controller_type),
            ("proportional_gain", self.proportional_gain),
            ("integral_time", self.integral_time),
            ("derivative_time", self.derivative_time),
            ("control_limit", self.control_limit),
            ("setpoint", self.setpoint),
        ):
            variable.set(state[key])
        self._set_controller_mode(bool(state["controller_enabled"]))
        self._update_controller_type()
        self._update_disturbance_type()
        self._update_input_states()
        self._clear_result_values()
        self._draw_control_diagram()
        self._draw_static_charts()

    def _draw_comparison(self):
        if not hasattr(self, "comparison_table") or self.current_page != "comparison":
            return
        runs = self._selected_comparison_runs()
        self._remove_controller_axis()
        self.primary_chart_title.set("Сравнение переходных процессов")
        self.primary_chart_subtitle.set("Закреплённые расчёты на одном графике")
        self._style_axis(self.disturbance_axis, "Время, с", "Концентрация")
        colors = ("#2563EB", "#F59E0B", "#16A34A", "#7C3AED", "#DB2777", "#0891B2")
        for index, run in enumerate(runs):
            self.disturbance_axis.plot(
                run["time"],
                run["response"],
                color=colors[index % len(colors)],
                linewidth=2.2,
                label=run["name"],
            )
        if runs:
            self._place_legend_above(self.disturbance_axis)
            self.disturbance_axis.margins(x=0.02, y=0.12)
        else:
            self.disturbance_axis.text(
                0.5,
                0.5,
                "Нет закреплённых опытов",
                transform=self.disturbance_axis.transAxes,
                ha="center",
                va="center",
                color=MUTED,
            )
        self._enable_legend_toggles(self.disturbance_axis, self.disturbance_canvas)
        self.disturbance_canvas.draw_idle()

        self.response_chart_title.set("Длительность установления")
        self.response_subtitle.set("Время после начала реакции объекта")
        self._style_axis(self.response_axis, "Опыт", "Время, с")
        values = [
            0.0 if run["settling_duration"] is None else run["settling_duration"]
            for run in runs
        ]
        positions = np.arange(len(runs))
        bars = self.response_axis.bar(
            positions,
            values,
            color=[colors[index % len(colors)] for index in range(len(runs))],
            alpha=0.85,
        )
        self.response_axis.set_xticks(positions)
        self.response_axis.set_xticklabels(
            [run["name"].split(":", 1)[0] for run in runs],
            rotation=20,
            ha="right",
        )
        for bar, run in zip(bars, runs, strict=True):
            label = "не достигнуто" if run["settling_duration"] is None else f"{run['settling_duration']:.1f} с"
            self.response_axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT,
            )
        self.response_axis.margins(y=0.18)
        self.response_canvas.draw_idle()
        self._charts_show_comparison = True

    def _export_graphs_png(self):
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить графики",
            defaultextension=".png",
            filetypes=(("PNG", "*.png"),),
            initialfile="absorption_result.png",
        )
        if not selected:
            return
        try:
            signal_path, response_path = save_graphs(
                self.disturbance_axis.figure,
                self.response_axis.figure,
                selected,
            )
        except OSError as error:
            self._set_status(f"Не удалось сохранить PNG: {error}", error=True)
            return
        self.export_summary.set(f"Сохранены: {signal_path.name}, {response_path.name}.")
        self._set_status("Графики сохранены в PNG")

    def _export_csv(self):
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Экспортировать точки",
            defaultextension=".csv",
            filetypes=(("CSV", "*.csv"),),
            initialfile="absorption_result.csv",
        )
        if not selected:
            return
        try:
            path = write_csv(selected, self.last_calculation)
        except OSError as error:
            self._set_status(f"Не удалось сохранить CSV: {error}", error=True)
            return
        self.export_summary.set(f"Точки сохранены: {path.name}.")
        self._set_status("Точки экспортированы в CSV")

    def _copy_protocol(self):
        protocol = self._build_protocol_text()
        self.root.clipboard_clear()
        self.root.clipboard_append(protocol)
        self.root.update_idletasks()
        self.export_summary.set("Параметры и результаты скопированы в буфер обмена.")
        self._set_status("Протокол скопирован")

    def _save_protocol(self):
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить протокол",
            defaultextension=".txt",
            filetypes=(("Текстовый файл", "*.txt"),),
            initialfile="absorption_protocol.txt",
        )
        if not selected:
            return
        try:
            path = write_protocol(selected, self._build_protocol_text())
        except OSError as error:
            self._set_status(f"Не удалось сохранить протокол: {error}", error=True)
            return
        self.export_summary.set(f"Протокол сохранён: {path.name}.")
        self._set_status("Протокол сохранён")

    def _save_html_report(self):
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить отчёт лабораторной работы",
            defaultextension=".html",
            filetypes=(("HTML", "*.html"),),
            initialfile="absorption_lab_report.html",
        )
        if not selected:
            return
        title = self.active_scenario.get().removeprefix("Применён: ").removesuffix(".")
        try:
            path = write_html_report(
                selected, self.last_calculation, title, self.current_lesson,
                self.assignment_evaluation, self.student_name.get(),
                self.student_conclusion.get(),
                (self.disturbance_axis.figure, self.response_axis.figure),
                prediction=self.last_prediction,
                comparison_runs=self.comparison_runs,
            )
        except OSError as error:
            self._set_status(f"Не удалось сохранить HTML-отчёт: {error}", error=True)
            return
        self.export_summary.set(f"HTML-отчёт сохранён: {path.name}.")
        self._set_status("HTML-отчёт сохранён")

    def _save_pdf_report(self):
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Сохранить отчёт лабораторной работы",
            defaultextension=".pdf",
            filetypes=(("PDF", "*.pdf"),),
            initialfile="absorption_lab_report.pdf",
        )
        if not selected:
            return
        title = self.active_scenario.get().removeprefix("Применён: ").removesuffix(".")
        try:
            path = write_pdf_report(
                selected, self.last_calculation, title, self.current_lesson,
                self.assignment_evaluation, self.student_name.get(),
                self.student_conclusion.get(),
                (self.disturbance_axis.figure, self.response_axis.figure),
                prediction=self.last_prediction,
                comparison_runs=self.comparison_runs,
            )
        except OSError as error:
            self._set_status(f"Не удалось сохранить PDF-отчёт: {error}", error=True)
            return
        self.export_summary.set(f"PDF-отчёт сохранён: {path.name}.")
        self._set_status("PDF-отчёт сохранён")

    def _build_protocol_text(self):
        title = self.active_scenario.get() if self.active_scenario.get().startswith("Применён:") else "Свободный расчёт"
        return build_protocol(self.last_calculation, title)

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
        details = []
        if "I" in controller_type:
            details.append("Интегратор защищён от насыщения.")
        if "D" in controller_type:
            details.append("Производная берётся по выходу без скачка от задания.")
        details.append("|u| задаёт предел воздействия.")
        self.controller_formula.set(
            f"{' '.join(details)}\n\n"
            "Выше показаны правила автоподбора. Для PID/PD при наличии "
            "запаздывания целевая динамика ускоряется вдвое."
        )
        if hasattr(self, "formula_panel"):
            self.formula_panel.set_controller_type(controller_type)
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
            prediction = self._read_prediction()
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

        result = run_simulation(
            self.chain,
            self.model_values,
            component_fraction,
            flow_fraction,
            dynamics,
            controller,
        )
        result["disturbance_type"] = self.disturbance_type.get()

        self.baseline_result.set(self._format_number(result["baseline"]))
        self.disturbance_result.set(
            f"{self._format_number(result['combined_fraction'])} "
            f"({result['combined_fraction'] * 100:+.1f}%)"
        )
        self.calculated_result.set(self._format_number(result["calculated"]))
        self._update_calculation_steps(
            component_fraction,
            flow_fraction,
            result["combined_fraction"],
            result["baseline"],
            result["calculated"],
        )
        self._draw_calculation_result(result)

        self.result_mode.set(result["result_mode"])
        self.final_result.set(self._format_number(result["final_response"][-1]))
        self._update_transition_metrics(
            result["metrics"],
            dynamics,
            result["response_start"],
        )
        self.last_calculation = result
        self.last_prediction = prediction
        self.add_comparison_button.configure(state="normal")
        for button in self.export_buttons:
            button.configure(state="normal")
        self.export_summary.set("Расчёт готов к экспорту.")
        self._evaluate_assignment(prediction, result["prediction_outcome"])
        settling_time = result["metrics"]["settling_time"]
        settling_summary = (
            "не установилось"
            if settling_time is None
            else f"установление {max(0.0, settling_time - result['response_start']):.1f} с"
        )
        active_name = (
            self.active_scenario.get().removeprefix("Применён: ").removesuffix(".")
            if self.active_scenario.get().startswith("Применён:")
            else "Свободный расчёт"
        )
        self.topbar_context.set(
            f"{active_name} · {result['result_mode'].lower()} · "
            f"отклонение {result['metrics']['maximum_deviation']:.4g} · {settling_summary}"
        )
        self.calculate_button_text.set("Пересчитать")
        self._show_page("scenarios" if prediction is not None else "results")
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
        self.last_calculation = None
        self.last_prediction = None
        if hasattr(self, "add_comparison_button"):
            self.add_comparison_button.configure(state="disabled")
        if hasattr(self, "export_buttons"):
            for button in self.export_buttons:
                button.configure(state="disabled")
        if hasattr(self, "export_summary"):
            self.export_summary.set("Сначала выполните расчёт.")
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
        if hasattr(self, "calculate_button_text"):
            self.calculate_button_text.set(
                "Проверить прогноз" if self.assignment_enabled.get() else "Рассчитать"
            )
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
        self._place_legend_above(self.response_axis)
        self.response_canvas.draw_idle()
        self._charts_show_comparison = False

    def _draw_last_calculation(self):
        if self.last_calculation is None:
            self._draw_static_charts()
        else:
            self._draw_calculation_result(self.last_calculation)

    def _draw_calculation_result(self, result):
        dynamics = result["dynamics"]
        controller = result["controller"]
        if controller is None:
            self._draw_disturbance(
                result["time"],
                result["profile"],
                result["component_fraction"],
                result["flow_fraction"],
                dynamics,
            )
            self._draw_response(
                result["time"],
                result["responses"],
                dynamics,
                result["metrics"],
            )
        else:
            self._draw_controller_signals(
                result["time"],
                result["error"],
                result["control"],
                dynamics,
            )
            self._draw_control_comparison(
                result["time"],
                result["responses"]["Совместное воздействие"],
                result["controlled_response"],
                controller["setpoint"],
                controller["controller_type"],
                dynamics,
                result["metrics"],
            )
        self._charts_show_comparison = False

    def _draw_disturbance(
        self,
        time,
        profile,
        component_fraction,
        flow_fraction,
        dynamics=None,
    ):
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
        self._annotate_timing(self.disturbance_axis, dynamics)
        self.disturbance_axis.margins(x=0.02, y=0.15)
        self._place_legend_above(self.disturbance_axis)
        self._enable_legend_toggles(self.disturbance_axis, self.disturbance_canvas)
        self.disturbance_canvas.draw_idle()

    def _draw_response(self, time, responses, dynamics=None, metrics=None):
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
        self._annotate_transition(self.response_axis, dynamics, metrics)
        self.response_axis.margins(x=0.02, y=0.12)
        self._place_legend_above(self.response_axis)
        self._enable_legend_toggles(self.response_axis, self.response_canvas)
        self.response_canvas.draw_idle()

    def _draw_controller_signals(self, time, error, control, dynamics=None):
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
        self._annotate_timing(self.disturbance_axis, dynamics)

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
        self._place_legend_above(
            self.disturbance_axis,
            [error_line, control_line],
            [error_line.get_label(), control_line.get_label()],
        )
        self.disturbance_canvas.draw_idle()

    def _draw_control_comparison(
        self,
        time,
        open_response,
        controlled_response,
        setpoint,
        controller_type,
        dynamics=None,
        metrics=None,
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
        self._annotate_transition(self.response_axis, dynamics, metrics)
        self.response_axis.margins(x=0.02, y=0.12)
        self._place_legend_above(self.response_axis)
        self._enable_legend_toggles(self.response_axis, self.response_canvas)
        self.response_canvas.draw_idle()

    @staticmethod
    def _place_legend_above(axis, handles=None, labels=None):
        if handles is None:
            handles, labels = axis.get_legend_handles_labels()
        if not handles:
            return
        axis.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            fontsize=8,
            ncol=len(handles),
            borderaxespad=0,
            columnspacing=1.4,
            handletextpad=0.6,
        )

    @staticmethod
    def _enable_legend_toggles(axis, canvas):
        callback_id = getattr(canvas, "_legend_toggle_callback", None)
        if callback_id is not None:
            canvas.mpl_disconnect(callback_id)
        legend = axis.get_legend()
        if legend is None:
            canvas._legend_toggle_callback = None
            return
        artists_by_label = {
            artist.get_label(): artist
            for artist in axis.lines
            if artist.get_label() and not artist.get_label().startswith("_")
        }
        toggle_map = {}
        for legend_line, legend_text in zip(
            legend.get_lines(),
            legend.get_texts(),
            strict=False,
        ):
            artist = artists_by_label.get(legend_text.get_text())
            if artist is None:
                continue
            legend_line.set_picker(5)
            toggle_map[legend_line] = artist

        def toggle(event):
            artist = toggle_map.get(event.artist)
            if artist is None:
                return
            artist.set_visible(not artist.get_visible())
            event.artist.set_alpha(1.0 if artist.get_visible() else 0.25)
            canvas.draw_idle()

        canvas._legend_toggle_callback = canvas.mpl_connect("pick_event", toggle)

    def _annotate_transition(self, axis, dynamics, metrics):
        self._annotate_timing(
            axis,
            dynamics,
            None if metrics is None else metrics.get("settling_time"),
        )
        if metrics is None:
            return
        reference = max(
            metrics["maximum_deviation"],
            abs(metrics["steady_state"] - metrics["initial_value"]),
        )
        tolerance = reference * 0.05
        if tolerance > 0:
            axis.axhspan(
                metrics["steady_state"] - tolerance,
                metrics["steady_state"] + tolerance,
                color="#16A34A",
                alpha=0.08,
                label="Полоса ±5%",
            )

    @staticmethod
    def _annotate_timing(axis, dynamics, settling_time=None):
        if dynamics is None:
            return
        markers = [(dynamics["start_time"], "t₀", "#7C3AED")]
        delayed_start = dynamics["start_time"] + dynamics["delay"]
        if dynamics["delay"] > 0:
            markers.append((delayed_start, "t₀ + L", "#DB2777"))
        if settling_time is not None:
            markers.append((settling_time, "tуст", "#16A34A"))
        for index, (position, label, color) in enumerate(markers):
            axis.axvline(position, color=color, linestyle=":", linewidth=1.2, alpha=0.9)
            axis.text(
                position,
                0.98 - index * 0.09,
                label,
                color=color,
                fontsize=8,
                ha="left",
                va="top",
                transform=axis.get_xaxis_transform(),
            )

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
