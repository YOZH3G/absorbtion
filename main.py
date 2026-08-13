import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.interpolate import make_interp_spline

from calculations import calculate_xna, calculate_xog, combine_fractions
from validation import parse_fraction


BACKGROUND = "#F4F6F8"
CARD_BACKGROUND = "#FFFFFF"
BORDER = "#D7DCE2"
TEXT = "#1F2937"
MUTED = "#667085"
ACCENT = "#2563EB"
ACCENT_ACTIVE = "#1D4ED8"
ERROR = "#B42318"
SUCCESS = "#16A34A"

LEAN_GAS = "lean_gas"
RICH_ABSORBENT = "rich_absorbent"

GNA = 7800
XA = 0.5
XG = 0.5
GG = 1000
XOG_INITIAL = 0.8
XNA_INITIAL = 30
GOG = (GG * XG) / XOG_INITIAL
GA = (GNA * XNA_INITIAL) / XA


class AbsorptionApp(ttk.Frame):
    def __init__(self, root):
        super().__init__(root, style="App.TFrame", padding=(20, 16, 20, 12))
        self.root = root
        self.chain = LEAN_GAS

        self.component_enabled = tk.BooleanVar(value=False)
        self.flow_enabled = tk.BooleanVar(value=False)
        self.component_value = tk.StringVar()
        self.flow_value = tk.StringVar()
        self.component_error = tk.StringVar()
        self.flow_error = tk.StringVar()
        self.component_label = tk.StringVar()
        self.component_symbol = tk.StringVar()
        self.flow_label = tk.StringVar()
        self.flow_symbol = tk.StringVar()
        self.baseline_result = tk.StringVar(value="—")
        self.disturbance_result = tk.StringVar(value="—")
        self.calculated_result = tk.StringVar(value="—")
        self.response_subtitle = tk.StringVar()
        self.status_text = tk.StringVar(value="Готово к расчёту")

        self._configure_window()
        self._configure_styles()
        self._build_layout()
        self._select_chain(LEAN_GAS)

    def _configure_window(self):
        self.root.title("Анализ процесса абсорбции")
        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)
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
        self.columnconfigure(0, minsize=360)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Label(self, text="Анализ процесса абсорбции", style="Header.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 14)
        )

        controls = ttk.Frame(self, style="App.TFrame")
        controls.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        controls.columnconfigure(0, weight=1)

        results = ttk.Frame(self, style="App.TFrame")
        results.grid(row=1, column=1, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)
        results.rowconfigure(1, weight=1)

        self._build_chain_card(controls)
        self._build_disturbance_card(controls)
        self._build_result_card(controls)

        self.disturbance_axis, self.disturbance_canvas, self.disturbance_toolbar = self._build_chart_card(
            results,
            row=0,
            title="Возмущающее воздействие",
            subtitle="Изменение относительно базового уровня",
        )
        self.response_axis, self.response_canvas, self.response_toolbar = self._build_chart_card(
            results,
            row=1,
            title="Кривая разгона",
            subtitle_variable=self.response_subtitle,
        )

        status = ttk.Frame(self, style="App.TFrame")
        status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        status.columnconfigure(1, weight=1)
        self.status_dot = ttk.Label(status, text="●", style="StatusDot.TLabel")
        self.status_dot.grid(row=0, column=0, sticky="w")
        ttk.Label(status, textvariable=self.status_text, style="Status.TLabel").grid(row=0, column=1, sticky="w", padx=(5, 0))
        ttk.Label(status, text="Python 3.14", style="Status.TLabel").grid(row=0, column=2, sticky="e")

    def _card(self, parent, row, padding=(18, 16)):
        card = ttk.Frame(parent, style="Card.TFrame", padding=padding)
        card.grid(row=row, column=0, sticky="ew", pady=(0, 14))
        card.columnconfigure(0, weight=1)
        return card

    def _build_chain_card(self, parent):
        card = self._card(parent, 0)
        ttk.Label(card, text="Цепь управления", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        self.lean_button = ttk.Button(card, text="Обеднённый газ", command=lambda: self._select_chain(LEAN_GAS), style="Segment.TButton")
        self.lean_button.grid(row=1, column=0, sticky="ew")
        self.rich_button = ttk.Button(card, text="Насыщенный абсорбент", command=lambda: self._select_chain(RICH_ABSORBENT), style="Segment.TButton")
        self.rich_button.grid(row=1, column=1, sticky="ew", padx=(4, 0))
        card.columnconfigure((0, 1), weight=1)

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

        ttk.Label(card, text="Введите долю: 0.01 = +1%. Допустимый диапазон: 0.01–9.99", style="Muted.TLabel", wraplength=300).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(2, 14)
        )

        actions = ttk.Frame(card, style="CardBody.TFrame")
        actions.grid(row=6, column=0, columnspan=3, sticky="ew")
        actions.columnconfigure((0, 1), weight=1)
        self.calculate_button = ttk.Button(actions, text="Рассчитать", command=self._calculate, style="Primary.TButton", state="disabled")
        self.calculate_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(actions, text="Сбросить", command=self._reset, style="Secondary.TButton").grid(row=0, column=1, sticky="ew", padx=(6, 0))
        card.columnconfigure(0, weight=1)

    def _build_result_card(self, parent):
        card = self._card(parent, 2)
        ttk.Label(card, text="Результат", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        rows = (
            ("Базовое значение", self.baseline_result),
            ("Суммарная доля", self.disturbance_result),
            ("Расчётное значение", self.calculated_result),
        )
        for row, (label, variable) in enumerate(rows, start=1):
            ttk.Label(card, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=6)
            ttk.Label(card, textvariable=variable, style="ResultValue.TLabel").grid(row=row, column=1, sticky="e", pady=6)
        card.columnconfigure(0, weight=1)

    def _build_chart_card(self, parent, row, title, subtitle=None, subtitle_variable=None):
        card = ttk.Frame(parent, style="Card.TFrame", padding=(16, 12))
        card.grid(row=row, column=0, sticky="nsew", pady=(0, 8) if row == 0 else (8, 0))
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        header = ttk.Frame(card, style="CardBody.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        header.columnconfigure(0, weight=1)
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

        self._reset()

    def _update_input_states(self):
        self._set_entry_state(self.component_entry, self.component_enabled.get(), self.component_value)
        self._set_entry_state(self.flow_entry, self.flow_enabled.get(), self.flow_value)
        enabled = self.component_enabled.get() or self.flow_enabled.get()
        self.calculate_button.configure(state="normal" if enabled else "disabled")
        self._clear_errors()

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
        self.component_entry.configure(style="TEntry")
        self.flow_entry.configure(style="TEntry")

    def _read_fraction(self, enabled, value, error_variable, entry):
        if not enabled:
            return 0.0
        try:
            return parse_fraction(value.get())
        except ValueError as error:
            error_variable.set(str(error))
            entry.configure(style="Error.TEntry")
            raise

    def _calculate(self):
        self._clear_errors()
        try:
            component_fraction = self._read_fraction(
                self.component_enabled.get(), self.component_value, self.component_error, self.component_entry
            )
            flow_fraction = self._read_fraction(
                self.flow_enabled.get(), self.flow_value, self.flow_error, self.flow_entry
            )
        except ValueError:
            self._set_status("Исправьте значение", error=True)
            return

        combined_fraction = combine_fractions(component_fraction, flow_fraction)
        if self.chain == LEAN_GAS:
            baseline = XOG_INITIAL
            calculated = calculate_xog(GG, XG, GOG, component_fraction, flow_fraction)
            transition = calculated
        else:
            baseline = XNA_INITIAL
            calculated = calculate_xna(GA, GNA, XA, component_fraction, flow_fraction)
            transition = calculated - 1

        self.baseline_result.set(self._format_number(baseline))
        self.disturbance_result.set(f"{self._format_number(combined_fraction)} ({combined_fraction * 100:+.1f}%)")
        self.calculated_result.set(self._format_number(calculated))
        self._draw_disturbance(combined_fraction)
        self._draw_response(baseline, calculated, transition)
        self._set_status("Расчёт выполнен")

    def _reset(self):
        self.component_enabled.set(False)
        self.flow_enabled.set(False)
        self.component_value.set("")
        self.flow_value.set("")
        self.component_entry.configure(state="disabled")
        self.flow_entry.configure(state="disabled")
        self.calculate_button.configure(state="disabled")
        self._clear_errors()
        self.baseline_result.set("—")
        self.disturbance_result.set("—")
        self.calculated_result.set("—")
        self._draw_static_charts()
        self._set_status("Готово к расчёту")

    def _draw_static_charts(self):
        baseline = XOG_INITIAL if self.chain == LEAN_GAS else XNA_INITIAL
        self._draw_disturbance(0.0)
        self._style_axis(self.response_axis, "Время", "Концентрация")
        self.response_axis.plot([0, 100], [baseline, baseline], color=ACCENT, linewidth=2)
        self.response_canvas.draw_idle()

    def _draw_disturbance(self, combined_fraction):
        base_level = 3
        new_level = base_level + combined_fraction
        self._style_axis(self.disturbance_axis, "Время", "Уровень воздействия")
        self.disturbance_axis.step([0, 10, 100], [base_level, new_level, new_level], where="post", color=ACCENT, linewidth=2)
        self.disturbance_axis.margins(x=0.02, y=0.15)
        self.disturbance_canvas.draw_idle()

    def _draw_response(self, baseline, calculated, transition):
        x = np.arange(0, 101, 10)
        y = [baseline, baseline, transition] + [calculated] * 8
        spline = make_interp_spline(x, y, k=2)
        x_smooth = np.linspace(0, 100, 200)
        y_smooth = np.maximum(spline(x_smooth), baseline)

        self._style_axis(self.response_axis, "Время", "Концентрация")
        self.response_axis.plot(x_smooth, y_smooth, color=ACCENT, linewidth=2)
        self.response_axis.margins(x=0.02, y=0.12)
        self.response_canvas.draw_idle()

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


def main():
    root = tk.Tk()
    AbsorptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
