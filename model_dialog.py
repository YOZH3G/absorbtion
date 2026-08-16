import tkinter as tk
from tkinter import ttk

from validation import parse_positive_number


class ModelParametersDialog(tk.Toplevel):
    def __init__(
        self,
        parent,
        values,
        defaults,
        on_apply,
        on_close,
        format_number,
        background,
    ):
        super().__init__(parent)
        self._defaults = defaults
        self._on_apply = on_apply
        self._on_close = on_close
        self._format_number = format_number
        self._parsed_values = None
        self._variables = {
            key: tk.StringVar(value=format_number(value))
            for key, value in values.items()
        }
        self._errors = {key: tk.StringVar() for key in self._variables}
        self._entries = {}

        self.title("Параметры математической модели")
        self.configure(background=background)
        self.resizable(False, False)
        self.transient(parent)
        self._build_content()
        self._bind_events()
        self._validate_values()
        self._center(parent)

    def _build_content(self):
        content = ttk.Frame(self, style="App.TFrame", padding=20)
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
        self._build_parameter_group(content, 0, "Газовая часть", gas_specs)
        self._build_parameter_group(content, 1, "Абсорбент", absorbent_specs)

        derived = ttk.Frame(content, style="Card.TFrame", padding=(16, 12))
        derived.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        derived.columnconfigure((1, 3), weight=1)
        ttk.Label(derived, text="Расчётные значения", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8)
        )
        self._gog_value = tk.StringVar(value="—")
        self._ga_value = tk.StringVar(value="—")
        ttk.Label(derived, text="Gог — расход обеднённого газа", style="Body.TLabel").grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(derived, textvariable=self._gog_value, style="ResultValue.TLabel").grid(
            row=1, column=1, sticky="w", padx=(8, 28)
        )
        ttk.Label(derived, text="Gа — расход абсорбента", style="Body.TLabel").grid(
            row=1, column=2, sticky="w"
        )
        ttk.Label(derived, textvariable=self._ga_value, style="ResultValue.TLabel").grid(
            row=1, column=3, sticky="w", padx=(8, 0)
        )

        actions = ttk.Frame(content, style="App.TFrame")
        actions.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        actions.columnconfigure(1, weight=1)
        ttk.Button(
            actions,
            text="По умолчанию",
            command=self._restore_defaults,
            style="Secondary.TButton",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            actions,
            text="Отмена",
            command=self._close,
            style="Secondary.TButton",
        ).grid(row=0, column=2, padx=(8, 8))
        self._apply_button = ttk.Button(
            actions,
            text="Применить",
            command=self._apply_values,
            style="Primary.TButton",
        )
        self._apply_button.grid(row=0, column=3)

    def _build_parameter_group(self, parent, column, title, specs):
        group = ttk.Frame(parent, style="Card.TFrame", padding=(16, 14))
        group.grid(row=2, column=column, sticky="nsew", padx=(0, 7) if column == 0 else (7, 0))
        group.columnconfigure(0, weight=1)
        ttk.Label(group, text=title, style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )
        for index, (key, label, symbol) in enumerate(specs):
            row = 1 + index * 2
            ttk.Label(group, text=label, style="Body.TLabel", wraplength=220).grid(
                row=row, column=0, sticky="w"
            )
            ttk.Label(group, text=symbol, style="Body.TLabel").grid(
                row=row, column=1, padx=(10, 8)
            )
            entry = ttk.Entry(group, textvariable=self._variables[key], width=11)
            entry.grid(row=row, column=2, sticky="e")
            self._entries[key] = entry
            ttk.Label(group, textvariable=self._errors[key], style="Error.TLabel").grid(
                row=row + 1, column=0, columnspan=3, sticky="w", pady=(2, 7)
            )

    def _bind_events(self):
        for variable in self._variables.values():
            variable.trace_add("write", self._validate_values)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        self.bind("<Return>", lambda _event: self._apply_values())

    def _validate_values(self, *_):
        parsed = {}
        valid = True
        for key, variable in self._variables.items():
            try:
                parsed[key] = parse_positive_number(variable.get())
                self._errors[key].set("")
                self._entries[key].configure(style="TEntry")
            except ValueError as error:
                self._errors[key].set(str(error))
                self._entries[key].configure(style="Error.TEntry")
                valid = False

        if valid:
            self._gog_value.set(self._format_number(parsed["gg"] * parsed["xg"] / parsed["xog_initial"]))
            self._ga_value.set(self._format_number(parsed["gna"] * parsed["xna_initial"] / parsed["xa"]))
            self._parsed_values = parsed
            self._apply_button.configure(state="normal")
        else:
            self._gog_value.set("—")
            self._ga_value.set("—")
            self._parsed_values = None
            self._apply_button.configure(state="disabled")

    def _restore_defaults(self):
        for key, value in self._defaults.items():
            self._variables[key].set(self._format_number(value))

    def _apply_values(self):
        self._validate_values()
        if self._parsed_values is None:
            return
        values = self._parsed_values.copy()
        self._close()
        self._on_apply(values)

    def _close(self):
        if self.grab_current() is self:
            self.grab_release()
        self._on_close()
        self.destroy()

    def _center(self, parent):
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_reqwidth()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_reqheight()) // 2
        self.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        self.grab_set()
        self.lift()
        self.focus_force()
        self._entries["gg"].focus_set()
