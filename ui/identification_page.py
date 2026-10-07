"""Experiment import and fitting controls; calculation is owned by the app worker."""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from app.identification import experiment_columns, export_identification, identify_step, read_experiment


class IdentificationPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, style="CardBody.TFrame", padding=12)
        self.app = app
        self.headers, self.rows = [], []
        self.generation = 0
        self.result = None
        self.source = ""
        self.columns = [tk.StringVar() for _ in range(3)]
        self.units = [tk.StringVar(value=value) for value in ("с", "как в файле", "как в файле")]
        self.delimiter = tk.StringVar(value=";")
        self.view = tk.StringVar(value="Отклик")
        self.summary = tk.StringVar(value="Импортируйте CSV с одной постоянной ступенью. Нужны минимум 5 точек до изменения и 12 после него.")
        self.columnconfigure(1, weight=1)
        ttk.Label(self, text="Идентификация по CSV", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self, text="Разделитель CSV", style="Body.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Combobox(self, textvariable=self.delimiter, values=(";", ",", "табуляция"), state="readonly", width=12).grid(row=1, column=1, sticky="ew")
        ttk.Button(self, text="Импортировать CSV", command=self.import_file).grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Button(self, text="Учебный пример", command=lambda: self.load_path(Path(__file__).resolve().parent.parent / "data" / "identification_step.csv")).grid(row=3, column=0, columnspan=2, sticky="ew")
        self.boxes = []
        for index, label in enumerate(("Время", "Вход", "Выход")):
            row = 4 + index * 2
            ttk.Label(self, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=(10, 0))
            box = ttk.Combobox(self, textvariable=self.columns[index], state="readonly")
            box.grid(row=row, column=1, sticky="ew", pady=(10, 0))
            self.boxes.append(box)
            ttk.Label(self, text="Единицы", style="Hint.TLabel").grid(row=row + 1, column=0, sticky="w")
            ttk.Combobox(self, textvariable=self.units[index], values=("с", "мин") if index == 0 else ("как в файле", "%"),
                         state="readonly", width=12).grid(row=row + 1, column=1, sticky="ew")
        ttk.Button(self, text="Оценить K, T и L", command=self.calculate).grid(row=10, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Combobox(self, textvariable=self.view, values=("Отклик", "Остаток"), state="readonly").grid(row=11, column=0, columnspan=2, sticky="ew")
        ttk.Label(self, textvariable=self.summary, style="Body.TLabel", wraplength=320).grid(row=12, column=0, columnspan=2, sticky="ew", pady=10)
        self.apply_button = ttk.Button(self, text="Применить T и L", command=self.apply, state="disabled")
        self.apply_button.grid(row=13, column=0, columnspan=2, sticky="ew")
        self.export_buttons = []
        for row, fmt in enumerate(("csv", "json"), 14):
            button = ttk.Button(self, text=f"Экспорт {fmt.upper()}", command=lambda fmt=fmt: self.export(fmt), state="disabled")
            button.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)
            self.export_buttons.append(button)
        ttk.Label(self, text="Маршрут: эксперимент → модель и остаток → T и L → автоподбор регулятора. K описывает эксперимент; баланс абсорбера сохраняет своё усиление. Начните с учебного примера, затем импортируйте собственный опыт.",
                  style="Hint.TLabel", wraplength=320).grid(row=16, column=0, columnspan=2, sticky="ew", pady=12)
        for variable in (*self.columns, *self.units, self.delimiter):
            variable.trace_add("write", self.changed)
        self.view.trace_add("write", lambda *_: self.draw() if self.result is not None else None)

    def signature(self):
        return (self.generation, *(v.get() for v in (*self.columns, *self.units, self.delimiter)))

    def changed(self, *_):
        self.app._invalidate_task()
        if self.result is not None:
            self.apply_button.configure(state="disabled")
            self.summary.set("Выбор изменён — повторите оценку. Графики и экспорт сохраняют последний успешный опыт.")

    def import_file(self):
        path = filedialog.askopenfilename(title="Эксперимент CSV", filetypes=(("CSV", "*.csv"),))
        if path:
            self.load_path(path)

    def load_path(self, path):
        try:
            delimiter = "\t" if self.delimiter.get() == "табуляция" else self.delimiter.get()
            headers, rows = read_experiment(path, delimiter)
        except (ValueError, OSError, UnicodeError) as error:
            self.summary.set(str(error))
            return
        self.headers, self.rows, self.source = headers, rows, str(path)
        self.generation += 1
        self.changed()
        for index, box in enumerate(self.boxes):
            box.configure(values=headers)
            self.columns[index].set(headers[index])
        self.summary.set(f"{Path(path).name}: {len(rows)} строк. Проверьте столбцы и единицы, затем оцените модель.")

    def calculate(self):
        try:
            columns = tuple(v.get() for v in self.columns)
            units = tuple(v.get() for v in self.units)
            arrays = experiment_columns(self.headers, self.rows, columns, units)
        except ValueError as error:
            self.summary.set(str(error))
            return
        metadata = dict(source=Path(self.source).name, columns=columns, units=units)
        self.app._start_task("Идентификация", lambda cancel, progress: dict(
            identify_step(*arrays, cancel=cancel, progress=progress), **metadata), self.accept, "identification")

    def accept(self, result):
        self.result = result
        self.summary.set(f"K = {result['gain']:.6g}; T = {result['time_constant']:.6g} с; L = {result['delay']:.6g} с. "
                         f"RMSE = {result['rmse']:.5g}; ошибка {result['normalized_rmse'] * 100:.3g}% амплитуды. "
                         f"{result['source']}. Время ступени определено внутри интервала {result['step_interval'][0]:g}–{result['step_interval'][1]:g} с.")
        self.apply_button.configure(state="normal")
        for button in self.export_buttons:
            button.configure(state="normal")
        self.app._set_status("Идентификация выполнена")
        self.draw()

    def draw(self):
        if self.result is None or self.app.current_page != "identification":
            return
        app, result = self.app, self.result
        app._remove_controller_axis()
        app._style_axis(app.disturbance_axis, "Время, с", "Вход, приведённые единицы")
        app.disturbance_axis.plot(result["time"], result["input"], label="Вход", color="#16A34A")
        app.primary_chart_title.set("Экспериментальный вход")
        app.primary_chart_subtitle.set(result["source"])
        residual = self.view.get() == "Остаток"
        app._style_axis(app.response_axis, "Время, с", "Остаток, приведённые единицы" if residual else "Выход, приведённые единицы")
        if residual:
            app.response_axis.plot(result["time"], result["residual"], label="Эксперимент − модель", color="#F59E0B")
            app.response_axis.axhline(0, color="#64748B", linewidth=.8)
        else:
            app.response_axis.plot(result["time"], result["output"], label="Эксперимент", color="#64748B")
            app.response_axis.plot(result["time"], result["model"], label="Модель", color="#2563EB")
        app.response_chart_title.set("Остаток идентификации" if residual else "Эксперимент и модель")
        app.response_subtitle.set(f"K={result['gain']:.5g}; T={result['time_constant']:.5g} с; L={result['delay']:.5g} с")
        for axis, canvas in ((app.disturbance_axis, app.disturbance_canvas), (app.response_axis, app.response_canvas)):
            app._place_legend_above(axis)
            app._enable_legend_toggles(axis, canvas)
            canvas.draw_idle()
        app._charts_show_comparison = False

    def apply(self):
        if self.result is not None:
            self.app.time_constant.set(f"{self.result['time_constant']:.12g}")
            self.app.delay.set(f"{self.result['delay']:.12g}")
            self.app._show_page("controller")
            self.app._set_status("T и L применены. Выполните автоподбор и расчёт опыта.")

    def export(self, fmt):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(title="Результат идентификации", defaultextension=f".{fmt}", initialfile=f"identification.{fmt}")
        if path:
            try:
                export_identification(path, self.result, fmt)
                self.app._set_status("Идентификация экспортирована")
            except OSError as error:
                self.summary.set(str(error))
