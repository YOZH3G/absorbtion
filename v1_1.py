import PySimpleGUI as sg
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from calculations import calculate_xna, calculate_xog, combine_fractions
from validation import parse_fraction

# Настройки шрифта
sg.set_options(font=("SF Pro Display",14))

# Переменная chain для интерфейса программы
chain = ["Концентрация в обедненном газе","Концентрация в насыщенном абсорбенте"]
# Переменные для выполнения логики
events_cbvv = ["CB_VV1", "CB_VV2", "CB_VV3"]
events_invv = ["IN_VV1","IN_VV2","IN_VV3"]
# Список static_vv - ось ординат для графиков статики; 
# x_static = ось абсцисс для всех графиков статики. 
static_vv = [0 for i in range(100)]
x_static = np.linspace(0,100,100)
# Исходные значения для работы логики, можно менять
Gna = 7800
Xa = 0.5
Xg = 0.5
Gg = 1000
Xog_v = 0.8
Xna_v = 30
# Xog = ось ординат для статики по обеднённому газу, из неё же берётся исх. значение концетрации обеднённого газа;
# Xna = ось ординат для статики по насыщенному абсорбенту, из неё же берётся исх. значение концетрации насыщенного абсорбента.
Xog = [Xog_v for i in range(100)]
Xna = [Xna_v for i in range(100)]
# Формулы для расчета Gog и Ga
Gog = (Gg * Xg) / Xog[0]
Ga = (Gna * Xna[0])/Xa



# Служебные функции для отрисовки графиков
def create_plot(x,y):
    plt.cla()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.plot(x, y)
def create_step(x,y):
    plt.cla()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.step(x, y)
def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)

def draw_graph(window, figure_number, x, y, canvas_key, toolbar_key, step=False):
    plt.figure(figure_number)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(428 / DPI, 200 / DPI)
    if step:
        create_step(x, y)
    else:
        create_plot(x, y)
    draw_figure_w_toolbar(window[canvas_key].TKCanvas, fig, window[toolbar_key].TKCanvas)

def draw_static_graphs(window, selected_chain):
    draw_graph(window, 1, x_static, static_vv, 'VV_GRAPH', 'TBVV')
    output = Xog if selected_chain == 1 else Xna
    draw_graph(window, 2, x_static, output, 'DP_GRAPH', 'TBDP')

def reset_disturbance_controls(window):
    for checkbox_key, input_key in zip(events_cbvv, events_invv):
        window[checkbox_key].update(value=False, disabled=True)
        window[input_key].update(value="", disabled=True)
    window["OK"].update(disabled=True)

# Класс для панели под графиком
class Toolbar(NavigationToolbar2Tk):
    toolitems = [t for t in NavigationToolbar2Tk.toolitems]
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)

# Цепь управления и её выбор
chaintext=sg.Text("Цепь управления")
chaincombo=sg.Combo(values = chain, size = (34, 10), key = "CB_CHAIN", enable_events = True, readonly = True)

# Наличие возмущающих воздействий и их выбор
vv=sg.Text("Возмущающие воздействия", justification = "center")
vv_y = sg.Radio("Есть", "VV_R", key = "VV_YES", enable_events = True, disabled = True)
vv_n = sg.Radio("Нет", "VV_R", key = "VV_NO", enable_events = True, disabled = True)
vv_1 = sg.Checkbox("Доля извлекаемого \nкомпонента в исходном газе (Xг)", key = "CB_VV1", enable_events = True, visible = True, disabled = True, size = (27, 1))
vv_2 = sg.Checkbox("Исходный состав абсорбера (Ха)", key = "CB_VV2", enable_events = True, visible = True, disabled = True, size = (27, 1))
vv_3 = sg.Checkbox("Расход газовой смеси/абсорбента", key = "CB_VV3", enable_events = True, visible = True, disabled = True, size = (27, 1))
vv_1input = sg.Input(key = "IN_VV1", size = (7,1), visible = True, disabled = True)
vv_2input = sg.Input(key = "IN_VV2", size = (7,1), visible = True, disabled = True)
vv_3input = sg.Input(key = "IN_VV3", size = (7,1), visible = True, disabled = True)
frame_vv = sg.Frame("", [[vv_1, vv_1input], [vv_2, vv_2input, sg.Button("OK", key = "OK", enable_events = True, disabled = True)], [vv_3, vv_3input]])

# Графики
vv_graph = sg.Canvas(key='VV_GRAPH', size = (100, 100))
tb_vv = sg.Canvas(key = "TBVV", size = (1, 1))
dp_graph = sg.Canvas(key='DP_GRAPH', size = (100, 100))
tb_dp = sg.Canvas(key = "TBDP", size = (1, 1))
# Вывод окна
layout = [[sg.Text(" ", size = (5,1))], 
        [sg.Text(" ", size = (5,1)), chaintext, chaincombo], 
        [sg.Text(" ", size = (5,1))],
        [sg.Text(" ", size = (5,1)), vv, vv_y, vv_n], 
        [sg.Text(" ", size = (5,1)), frame_vv],
        [sg.Text(" ", size = (5,1)), sg.Text("Возмущающие воздействия")], 
        [sg.Text(" ", size = (5,1)),vv_graph],
        [sg.Text(" ", size = (5,1)),tb_vv], 
        [sg.Text(" ", size = (5,1)), sg.Text("Кривая разгона DP")],
        [sg.Text(" ", size = (5,1)),dp_graph],
        [sg.Text(" ", size = (5,1)),tb_dp], 
        [sg.Button("Exit")]]
layout = [[sg.Titlebar('Абсорбция',
                           background_color="#2196f2",
                           text_color='white',
                           k='-TITLEBAR-')]] + layout
window = sg.Window('Абсорбция', layout, finalize = True)


# Обработка событий
while True:
    event, values = window.read()
    # print (event, values)
    if event in (sg.WIN_CLOSED, 'Exit'): 
        break
# Выбор цепи управления и отображение статической характеристики
    if event == "CB_CHAIN":
        if values["CB_CHAIN"] == "Концентрация в обедненном газе":
            sel_chain = 1
        elif values["CB_CHAIN"] == "Концентрация в насыщенном абсорбенте":
            sel_chain = 2
        window["VV_YES"].update(disabled=False)
        window["VV_NO"].update(value=True, disabled=False)
        reset_disturbance_controls(window)
        draw_static_graphs(window, sel_chain)
# Проверка наличия возмущающих воздействий и управление полями ввода
    if event == "VV_NO":
        reset_disturbance_controls(window)
        draw_static_graphs(window, sel_chain)

    if event == "VV_YES":
        reset_disturbance_controls(window)
        component_index = 0 if sel_chain == 1 else 1
        window[events_cbvv[component_index]].update(disabled=False, visible=True)
        window["CB_VV3"].update(disabled=False, visible=True)
        flow_label = "Расход исх. газовой смеси (Gг)" if sel_chain == 1 else "Расход абсорбента (Gа)"
        window["CB_VV3"].update(text=flow_label)

    if event in events_cbvv:
        input_key = events_invv[events_cbvv.index(event)]
        if values[event]:
            window[input_key].update(disabled=False)
        else:
            window[input_key].update(value="", disabled=True)
        active_indices = (0, 2) if sel_chain == 1 else (1, 2)
        has_selected_disturbance = any(values[events_cbvv[index]] for index in active_indices)
        window["OK"].update(disabled=not has_selected_disturbance)

# Изменение цепи управления с возмущающими воздействиями и вывод графиков
    if event == "OK":
        active_indices = (0, 2) if sel_chain == 1 else (1, 2)
        selected_indices = [index for index in active_indices if values[events_cbvv[index]]]
        if not selected_indices:
            sg.popup("Выберите хотя бы одно возмущающее воздействие.")
            continue

        disturbances = [0.0, 0.0, 0.0]
        try:
            for index in selected_indices:
                disturbances[index] = parse_fraction(values[events_invv[index]])
        except ValueError as error:
            sg.popup(str(error))
            window[events_invv[index]].update("")
            continue

        component_index = 0 if sel_chain == 1 else 1
        component_fraction = disturbances[component_index]
        flow_fraction = disturbances[2]
        combined_fraction = combine_fractions(component_fraction, flow_fraction)

        x = [i * 10 for i in range(11)]
        y_start = 3
        y_new = y_start + combined_fraction
        y_graph = np.array([y_start, y_start] + [y_new] * 9)
        draw_graph(window, 1, x, y_graph, 'VV_GRAPH', 'TBVV', step=True)

        if sel_chain == 1:
            output_start = Xog[0]
            output_new = calculate_xog(Gg, Xg, Gog, component_fraction, flow_fraction)
            transition_value = output_new
        else:
            output_start = Xna[0]
            output_new = calculate_xna(Ga, Gna, Xa, component_fraction, flow_fraction)
            transition_value = output_new - 1

        y = [output_start, output_start, transition_value] + [output_new] * 8
        spl = make_interp_spline(x, y, k=2)
        xnew = np.linspace(0, 100, 200)
        y_smooth = spl(xnew)
        y_smooth[y_smooth < output_start] = output_start
        draw_graph(window, 2, xnew, y_smooth, 'DP_GRAPH', 'TBDP')
    
                        




window.close()
