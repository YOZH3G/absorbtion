import PySimpleGUI as sg
import numpy as np
import math
from matplotlib import use as use_agg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

sg.set_options(font=("Arial Bold",14))
chain = ["Концентрация в обедненном газе","Концентрация в насыщенном абсорбенте"]
sel_chain = "Концентрация в обедненном газе"
x = np.arange(0,4*np.pi,0.1)
current_func = np.sin(x)
func_a = np.sin(x)
func_b = np.cos(x)

#Служебные функции для создания графиков
def create_plot(x,y,title):
    ax.cla()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid()
    plt.plot(x, y)
    fig.canvas.draw()
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

empty_space = sg.Text(" ", size = (5,1))
#Цепь управления и её выбор
chaintext=sg.Text("Цепь управления")
chaincombo=sg.Combo(values = chain, default_value = sel_chain, size = (34, 10), key = "CB_CHAIN", enable_events = True, readonly = True)

#Наличие возмущающих воздействий и их выбор
vv=sg.Text("Возмущающие воздействия", justification = "center")
vv_y = sg.Radio("Есть", "VV_R", key = "VV_YES", enable_events = True)
vv_n = sg.Radio("Нет", "VV_R", key = "VV_NO", enable_events = True, default = True)
vv_1 = sg.Checkbox("Доля извлекаемого \nкомпонента в исходном газе (Xг)", key = "CB_VV1", enable_events = True, visible = True, disabled = True, size = (27, 1))
vv_2 = sg.Checkbox("Исходный состав абсорбера (Ха)", key = "CB_VV2", enable_events = True, visible = True, disabled = True, size = (27, 1))
vv_3 = sg.Checkbox("Расход газовой смеси", key = "CB_VV3", enable_events = True, visible = True, disabled = True, size = (27, 1))
vv_1input = sg.Input(key = "IN_VV1", size = (7,1), visible = True, disabled = True)
vv_2input = sg.Input(key = "IN_VV2", size = (7,1), visible = True, disabled = True)
vv_3input = sg.Input(key = "IN_VV3", size = (7,1), visible = True, disabled = True)
frame_vv = sg.Frame("", [[vv_1, vv_1input], [vv_2, vv_2input], [vv_3, vv_3input]])

#Графики
vv_graph = sg.Graph(canvas_size = (200, 200), graph_bottom_left=(0, 0), graph_top_right=(200, 200), key='VV_GRAPH')
dp_graph = sg.Graph((640, 480), (0, 0), (200, 200), key='DP_GRAPH')

#Форма
helloworld = "hello"
layout = [[sg.Text(" ", size = (5,1))], 
        [sg.Text(" ", size = (5,1)), chaintext, chaincombo], 
        [sg.Text(" ", size = (5,1))],
        [sg.Text(" ", size = (5,1)), vv, vv_y, vv_n], 
        [sg.Text(" ", size = (5,1)), frame_vv],
        [sg.Text(" ", size = (5,1)), sg.Text("Возмущающие воздействия")], 
        [sg.Text(" ", size = (5,1)),vv_graph], 
        [sg.Button("Exit")]]


window = sg.Window('Абсорбция', layout, finalize = True)

fig, ax = plt.subplots(figsize=(3, 2))


canvas = FigureCanvasTkAgg(fig, window['VV_GRAPH'].Widget)
plot_widget = canvas.get_tk_widget()
plot_widget.grid(row=0, column=0)

#Обработка событий
while True:
    event, values = window.read()
    print (event, values)
    if event in (sg.WIN_CLOSED, 'Exit'): 
        break
#Выбор цепи управления
    if values["CB_CHAIN"] == "Концентрация в обедненном газе":
        current_func = func_a
        sel_chain = values[event]
        create_plot(x, current_func, "Возмущающие воздействия")
    if values["CB_CHAIN"] == "Концентрация в насыщенном абсорбенте":
        plt.cla()
        current_func = func_b
        sel_chain = values[event]
        create_plot(x, current_func, "Возмущающие воздействия")
#Проверка на наличие возмущающих воздействий и их задача в случае наличия
    if values["CB_VV1"] == True:
        window.Element("IN_VV1").update(disabled = False)
    else:
        window.Element("IN_VV1").update(disabled = True)
        window.Element("IN_VV1").update("")

    if values["CB_VV2"] == True:
        window.Element("IN_VV2").update(disabled = False)
    else:
        window.Element("IN_VV2").update(disabled = True)
        window.Element("IN_VV2").update("")

    if values["CB_VV3"] == True:
        window.Element("IN_VV3").update(disabled = False)
    else:
        window.Element("IN_VV3").update(disabled = True)
        window.Element("IN_VV3").update("")

    if event == "VV_NO":
        window.Element("CB_VV1").update(disabled = True)
        window.Element("IN_VV1").update(disabled = True)
        window.Element("CB_VV1").update(False)
        window.Element("CB_VV2").update(disabled = True)
        window.Element("IN_VV2").update(disabled = True)
        window.Element("CB_VV2").update(False)
        window.Element("CB_VV3").update(disabled = True)
        window.Element("IN_VV3").update(disabled = True)
        window.Element("CB_VV3").update(False)

    if event == "VV_YES":
        window.Element("CB_VV1").update(disabled = False, visible = True)
        window.Element("CB_VV2").update(disabled = False, visible = True)
        window.Element("CB_VV3").update(disabled = False, visible = True)




window.close()