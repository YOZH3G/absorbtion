import unittest

from matplotlib.backend_bases import KeyEvent, MouseEvent, PickEvent, ResizeEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from app.plotting import adaptive_legend
from main import DynamicsControlApp


LABELS = ("Исходный режим", "Только состав", "Только расход", "Совместное воздействие", "Без регулятора")


def chart(width):
    figure = Figure(figsize=(width / 100, 3.4), dpi=100, constrained_layout=True)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    for index, label in enumerate(LABELS):
        axis.plot([0, 1], [index, index + 1], label=label)
    axis.axhspan(1, 2, alpha=0.1, label="Полоса ±5%")
    return figure, canvas, axis


class LegendTests(unittest.TestCase):
    def test_legends_fit_narrow_medium_and_wide_charts(self):
        for width in (450, 700, 1000):
            with self.subTest(width=width):
                figure, canvas, axis = chart(width)
                adaptive_legend(axis)
                canvas.draw()
                bounds = axis.get_legend().get_window_extent(canvas.get_renderer())
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertLessEqual(bounds.x1, width)
                self.assertLessEqual(bounds.y1, figure.bbox.height)
                self.assertGreater(axis.get_window_extent().height, figure.bbox.height * 0.5)

    def test_text_keyboard_and_band_toggle_with_visibility_preserved_on_resize(self):
        figure, canvas, axis = chart(1000)
        DynamicsControlApp._place_legend_above(axis)
        DynamicsControlApp._enable_legend_toggles(axis, canvas)
        canvas.draw()
        text = axis.get_legend().get_texts()[0]
        mouse = MouseEvent("button_press_event", canvas, 10, 10, button=1)
        canvas.callbacks.process("pick_event", PickEvent("pick_event", canvas, mouse, text))
        self.assertFalse(axis.lines[0].get_visible())
        self.assertEqual(text.get_alpha(), 0.25)
        figure.set_size_inches(4.5, 3.4)
        canvas.callbacks.process("resize_event", ResizeEvent("resize_event", canvas))
        self.assertFalse(axis.lines[0].get_visible())
        canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="1"))
        self.assertTrue(axis.lines[0].get_visible())
        canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="6"))
        self.assertFalse(axis.patches[0].get_visible())


if __name__ == "__main__":
    unittest.main()
