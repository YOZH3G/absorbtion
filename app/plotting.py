"""Legends shared by interactive charts and exported reports."""

import textwrap

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties


def adaptive_legend(axis, handles=None, labels=None):
    if handles is None:
        handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return None
    figure = axis.figure
    available = max(100, figure.bbox.width - 110)
    font = FontProperties(size=8)
    renderer = RendererAgg(int(figure.bbox.width), int(figure.bbox.height), figure.dpi)
    display = []
    for label in labels:
        width = max(1, int(available / (4.8 * figure.dpi / 100)))
        wrapped = textwrap.fill(str(label), width=width)
        while width > 1 and max(renderer.get_text_width_height_descent(line, font, False)[0]
                                for line in wrapped.splitlines()) > available - 44 * figure.dpi / 100:
            width -= 1
            wrapped = textwrap.fill(str(label), width=width)
        display.append(wrapped)
    widest = max(renderer.get_text_width_height_descent(line, font, False)[0]
                 for label in display for line in label.splitlines())
    columns = min(len(handles), max(1, int(available / (widest + 44 * figure.dpi / 100))))
    legend = axis.legend(handles, display, loc="lower center", bbox_to_anchor=(0.5, 1.02),
                         frameon=False, fontsize=8, ncol=columns, borderaxespad=0,
                         columnspacing=1.4, handletextpad=0.6)
    legend._source_labels = tuple(labels)
    return legend
