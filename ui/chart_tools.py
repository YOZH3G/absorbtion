"""Time cursor shared by the two current chart figures."""

import numpy as np


class SharedTimeCursor:
    def __init__(self, charts):
        self.charts = charts
        self.markers = {}
        for canvas, _toolbar, _readout in charts:
            canvas.mpl_connect("motion_notify_event", self.move)
            canvas.mpl_connect("figure_leave_event", self.clear)

    def clear(self, _event=None):
        for canvas, _toolbar, readout in self.charts:
            readout.set("")
            for axis in canvas.figure.axes:
                marker = self.markers.get(axis)
                if marker is not None:
                    marker.set_visible(False)
            canvas.draw_idle()

    def move(self, event):
        if event.inaxes is None or event.xdata is None:
            self.clear()
            return
        if any(toolbar.mode for _canvas, toolbar, _readout in self.charts):
            self.clear()
            return
        if not self.is_time_axis(event.inaxes):
            self.clear()
            return
        time = float(event.xdata)
        for canvas, _toolbar, readout in self.charts:
            values = []
            for axis in canvas.figure.axes:
                if not self.is_time_axis(axis):
                    continue
                marker = self.markers.get(axis)
                if marker is None or marker not in axis.lines:
                    marker = axis.axvline(time, color="#64748B", linestyle=":", linewidth=.8,
                                          label="_time_cursor")
                    self.markers[axis] = marker
                marker.set_xdata([time, time])
                marker.set_visible(axis.get_xlim()[0] <= time <= axis.get_xlim()[1])
                for line in axis.lines:
                    label = line.get_label()
                    if label.startswith("_") or not line.get_visible():
                        continue
                    x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
                    if x.size < 2 or np.any(np.diff(x) <= 0) or not x[0] <= time <= x[-1]:
                        continue
                    values.append(f"{label}: {np.interp(time, x, y):.5g}")
            readout.set(f"t = {time:.5g} с; " + "; ".join(values))
            canvas.draw_idle()

    @staticmethod
    def is_time_axis(axis):
        return any(sibling.get_xlabel().startswith("Время")
                   for sibling in axis.get_shared_x_axes().get_siblings(axis))
