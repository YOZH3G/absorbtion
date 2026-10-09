"""Time cursor shared by the two current chart figures."""

import numpy as np


class SharedTimeCursor:
    def __init__(self, charts):
        self.charts = charts
        self.markers = {}
        self.backgrounds = {}
        for canvas, _toolbar, _readout in charts:
            canvas.mpl_connect("motion_notify_event", self.move)
            canvas.mpl_connect("figure_leave_event", self.clear)
            canvas.mpl_connect("draw_event", self._drawn)
            canvas.mpl_connect("resize_event", lambda event: self.backgrounds.pop(event.canvas, None))

    def _prune(self):
        axes = {axis for canvas, _toolbar, _readout in self.charts for axis in canvas.figure.axes}
        self.markers = {axis: marker for axis, marker in self.markers.items()
                        if axis in axes and marker in axis.lines}

    def _drawn(self, event):
        self._prune()
        canvas = event.canvas
        if canvas.is_saving():
            self.backgrounds.pop(canvas, None)
            return
        if canvas.supports_blit:
            self.backgrounds[canvas] = canvas.copy_from_bbox(canvas.figure.bbox)
            for axis in canvas.figure.axes:
                marker = self.markers.get(axis)
                if marker is not None and marker.get_visible():
                    axis.draw_artist(marker)

    def _render(self, canvas):
        background = self.backgrounds.get(canvas)
        if background is None or canvas.figure.stale:
            canvas.draw_idle()
            return
        canvas.restore_region(background)
        for axis in canvas.figure.axes:
            marker = self.markers.get(axis)
            if marker is not None and marker.get_visible():
                axis.draw_artist(marker)
        canvas.blit(canvas.figure.bbox)

    def clear(self, _event=None):
        self._prune()
        for canvas, _toolbar, readout in self.charts:
            if readout.get():
                readout.set("")
            changed = False
            for axis in canvas.figure.axes:
                marker = self.markers.get(axis)
                if marker is not None and marker.get_visible():
                    marker.set_visible(False)
                    changed = True
            if changed:
                self._render(canvas)

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
        self._prune()
        for canvas, _toolbar, readout in self.charts:
            values = []
            for axis in canvas.figure.axes:
                if not self.is_time_axis(axis):
                    continue
                marker = self.markers.get(axis)
                if marker is None or marker not in axis.lines:
                    marker = axis.axvline(time, color="#64748B", linestyle=":", linewidth=.8,
                                          label="_time_cursor", animated=canvas.supports_blit)
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
            text = f"t = {time:.5g} с; " + "; ".join(values)
            if readout.get() != text:
                readout.set(text)
            self._render(canvas)

    @staticmethod
    def is_time_axis(axis):
        return any(sibling.get_xlabel().startswith("Время")
                   for sibling in axis.get_shared_x_axes().get_siblings(axis))
