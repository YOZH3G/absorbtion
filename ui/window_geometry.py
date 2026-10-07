"""Keep initial windows within the available primary monitor work area."""

import re
import sys


def work_area(widget):
    if sys.platform == "win32":
        import ctypes
        from ctypes.wintypes import RECT
        area = RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(area), 0):
            return area.left, area.top, area.right, area.bottom
    return 0, 0, widget.winfo_screenwidth(), widget.winfo_screenheight()


def fit_geometry(geometry, area):
    match = re.fullmatch(r"(\d+)x(\d+)(?:([+-]\d+)([+-]\d+))?", geometry)
    if match is None:
        raise ValueError("Invalid window geometry")
    left, top, right, bottom = area
    width = min(max(1, int(match[1])), max(1, right - left - 16))
    height = min(max(1, int(match[2])), max(1, bottom - top - 48))
    x = left + (right - left - width) // 2 if match[3] is None else int(match[3])
    y = top + 8 if match[4] is None else int(match[4])
    x = min(max(left, x), max(left, right - width - 8))
    y = min(max(top, y), max(top, bottom - height - 40))
    return f"{width}x{height}{x:+d}{y:+d}"
