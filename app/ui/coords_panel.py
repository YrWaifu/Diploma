from __future__ import annotations
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np


class CoordinatesPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(180)
        self.setMaximumWidth(250)
        self._sticks_data: dict[int, dict] = {}
        self._cursor_x = 0.0
        self._cursor_y_pixel = 0
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QtGui.QColor(250, 250, 250))
        self.setPalette(pal)

    def update_stick_data(self, idx: int, color: QtGui.QColor, data_min: float, data_max: float,
                          y_pixel_top: int, y_pixel_bottom: int, x_data, y_data,
                          name: str = ""):
        self._sticks_data[idx] = {
            'color': color,
            'data_min': data_min,
            'data_max': data_max,
            'y_pixel_top': y_pixel_top,
            'y_pixel_bottom': y_pixel_bottom,
            'x_data': x_data,
            'y_data': y_data,
            'name': name,
        }
        self.update()

    def update_cursor_position(self, x_value: float, y_pixel: float):
        self._cursor_x = x_value
        self._cursor_y_pixel = y_pixel
        self.update()

    def clear_all(self):
        self._sticks_data.clear()
        self.update()

    def _calculate_y_value(self, idx: int):
        data = self._sticks_data.get(idx)
        if not data:
            return None
        x_data = data['x_data']
        y_data = data['y_data']
        if len(x_data) == 0:
            return None
        # Интерполяция Y в точке курсора (ось времени монотонна, searchsorted)
        try:
            x_arr = np.asarray(x_data)
            y_arr = np.asarray(y_data)
            i = int(np.searchsorted(x_arr, self._cursor_x, side='left'))
            if i <= 0:
                return float(y_arr[0])
            if i >= x_arr.size:
                return float(y_arr[-1])
            x0 = float(x_arr[i - 1]); x1 = float(x_arr[i])
            y0 = float(y_arr[i - 1]); y1 = float(y_arr[i])
            if x1 == x0:
                return y0
            t = (self._cursor_x - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))
        except Exception:
            idx_closest = min(range(len(x_data)), key=lambda j: abs(x_data[j] - self._cursor_x))
            return float(y_data[idx_closest])

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        font_title = p.font(); font_title.setPointSize(10); font_title.setBold(True)
        p.setFont(font_title); p.setPen(QtGui.QColor(60, 60, 60))
        p.drawText(10, 20, "Координаты")
        font_normal = p.font(); font_normal.setPointSize(9); font_normal.setBold(False)
        p.setFont(font_normal); p.setPen(QtGui.QColor(80, 80, 80))
        p.drawText(10, 45, f"X: {self._cursor_x:.4f}")
        y_offset = 70
        avail_w = max(40, self.width() - 28 - 6)  # ширина, доступная для текста
        fm = p.fontMetrics()
        for idx in sorted(self._sticks_data.keys()):
            data = self._sticks_data[idx]
            color = data['color']
            y_value = self._calculate_y_value(idx)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QBrush(color))
            p.drawRect(10, y_offset - 8, 12, 12)
            p.setPen(color); p.setFont(font_normal)
            series_name = data.get('name') or f"График {idx + 1}"
            # Имя ряда — отдельной строкой, сокращённое при необходимости.
            elided_name = fm.elidedText(series_name, QtCore.Qt.ElideMiddle, avail_w)
            p.drawText(28, y_offset + 3, elided_name)
            # Значение под именем.
            val_text = f"{y_value:.4f}" if y_value is not None else "---"
            p.drawText(28, y_offset + 18, val_text)
            p.setPen(QtGui.QColor(120, 120, 120))
            range_text = f"[{data['data_min']:.2g} … {data['data_max']:.2g}]"
            p.drawText(28, y_offset + 33, range_text)
            y_offset += 52
