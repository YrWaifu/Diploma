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
                          y_pixel_top: int, y_pixel_bottom: int, x_data, y_data):
        self._sticks_data[idx] = {
            'color': color,
            'data_min': data_min,
            'data_max': data_max,
            'y_pixel_top': y_pixel_top,
            'y_pixel_bottom': y_pixel_bottom,
            'x_data': x_data,
            'y_data': y_data,
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
        # интерполяция ближайшего значения (векторизовано)
        try:
            x_arr = np.asarray(x_data)
            y_arr = np.asarray(y_data)
            return float(np.interp(self._cursor_x, x_arr, y_arr))
        except Exception:
            # fallback на поиск ближайшей точки
            idx_closest = min(range(len(x_data)), key=lambda i: abs(x_data[i] - self._cursor_x))
            return y_data[idx_closest]

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
        for idx in sorted(self._sticks_data.keys()):
            data = self._sticks_data[idx]
            color = data['color']
            y_value = self._calculate_y_value(idx)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QBrush(color))
            p.drawRect(10, y_offset - 8, 12, 12)
            p.setPen(color); p.setFont(font_normal)
            text = f"График {idx + 1}: {y_value:.4f}" if y_value is not None else f"График {idx + 1}: ---"
            p.drawText(28, y_offset + 3, text)
            p.setPen(QtGui.QColor(120, 120, 120))
            range_text = f"  [{data['data_min']:.2g} ... {data['data_max']:.2g}]"
            p.drawText(28, y_offset + 18, range_text)
            y_offset += 40