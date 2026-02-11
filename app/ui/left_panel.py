from __future__ import annotations
from typing import Optional
from PyQt5 import QtCore, QtGui, QtWidgets


class Stick:
    __slots__ = ("x", "y1", "y2", "color", "data_min", "data_max", "name")

    def __init__(
        self,
        x: int,
        y1: int,
        y2: int,
        color: Optional[QtGui.QColor] = None,
        data_min: float = -1.0,
        data_max: float = 1.0,
        name: str = "",
    ):
        self.x = int(x)
        self.y1 = int(min(y1, y2))
        self.y2 = int(max(y1, y2))
        self.color = color or QtGui.QColor(50, 90, 200)
        self.data_min = float(data_min)
        self.data_max = float(data_max)
        self.name = str(name)


class DotsCanvas(QtWidgets.QWidget):
    stickCreated = QtCore.pyqtSignal(int, int, int, QtGui.QColor)  # idx, y1, y2, color
    stickUpdated = QtCore.pyqtSignal(int, int, int)                # idx, y1, y2

    def __init__(self, is_left: bool = False, parent=None, grid_spacing: int = 18, grid_margin: int = 12, dot_size: int = 2):
        super().__init__(parent)
        self.is_left = is_left
        self.setMouseTracking(True)
        self.setMinimumSize(220, 220)

        self._sticks: list[Stick] = []
        self._draft_stick: Optional[Stick] = None
        self._active_idx: Optional[int] = None
        self._drag_mode: Optional[str] = None  # 'top' | 'bottom' | 'move' | None
        self._last_mouse_pos: Optional[QtCore.QPoint] = None

        self._grid_spacing = int(grid_spacing)
        self._margin = int(grid_margin)
        self._dot_size = int(dot_size)
        self._grid_points: list[QtCore.QPoint] = []
        self._recalc_grid()

    # public API
    @property
    def margin(self) -> int:
        return self._margin

    def stick_count(self) -> int:
        return len(self._sticks)

    def add_stick(self, stick: Stick) -> int:
        self._sticks.append(stick)
        self.update()
        return len(self._sticks) - 1

    def get_stick(self, idx: int) -> Optional[Stick]:
        """Возвращает стик по индексу или None."""
        if 0 <= idx < len(self._sticks):
            return self._sticks[idx]
        return None

    def clear_all(self) -> None:
        self._sticks.clear()
        self._draft_stick = None
        self._active_idx = None
        self._drag_mode = None
        self._last_mouse_pos = None
        self.update()

    # grid
    def _recalc_grid(self) -> None:
        w = max(1, self.width())
        h = max(1, self.height())
        m = self._margin
        s = max(4, self._grid_spacing)
        pts: list[QtCore.QPoint] = []
        y = m
        while y <= h - m:
            x = m
            while x <= w - m:
                pts.append(QtCore.QPoint(x, y))
                x += s
            y += s
        self._grid_points = pts

    # hit test
    def _hit_stick(self, pos: QtCore.QPoint):
        if not self._sticks:
            return None, None
        x_tol, y_tol = 6, 8
        for i, s in enumerate(self._sticks):
            if abs(pos.x() - s.x) <= x_tol and s.y1 - y_tol <= pos.y() <= s.y2 + y_tol:
                if abs(pos.y() - s.y1) <= y_tol:
                    return i, 'top'
                if abs(pos.y() - s.y2) <= y_tol:
                    return i, 'bottom'
                return i, 'move'
        return None, None

    # input
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if not self.is_left or e.button() != QtCore.Qt.LeftButton:
            return
        idx, part = self._hit_stick(e.pos())
        if idx is not None and part in ('top', 'bottom', 'move'):
            self._active_idx = idx
            self._drag_mode = part
            self._last_mouse_pos = e.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor if part == 'move' else QtCore.Qt.SizeVerCursor)
            return

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if not self.is_left:
            return
        if self._drag_mode and self._active_idx is not None:
            s = self._sticks[self._active_idx]
            h, w = self.height(), self.width()
            m = self._margin
            if self._last_mouse_pos is None:
                self._last_mouse_pos = e.pos()

            if self._drag_mode == 'move':
                dx = int(e.pos().x() - self._last_mouse_pos.x())
                dy = int(e.pos().y() - self._last_mouse_pos.y())
                self._last_mouse_pos = e.pos()
                span = s.y2 - s.y1
                new_y1 = max(m, min(h - m - span, s.y1 + dy))
                new_x = max(m, min(w - m, s.x + dx))
                s.x, s.y1, s.y2 = new_x, new_y1, new_y1 + span
            elif self._drag_mode == 'top':
                y = max(m, min(h - m, e.pos().y()))
                s.y1 = min(y, s.y2)
            elif self._drag_mode == 'bottom':
                y = max(m, min(h - m, e.pos().y()))
                s.y2 = max(y, s.y1)

            self.update()
            self.stickUpdated.emit(self._active_idx, s.y1, s.y2)
            return

        idx, part = self._hit_stick(e.pos())
        if idx is not None:
            self.setCursor(QtCore.Qt.OpenHandCursor if part == 'move' else QtCore.Qt.SizeVerCursor)
        else:
            self.unsetCursor()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if not self.is_left or e.button() != QtCore.Qt.LeftButton:
            return
        if self._drag_mode and self._active_idx is not None:
            s = self._sticks[self._active_idx]
            self.stickUpdated.emit(self._active_idx, s.y1, s.y2)
            self._drag_mode = None
            self._active_idx = None
            self._last_mouse_pos = None
            self.unsetCursor()

    # paint
    def paint_grid(self, p: QtGui.QPainter):
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255))
        if self._grid_points:
            pen = QtGui.QPen(QtGui.QColor(220, 220, 220))
            pen.setWidth(max(1, self._dot_size))
            p.setPen(pen)
            p.drawPoints(QtGui.QPolygon(self._grid_points))

    def _draw_handle(self, p: QtGui.QPainter, pt: QtCore.QPoint, color: QtGui.QColor):
        r = 5
        pen = QtGui.QPen(color, 2)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        p.setPen(pen)
        p.setBrush(brush)
        p.drawEllipse(QtCore.QRect(pt.x() - r, pt.y() - r, 2 * r, 2 * r))

    def _draw_scale(self, p: QtGui.QPainter, x: int, y_top: int, y_bottom: int, color: QtGui.QColor,
                    data_min: float, data_max: float):
        span = y_bottom - y_top
        data_range = max(1e-9, data_max - data_min)
        num_ticks = 2 if span < 80 else 3 if span < 140 else 5 if span < 220 else 9
        ticks = [data_min + (data_max - data_min) * i / (num_ticks - 1) for i in range(num_ticks)]
        tick_len = 6
        scale_x0 = x - 12
        pen_tick = QtGui.QPen(color)
        pen_tick.setWidth(1)
        p.setPen(pen_tick)
        for val in ticks:
            t = (val - data_min) / data_range
            y = int(y_top + (1 - t) * span)
            p.drawLine(scale_x0 - tick_len, y, scale_x0, y)
            rect = QtCore.QRect(scale_x0 - tick_len - 36, y - 10, 34, 20)
            p.drawText(rect, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, f"{val:.2g}")

    def paint_sticks(self, p: QtGui.QPainter):
        font = p.font()
        font.setPointSize(9)
        p.setFont(font)
        for s in self._sticks:
            pen = QtGui.QPen(s.color, 3)
            p.setPen(pen)
            x = max(self._margin, min(self.width() - self._margin, s.x))
            y1 = max(self._margin, min(self.height() - self._margin, s.y1))
            y2 = max(self._margin, min(self.height() - self._margin, s.y2))
            p.drawLine(x, y1, x, y2)
            self._draw_handle(p, QtCore.QPoint(x, y1), s.color)
            self._draw_handle(p, QtCore.QPoint(x, y2), s.color)
            self._draw_scale(p, x, y1, y2, s.color, s.data_min, s.data_max)
            # Подпись графика вдоль стика (вертикально, снизу вверх).
            if s.name:
                p.save()
                try:
                    scale_x0 = x - 12
                    text_x = scale_x0 + 40
                    text_y = int((y1 + y2) / 2)

                    # Доступная длина текста = высота стика (текст повёрнут на –90°)
                    avail = max(30, y2 - y1)
                    fm = p.fontMetrics()
                    elided = fm.elidedText(s.name, QtCore.Qt.ElideMiddle, avail)

                    p.translate(text_x, text_y)
                    p.rotate(-90)
                    half = avail // 2
                    rect = QtCore.QRect(-half, -20, avail, 40)
                    p.setPen(QtGui.QPen(s.color))
                    p.drawText(rect, QtCore.Qt.AlignCenter, elided)
                finally:
                    p.restore()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.paint_grid(p)
        if self.is_left:
            self.paint_sticks(p)

    def resizeEvent(self, e):
        self._recalc_grid()
        super().resizeEvent(e)


