# -*- coding: utf-8 -*-
# requirements: PyQt5, pyqtgraph, pandas, openpyxl
# run: python app.py

import sys
import math
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pandas as pd


# --------------------------
# МОДЕЛЬ ПАЛКИ (левая панель)
# --------------------------
class Stick:
    __slots__ = ("x", "y1", "y2", "color", "data_min", "data_max")

    def __init__(self, x: int, y1: int, y2: int, color: QtGui.QColor = None, data_min: float = -1.0,
                 data_max: float = 1.0):
        self.x = int(x)
        self.y1 = int(min(y1, y2))
        self.y2 = int(max(y1, y2))
        self.color = color or QtGui.QColor(50, 90, 200)
        self.data_min = float(data_min)
        self.data_max = float(data_max)


# ------------------------------------
# ЛЕВАЯ ПАНЕЛЬ: кастомная, как и раньше
# ------------------------------------
class DotsCanvas(QtWidgets.QWidget):
    """
    Белый фон, точечная сетка. Можно:
    - создать палку (ЛКМ и потянуть),
    - тянуть за верх/низ,
    - двигать целиком по X и по Y за центр.
    У каждой палки рисуется своя шкала [-1..1] с адаптивными подписями и её же цветом.
    """
    stickCreated = QtCore.pyqtSignal(int, int, int, QtGui.QColor)  # idx, y1, y2, color
    stickUpdated = QtCore.pyqtSignal(int, int, int)  # idx, y1, y2

    def __init__(self, is_left=False, parent=None, grid_spacing=18, grid_margin=12, dot_size=2):
        super().__init__(parent)
        self.is_left = is_left
        self.setMouseTracking(True)
        self.setMinimumSize(220, 220)

        # палки
        self._sticks = []
        self._draft_stick = None
        self._active_idx = None
        self._drag_mode = None  # 'top' | 'bottom' | 'move' | None
        self._last_mouse_pos = None  # QPoint

        # сетка-точки
        self._grid_spacing = int(grid_spacing)
        self._margin = int(grid_margin)
        self._dot_size = int(dot_size)
        self._grid_points = []
        self._recalc_grid()

    # API
    def add_stick(self, stick: Stick):
        self._sticks.append(stick)
        self.update()
        return len(self._sticks) - 1

    def clear_all(self):
        self._sticks.clear()
        self._draft_stick = None
        self._active_idx = None
        self._drag_mode = None
        self._last_mouse_pos = None
        self.update()

    # сетка
    def _recalc_grid(self):
        w = max(1, self.width())
        h = max(1, self.height())
        m = self._margin
        s = max(4, self._grid_spacing)
        pts = []
        y = m
        while y <= h - m:
            x = m
            while x <= w - m:
                pts.append(QtCore.QPoint(x, y))
                x += s
            y += s
        self._grid_points = pts

    # хиттест палки
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

    # ввод
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
        # Создание новой палки отключено
        # pos = e.pos()
        # self._draft_stick = Stick(pos.x(), pos.y(), pos.y())
        # self.update()

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

        # Создание новой палки отключено
        # if self._draft_stick:
        #     self._draft_stick.y2 = e.pos().y()
        #     self.update()
        #     return

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
            return
        # Создание новой палки отключено
        # if self._draft_stick:
        #     hue = (len(self._sticks) * 37) % 360
        #     color = QtGui.QColor.fromHsv(hue, 220, 220)
        #     self._draft_stick.color = color
        #     idx = self.add_stick(self._draft_stick)
        #     s = self._sticks[idx]
        #     self._draft_stick = None
        #     self.update()
        #     self.stickCreated.emit(idx, s.y1, s.y2, color)

    # рендер
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

    def _draw_scale(self, p: QtGui.QPainter, x: int, y_top: int, y_bottom: int, color: QtGui.QColor, data_min: float,
                    data_max: float):
        span = y_bottom - y_top
        data_range = data_max - data_min
        if data_range < 1e-9:
            data_range = 1.0

        # Определяем количество меток в зависимости от размера
        if span < 80:
            num_ticks = 2
        elif span < 140:
            num_ticks = 3
        elif span < 220:
            num_ticks = 5
        else:
            num_ticks = 9

        # Создаем метки равномерно распределенные между data_min и data_max
        ticks = [data_min + (data_max - data_min) * i / (num_ticks - 1) for i in range(num_ticks)]

        tick_len = 6
        scale_x0 = x - 12
        pen_tick = QtGui.QPen(color)
        pen_tick.setWidth(1)
        p.setPen(pen_tick)
        for val in ticks:
            # Нормализуем значение в диапазон [0, 1]
            t = (val - data_min) / data_range
            # Инвертируем, так как y_top находится вверху
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

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.paint_grid(p)
        if self.is_left:
            self.paint_sticks(p)

    def resizeEvent(self, e):
        self._recalc_grid()
        super().resizeEvent(e)


# ---------------------------------------------------------
# ПРАВАЯ ПАНЕЛЬ: pyqtgraph с кастомным ViewBox для зума ЛКМ
# ---------------------------------------------------------
class XZoomViewBox(pg.ViewBox):
    """
    Переопределяем поведение:
    - ЛКМ: зум по X с помощью двух вертикальных линий.
    - ПКМ: по умолчанию контекстное меню отключено, но оставим как есть.
    """
    sigZoomRange = QtCore.pyqtSignal(float, float)  # x_min, x_max для зума

    def __init__(self, *args, **kwargs):
        super().__init__(*args, enableMenu=False, **kwargs)
        self._zoom_line1 = None  # первая вертикальная линия
        self._zoom_line2 = None  # вторая вертикальная линия
        self.setMouseEnabled(x=False, y=False)  # сами рулём

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            if ev.isStart():
                # Начало - создаем первую линию
                self._startPos = ev.pos()
                start_point = self.mapSceneToView(QtCore.QPointF(self._startPos))

                if self._zoom_line1 is None:
                    self._zoom_line1 = pg.InfiniteLine(
                        pos=start_point.x(),
                        angle=90,
                        movable=False,
                        pen=pg.mkPen('blue', width=2, style=QtCore.Qt.DashLine)
                    )
                    self.addItem(self._zoom_line1)
                else:
                    self._zoom_line1.setPos(start_point.x())
                    self._zoom_line1.show()

                if self._zoom_line2 is None:
                    self._zoom_line2 = pg.InfiniteLine(
                        pos=start_point.x(),
                        angle=90,
                        movable=False,
                        pen=pg.mkPen('blue', width=2, style=QtCore.Qt.DashLine)
                    )
                    self.addItem(self._zoom_line2)
                else:
                    self._zoom_line2.setPos(start_point.x())
                    self._zoom_line2.show()

            elif ev.isFinish():
                # Завершение - применяем зум
                if not hasattr(self, "_startPos"):
                    if self._zoom_line1:
                        self._zoom_line1.hide()
                    if self._zoom_line2:
                        self._zoom_line2.hide()
                    return

                start = self._startPos
                end = ev.pos()

                # Скрываем линии
                if self._zoom_line1:
                    self._zoom_line1.hide()
                if self._zoom_line2:
                    self._zoom_line2.hide()

                # Проверяем минимальное расстояние
                if abs(end.x() - start.x()) < 6:
                    return

                # Конвертируем в координаты данных
                p0 = self.mapSceneToView(QtCore.QPointF(start))
                p1 = self.mapSceneToView(QtCore.QPointF(end))
                xmin, xmax = sorted([p0.x(), p1.x()])

                # Посылаем сигнал с новым диапазоном
                self.sigZoomRange.emit(float(xmin), float(xmax))
            else:
                # Перемещение - обновляем вторую линию
                if self._zoom_line2 and hasattr(self, "_startPos"):
                    end_point = self.mapSceneToView(QtCore.QPointF(ev.pos()))
                    self._zoom_line2.setPos(end_point.x())
        else:
            # игнорим другие кнопки — пусть PlotWidget обработает по умолчанию (колесо и т.п.)
            ev.ignore()


class DataPlot(pg.PlotWidget):
    """
    Правая зона на pyqtgraph:
    - Отображение данных из файлов.
    - Общий X в [-R..R] с нулём в центре.
    - ЛКМ-резинка: зум по X.
    - По вертикали каждая кривая живёт в своём пиксельном диапазоне (invertY=True).
    - Перекрестие при движении мыши.
    """
    sigXHalfRangeChanged = QtCore.pyqtSignal(float)
    sigCursorMoved = QtCore.pyqtSignal(float, int)  # x_value, y_pixel

    def __init__(self, parent=None):
        # 1) ВАЖНО: состояние — ДО super().__init__()
        self.vb = XZoomViewBox()
        self._curves = {}  # idx -> PlotDataItem
        self._plots = {}  # idx -> dict(x_data, y_data, y_top, y_bottom, color)
        self._x_half_range = 10.0  # для совместимости
        self._x_min = -10.0  # левая граница отображения
        self._x_max = 10.0  # правая граница отображения
        self._axis_band = 24
        self._axis_font_px = 9
        self._initialized = False  # флажок готовности

        # Перекрестие
        self._crosshair_enabled = True
        self._crosshair_pos = None  # QPointF в координатах данных

        # 2) Конструируем PlotWidget
        super().__init__(parent=parent, viewBox=self.vb, background="w")
        pg.setConfigOptions(antialias=True)

        # 3) Оформление
        self.showGrid(x=True, y=True, alpha=0.15)
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().hideAxis('left')
        ax = self.getPlotItem().getAxis('bottom')
        ax.setPen(pg.mkPen(120, 120, 120))
        ax.setTextPen(pg.mkPen(120, 120, 120))

        # Y сверху вниз как в левой панели
        self.getViewBox().invertY(True)

        # Линии перекрестия
        self._vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
        self._hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
        self.addItem(self._vLine, ignoreBounds=True)
        self.addItem(self._hLine, ignoreBounds=True)
        self._vLine.setVisible(False)
        self._hLine.setVisible(False)

        # 4) Сигналы
        self.vb.sigZoomRange.connect(self.set_x_range_direct)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

        # 5) Финальный запуск
        self._initialized = True
        self._apply_ranges()
        self._replot_all()

    # публичные параметры
    def set_x_half_range(self, R: float):
        """Устанавливает симметричный диапазон [-R, R] (для совместимости)"""
        self._x_half_range = max(0.001, float(R))
        self._x_min = -self._x_half_range
        self._x_max = self._x_half_range
        if self._initialized:
            self._apply_ranges()
            self._replot_all()
        self.sigXHalfRangeChanged.emit(self._x_half_range)

    def set_x_range_direct(self, xmin: float, xmax: float):
        """Устанавливает диапазон X напрямую (используется при зуме)"""
        self._x_min = float(xmin)
        self._x_max = float(xmax)
        # Обновляем полудиапазон для совместимости (для спинбокса)
        self._x_half_range = max(abs(xmin), abs(xmax))
        if self._initialized:
            self._apply_ranges()
            self._replot_all()
        # НЕ эмитим сигнал, чтобы не было циклической связи со спинбоксом

    # данные от левой панели
    def add_or_update_plot(self, idx: int, x_data, y_data, y1: int, y2: int, color: QtGui.QColor):
        self._plots[idx] = dict(x_data=x_data, y_data=y_data, y_top=min(y1, y2), y_bottom=max(y1, y2), color=color)
        if idx not in self._curves:
            self._curves[idx] = self.plot(pen=pg.mkPen(color, width=2))
        else:
            self._curves[idx].setPen(pg.mkPen(color, width=2))
        if self._initialized:
            self._replot(idx)

    def update_plot_v_range(self, idx: int, y1: int, y2: int):
        if idx in self._plots:
            self._plots[idx]['y_top'] = min(y1, y2)
            self._plots[idx]['y_bottom'] = max(y1, y2)
            if self._initialized:
                self._replot(idx)

    def clear_all(self):
        for c in self._curves.values():
            self.removeItem(c)
        self._curves.clear()
        self._plots.clear()

    # обработка перекрестия
    def _on_mouse_moved(self, pos):
        if not self._crosshair_enabled:
            return
        # pos это QPointF в координатах сцены
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.getViewBox().mapSceneToView(pos)
            self._crosshair_pos = mouse_point
            self._vLine.setPos(mouse_point.x())
            self._hLine.setPos(mouse_point.y())
            self._vLine.setVisible(True)
            self._hLine.setVisible(True)
            # Посылаем сигнал с координатами
            self.sigCursorMoved.emit(mouse_point.x(), mouse_point.y())
        else:
            self._vLine.setVisible(False)
            self._hLine.setVisible(False)
            self._crosshair_pos = None

    # внутреннее
    def _apply_ranges(self):
        # На случай ранних вызовов до полной инициализации
        if not hasattr(self, "_x_min") or not hasattr(self, "_x_max"):
            return
        self.setXRange(self._x_min, self._x_max, padding=0)
        H = max(1, self.viewport().height())
        self.setYRange(0, H, padding=0)

    def resizeEvent(self, e):
        # сначала базовый обработчик, потом наша логика
        super().resizeEvent(e)
        if not self._initialized:
            return
        self._apply_ranges()
        self._replot_all()

    def _replot_all(self):
        for idx in list(self._plots.keys()):
            self._replot(idx)

    def _replot(self, idx: int):
        if idx not in self._plots or idx not in self._curves:
            return
        plot_data = self._plots[idx]
        y_top, y_bot = plot_data["y_top"], plot_data["y_bottom"]
        span = max(1, y_bot - y_top)

        x_data = plot_data['x_data']
        y_data = plot_data['y_data']

        if y_data.size == 0:
            self._curves[idx].setData([], [])
            return

        y_min, y_max = min(y_data), max(y_data)
        y_data_span = y_max - y_min
        if y_data_span < 1e-9:
            y_data_span = 1.0

        # Нормализуем Y в [0,1], затем растягиваем на [y_top, y_bot]
        # Инвертируем, чтобы максимум был вверху (как на шкале слева)
        ys = [y_top + ((y_max - y) / y_data_span) * span for y in y_data]

        self._curves[idx].setData(x_data, ys)


# ------------------------------------------------
# ПАНЕЛЬ КООРДИНАТ (справа): отображение значений
# ------------------------------------------------
class CoordinatesPanel(QtWidgets.QWidget):
    """
    Панель справа, показывает значения Y для каждой шкалы при текущей позиции курсора.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(180)
        self.setMaximumWidth(250)

        # Данные
        self._sticks_data = {}  # idx -> dict(color, data_min, data_max, y_pixel_top, y_pixel_bottom)
        self._cursor_x = 0.0
        self._cursor_y_pixel = 0

        # Фон
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QtGui.QColor(250, 250, 250))
        self.setPalette(pal)

    def update_stick_data(self, idx: int, color: QtGui.QColor, data_min: float, data_max: float,
                          y_pixel_top: int, y_pixel_bottom: int, x_data, y_data):
        """Обновляет информацию о палке/графике"""
        self._sticks_data[idx] = {
            'color': color,
            'data_min': data_min,
            'data_max': data_max,
            'y_pixel_top': y_pixel_top,
            'y_pixel_bottom': y_pixel_bottom,
            'x_data': x_data,
            'y_data': y_data
        }
        self.update()

    def update_cursor_position(self, x_value: float, y_pixel: float):
        """Обновляет позицию курсора"""
        self._cursor_x = x_value
        self._cursor_y_pixel = y_pixel
        self.update()

    def clear_all(self):
        """Очистить все данные"""
        self._sticks_data.clear()
        self.update()

    def _calculate_y_value(self, idx: int) -> float:
        """Вычисляет значение Y для данной палки в позиции курсора"""
        if idx not in self._sticks_data:
            return None

        data = self._sticks_data[idx]
        x_data = data['x_data']
        y_data = data['y_data']

        # Находим ближайшую точку по X
        if len(x_data) == 0:
            return None

        # Простая интерполяция - находим ближайшую точку
        idx_closest = None
        min_dist = float('inf')
        for i, x_val in enumerate(x_data):
            dist = abs(x_val - self._cursor_x)
            if dist < min_dist:
                min_dist = dist
                idx_closest = i

        if idx_closest is not None:
            return y_data[idx_closest]
        return None

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        # Заголовок
        font_title = p.font()
        font_title.setPointSize(10)
        font_title.setBold(True)
        p.setFont(font_title)
        p.setPen(QtGui.QColor(60, 60, 60))
        p.drawText(10, 20, "Координаты")

        # Значение X
        font_normal = p.font()
        font_normal.setPointSize(9)
        font_normal.setBold(False)
        p.setFont(font_normal)
        p.setPen(QtGui.QColor(80, 80, 80))
        p.drawText(10, 45, f"X: {self._cursor_x:.4f}")

        # Значения для каждой шкалы
        y_offset = 70
        for idx in sorted(self._sticks_data.keys()):
            data = self._sticks_data[idx]
            color = data['color']

            # Вычисляем значение Y
            y_value = self._calculate_y_value(idx)

            # Рисуем цветной индикатор
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QBrush(color))
            p.drawRect(10, y_offset - 8, 12, 12)

            # Рисуем текст
            p.setPen(color)
            p.setFont(font_normal)
            if y_value is not None:
                text = f"График {idx + 1}: {y_value:.4f}"
            else:
                text = f"График {idx + 1}: ---"
            p.drawText(28, y_offset + 3, text)

            # Диапазон
            p.setPen(QtGui.QColor(120, 120, 120))
            range_text = f"  [{data['data_min']:.2g} ... {data['data_max']:.2g}]"
            p.drawText(28, y_offset + 18, range_text)

            y_offset += 40


# ------------------------
# ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ
# ------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ODiploma — правая панель на pyqtgraph, всё быстро и по делу")
        self.resize(1400, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # панели
        self.left = DotsCanvas(is_left=True)
        self.right = DataPlot()
        self.coords = CoordinatesPanel()

        # связи
        self.left.stickUpdated.connect(self.on_stick_updated)
        self.right.sigCursorMoved.connect(self.coords.update_cursor_position)

        layout.addWidget(self.left, 1)  # левая панель ~ 1/6
        layout.addWidget(self.right, 4)  # правая панель ~ 2/3
        layout.addWidget(self.coords, 1)  # панель координат ~ 1/6

        # тулбар
        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

        clear_action = QtWidgets.QAction("Очистить", self)
        clear_action.triggered.connect(self.clear_all)
        toolbar.addAction(clear_action)

        load_action = QtWidgets.QAction("Загрузить XLSX", self)
        load_action.triggered.connect(self.load_data)
        toolbar.addAction(load_action)

        reset_x_action = QtWidgets.QAction("Сброс X", self)
        reset_x_action.setToolTip("Сбросить масштаб по X к дефолту (также левой панелью зум)")
        reset_x_action.triggered.connect(lambda: self.right.set_x_half_range(10.0))
        toolbar.addAction(reset_x_action)

        zoom_out_action = QtWidgets.QAction("Отдалить", self)
        zoom_out_action.setToolTip("Увеличить диапазон по X (отдалить)")
        zoom_out_action.triggered.connect(self.zoom_out_right_panel)
        toolbar.addAction(zoom_out_action)

        # Полудиапазон X
        self._xhalf_spin = QtWidgets.QDoubleSpinBox()
        self._xhalf_spin.setRange(0.001, 1e9)
        self._xhalf_spin.setDecimals(3)
        self._xhalf_spin.setValue(10.0)
        self._xhalf_spin.valueChanged.connect(lambda v: self.right.set_x_half_range(v))
        self.right.sigXHalfRangeChanged.connect(self._xhalf_spin.setValue)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel("Полудиапазон X:"))
        toolbar.addWidget(self._xhalf_spin)

        # подсказка по зуму
        hint = QtWidgets.QLabel("ЛКМ: выделить по X для увеличения. Колесо: стандартный масштаб. ПКМ: меню pyqtgraph.")
        hint.setStyleSheet("color: #666;")
        toolbar.addSeparator()
        toolbar.addWidget(hint)

    def load_data(self):
        filepaths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Выберите XLSX файлы", "", "Excel Files (*.xlsx)"
        )
        if not filepaths:
            return

        for fpath in filepaths:
            try:
                df = pd.read_excel(fpath, header=None)
                if df.shape[1] < 2:
                    # Можно показать QMessageBox
                    print(f"Файл {fpath} содержит менее 2 колонок.")
                    continue

                x_data = df.iloc[:, 0].to_numpy()
                y_data = df.iloc[:, 1].to_numpy()

                # Получаем реальные min и max значения из данных
                data_min = float(y_data.min())
                data_max = float(y_data.max())

                idx = len(self.left._sticks)
                hue = (idx * 47) % 360
                color = QtGui.QColor.fromHsv(hue, 220, 220)

                # Создаём палку на левой панели
                h = self.left.height()
                margin = self.left._margin
                stick_h = max(100, h - margin * 2 - 100)
                y1 = (h - stick_h) / 2
                y2 = y1 + stick_h
                x = margin + 30 + (idx * 40) % (self.left.width() - margin * 2 - 60)

                new_stick = Stick(x, y1, y2, color, data_min, data_max)
                self.left.add_stick(new_stick)

                # Добавляем график на правую панель
                self.right.add_or_update_plot(idx, x_data, y_data, y1, y2, color)

                # Обновляем данные в панели координат
                self.coords.update_stick_data(idx, color, data_min, data_max, y1, y2, x_data, y_data)

            except Exception as e:
                # Можно показать QMessageBox
                print(f"Не удалось загрузить файл {fpath}: {e}")

    # события от левой панели
    def on_stick_updated(self, idx: int, y1: int, y2: int):
        self.right.update_plot_v_range(idx, y1, y2)
        # Обновляем также координаты, если есть данные для этого индекса
        if idx in self.coords._sticks_data:
            data = self.coords._sticks_data[idx]
            self.coords.update_stick_data(idx, data['color'], data['data_min'], data['data_max'],
                                          y1, y2, data['x_data'], data['y_data'])

    def clear_all(self):
        self.left.clear_all()
        self.right.clear_all()
        self.coords.clear_all()

    def zoom_out_right_panel(self):
        current_range = self.right._x_half_range
        self.right.set_x_half_range(current_range * 1.5)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # PyQt стиль пускай будет Fusion, чтобы без сюрпризов
    app.setStyle("Fusion")

    # Немного косметики pyqtgraph
    pg.setConfigOptions(background=None, foreground='k')

    w = MainWindow()
    w.showMaximized()  # Открываем на весь экран
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
