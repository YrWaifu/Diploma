from __future__ import annotations
from PyQt5 import QtCore
import pyqtgraph as pg
import numpy as np
import time


class XZoomViewBox(pg.ViewBox):
    sigZoomRange = QtCore.pyqtSignal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, enableMenu=False, **kwargs)
        self._zoom_line1 = None
        self._zoom_line2 = None
        self.setMouseEnabled(x=False, y=False)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton:
            ev.accept()

            if ev.isStart():
                self._startPos = ev.pos()
                start_point = self.mapSceneToView(self._startPos)
                if self._zoom_line1 is None:
                    self._zoom_line1 = pg.InfiniteLine(pos=start_point.x(), angle=90, movable=False,
                                                       pen=pg.mkPen('blue', width=2, style=QtCore.Qt.DashLine))
                    self.addItem(self._zoom_line1)
                else:
                    self._zoom_line1.setPos(start_point.x())
                    self._zoom_line1.show()
                if self._zoom_line2 is None:
                    self._zoom_line2 = pg.InfiniteLine(pos=start_point.x(), angle=90, movable=False,
                                                       pen=pg.mkPen('blue', width=2, style=QtCore.Qt.DashLine))
                    self.addItem(self._zoom_line2)
                else:
                    self._zoom_line2.setPos(start_point.x())
                    self._zoom_line2.show()
            elif ev.isFinish():
                if not hasattr(self, "_startPos"):
                    if self._zoom_line1:
                        self._zoom_line1.hide()
                    if self._zoom_line2:
                        self._zoom_line2.hide()
                    return

                start = self._startPos
                end = ev.pos()

                if self._zoom_line1:
                    self._zoom_line1.hide()
                if self._zoom_line2:
                    self._zoom_line2.hide()
                if abs(end.x() - start.x()) < 6:
                    return

                p0 = self.mapSceneToView(start)
                p1 = self.mapSceneToView(end)
                xmin, xmax = sorted([p0.x(), p1.x()])
                self.sigZoomRange.emit(float(xmin), float(xmax))
            else:
                if self._zoom_line2 and hasattr(self, "_startPos"):
                    end_point = self.mapSceneToView(ev.pos())
                    self._zoom_line2.setPos(end_point.x())
        else:
            ev.ignore()


class DataPlot(pg.PlotWidget):
    sigXHalfRangeChanged = QtCore.pyqtSignal(float)
    sigCursorMoved = QtCore.pyqtSignal(float, int)
    # Сигнал: списки времён начала и окончания режимов обновились (после перетаскивания линии).
    sigModeTimesChanged = QtCore.pyqtSignal(list, list)
    # Сигнал: видимый x-диапазон изменился → (x_min, x_max, data_x_max).
    sigXRangeChanged = QtCore.pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        self.vb = XZoomViewBox()
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._plots: dict[int, dict] = {}
        self._mode_lines_start: list[pg.InfiniteLine] = []
        self._mode_lines_end: list[pg.InfiniteLine] = []
        self._mode_off_regions: list[pg.LinearRegionItem] = []
        self._labels: dict[int, pg.TextItem] = {}
        self._x_half_range = 10.0
        self._x_min = 0.0
        self._x_max = 20.0
        self._data_x_max = 20.0  # максимум по данным (полный диапазон)
        self._initialized = False
        # Частота отрисовки линий курсора не ограничивается, чтобы движение было плавным.
        # Частота обновления координат (emit в панель) ограничена отдельно.
        self._last_coords_emit_ts = 0.0
        self._coords_emit_interval = 0.02  # ~50 Гц для вычислений координат
        self._max_points_factor = 1.0  # целим ~1 точку на пиксель

        super().__init__(parent=parent, viewBox=self.vb, background="w")
        pg.setConfigOptions(antialias=True)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().hideAxis('left')
        ax = self.getPlotItem().getAxis('bottom')
        ax.setPen(pg.mkPen(120, 120, 120))
        ax.setTextPen(pg.mkPen(120, 120, 120))
        self.getViewBox().invertY(True)

        self._vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
        self._hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
        self.addItem(self._vLine, ignoreBounds=True)
        self.addItem(self._hLine, ignoreBounds=True)
        self._vLine.setVisible(False)
        self._hLine.setVisible(False)

        self.vb.sigZoomRange.connect(self.set_x_range_direct)
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._initialized = True
        self._apply_ranges()
        self._replot_all()

    def set_cursor_sample_rate_hz(self, hz: float):
        # Задаёт частоту обновления координат (не линий) в Гц
        f = max(5.0, float(hz))  # защитимся от слишком низких значений
        self._coords_emit_interval = 1.0 / f

    def set_points_per_pixel(self, ppp: float):
        self._max_points_factor = max(0.1, float(ppp))
        if self._initialized:
            self._replot_all()

    def set_x_half_range(self, R: float):
        self._x_half_range = max(0.001, float(R))
        # Всегда держим левую границу в 0 и масштабируем вправо
        self._x_min = 0.0
        self._x_max = 2.0 * self._x_half_range
        if self._initialized:
            self._apply_ranges()
            self._replot_all()
        self.sigXHalfRangeChanged.emit(self._x_half_range)

    def set_x_range_direct(self, xmin: float, xmax: float):
        x0 = max(0.0, float(xmin))
        x1 = float(xmax)
        if not np.isfinite(x0) or not np.isfinite(x1):
            return
        if x1 <= x0:
            x1 = x0 + 1.0
        self._x_min = 0.0 if x0 <= 1e-12 else x0
        self._x_max = x1
        self._x_half_range = 0.5 * (self._x_max - self._x_min)
        if self._initialized:
            self._apply_ranges()
            self._replot_all()
        self.sigXRangeChanged.emit(self._x_min, self._x_max, self._data_x_max)

    def add_or_update_plot(self, idx: int, x_data, y_data, y1: int, y2: int, color, y_min: float | None = None, y_max: float | None = None):
        was_empty = len(self._plots) == 0
        # Кэшируем глобальные min/max для ускорения реплота на больших массивах
        if y_min is None or y_max is None:
            try:
                y_min = float(np.min(y_data)) if getattr(y_data, 'size', 0) else 0.0
                y_max = float(np.max(y_data)) if getattr(y_data, 'size', 0) else 1.0
            except Exception:
                y_min, y_max = 0.0, 1.0
        self._plots[idx] = dict(
            x_data=x_data,
            y_data=y_data,
            y_top=min(y1, y2),
            y_bottom=max(y1, y2),
            color=color,
            y_data_min=y_min,
            y_data_max=y_max,
        )
        # Обновляем глобальный максимум по X среди всех графиков
        try:
            x_arr = np.asarray(x_data)
            if x_arr.size:
                xm = float(np.max(x_arr))
                if np.isfinite(xm) and xm > self._data_x_max:
                    self._data_x_max = xm
        except Exception:
            pass
        if idx not in self._curves:
            self._curves[idx] = self.plot(pen=pg.mkPen(color, width=2), clipToView=True, autoDownsample=True)
        else:
            self._curves[idx].setPen(pg.mkPen(color, width=2))
        if self._initialized:
            self._replot(idx)
            if was_empty:
                self.set_x_zero_to_data_max(padding_ratio=0.02)

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
        # Удаляем режимные маркеры и подписи, если они есть
        self.clear_mode_markers()
        for lbl in self._labels.values():
            try:
                self.removeItem(lbl)
            except Exception:
                pass
        self._labels.clear()

    def _on_mouse_moved(self, pos):
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.getViewBox().mapSceneToView(pos)
            # Линии сдвигаем без троттлинга для максимально плавного хода
            self._vLine.setPos(mouse_point.x())
            self._hLine.setPos(mouse_point.y())
            self._vLine.setVisible(True)
            self._hLine.setVisible(True)
            # Троттлим только расчёт и отправку координат в панель
            now = time.monotonic()
            if now - self._last_coords_emit_ts >= self._coords_emit_interval:
                self._last_coords_emit_ts = now
                self.sigCursorMoved.emit(mouse_point.x(), mouse_point.y())
        else:
            self._vLine.setVisible(False)
            self._hLine.setVisible(False)

    def _apply_ranges(self):
        self.setXRange(self._x_min, self._x_max, padding=0)
        H = max(1, self.viewport().height())
        self.setYRange(0, H, padding=0)

    def resizeEvent(self, e):
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
        if getattr(y_data, 'size', 0) == 0:
            self._curves[idx].setData([], [])
            return
        # ограничиваем число точек под текущую ширину вьюпорта
        vp_w = max(1, int(self.viewport().width()))
        target_pts = max(1, int(vp_w * self._max_points_factor))

        x_arr = np.asarray(x_data)
        y_arr = np.asarray(y_data)
        # видимая выборка по X (x отсортирован по времени)
        try:
            i0, i1 = int(np.searchsorted(x_arr, self._x_min, side='left')), int(np.searchsorted(x_arr, self._x_max, side='right'))
        except Exception:
            i0, i1 = 0, x_arr.size
        i0 = max(0, min(i0, x_arr.size))
        i1 = max(i0, min(i1, x_arr.size))
        if i1 - i0 <= 0:
            self._curves[idx].setData([], [])
            return

        x_vis = x_arr[i0:i1]
        y_vis = y_arr[i0:i1]
        n_vis = x_vis.size
        if n_vis > target_pts:
            stride = int(np.ceil(n_vis / target_pts))
            x_vis = x_vis[::stride]
            y_vis = y_vis[::stride]

        # нормализация Y в пиксели панели
        y_min = float(plot_data.get('y_data_min', np.min(y_arr)))
        y_max = float(plot_data.get('y_data_max', np.max(y_arr)))
        y_data_span = max(1e-9, y_max - y_min)
        ys = y_top + ((y_max - y_vis) / y_data_span) * span
        self._curves[idx].setData(x_vis, ys, antialias=False)
        # Обновляем положение подписи графика (если есть)
        self._update_label_position(idx)

    def set_x_zero_to_data_max(self, padding_ratio: float = 0.02):
        if not self._plots:
            self._data_x_max = 20.0
            self.set_x_range_direct(0.0, 20.0)
            return
        x_maxs = []
        for p in self._plots.values():
            x_arr = np.asarray(p['x_data'])
            if x_arr.size:
                x_maxs.append(float(np.max(x_arr)))
        if not x_maxs:
            self._data_x_max = 20.0
            self.set_x_range_direct(0.0, 20.0)
            return
        xmax = max(x_maxs)
        if not np.isfinite(xmax):
            self._data_x_max = 20.0
            self.set_x_range_direct(0.0, 20.0)
            return
        self._data_x_max = xmax
        span = max(1e-12, xmax - 0.0)
        pad_right = span * float(padding_ratio)
        self.set_x_range_direct(0.0, xmax + pad_right)

    # Режимные линии и маркеры
    def clear_mode_markers(self):
        """Удаляет все вертикальные линии и области, связанные с режимами."""
        for ln in self._mode_lines_start:
            try:
                self.removeItem(ln)
            except Exception:
                pass
        for ln in self._mode_lines_end:
            try:
                self.removeItem(ln)
            except Exception:
                pass
        self._mode_lines_start.clear()
        self._mode_lines_end.clear()
        for r in self._mode_off_regions:
            try:
                self.removeItem(r)
            except Exception:
                pass
        self._mode_off_regions.clear()

    def add_mode_markers(self, start_times: list[float], end_times: list[float]):
        """
        Добавляет вертикальные линии на моменты начала/окончания режима.

        start_times — начала (синие линии),
        end_times — окончания (красные линии).
        """
        self.clear_mode_markers()
        # Синие — начало режима (перетаскиваемые)
        pen_start = pg.mkPen((50, 100, 200, 200), width=2, style=QtCore.Qt.DashLine)
        hover_start = pg.mkPen((30, 70, 220, 255), width=3, style=QtCore.Qt.SolidLine)
        for t in start_times:
            try:
                ln = pg.InfiniteLine(pos=float(t), angle=90, movable=True,
                                     pen=pen_start, hoverPen=hover_start)
                ln.sigPositionChangeFinished.connect(self._on_mode_line_moved)
                self.addItem(ln)
                self._mode_lines_start.append(ln)
            except Exception:
                continue
        # Красные — окончание режима (перетаскиваемые)
        pen_end = pg.mkPen((200, 60, 60, 200), width=2, style=QtCore.Qt.DashLine)
        hover_end = pg.mkPen((220, 30, 30, 255), width=3, style=QtCore.Qt.SolidLine)
        for t in end_times:
            try:
                ln = pg.InfiniteLine(pos=float(t), angle=90, movable=True,
                                     pen=pen_end, hoverPen=hover_end)
                ln.sigPositionChangeFinished.connect(self._on_mode_line_moved)
                self.addItem(ln)
                self._mode_lines_end.append(ln)
            except Exception:
                continue

        self._rebuild_mode_regions()

    def _get_mode_times(self) -> tuple[list[float], list[float]]:
        """Считывает текущие позиции линий-маркеров режимов."""
        starts = sorted(float(ln.value()) for ln in self._mode_lines_start)
        ends = sorted(float(ln.value()) for ln in self._mode_lines_end)
        return starts, ends

    def get_mode_times(self) -> tuple[list[float], list[float]]:
        """Публичный доступ к актуальным временам начала/окончания режимов."""
        return self._get_mode_times()

    def _on_mode_line_moved(self):
        """Обработчик окончания перетаскивания любой режимной линии."""
        self._rebuild_mode_regions()
        starts, ends = self._get_mode_times()
        self.sigModeTimesChanged.emit(starts, ends)

    def _rebuild_mode_regions(self):
        """Перестраивает серые области «выключенного режима» по текущим позициям линий."""
        # Удаляем старые регионы
        for r in self._mode_off_regions:
            try:
                self.removeItem(r)
            except Exception:
                pass
        self._mode_off_regions.clear()

        starts, ends = self._get_mode_times()
        if not starts and not ends:
            return

        try:
            # Собираем все события на временной оси
            events: list[tuple[str, float]] = []
            for s in starts:
                events.append(('s', s))
            for e in ends:
                events.append(('e', e))
            events.sort(key=lambda ev: ev[1])

            # Определяем границы данных по X
            x_lo = self._x_min
            x_hi = self._x_max
            if self._plots:
                x_arr_mins: list[float] = []
                x_arr_maxs: list[float] = []
                for p in self._plots.values():
                    xa = np.asarray(p['x_data'])
                    if xa.size:
                        x_arr_mins.append(float(xa[0]))
                        x_arr_maxs.append(float(xa[-1]))
                if x_arr_mins:
                    x_lo = min(x_lo, min(x_arr_mins))
                if x_arr_maxs:
                    x_hi = max(x_hi, max(x_arr_maxs))

            # Формируем интервалы «на минимуме».
            plateau: list[tuple[float, float]] = []
            if events[0][0] == 's' and events[0][1] > x_lo:
                plateau.append((x_lo, events[0][1]))

            for k in range(len(events) - 1):
                if events[k][0] == 'e' and events[k + 1][0] == 's':
                    a, b = events[k][1], events[k + 1][1]
                    if b > a:
                        plateau.append((a, b))

            if events[-1][0] == 'e' and events[-1][1] < x_hi:
                plateau.append((events[-1][1], x_hi))

            if not plateau:
                return

            brush = pg.mkBrush(210, 210, 210, 120)
            pen_none = pg.mkPen(None)
            for a, b in plateau:
                if b <= a:
                    continue
                region = pg.LinearRegionItem(
                    values=(a, b),
                    orientation=pg.LinearRegionItem.Vertical,
                    movable=False,
                    brush=brush,
                )
                region.setZValue(5)
                for line in region.lines:
                    line.setPen(pen_none)
                    line.setHoverPen(pen_none)
                    line.setMovable(False)
                self.addItem(region)
                self._mode_off_regions.append(region)
        except Exception:
            return

    # Подписи справа от графика
    def set_plot_label(self, idx: int, text: str, color) -> None:
        """Создаёт или обновляет подпись графика вдоль шкалы справа."""
        if idx not in self._plots:
            return
        if idx not in self._labels:
            lbl = pg.TextItem(text=text, anchor=(1.0, 0.5))
            self._labels[idx] = lbl
            self.addItem(lbl)
        else:
            lbl = self._labels[idx]
            lbl.setText(text)
        try:
            lbl.setColor(color)
        except Exception:
            pass
        self._update_label_position(idx)

    def _update_label_position(self, idx: int) -> None:
        """Обновляет положение подписи для заданного графика по текущему виду."""
        if idx not in self._labels or idx not in self._plots:
            return
        plot_data = self._plots[idx]
        y_top, y_bot = plot_data["y_top"], plot_data["y_bottom"]
        y_center = 0.5 * (y_top + y_bot)
        span_x = max(1e-6, float(self._x_max - self._x_min))
        x_pos = float(self._x_max - 0.02 * span_x)
        # Пытаемся просто поставить подпись в нужное место; ошибки здесь не критичны.
        self._labels[idx].setPos(x_pos, y_center)