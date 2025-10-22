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

    def __init__(self, parent=None):
        self.vb = XZoomViewBox()
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._plots: dict[int, dict] = {}
        self._x_half_range = 10.0
        self._x_min = -10.0
        self._x_max = 10.0
        self._initialized = False
        self._last_mouse_event_ts = 0.0

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

    def set_x_half_range(self, R: float):
        self._x_half_range = max(0.001, float(R))
        # Масштабирование вокруг текущего центра видимого диапазона, а не вокруг нуля
        center = 0.5 * (self._x_min + self._x_max)
        if not np.isfinite(center):
            center = 0.0
        self._x_min = center - self._x_half_range
        self._x_max = center + self._x_half_range
        if self._initialized:
            self._apply_ranges()
            self._replot_all()
        self.sigXHalfRangeChanged.emit(self._x_half_range)

    def set_x_range_direct(self, xmin: float, xmax: float):
        self._x_min = float(xmin)
        self._x_max = float(xmax)
        self._x_half_range = max(abs(xmin), abs(xmax))
        if self._initialized:
            self._apply_ranges()
            self._replot_all()

    def add_or_update_plot(self, idx: int, x_data, y_data, y1: int, y2: int, color):
        self._plots[idx] = dict(x_data=x_data, y_data=y_data, y_top=min(y1, y2), y_bottom=max(y1, y2), color=color)
        if idx not in self._curves:
            self._curves[idx] = self.plot(pen=pg.mkPen(color, width=2), clipToView=True, autoDownsample=True)
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

    def _on_mouse_moved(self, pos):
        now = time.monotonic()
        if now - self._last_mouse_event_ts < 0.012:  # ~80 Гц, троттлинг для плавности
            return
        self._last_mouse_event_ts = now
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.getViewBox().mapSceneToView(pos)
            self._vLine.setPos(mouse_point.x())
            self._hLine.setPos(mouse_point.y())
            self._vLine.setVisible(True)
            self._hLine.setVisible(True)
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
        y_arr = np.asarray(y_data)
        y_min = float(np.min(y_arr))
        y_max = float(np.max(y_arr))
        y_data_span = max(1e-9, y_max - y_min)
        ys = y_top + ((y_max - y_arr) / y_data_span) * span
        self._curves[idx].setData(x_data, ys, antialias=False)