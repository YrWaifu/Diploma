from __future__ import annotations
from pathlib import Path
from typing import List, Dict
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np

from app.ui.left_panel import DotsCanvas, Stick
from app.ui.plot import DataPlot
from app.ui.coords_panel import CoordinatesPanel
from app.io.readers.xlsx_reader import read_multicolumn_xlsx
from app.io.readers.csv_reader import read_multicolumn_csv
from app.io.readers.parquet_reader import read_multicolumn_parquet


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ODiploma — правая панель на pyqtgraph, всё быстро и по делу")
        self.resize(1400, 700)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(16)

        self.left = DotsCanvas(is_left=True)
        self.right = DataPlot()
        self.coords = CoordinatesPanel()

        # Хранилище импортированных данных (без отрисовки)
        # self._datasets: список словарей { 'path': Path, 'series': List[descriptor] }
        # descriptor: { 'key': str, 'file': Path, 'name': str, 'x': np.ndarray, 'y': np.ndarray }
        self._datasets: List[Dict] = []
        self._plotted_keys: set[str] = set()

        self.left.stickUpdated.connect(self.on_stick_updated)
        self.right.sigCursorMoved.connect(self.coords.update_cursor_position)

        layout.addWidget(self.left, 1)
        layout.addWidget(self.right, 4)
        layout.addWidget(self.coords, 1)

        toolbar = QtWidgets.QToolBar(); toolbar.setMovable(False)
        self.addToolBar(QtCore.Qt.TopToolBarArea, toolbar)

        clear_action = QtWidgets.QAction("Очистить", self)
        clear_action.triggered.connect(self.clear_all)
        toolbar.addAction(clear_action)

        load_action = QtWidgets.QAction("Загрузить XLSX", self)
        load_action.triggered.connect(self.load_data)
        toolbar.addAction(load_action)

        add_plot_action = QtWidgets.QAction("Добавить график…", self)
        add_plot_action.setToolTip("Выбрать и добавить один из загруженных рядов")
        add_plot_action.triggered.connect(self.add_plot_dialog)
        toolbar.addAction(add_plot_action)

        reset_x_action = QtWidgets.QAction("Сброс X", self)
        reset_x_action.setToolTip("Сбросить масштаб по X к дефолту")
        reset_x_action.triggered.connect(lambda: self.right.set_x_half_range(10.0))
        toolbar.addAction(reset_x_action)

        zoom_out_action = QtWidgets.QAction("Отдалить", self)
        zoom_out_action.setToolTip("Увеличить диапазон по X (отдалить)")
        zoom_out_action.triggered.connect(self.zoom_out_right_panel)
        toolbar.addAction(zoom_out_action)

        self._xhalf_spin = QtWidgets.QDoubleSpinBox()
        self._xhalf_spin.setRange(0.001, 1e9); self._xhalf_spin.setDecimals(3); self._xhalf_spin.setValue(10.0)
        self._xhalf_spin.valueChanged.connect(lambda v: self.right.set_x_half_range(v))
        self.right.sigXHalfRangeChanged.connect(self._xhalf_spin.setValue)
        toolbar.addSeparator(); toolbar.addWidget(QtWidgets.QLabel("Полудиапазон X:")); toolbar.addWidget(self._xhalf_spin)

        hint = QtWidgets.QLabel("ЛКМ: выделить по X для увеличения. Колесо: масштаб. ПКМ: меню pyqtgraph.")
        hint.setStyleSheet("color: #666;")
        toolbar.addSeparator(); toolbar.addWidget(hint)

    # UI actions
    def load_data(self):
        filepaths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Выберите файлы",
            "",
            "Data Files (*.xlsx *.csv *.parquet *.parq);;Excel (*.xlsx);;CSV (*.csv);;Parquet (*.parquet *.parq)"
        )
        if not filepaths:
            return
        imported = 0
        for fpath in filepaths:
            try:
                suffix = Path(fpath).suffix.lower()
                if suffix == '.xlsx':
                    ds = read_multicolumn_xlsx(fpath)
                elif suffix == '.csv':
                    ds = read_multicolumn_csv(fpath)
                elif suffix in ('.parquet', '.parq'):
                    ds = read_multicolumn_parquet(fpath)
                else:
                    raise ValueError(f"Неподдерживаемое расширение: {suffix}")
                file_path = Path(fpath)
                series_list = []
                for name, y in ds.series.items():
                    key = f"{file_path}|{name}"
                    series_list.append({
                        'key': key,
                        'file': file_path,
                        'name': name,
                        'x': ds.x,
                        'y': y,
                    })
                self._datasets.append({'path': file_path, 'series': series_list})
                imported += len(series_list)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить файл\n{fpath}\n\n{e}")

        if imported:
            QtWidgets.QMessageBox.information(self, "Импорт завершён", f"Загружено рядов: {imported}. Теперь вы можете добавить их через ‘Добавить график…’")

    def add_plot_dialog(self):
        # подготовка списка доступных, ещё не добавленных рядов
        available = []
        for entry in self._datasets:
            for desc in entry['series']:
                if desc['key'] not in self._plotted_keys:
                    available.append(desc)

        if not available:
            QtWidgets.QMessageBox.information(self, "Нет доступных рядов", "Сначала загрузите XLSX, или все ряды уже добавлены.")
            return

        items = [f"{Path(d['file']).name} — {d['name']}" for d in available]
        item, ok = QtWidgets.QInputDialog.getItem(self, "Добавить график", "Выберите ряд:", items, 0, False)
        if not ok:
            return
        chosen_idx = items.index(item)
        chosen = available[chosen_idx]
        self._add_series_to_plot(chosen)

    def _add_series_to_plot(self, desc: Dict):
        x_data = desc['x']
        y_data = desc['y']
        data_min, data_max = float(np.min(y_data)), float(np.max(y_data)) if hasattr(y_data, 'size') and y_data.size else (0.0, 1.0)
        idx = self.left.stick_count()
        hue = (idx * 47) % 360
        color = QtGui.QColor.fromHsv(hue, 220, 220)

        # геометрия левой панели
        h = self.left.height(); margin = self.left.margin
        stick_h = max(100, h - margin * 2 - 100)
        y1 = int((h - stick_h) / 2); y2 = int(y1 + stick_h)
        avail_w = max(1, self.left.width() - margin * 2 - 60)
        x = int(margin + 30 + (idx * 40) % max(1, avail_w))

        new_stick = Stick(x, y1, y2, color, data_min, data_max)
        idx_added = self.left.add_stick(new_stick)

        self.right.add_or_update_plot(idx_added, x_data, y_data, y1, y2, color)
        self.coords.update_stick_data(idx_added, color, data_min, data_max, y1, y2, x_data, y_data)
        self._plotted_keys.add(desc['key'])

    def on_stick_updated(self, idx: int, y1: int, y2: int):
        self.right.update_plot_v_range(idx, y1, y2)
        if idx in self.coords._sticks_data:
            data = self.coords._sticks_data[idx]
            self.coords.update_stick_data(idx, data['color'], data['data_min'], data['data_max'], y1, y2,
                                          data['x_data'], data['y_data'])

    def clear_all(self):
        self.left.clear_all(); self.right.clear_all(); self.coords.clear_all()
        # Не очищаем импортированные данные: их можно добавлять повторно

    def zoom_out_right_panel(self):
        current_range = self.right._x_half_range
        self.right.set_x_half_range(current_range * 1.5)