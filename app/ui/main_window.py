from __future__ import annotations
from pathlib import Path
from typing import List, Dict
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
import tempfile

from app.ui.left_panel import DotsCanvas, Stick
from app.ui.plot import DataPlot
from app.ui.coords_panel import CoordinatesPanel
from app.io.series_provider import make_series_provider


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

        # Настройки пользователя
        self._settings = QtCore.QSettings("ODiploma", "ODiploma")

        # Хранилище импортированных данных (без отрисовки)
        # self._datasets: список словарей { 'path': Path, 'series': List[descriptor] }
        # descriptor: { 'key': str, 'file': Path, 'name': str, 'x': np.ndarray, 'y': np.ndarray }
        self._datasets: List[Dict] = []
        self._plotted_keys: set[str] = set()
        # Провайдеры ленивой загрузки по файлам
        self._providers: Dict[Path, object] = {}
        # MRU-порядок графиков, которые хранят Y в RAM (индексы слева-направо: старые->новые)
        self._memory_mru: List[int] = []
        # Временные файлы для пониженных в memmap графиков: idx -> (y_tmp_path)
        self._plot_tmp_paths: Dict[int, str] = {}
        # Порог RAM-графиков (может быть переопределён настройками)
        self._max_memory_series: int = 7

        # Применяем сохранённые настройки (режим загрузки, плотность точек и т.п.)
        self._load_user_settings()

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

        # Кнопка "Оптимизация" с меню
        self._load_mode = getattr(self, "_load_mode", "lazy")  # "lazy" | "preload_all"
        opt_btn = QtWidgets.QToolButton(self)
        opt_btn.setText("Оптимизация")
        # Клик по тексту сразу открывает меню; убираем индикатор-стрелку
        opt_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        opt_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        opt_btn.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        opt_menu = QtWidgets.QMenu(opt_btn)
        act_mode = opt_menu.addAction("Режим загрузки…")
        act_mode.triggered.connect(self._show_optimization_mode_dialog)
        act_preload = opt_menu.addAction("Предзагрузить все серии сейчас")
        act_preload.triggered.connect(self._preload_all_series)
        opt_btn.setMenu(opt_menu)
        toolbar.addSeparator(); toolbar.addWidget(opt_btn)

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
        was_canceled = False
        new_file_paths: List[Path] = []
        # В режиме предзагрузки НЕ показываем общий прогресс заголовков,
        # чтобы не дублировать с детальным прогрессом по файлу
        use_headers_progress = self._load_mode != "preload_all"
        progress = None
        if use_headers_progress:
            progress = QtWidgets.QProgressDialog("Чтение заголовков…", "Отмена", 0, len(filepaths), self)
            progress.setWindowTitle("Импорт файлов")
            progress.setWindowModality(QtCore.Qt.ApplicationModal)
            progress.setMinimumDuration(300)
        for i, fpath in enumerate(filepaths):
            try:
                file_path = Path(fpath)
                if progress is not None:
                    progress.setLabelText(f"Чтение: {file_path.name}")
                    progress.setValue(i)
                    QtWidgets.QApplication.processEvents()
                    if progress.wasCanceled():
                        break
                provider = make_series_provider(file_path)
                info = provider.list_series()
                self._providers[file_path] = provider

                series_list = []
                for name in info.y_names:
                    key = f"{file_path}|{name}"
                    series_list.append({
                        'key': key,
                        'file': file_path,
                        'name': name,
                    })
                self._datasets.append({'path': file_path, 'series': series_list, 'x_name': info.x_name})
                new_file_paths.append(file_path)
                imported += len(info.y_names)
                # если включён режим предзагрузки — подготовим X и все Y сразу
                if self._load_mode == "preload_all":
                    # Покажем только детальный прогресс на файл (без общего)
                    canceled = self._preload_provider_series(file_path, provider, series_list, None, 0)
                    if canceled:
                        was_canceled = True
                        break
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить файл\n{fpath}\n\n{e}")
        if progress is not None:
            progress.setValue(len(filepaths))
            if progress.wasCanceled():
                was_canceled = True
            progress.close()

        if was_canceled:
            self._rollback_new_imports(new_file_paths)
            QtWidgets.QMessageBox.information(self, "Отмена", "Импорт и предзагрузка отменены пользователем.")
            return

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
        file_path: Path = desc['file']
        provider = self._providers.get(file_path)
        if provider is None:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Для файла {file_path} не найден провайдер данных.")
            return
        # Если X и Y уже предзагружены — не показываем прогресс, сразу добавляем
        try:
            if getattr(provider, "is_x_cached", lambda: False)() and getattr(provider, "is_y_cached", lambda _n: False)(desc['name']):
                x_data = provider.ensure_x_loaded()
                y_data = provider.load_y(desc['name'])
                self._finalize_added_series(desc, x_data, y_data)
                return
        except Exception:
            pass
        # Запускаем загрузку X/Y в фоновом потоке, чтобы не блокировать GUI
        progress_dlg = QtWidgets.QProgressDialog("Подготовка ряда…", "Отмена", 0, 0, self)
        progress_dlg.setWindowTitle("Загрузка ряда")
        progress_dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        progress_dlg.setMinimumDuration(200)
        progress_dlg.setAutoClose(True)
        progress_dlg.setLabelText(f"Подготовка «{desc['name']}»…")

        class LoadSeriesWorker(QtCore.QObject):
            finished = QtCore.pyqtSignal(object, object, object)  # x, y, err
            setRange = QtCore.pyqtSignal(int, str)
            setValue = QtCore.pyqtSignal(int, str)
            def __init__(self, provider_obj, yname: str):
                super().__init__()
                self.provider_obj = provider_obj
                self.yname = yname
            @QtCore.pyqtSlot()
            def run(self):
                try:
                    # оценим строки, чтобы показать точный прогресс
                    total_rows = None
                    try:
                        # для CSV есть estimate_rows
                        if hasattr(self.provider_obj, "list_series"):
                            info = self.provider_obj.list_series()
                            xname = getattr(info, "x_name", None)
                        else:
                            xname = None
                        if hasattr(self.provider_obj, "estimate_rows"):
                            total_rows = int(self.provider_obj.estimate_rows(xname))
                    except Exception:
                        total_rows = None
                    if total_rows and total_rows > 0:
                        self.setRange.emit(total_rows * 2, "Подготовка X…")
                    # колбэки прогресса
                    base = 0
                    def progress_x(done, total, _label):
                        try:
                            self.setValue.emit(min(done, total) if total else done, "Подготовка X…")
                        except Exception:
                            pass
                    x_local = self.provider_obj.ensure_x_loaded(progress=progress_x, total_hint=total_rows)
                    base = int(total_rows) if total_rows else len(x_local) if hasattr(x_local, "__len__") else 0
                    def progress_y(done, total, _label):
                        try:
                            self.setValue.emit(base + (min(done, total) if total else done), f"Чтение «{self.yname}»…")
                        except Exception:
                            pass
                    y_local = self.provider_obj.load_y(self.yname, progress=progress_y, total_hint=total_rows)
                    # финальное значение
                    if total_rows and total_rows > 0:
                        self.setValue.emit(total_rows * 2, "Готово")
                    self.finished.emit(x_local, y_local, None)
                except Exception as exc:
                    self.finished.emit(None, None, exc)
            def cancel(self):
                try:
                    # поддерживается CSV-провайдер (стриминговая отмена)
                    if hasattr(self.provider_obj, "request_cancel"):
                        self.provider_obj.request_cancel()
                except Exception:
                    pass

        worker = LoadSeriesWorker(provider, desc['name'])
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def on_finished(x_data, y_data, err):
            progress_dlg.close()
            thread.quit()
            thread.wait()
            worker.deleteLater()
            thread.deleteLater()
            if err is not None or x_data is None or y_data is None:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось загрузить ряд\n{file_path}\n\n{err}")
                return
            self._finalize_added_series(desc, x_data, y_data)

        progress_dlg.canceled.connect(worker.cancel)
        worker.setRange.connect(lambda mx, text: (progress_dlg.setRange(0, mx), progress_dlg.setLabelText(text)))
        worker.setValue.connect(lambda val, text: (progress_dlg.setValue(val), progress_dlg.setLabelText(text)))
        worker.finished.connect(on_finished)
        thread.start()

    def _finalize_added_series(self, desc: Dict, x_data, y_data):
        file_path: Path = desc['file']
        # Получаем min/max без полной загрузки в RAM: используем кэш провайдера или быструю выборку
        provider = self._providers.get(file_path)
        data_min, data_max = 0.0, 1.0
        if provider is not None:
            mm = getattr(provider, "get_y_min_max", None)
            if callable(mm):
                pair = provider.get_y_min_max(desc['name'])
                if pair is not None:
                    data_min, data_max = float(pair[0]), float(pair[1])
        if data_max == data_min or not np.isfinite(data_min) or not np.isfinite(data_max):
            # fallback: быстрая оценка по подвыборке
            try:
                y_arr = np.asarray(y_data)
                n = y_arr.size
                if n > 0:
                    stride = max(1, n // 100000)  # до 100k точек
                    s = y_arr[::stride]
                    data_min = float(np.nanmin(s))
                    data_max = float(np.nanmax(s))
                    if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min == data_max:
                        data_min, data_max = 0.0, 1.0
            except Exception:
                data_min, data_max = 0.0, 1.0
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

        self.right.add_or_update_plot(idx_added, x_data, y_data, y1, y2, color, y_min=data_min, y_max=data_max)
        self.coords.update_stick_data(idx_added, color, data_min, data_max, y1, y2, x_data, y_data)
        self._plotted_keys.add(desc['key'])
        # если ряд уже был memmap — учтём путь для корректной очистки
        try:
            import numpy as _np
            if isinstance(y_data, _np.memmap):
                self._plot_tmp_paths[idx_added] = str(getattr(y_data, 'filename', ''))
        except Exception:
            pass
        # отметим, что этот график хранится в RAM (пока)
        self._memory_mru.append(idx_added)
        self._enforce_memory_limit()

    def on_stick_updated(self, idx: int, y1: int, y2: int):
        self.right.update_plot_v_range(idx, y1, y2)
        if idx in self.coords._sticks_data:
            data = self.coords._sticks_data[idx]
            self.coords.update_stick_data(idx, data['color'], data['data_min'], data['data_max'], y1, y2,
                                          data['x_data'], data['y_data'])

    def clear_all(self):
        self.left.clear_all(); self.right.clear_all(); self.coords.clear_all()
        # Не очищаем импортированные данные: их можно добавлять повторно
        # Чистим временные файлы меммапов Y
        for p in list(self._plot_tmp_paths.values()):
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass
        self._plot_tmp_paths.clear()
        # Чистим X-меммапы провайдеров
        for prov in list(self._providers.values()):
            try:
                prov.cleanup()
            except Exception:
                pass
        self._memory_mru.clear()

    def zoom_out_right_panel(self):
        current_range = self.right._x_half_range
        self.right.set_x_half_range(current_range * 1.5)

    # --- memory management: demote older Y arrays to memmap when exceeding threshold ---
    def _enforce_memory_limit(self):
        # оставляем только последние self._max_memory_series индексов в RAM
        while len(self._memory_mru) > self._max_memory_series:
            idx_to_demote = self._memory_mru.pop(0)
            self._demote_plot_to_memmap(idx_to_demote)

    def _demote_plot_to_memmap(self, idx: int):
        # если уже понижен — ничего не делаем
        if idx in self._plot_tmp_paths:
            return
        plot_data = self.right._plots.get(idx) if hasattr(self.right, "_plots") else None
        if not plot_data:
            return
        # Индикатор переноса на диск (без отмены)
        pdialog = QtWidgets.QProgressDialog("Перенос графика на диск…", "", 0, 0, self)
        pdialog.setWindowTitle("Оптимизация памяти")
        pdialog.setWindowModality(QtCore.Qt.ApplicationModal)
        pdialog.setMinimumDuration(0)
        QtWidgets.QApplication.processEvents()
        x_data = plot_data['x_data']
        y_data = plot_data['y_data']
        y_top = plot_data['y_top']; y_bottom = plot_data['y_bottom']
        color = plot_data['color']
        # сохраняем Y в tmp и открываем как memmap
        try:
            # если уже memmap — ничего делать не нужно
            try:
                import numpy as _np
                if isinstance(y_data, _np.memmap):
                    self._plot_tmp_paths[idx] = str(getattr(y_data, 'filename', ''))
                    return
            except Exception:
                pass
            tmp = tempfile.NamedTemporaryFile(prefix=f"odiploma_y_{idx}_", suffix=".npy", delete=False)
            tmp_path = tmp.name
            tmp.close()
            np.save(tmp_path, np.asarray(y_data))
            y_mem = np.load(tmp_path, mmap_mode="r")
            # переустанавливаем график на memmap
            # сохраняем прежние min/max, чтобы не сканировать
            y_min = float(plot_data.get('y_data_min', 0.0))
            y_max = float(plot_data.get('y_data_max', 1.0))
            self.right.add_or_update_plot(idx, x_data, y_mem, y_top, y_bottom, color, y_min=y_min, y_max=y_max)
            # обновляем координатную панель
            if idx in self.coords._sticks_data:
                data = self.coords._sticks_data[idx]
                self.coords.update_stick_data(idx, data['color'], data['data_min'], data['data_max'],
                                              y_top, y_bottom, x_data, y_mem)
            self._plot_tmp_paths[idx] = tmp_path
        except Exception as e:
            # если не удалось, возвращаем idx в конец MRU (считаем что он остаётся в RAM)
            self._memory_mru.append(idx)
        finally:
            pdialog.close()

    def closeEvent(self, event):
        # чистим ресурсы перед закрытием
        try:
            self._save_user_settings()
            self.clear_all()
        finally:
            super().closeEvent(event)

    # --- Optimization UI ---
    def _show_optimization_mode_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Режим оптимизации")
        lay = QtWidgets.QVBoxLayout(dlg)
        rb_lazy = QtWidgets.QRadioButton("Подгружать по одному графику (рекомендовано)")
        rb_pre = QtWidgets.QRadioButton("Выгружать все серии сразу (дольше старт, больше I/O)")
        rb_lazy.setChecked(self._load_mode == "lazy")
        rb_pre.setChecked(self._load_mode == "preload_all")
        lay.addWidget(rb_lazy); lay.addWidget(rb_pre)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        lay.addWidget(btns)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._load_mode = "preload_all" if rb_pre.isChecked() else "lazy"
            self._save_user_settings()
            if self._load_mode == "preload_all":
                ans = QtWidgets.QMessageBox.question(self, "Предзагрузка",
                    "Предзагрузить сразу все серии для уже импортированных файлов?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                if ans == QtWidgets.QMessageBox.Yes:
                    self._preload_all_series()

    def _preload_all_series(self):
        # Предзагрузка по файлам с детальным прогрессом на каждый файл
        has_series = any(len(entry.get('series', [])) > 0 for entry in self._datasets)
        if not has_series:
            QtWidgets.QMessageBox.information(self, "Нет данных", "Сначала загрузите файлы.")
            return
        for entry in self._datasets:
            file_path: Path = entry['path']
            provider = self._providers.get(file_path)
            if not provider:
                continue
            canceled = self._preload_provider_series(file_path, provider, entry.get('series', []), None, 0)
            if canceled:
                QtWidgets.QMessageBox.information(self, "Отмена", f"Предзагрузка файла {Path(file_path).name} отменена.")
                break

    def _preload_provider_series(self, file_path: Path, provider, series_list: list, progress: QtWidgets.QProgressDialog | None = None, current_step: int = 0) -> bool:
        # Готовим подробный прогресс: сначала показываем диалог сразу, затем оцениваем строки
        local_progress = progress
        if local_progress is None:
            local_progress = QtWidgets.QProgressDialog(f"{file_path.name}: инициализация…", "Отмена", 0, 0, self)
            local_progress.setWindowTitle("Предзагрузка файла")
            local_progress.setWindowModality(QtCore.Qt.ApplicationModal)
            local_progress.setMinimumDuration(0)
            local_progress.setValue(0)
            QtWidgets.QApplication.processEvents()

        # Оценка количества строк (может занять время) — показываем статус сразу
        local_progress.setLabelText(f"{file_path.name}: оценка объёма данных…")
        QtWidgets.QApplication.processEvents()

        total_rows = 0
        try:
            if hasattr(provider, "estimate_rows"):
                total_rows = int(provider.estimate_rows())
            else:
                # попробуем получить длину X (может быть тяжело для некоторых форматов)
                x_tmp = provider.ensure_x_loaded()
                try:
                    total_rows = len(x_tmp)
                except Exception:
                    total_rows = 0
        except InterruptedError:
            # пользователь отменил во время оценки — прерываем
            if progress is None and local_progress is not None:
                local_progress.close()
            return True
        except Exception:
            total_rows = 0

        # Теперь можно установить точный диапазон прогресса
        units_total = total_rows * (1 + len(series_list))
        if units_total > 0:
            local_progress.setRange(0, units_total)
            local_progress.setValue(0)
        else:
            local_progress.setRange(0, 0)
        QtWidgets.QApplication.processEvents()

        # Подготавливаем X
        try:
            if local_progress:
                local_progress.setLabelText(f"{file_path.name}: Подготовка X…")
                QtWidgets.QApplication.processEvents()
            # прогресс по X: базовый offset = 0
            base_offset = 0
            def cb_x(done, total, label):
                if local_progress and units_total > 0:
                    # done уже в "строках"
                    local_progress.setValue(base_offset + int(done))
                    local_progress.setLabelText(f"{file_path.name}: X {done}/{total} ({int((done/max(1,total))*100)}%)")
                    QtWidgets.QApplication.processEvents()
            provider.ensure_x_loaded(progress=cb_x, total_hint=total_rows or None)
        except InterruptedError:
            # отмена пользователем
            if progress is None and local_progress is not None:
                local_progress.close()
            return True
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось подготовить X для {file_path}\n\n{e}")
            if progress is None and local_progress is not None:
                local_progress.close()
            return False

        # Для каждой серии — подготовить Y (накапливаем offset)
        base_offset = total_rows if total_rows > 0 else 0
        for desc in series_list:
            if local_progress and local_progress.wasCanceled():
                try:
                    if hasattr(provider, "request_cancel"):
                        provider.request_cancel()
                except Exception:
                    pass
                if progress is None and local_progress is not None:
                    local_progress.close()
                return True
            if local_progress:
                local_progress.setLabelText(f"{file_path.name}: {desc['name']}…")
                QtWidgets.QApplication.processEvents()
            try:
                def cb_y(done, total, label):
                    if local_progress and units_total > 0:
                        local_progress.setValue(base_offset + int(done))
                        local_progress.setLabelText(f"{file_path.name}: {desc['name']} {done}/{total} ({int((done/max(1,total))*100)}%)")
                        QtWidgets.QApplication.processEvents()
                provider.load_y(desc['name'], progress=cb_y, total_hint=total_rows or None)
                base_offset += (total_rows if total_rows > 0 else 0)
            except InterruptedError:
                if progress is None and local_progress is not None:
                    local_progress.close()
                return True
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось подготовить «{desc['name']}»\n{file_path}\n\n{e}")
                continue
        if local_progress and units_total > 0:
            local_progress.setValue(units_total)
        if progress is None and local_progress is not None:
            local_progress.close()
        return False

    def _rollback_new_imports(self, file_paths: List[Path]):
        # Удаляем datasets и провайдеры добавленные в этом запуске
        paths_set = set(Path(p) for p in file_paths)
        # remove datasets entries
        self._datasets = [d for d in self._datasets if Path(d.get('path')) not in paths_set]
        # cleanup providers
        for p in list(paths_set):
            prov = self._providers.pop(p, None)
            if prov is not None:
                try:
                    prov.cleanup()
                except Exception:
                    pass

    # --- Settings ---
    def _load_user_settings(self):
        try:
            mode = str(self._settings.value("load_mode", "lazy"))
            self._load_mode = mode if mode in ("lazy", "preload_all") else "lazy"
        except Exception:
            self._load_mode = "lazy"
        try:
            ppp = float(self._settings.value("points_per_pixel", 1.0))
            self.right.set_points_per_pixel(ppp)
        except Exception:
            pass
        try:
            cursor_hz = float(self._settings.value("cursor_hz", 50.0))
            self.right.set_cursor_sample_rate_hz(cursor_hz)
        except Exception:
            pass
        try:
            self._max_memory_series = int(self._settings.value("max_memory_series", self._max_memory_series))
        except Exception:
            pass

    def _save_user_settings(self):
        try:
            self._settings.setValue("load_mode", self._load_mode)
            ppp = float(getattr(self.right, "_max_points_factor", 1.0))
            self._settings.setValue("points_per_pixel", ppp)
            hz = 0.0
            try:
                interval = float(getattr(self.right, "_coords_emit_interval", 0.02))
                hz = (1.0 / interval) if interval > 0 else 50.0
            except Exception:
                hz = 50.0
            self._settings.setValue("cursor_hz", hz)
            self._settings.setValue("max_memory_series", int(self._max_memory_series))
        except Exception:
            pass