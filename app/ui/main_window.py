from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np
import tempfile
import time

from app.ui.left_panel import DotsCanvas, Stick
from app.ui.plot import DataPlot
from app.ui.coords_panel import CoordinatesPanel
from app.io.series_provider import make_series_provider
from app.io.project_store import save_project, load_project, is_project_file, PROJECT_EXT
from app.processing import (
    compute_strain_characteristics,
    StrainResult,
    compute_vibrometry,
    VibrometryResult,
    estimate_fs_from_time,
)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ODiploma")
        self.resize(1400, 700)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(16)

        self.left = DotsCanvas(is_left=True)
        self.right = DataPlot()
        self.coords = CoordinatesPanel()

        # ????????? ????????????
        self._settings = QtCore.QSettings("ODiploma", "ODiploma")

        # ????????? ??????????????? ?????? (??? ?????????)
        # self._datasets: ?????? ???????? { 'path': Path, 'series': List[descriptor] }
        # descriptor: { 'key': str, 'file': Path, 'name': str, 'x': np.ndarray, 'y': np.ndarray }
        self._datasets: List[Dict] = []
        self._plotted_keys: set[str] = set()
        # ?????????? ??????? ???????? ?? ??????
        self._providers: Dict[Path, object] = {}
        # MRU-??????? ????????, ??????? ?????? Y ? RAM (??????? ?????-???????: ??????->?????)
        self._memory_mru: List[int] = []
        # ????????? ????? ??? ?????????? ? memmap ????????: idx -> (y_tmp_path)
        self._plot_tmp_paths: Dict[int, str] = {}
        # ????? RAM-???????? (????? ???? ????????????? ???????????)
        self._max_memory_series: int = 7
        # для сохранения проекта: idx -> (path, series_name)
        self._plot_index_to_source: Dict[int, tuple] = {}
        # путь текущего проекта (для «Сохранить»)
        self._project_path: Path | None = None
        # проект изменён после последнего сохранения (для подтверждения при закрытии)
        self._project_dirty: bool = False

        # загрузка пользовательских настроек
        self._load_user_settings()

        self.left.stickUpdated.connect(self.on_stick_updated)
        self.right.sigCursorMoved.connect(self.coords.update_cursor_position)

        layout.addWidget(self.left, 1)
        layout.addWidget(self.right, 4)
        layout.addWidget(self.coords, 1)

        self._load_mode = getattr(self, "_load_mode", "lazy")  # "lazy" | "preload_all"
        self._init_menu()
        self._init_chart_toolbar()

        self.left.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.left.customContextMenuRequested.connect(lambda pos: self._show_plot_context_menu(self.left, pos))
        self.right.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.right.customContextMenuRequested.connect(lambda pos: self._show_plot_context_menu(self.right, pos))


    def _init_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("Файл")
        open_action = file_menu.addAction("Открыть")
        open_action.triggered.connect(self.load_data)
        save_action = file_menu.addAction("Сохранить")
        save_action.setShortcut(QtGui.QKeySequence.Save)  # Ctrl+S
        save_action.triggered.connect(self._save_project)
        save_as_action = file_menu.addAction("Сохранить как...")
        save_as_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._save_project_as)
        export_action = file_menu.addAction("Экспорт")
        export_action.triggered.connect(self._export_results)

        calc_menu = menu.addMenu("Расчеты")
        for name in [
            "Выгрузить и отобразить",
            "Калькулятор",
            "Тензометрирование (прочность)",
            "Виброметрирование (вибрации)",
        ]:
            action = calc_menu.addAction(name)
            action.triggered.connect(lambda _checked=False, n=name: self._open_calculation_dialog(n))

    def _init_chart_toolbar(self):
        """Панель с кнопками управления графиками справа (рядом с Файл/Расчеты)."""
        toolbar = QtWidgets.QToolBar(self)
        toolbar.setMovable(False)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        btn_zoom_in = QtWidgets.QPushButton("+")
        btn_zoom_in.setToolTip("Приблизить (уменьшить диапазон по X)")
        btn_zoom_in.setFixedSize(32, 24)
        btn_zoom_out = QtWidgets.QPushButton("−")
        btn_zoom_out.setToolTip("Отдалить (увеличить диапазон по X)")
        btn_zoom_out.setFixedSize(32, 24)
        btn_reset = QtWidgets.QPushButton("Сброс")
        btn_reset.setToolTip("Сбросить вид: левая граница 0, правая — максимум по всем графикам")
        btn_reset.setFixedHeight(24)
        toolbar.addWidget(btn_zoom_in)
        toolbar.addWidget(btn_zoom_out)
        toolbar.addWidget(btn_reset)
        btn_zoom_in.clicked.connect(self._zoom_in_charts)
        btn_zoom_out.clicked.connect(self._zoom_out_charts)
        btn_reset.clicked.connect(self._reset_charts_view)
        self.addToolBar(toolbar)

    def _show_plot_context_menu(self, widget: QtWidgets.QWidget, pos: QtCore.QPoint):
        menu = QtWidgets.QMenu(self)
        add_action = menu.addAction("Добавить график")
        add_action.triggered.connect(self.add_plot_dialog)
        clear_action = menu.addAction("Очистить")
        clear_action.triggered.connect(self.clear_plots_only)
        if not self._has_available_series():
            add_action.setEnabled(False)
        menu.exec_(widget.mapToGlobal(pos))

    def _has_available_series(self) -> bool:
        for entry in self._datasets:
            for desc in entry['series']:
                if desc['key'] not in self._plotted_keys:
                    return True
        return False

    def _get_active_plots(self) -> list[tuple[int, str]]:
        plots = []
        if hasattr(self.right, "_plots"):
            for order, idx in enumerate(sorted(self.right._plots.keys())):
                plots.append((idx, f"График {order + 1}"))
        return plots

    def get_plot_series_data(self, plot_idx: int):
        """Возвращает (x_data, y_data) для графика с индексом plot_idx или None."""
        if not hasattr(self.right, "_plots"):
            return None
        plot_data = self.right._plots.get(plot_idx)
        if not plot_data:
            return None
        x_raw = plot_data.get("x_data")
        y_raw = plot_data.get("y_data")
        if x_raw is None or y_raw is None:
            return None
        x = np.asarray(x_raw, dtype=float)
        y = np.asarray(y_raw, dtype=float)
        if y.size == 0:
            return None
        return (x, y)

    def _open_calculation_dialog(self, title: str):
        plots = self._get_active_plots()
        if title == "Тензометрирование (прочность)":
            dlg = TensometryDialog(plots, self)
        elif title == "Виброметрирование (вибрации)":
            dlg = VibrometryDialog(plots, self)
        else:
            dlg = CalculationDialog(title, plots, self)
        dlg.exec_()

    def get_project_state(self) -> Dict[str, Any]:
        """Состояние проекта для сохранения: источники, графики (путь + ряд + стик), вид по X."""
        sources = []
        seen = set()
        for entry in self._datasets:
            p = entry["path"]
            path_str = str(p.resolve())
            if path_str not in seen:
                seen.add(path_str)
                sources.append({"path": path_str})

        plots = []
        for idx in sorted(getattr(self.right, "_plots", {}).keys()):
            src = self._plot_index_to_source.get(idx)
            if not src:
                continue
            path, series_name = src
            stick = self.left.get_stick(idx)
            if not stick:
                continue
            color_hex = stick.color.name() if stick.color else "#2d5ac8"
            if not color_hex.startswith("#"):
                color_hex = "#2d5ac8"
            plots.append({
                "source_path": str(Path(path).resolve()),
                "series_name": series_name,
                "stick": {
                    "x": stick.x,
                    "y1": stick.y1,
                    "y2": stick.y2,
                    "color_hex": color_hex,
                    "data_min": stick.data_min,
                    "data_max": stick.data_max,
                },
            })

        view = {}
        if hasattr(self.right, "_x_min") and hasattr(self.right, "_x_max"):
            view = {"x_min": float(self.right._x_min), "x_max": float(self.right._x_max)}

        return {"version": 1, "sources": sources, "plots": plots, "view": view}

    def apply_project_state(self, state: Dict[str, Any]) -> None:
        """Восстанавливает workspace из сохранённого состояния. Данные подгружаются из файлов."""
        self.clear_all()
        sources = state.get("sources", [])
        plots = state.get("plots", [])
        if not sources and not plots:
            return

        # Регистрируем источники (провайдеры + список рядов в _datasets)
        for s in sources:
            path_str = s.get("path", "")
            if not path_str:
                continue
            file_path = Path(path_str)
            if not file_path.exists():
                QtWidgets.QMessageBox.warning(
                    self, "Файл не найден",
                    f"Файл источника не найден:\n{file_path}\nРяды из него не будут восстановлены."
                )
                continue
            try:
                provider = make_series_provider(file_path)
                info = provider.list_series()
                self._providers[file_path] = provider
                series_list = [
                    {"key": f"{file_path}|{name}", "file": file_path, "name": name}
                    for name in info.y_names
                ]
                self._datasets.append({"path": file_path, "series": series_list, "x_name": info.x_name})
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка загрузки источника",
                    f"Не удалось открыть файл:\n{file_path}\n\n{e}"
                )

        if not plots:
            view = state.get("view", {})
            if view:
                x_min = view.get("x_min")
                x_max = view.get("x_max")
                if x_min is not None and x_max is not None and hasattr(self.right, "set_x_range_direct"):
                    self.right.set_x_range_direct(float(x_min), float(x_max))
            return

        # Восстанавливаем графики по порядку: загрузка данных и добавление с сохранённой геометрией стика
        progress_dlg = QtWidgets.QProgressDialog("Восстановление проекта...", "Отмена", 0, len(plots), self)
        progress_dlg.setWindowTitle("Открытие проекта")
        progress_dlg.setMinimumDuration(300)
        progress_dlg.setWindowModality(QtCore.Qt.ApplicationModal)

        for i, plot in enumerate(plots):
            progress_dlg.setValue(i)
            progress_dlg.setLabelText(f"Загрузка ряда {i + 1}/{len(plots)}...")
            QtWidgets.QApplication.processEvents()
            if progress_dlg.wasCanceled():
                break

            path_str = plot.get("source_path", "")
            series_name = plot.get("series_name", "")
            stick_override = plot.get("stick")
            file_path = Path(path_str)

            desc = None
            for entry in self._datasets:
                if entry["path"] == file_path:
                    for s in entry["series"]:
                        if s["name"] == series_name:
                            desc = s
                            break
                    break
            if not desc:
                continue

            provider = self._providers.get(file_path)
            if not provider:
                continue
            try:
                x_data = provider.ensure_x_loaded()
                y_data = provider.load_y(series_name)
                self._finalize_added_series(desc, x_data, y_data, stick_override=stick_override)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Ошибка",
                    f"Не удалось загрузить ряд «{series_name}» из {file_path.name}\n\n{e}"
                )

        progress_dlg.setValue(len(plots))
        progress_dlg.close()

        view = state.get("view", {})
        if view and hasattr(self.right, "set_x_range_direct"):
            x_min = view.get("x_min")
            x_max = view.get("x_max")
            if x_min is not None and x_max is not None:
                self.right.set_x_range_direct(float(x_min), float(x_max))
            elif getattr(self.right, "_plots", {}):
                self.right.set_x_zero_to_data_max(padding_ratio=0.02)

    def _save_project(self) -> bool:
        """Сохраняет проект. Возвращает True, если сохранение выполнено (или не требовалось)."""
        if self._project_path is not None and self._project_path.exists():
            state = self.get_project_state()
            try:
                save_project(state, self._project_path)
                self.setWindowTitle(f"ODiploma — {self._project_path.name}")
                self._project_dirty = False
                return True
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка сохранения", str(e))
                return False
        return self._save_project_as()

    def _save_project_as(self) -> bool:
        """Сохранить как... Возвращает True, если файл сохранён."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить проект",
            "",
            "Все файлы (*.*);;Проект SecSig (*.secsig)",
        )
        if not path:
            return False
        path = Path(path)
        if path.suffix.lower() != PROJECT_EXT:
            path = path.with_suffix(PROJECT_EXT)
        state = self.get_project_state()
        try:
            save_project(state, path)
            self._project_path = path
            self.setWindowTitle(f"ODiploma — {path.name}")
            self._project_dirty = False
            return True
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка сохранения", str(e))
            return False

    def _export_results(self):
        QtWidgets.QMessageBox.information(self, "Экспорт", "Экспорт пока не реализован.")

    def clear_plots_only(self):
        self.left.clear_all()
        self.right.clear_all()
        self.coords.clear_all()
        self._plot_index_to_source.clear()
        for p in list(self._plot_tmp_paths.values()):
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass
        self._plot_tmp_paths.clear()
        self._plotted_keys.clear()
        self._memory_mru.clear()
        self._project_dirty = True

    # UI actions
    def load_data(self):
        filepaths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Выберите файлы или проект",
            "",
            "Все файлы (*.*);;Проект SecSig (*.secsig);;Данные (*.xlsx *.csv *.parquet *.parq);;Excel (*.xlsx);;CSV (*.csv);;Parquet (*.parquet *.parq)",
        )
        if not filepaths:
            return

        # Один файл .secsig — открыть проект
        if len(filepaths) == 1 and is_project_file(Path(filepaths[0])):
            path = Path(filepaths[0])
            try:
                state = load_project(path)
                self.apply_project_state(state)
                self._project_path = path
                self._project_dirty = False
                self.setWindowTitle(f"ODiploma — {path.name}")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка открытия проекта", str(e))
            return

        # Импорт данных: сбрасываем путь проекта (сохранение будет «Сохранить как»)
        self._project_path = None
        imported = 0
        was_canceled = False
        new_file_paths: List[Path] = []
        # В режиме предзагрузки не показываем общий прогресс заголовков,
        # чтобы не дублировать с детальным прогрессом по файлу
        use_headers_progress = self._load_mode != "preload_all"
        progress = None
        if use_headers_progress:
            progress = QtWidgets.QProgressDialog("Чтение заголовков...", "Отмена", 0, len(filepaths), self)
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
                # Если включён режим предзагрузки — подготовим X и все Y сразу
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
            self._project_dirty = True
            QtWidgets.QMessageBox.information(
                self,
                "Импорт завершён",
                f"Загружено рядов: {imported}. Теперь вы можете добавить их через 'Добавить график…'"
            )

    def add_plot_dialog(self):
        # Подготовка списка доступных, ещё не добавленных рядов
        available = []
        for entry in self._datasets:
            for desc in entry['series']:
                if desc['key'] not in self._plotted_keys:
                    available.append(desc)

        if not available:
            QtWidgets.QMessageBox.information(self, "Нет доступных рядов", "Сначала загрузите файлы, или все ряды уже добавлены.")
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
        progress_dlg = QtWidgets.QProgressDialog("Подготовка ряда...", "Отмена", 0, 0, self)
        progress_dlg.setWindowTitle("Загрузка ряда")
        progress_dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        progress_dlg.setMinimumDuration(200)
        progress_dlg.setAutoClose(True)
        progress_dlg.setLabelText(f"Подготовка «{desc['name']}»...")

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
                    # Оценим строки, чтобы показать точный прогресс
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
                        self.setRange.emit(total_rows * 2, "Подготовка X...")
                    # колбэки прогресса
                    base = 0
                    def progress_x(done, total, _label):
                        try:
                            self.setValue.emit(min(done, total) if total else done, "Подготовка X...")
                        except Exception:
                            pass
                    x_local = self.provider_obj.ensure_x_loaded(progress=progress_x, total_hint=total_rows)
                    base = int(total_rows) if total_rows else len(x_local) if hasattr(x_local, "__len__") else 0
                    def progress_y(done, total, _label):
                        try:
                            self.setValue.emit(base + (min(done, total) if total else done), f"Чтение «{self.yname}»...")
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

    def _finalize_added_series(self, desc: Dict, x_data, y_data, stick_override: Optional[Dict[str, Any]] = None):
        file_path: Path = desc['file']
        # Диапазон Y по реальным данным (или из stick_override при восстановлении проекта).
        data_min, data_max = 0.0, 1.0
        if stick_override:
            data_min = float(stick_override.get("data_min", 0.0))
            data_max = float(stick_override.get("data_max", 1.0))
        else:
            try:
                y_arr = np.asarray(y_data)
                n = y_arr.size
                if n > 0:
                    stride = max(1, n // 100000)
                    s = y_arr[::stride]
                    data_min = float(np.nanmin(s))
                    data_max = float(np.nanmax(s))
                    if not np.isfinite(data_min) or not np.isfinite(data_max):
                        data_min, data_max = 0.0, 1.0
                    elif data_min == data_max:
                        data_min, data_max = data_min - 0.5, data_min + 0.5
            except Exception:
                data_min, data_max = 0.0, 1.0

        if stick_override:
            x = int(stick_override.get("x", 50))
            y1 = int(stick_override.get("y1", 50))
            y2 = int(stick_override.get("y2", 350))
            hex_color = stick_override.get("color_hex", "#2d5ac8")
            color = QtGui.QColor(hex_color) if hex_color.startswith("#") else QtGui.QColor(hex_color)
            if not color.isValid():
                color = QtGui.QColor.fromHsv((self.left.stick_count() * 47) % 360, 220, 220)
        else:
            idx = self.left.stick_count()
            hue = (idx * 47) % 360
            color = QtGui.QColor.fromHsv(hue, 220, 220)
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
        self._plot_index_to_source[idx_added] = (file_path, desc['name'])
        self._project_dirty = True
        # путь к memmap для уже выгруженных на диск рядов
        try:
            import numpy as _np
            if isinstance(y_data, _np.memmap):
                self._plot_tmp_paths[idx_added] = str(getattr(y_data, 'filename', ''))
        except Exception:
            pass
        # ???????, ??? ???? ?????? ???????? ? RAM (????)
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
        # ?? ??????? ??????????????? ??????: ?? ????? ????????? ????????
        # ?????? ????????? ????? ???????? Y
        for p in list(self._plot_tmp_paths.values()):
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass
        self._plot_tmp_paths.clear()
        self._plot_index_to_source.clear()
        for prov in list(self._providers.values()):
            try:
                prov.cleanup()
            except Exception:
                pass
        self._memory_mru.clear()
        self._project_dirty = True

    def _zoom_in_charts(self):
        """Приблизить: уменьшить видимый диапазон по X."""
        current_range = self.right._x_half_range
        self.right.set_x_half_range(current_range / 1.5)

    def _zoom_out_charts(self):
        """Отдалить: увеличить видимый диапазон по X."""
        current_range = self.right._x_half_range
        self.right.set_x_half_range(current_range * 1.5)

    def _reset_charts_view(self):
        """Сброс: левая граница 0, правая — максимум среди всех графиков (с небольшим отступом)."""
        self.right.set_x_zero_to_data_max(padding_ratio=0.02)

    def zoom_out_right_panel(self):
        current_range = self.right._x_half_range
        self.right.set_x_half_range(current_range * 1.5)

    # --- memory management: demote older Y arrays to memmap when exceeding threshold ---
    def _enforce_memory_limit(self):
        # ????????? ?????? ????????? self._max_memory_series ???????? ? RAM
        while len(self._memory_mru) > self._max_memory_series:
            idx_to_demote = self._memory_mru.pop(0)
            self._demote_plot_to_memmap(idx_to_demote)

    def _demote_plot_to_memmap(self, idx: int):
        # ???? ??? ??????? ? ?????? ?? ??????
        if idx in self._plot_tmp_paths:
            return
        plot_data = self.right._plots.get(idx) if hasattr(self.right, "_plots") else None
        if not plot_data:
            return
        # ????????? ???????? ?? ???? (??? ??????)
        pdialog = QtWidgets.QProgressDialog("Перенос графика на диск...", "", 0, 0, self)
        pdialog.setWindowTitle("Оптимизация памяти")
        pdialog.setWindowModality(QtCore.Qt.ApplicationModal)
        pdialog.setMinimumDuration(0)
        QtWidgets.QApplication.processEvents()
        x_data = plot_data['x_data']
        y_data = plot_data['y_data']
        y_top = plot_data['y_top']; y_bottom = plot_data['y_bottom']
        color = plot_data['color']
        # ????????? Y ? tmp ? ????????? ??? memmap
        try:
            # ???? ??? memmap ? ?????? ?????? ?? ?????
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
            # ????????????????? ?????? ?? memmap
            # ????????? ??????? min/max, ????? ?? ???????????
            y_min = float(plot_data.get('y_data_min', 0.0))
            y_max = float(plot_data.get('y_data_max', 1.0))
            self.right.add_or_update_plot(idx, x_data, y_mem, y_top, y_bottom, color, y_min=y_min, y_max=y_max)
            # ????????? ???????????? ??????
            if idx in self.coords._sticks_data:
                data = self.coords._sticks_data[idx]
                self.coords.update_stick_data(idx, data['color'], data['data_min'], data['data_max'],
                                              y_top, y_bottom, x_data, y_mem)
            self._plot_tmp_paths[idx] = tmp_path
        except Exception as e:
            # ???? ?? ???????, ?????????? idx ? ????? MRU (??????? ??? ?? ???????? ? RAM)
            self._memory_mru.append(idx)
        finally:
            pdialog.close()

    def closeEvent(self, event):
        if self._project_dirty:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Подтверждение",
                "Проект изменён. Сохранить изменения перед закрытием?",
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Save,
            )
            if reply == QtWidgets.QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QtWidgets.QMessageBox.Save:
                if not self._save_project():
                    event.ignore()
                    return
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
        # ???????????? ?? ?????? ? ????????? ?????????? ?? ?????? ????
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
        # ??????? ????????? ????????: ??????? ?????????? ?????? ?????, ????? ????????? ??????
        local_progress = progress
        if local_progress is None:
            local_progress = QtWidgets.QProgressDialog(f"{file_path.name}: инициализация...", "Отмена", 0, 0, self)
            local_progress.setWindowTitle("Предзагрузка файла")
            local_progress.setWindowModality(QtCore.Qt.ApplicationModal)
            local_progress.setMinimumDuration(0)
            local_progress.setValue(0)
            QtWidgets.QApplication.processEvents()

        # ?????? ?????????? ????? (????? ?????? ?????) ? ?????????? ?????? ?????
        local_progress.setLabelText(f"{file_path.name}: оценка объёма данных...")
        QtWidgets.QApplication.processEvents()

        total_rows = 0
        try:
            if hasattr(provider, "estimate_rows"):
                total_rows = int(provider.estimate_rows())
            else:
                # ????????? ???????? ????? X (????? ???? ?????? ??? ????????? ????????)
                x_tmp = provider.ensure_x_loaded()
                try:
                    total_rows = len(x_tmp)
                except Exception:
                    total_rows = 0
        except InterruptedError:
            # ???????????? ??????? ?? ????? ?????? ? ?????????
            if progress is None and local_progress is not None:
                local_progress.close()
            return True
        except Exception:
            total_rows = 0

        # ?????? ????? ?????????? ?????? ???????? ?????????
        units_total = total_rows * (1 + len(series_list))
        if units_total > 0:
            local_progress.setRange(0, units_total)
            local_progress.setValue(0)
        else:
            local_progress.setRange(0, 0)
        QtWidgets.QApplication.processEvents()

        # ?????????????? X
        try:
            if local_progress:
                local_progress.setLabelText(f"{file_path.name}: Подготовка X...")
                QtWidgets.QApplication.processEvents()
            # ???????? ?? X: ??????? offset = 0
            base_offset = 0
            def cb_x(done, total, label):
                if local_progress and units_total > 0:
                    # done ??? ? "???????"
                    local_progress.setValue(base_offset + int(done))
                    local_progress.setLabelText(f"{file_path.name}: X {done}/{total} ({int((done/max(1,total))*100)}%)")
                    QtWidgets.QApplication.processEvents()
            provider.ensure_x_loaded(progress=cb_x, total_hint=total_rows or None)
        except InterruptedError:
            # ?????? ?????????????
            if progress is None and local_progress is not None:
                local_progress.close()
            return True
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось подготовить X для {file_path}\n\n{e}")
            if progress is None and local_progress is not None:
                local_progress.close()
            return False

        # ??? ?????? ????? ? ??????????? Y (??????????? offset)
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
                local_progress.setLabelText(f"{file_path.name}: {desc['name']}?")
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
        # ??????? datasets ? ?????????? ??????????? ? ???? ???????
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

class CalculationWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(str)

    def __init__(self, steps: int = 100, delay_s: float = 0.02):
        super().__init__()
        self._steps = int(steps)
        self._delay_s = float(delay_s)

    @QtCore.pyqtSlot()
    def run(self):
        for i in range(self._steps + 1):
            time.sleep(self._delay_s)
            self.progress.emit(i)
        self.finished.emit("Расчет завершен. Логика режима пока не подключена.")


class TensometryWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal(str)
    strain_result = QtCore.pyqtSignal(object)  # StrainResult

    def __init__(self, y_data, miner_exponent: float = 5.0):
        super().__init__()
        self._y = np.asarray(y_data, dtype=float)
        self._miner_exponent = float(miner_exponent)

    @QtCore.pyqtSlot()
    def run(self):
        self.progress.emit(10)
        r = compute_strain_characteristics(self._y, miner_exponent=self._miner_exponent)
        self.progress.emit(100)
        self.strain_result.emit(r)
        self.finished.emit("Расчет характеристик тензосигнала завершен.")


class TensometryBatchWorker(QtCore.QObject):
    """Воркер: список (метка, массив y), показатель Минера → по одному результат на график."""
    progress = QtCore.pyqtSignal(int)
    one_result = QtCore.pyqtSignal(str, object)  # label, StrainResult
    finished = QtCore.pyqtSignal(str)

    def __init__(self, items: list, miner_exponent: float = 5.0):
        super().__init__()
        self._items = list(items)  # [(label, y_array), ...]
        self._miner_exponent = float(miner_exponent)

    @QtCore.pyqtSlot()
    def run(self):
        n = len(self._items)
        for i, (label, y_arr) in enumerate(self._items):
            y = np.asarray(y_arr, dtype=float)
            r = compute_strain_characteristics(y, miner_exponent=self._miner_exponent)
            self.one_result.emit(label, r)
            self.progress.emit(int((i + 1) * 100 / n) if n else 100)
        self.finished.emit("Расчет завершен.")


class TensometryDialog(QtWidgets.QDialog):
    """Одно окно: выбор графиков (галочки), параметр Минера, кнопка «Рассчитать», блок результатов."""

    def __init__(self, plots: list[tuple[int, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Тензометрирование (прочность)")
        self.resize(520, 560)
        self._plots = plots
        self._worker_thread = None
        self._worker = None

        layout = QtWidgets.QVBoxLayout(self)

        # --- Графики для анализа ---
        grp_graphs = QtWidgets.QGroupBox("Графики для анализа")
        grp_graphs_lay = QtWidgets.QVBoxLayout(grp_graphs)
        btn_row = QtWidgets.QHBoxLayout()
        btn_select_all = QtWidgets.QPushButton("Выбрать все")
        btn_select_none = QtWidgets.QPushButton("Снять все")
        btn_select_all.clicked.connect(self._check_all_graphs)
        btn_select_none.clicked.connect(self._uncheck_all_graphs)
        btn_row.addWidget(btn_select_all)
        btn_row.addWidget(btn_select_none)
        btn_row.addStretch(1)
        grp_graphs_lay.addLayout(btn_row)
        self._graph_list = QtWidgets.QListWidget()
        self._graph_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for idx, label in plots:
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, idx)
            self._graph_list.addItem(item)
        if not plots:
            self._graph_list.addItem(QtWidgets.QListWidgetItem("Нет активных графиков"))
        grp_graphs_lay.addWidget(self._graph_list)
        layout.addWidget(grp_graphs)

        # --- Параметры ---
        grp_params = QtWidgets.QGroupBox("Параметры")
        params_lay = QtWidgets.QHBoxLayout(grp_params)
        params_lay.addWidget(QtWidgets.QLabel("Показатель Минера (m):"))
        self._miner_spin = QtWidgets.QDoubleSpinBox()
        self._miner_spin.setRange(2.0, 20.0)
        self._miner_spin.setValue(5.0)
        self._miner_spin.setDecimals(1)
        params_lay.addWidget(self._miner_spin)
        params_lay.addStretch(1)
        layout.addWidget(grp_params)

        # --- Рассчитать ---
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._calc_btn = QtWidgets.QPushButton("Рассчитать")
        self._calc_btn.clicked.connect(self._run_calculation)
        layout.addWidget(self._progress)
        layout.addWidget(self._calc_btn)

        # --- Результаты ---
        grp_results = QtWidgets.QGroupBox("Результаты")
        results_lay = QtWidgets.QVBoxLayout(grp_results)
        self._results_text = QtWidgets.QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlaceholderText("Выберите графики, задайте параметр Минера и нажмите «Рассчитать».")
        self._results_text.setMinimumHeight(200)
        results_lay.addWidget(self._results_text)
        layout.addWidget(grp_results)

        # --- Закрыть ---
        btn_close = QtWidgets.QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def _check_all_graphs(self):
        for i in range(self._graph_list.count()):
            item = self._graph_list.item(i)
            if item.data(QtCore.Qt.UserRole) is not None:
                item.setCheckState(QtCore.Qt.Checked)

    def _uncheck_all_graphs(self):
        for i in range(self._graph_list.count()):
            self._graph_list.item(i).setCheckState(QtCore.Qt.Unchecked)

    def _run_calculation(self):
        main_win = self.parent()
        if not main_win or not hasattr(main_win, "get_plot_series_data"):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Нет доступа к данным графика.")
            return
        checked = []
        for i in range(self._graph_list.count()):
            item = self._graph_list.item(i)
            idx = item.data(QtCore.Qt.UserRole)
            if idx is None:
                continue
            if item.flags() & QtCore.Qt.ItemIsUserCheckable and item.checkState() == QtCore.Qt.Checked:
                label = item.text()
                data = main_win.get_plot_series_data(idx)
                if data is not None:
                    x_data, y_data = data
                    checked.append((label, y_data))
                else:
                    checked.append((label, None))
        valid = [(lbl, y) for lbl, y in checked if y is not None]
        if not valid:
            msg = "Нет выбранных графиков с данными." if not checked else "Не удалось получить данные выбранных графиков."
            QtWidgets.QMessageBox.warning(self, "Ошибка", msg)
            return
        miner = self._miner_spin.value()
        self._results_text.clear()
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._calc_btn.setEnabled(False)

        self._worker_thread = QtCore.QThread(self)
        self._worker = TensometryBatchWorker(valid, miner_exponent=miner)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.one_result.connect(self._append_result)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _append_result(self, label: str, r: StrainResult):
        block = [
            f"——— {label} ———",
            f"  Минимум: {r.y_min:.6g}",
            f"  Максимум: {r.y_max:.6g}",
            f"  Среднее: {r.y_mean:.6g}",
            f"  Число циклов: {r.n_cycles}",
            f"  Макс. полуразмах: {r.max_half_range:.6g}",
            f"  Мин. квазистатическое: {r.min_quasi_static:.6g}",
            f"  Макс. квазистатическое: {r.max_quasi_static:.6g}",
            f"  Эквив. полуразмах (Минера): {r.equivalent_half_range:.6g}",
            "",
        ]
        self._results_text.append("\n".join(block))

    def _on_batch_finished(self, message: str):
        self._progress.setValue(100)
        self._progress.setVisible(False)
        self._calc_btn.setEnabled(True)
        self._worker_thread = None
        self._worker = None


class VibrometryBatchWorker(QtCore.QObject):
    """Воркер: список (метка, x, y), полоса f1–f2, длина сегмента → по одному VibrometryResult на график."""
    progress = QtCore.pyqtSignal(int)
    one_result = QtCore.pyqtSignal(str, object)  # label, VibrometryResult
    finished = QtCore.pyqtSignal(str)

    def __init__(
        self,
        items: list,
        band_f1_hz: float | None = None,
        band_f2_hz: float | None = None,
        segment_length: int | None = None,
    ):
        super().__init__()
        self._items = list(items)  # [(label, x_array, y_array), ...]
        self._band_f1 = float(band_f1_hz) if band_f1_hz is not None else None
        self._band_f2 = float(band_f2_hz) if band_f2_hz is not None else None
        self._segment_length = int(segment_length) if segment_length and segment_length >= 4 else None

    @QtCore.pyqtSlot()
    def run(self):
        n = len(self._items)
        for i, (label, x_arr, y_arr) in enumerate(self._items):
            x = np.asarray(x_arr, dtype=float)
            y = np.asarray(y_arr, dtype=float)
            fs = estimate_fs_from_time(x)
            if not np.isfinite(fs) or fs <= 0:
                fs = 1000.0
            r = compute_vibrometry(
                y,
                fs_hz=fs,
                band_f1_hz=self._band_f1,
                band_f2_hz=self._band_f2,
                segment_length=self._segment_length,
            )
            self.one_result.emit(label, r)
            self.progress.emit(int((i + 1) * 100 / n) if n else 100)
        self.finished.emit("Расчет завершен.")


class VibrometryDialog(QtWidgets.QDialog):
    """Одно окно: выбор графиков (галочки), полоса частот, кнопка «Рассчитать», блок результатов."""

    def __init__(self, plots: list[tuple[int, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Виброметрирование (вибрации)")
        self.resize(540, 600)
        self._plots = plots
        self._worker_thread = None
        self._worker = None

        layout = QtWidgets.QVBoxLayout(self)

        # --- Графики для анализа ---
        grp_graphs = QtWidgets.QGroupBox("Графики для анализа")
        grp_graphs_lay = QtWidgets.QVBoxLayout(grp_graphs)
        btn_row = QtWidgets.QHBoxLayout()
        btn_select_all = QtWidgets.QPushButton("Выбрать все")
        btn_select_none = QtWidgets.QPushButton("Снять все")
        btn_select_all.clicked.connect(self._check_all_graphs)
        btn_select_none.clicked.connect(self._uncheck_all_graphs)
        btn_row.addWidget(btn_select_all)
        btn_row.addWidget(btn_select_none)
        btn_row.addStretch(1)
        grp_graphs_lay.addLayout(btn_row)
        self._graph_list = QtWidgets.QListWidget()
        self._graph_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for idx, label in plots:
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, idx)
            self._graph_list.addItem(item)
        if not plots:
            self._graph_list.addItem(QtWidgets.QListWidgetItem("Нет активных графиков"))
        grp_graphs_lay.addWidget(self._graph_list)
        layout.addWidget(grp_graphs)

        # --- Параметры ---
        grp_params = QtWidgets.QGroupBox("Параметры")
        params_lay = QtWidgets.QFormLayout(grp_params)
        self._f1_spin = QtWidgets.QDoubleSpinBox()
        self._f1_spin.setRange(0.0, 1e6)
        self._f1_spin.setValue(0.0)
        self._f1_spin.setDecimals(2)
        self._f1_spin.setSuffix(" Гц")
        self._f2_spin = QtWidgets.QDoubleSpinBox()
        self._f2_spin.setRange(0.0, 1e6)
        self._f2_spin.setValue(1000.0)
        self._f2_spin.setDecimals(2)
        self._f2_spin.setSuffix(" Гц")
        self._use_band_cb = QtWidgets.QCheckBox("СКЗ в полосе частот (по PSD Уэлча)")
        self._use_band_cb.setChecked(True)
        params_lay.addRow("Полоса частот f₁:", self._f1_spin)
        params_lay.addRow("Полоса частот f₂:", self._f2_spin)
        params_lay.addRow("", self._use_band_cb)
        layout.addWidget(grp_params)

        # --- Рассчитать ---
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._calc_btn = QtWidgets.QPushButton("Рассчитать")
        self._calc_btn.clicked.connect(self._run_calculation)
        layout.addWidget(self._progress)
        layout.addWidget(self._calc_btn)

        # --- Результаты ---
        grp_results = QtWidgets.QGroupBox("Результаты")
        results_lay = QtWidgets.QVBoxLayout(grp_results)
        self._results_text = QtWidgets.QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlaceholderText(
            "Выберите графики, при необходимости задайте полосу частот и нажмите «Рассчитать»."
        )
        self._results_text.setMinimumHeight(220)
        results_lay.addWidget(self._results_text)
        layout.addWidget(grp_results)

        btn_close = QtWidgets.QPushButton("Закрыть")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def _check_all_graphs(self):
        for i in range(self._graph_list.count()):
            item = self._graph_list.item(i)
            if item.data(QtCore.Qt.UserRole) is not None:
                item.setCheckState(QtCore.Qt.Checked)

    def _uncheck_all_graphs(self):
        for i in range(self._graph_list.count()):
            self._graph_list.item(i).setCheckState(QtCore.Qt.Unchecked)

    def _run_calculation(self):
        main_win = self.parent()
        if not main_win or not hasattr(main_win, "get_plot_series_data"):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Нет доступа к данным графика.")
            return
        checked = []
        for i in range(self._graph_list.count()):
            item = self._graph_list.item(i)
            idx = item.data(QtCore.Qt.UserRole)
            if idx is None:
                continue
            if item.flags() & QtCore.Qt.ItemIsUserCheckable and item.checkState() == QtCore.Qt.Checked:
                label = item.text()
                data = main_win.get_plot_series_data(idx)
                if data is not None:
                    x_data, y_data = data
                    checked.append((label, x_data, y_data))
        if not checked:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Нет выбранных графиков с данными.")
            return

        band_f1 = self._f1_spin.value() if self._use_band_cb.isChecked() else None
        band_f2 = self._f2_spin.value() if self._use_band_cb.isChecked() else None
        if self._use_band_cb.isChecked() and (band_f2 <= band_f1):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Полоса частот: f₂ должна быть больше f₁.")
            return

        self._results_text.clear()
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._calc_btn.setEnabled(False)

        self._worker_thread = QtCore.QThread(self)
        self._worker = VibrometryBatchWorker(
            checked,
            band_f1_hz=band_f1,
            band_f2_hz=band_f2,
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.one_result.connect(self._append_result)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _append_result(self, label: str, r: VibrometryResult):
        t = r.time
        lines = [
            f"——— {label} ———",
            f"  Частота дискретизации: {r.fs_hz:.2f} Гц, отсчётов: {r.n_samples}",
            "  Временные характеристики:",
            f"    Среднее (μ): {t.mean:.6g}",
            f"    СКЗ (x_rms): {t.rms:.6g}",
            f"    Пик (max |x₀|): {t.peak:.6g}",
            f"    Пик-пик: {t.peak_to_peak:.6g}",
            f"    Пик-фактор (CF): {t.crest_factor:.6g}",
        ]
        if r.rms_in_band is not None and r.band_f1_hz is not None and r.band_f2_hz is not None:
            lines.append(f"  СКЗ в полосе [{r.band_f1_hz:.2f}, {r.band_f2_hz:.2f}] Гц: {r.rms_in_band:.6g}")
        lines.append("")
        self._results_text.append("\n".join(lines))

    def _on_batch_finished(self, message: str):
        self._progress.setValue(100)
        self._progress.setVisible(False)
        self._calc_btn.setEnabled(True)
        self._worker_thread = None
        self._worker = None


class CalculationDialog(QtWidgets.QDialog):
    def __init__(self, title: str, plots: list[tuple[int, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 380)

        self._plots = plots
        self._worker_thread = None
        self._worker = None

        layout = QtWidgets.QVBoxLayout(self)
        self._stack = QtWidgets.QStackedWidget()
        layout.addWidget(self._stack)

        self._build_graph_page()
        self._page_mode = self._build_mode_page()
        self._page_format = self._build_format_page()
        self._page_run = self._build_run_page()

        self._stack.addWidget(self._page_graph)
        self._stack.addWidget(self._page_mode)
        self._stack.addWidget(self._page_format)
        self._stack.addWidget(self._page_run)

        btns = QtWidgets.QHBoxLayout()
        self._btn_back = QtWidgets.QPushButton("Назад")
        self._btn_next = QtWidgets.QPushButton("Продолжить")
        self._btn_close = QtWidgets.QPushButton("Закрыть")
        btns.addWidget(self._btn_back)
        btns.addWidget(self._btn_next)
        btns.addStretch(1)
        btns.addWidget(self._btn_close)
        layout.addLayout(btns)

        self._btn_back.clicked.connect(self._prev_step)
        self._btn_next.clicked.connect(self._next_step)
        self._btn_close.clicked.connect(self.reject)

        self._update_nav()

    def _build_graph_page(self) -> QtWidgets.QWidget:
        self._page_graph = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self._page_graph)
        lay.addWidget(QtWidgets.QLabel("Шаг 1. Выбор графика"))
        self._graph_combo = QtWidgets.QComboBox()
        if self._plots:
            for idx, label in self._plots:
                self._graph_combo.addItem(label, idx)
        else:
            self._graph_combo.addItem("Нет активных графиков")
            self._graph_combo.setEnabled(False)
        lay.addWidget(self._graph_combo)
        lay.addStretch(1)
        return self._page_graph

    def _build_mode_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.addWidget(QtWidgets.QLabel("Шаг 2. Режим и параметры"))
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItem("Основной режим")
        lay.addWidget(self._mode_combo)
        lay.addWidget(QtWidgets.QLabel("Параметры отсутствуют"))
        lay.addStretch(1)
        return page

    def _build_format_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.addWidget(QtWidgets.QLabel("Шаг 3. Формат результата"))
        self._format_combo = QtWidgets.QComboBox()
        self._format_combo.addItems([
            "Добавить как новый график",
            "Сохранить в файл",
        ])
        lay.addWidget(self._format_combo)
        lay.addStretch(1)
        return page

    def _build_run_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.addWidget(QtWidgets.QLabel("Шаг 4. Расчет"))
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._status = QtWidgets.QLabel("Готово к расчету")
        self._calc_btn = QtWidgets.QPushButton("Рассчитать")
        self._calc_btn.clicked.connect(self._start_calculation)
        self._results_text = QtWidgets.QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlaceholderText("Результаты появятся после расчета.")
        self._results_text.setMinimumHeight(120)
        lay.addWidget(self._progress)
        lay.addWidget(self._status)
        lay.addWidget(self._calc_btn)
        lay.addWidget(QtWidgets.QLabel("Результаты:"))
        lay.addWidget(self._results_text)
        lay.addStretch(1)
        return page

    def _update_nav(self):
        idx = self._stack.currentIndex()
        self._btn_back.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < self._stack.count() - 1)
        if not self._plots:
            self._btn_next.setEnabled(False)

    def _next_step(self):
        idx = self._stack.currentIndex()
        if idx < self._stack.count() - 1:
            self._stack.setCurrentIndex(idx + 1)
        self._update_nav()

    def _prev_step(self):
        idx = self._stack.currentIndex()
        if idx > 0:
            self._stack.setCurrentIndex(idx - 1)
        self._update_nav()

    def _start_calculation(self):
        if self._worker_thread is not None:
            return
        self._start_generic()

    def _start_generic(self):
        self._progress.setValue(0)
        self._status.setText("Выполняется расчет...")
        self._calc_btn.setEnabled(False)
        self._btn_back.setEnabled(False)
        self._btn_next.setEnabled(False)

        self._worker_thread = QtCore.QThread(self)
        self._worker = CalculationWorker()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_calc_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _on_calc_finished(self, message: str):
        self._status.setText(message)
        self._btn_back.setEnabled(True)
        self._btn_next.setEnabled(True)
        self._btn_close.setEnabled(True)
        self._worker_thread = None
        self._worker = None
