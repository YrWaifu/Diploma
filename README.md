# Diploma

# Цели архитектуры

* Масштабируемость: новые алгоритмы/графики/экспортёры подключаются без изменения ядра.
* Разделение ответственности: UI изолирован от вычислений и данных.
* Тестопригодность: алгоритмы и загрузчики тестируются без запуска GUI.
* Производительность: вычисления в потоках/задачах, без блокировки главного потока Qt.
* Портируемость: Windows/Linux/macOS; сборка в один файл через PyInstaller.

---

# Техстек (рекомендация, совместим с текущим PyQt5)

* **Python 3.11+**
* **PyQt5** (у вас уже есть; альтернативно PySide6, но оставляем PyQt5)
* **pyqtgraph** для интерактивной отрисовки
* **numpy / scipy / pandas** для математики и табличных данных
* **pydantic** (опционально) для валидации входных данных/конфигов
* **openpyxl / python-docx** для экспорта xlsx/docx
* **pytest + pytest-qt** для тестов

---

# Слои и ключевые пакеты

```
project_root/
  app/                          # исполняемый код приложения
    ui/                         # виджеты, окна, диалоги, ресурсы
      widgets/
      views/
      qss/                      # стили (опционально)
      icons/
    core/                       # доменная модель и контракты
      models/                   # TimeSeries, Dataset, Metadata, Units
      bus.py                    # EventBus/сигналы (тонкая обвязка вокруг Qt сигналов)
      repo/                     # абстракции репозиториев (в памяти/файлы)
      services/                 # фасады: PlotService, AnalysisService
      contracts/                # ABC: Algorithm, Exporter, DataSource
    io/                         # импорт/экспорт/хранилище
      readers/                  # xlsx/csv/json
      exporters/                # docx/xlsx/txt/png/svg
      project_store/            # сохранение проекта (.odproj)
    algorithms/                 # конкретные алгоритмы анализа/фильтры/детекторы событий
      filters/
      features/
      detectors/
    plotting/                   # объектная модель графиков и слоёв
      layers/                   # SeriesLayer, MarkerLayer, IntervalLayer
      styles/
      axes/
    runtime/                    # многопоточность, планировщик задач
      tasks.py                  # описание задач
      workers.py                # QThreadPool/QRunnable
    config/
      settings.py               # загрузка/валидация конфига
      defaults.yaml
    generators/                 # генераторы тестовых данных
    app.py                      # запуск, DI-контейнер (простая регистрация)
  tests/                        # юнит- и интеграционные тесты
  pyproject.toml
  README.md
```

---

# Доменная модель (минимум)

```python
# app/core/models/timeseries.py
from dataclasses import dataclass
import numpy as np

@dataclass
class TimeSeries:
    t: np.ndarray         # монотонное время (float секунд или np.datetime64)
    y: np.ndarray         # значения
    name: str = ""
    unit: str = ""
    meta: dict = None

    def window(self, t_min, t_max) -> "TimeSeries":
        mask = (self.t >= t_min) & (self.t <= t_max)
        return TimeSeries(self.t[mask], self.y[mask], self.name, self.unit, self.meta)
```

```python
# app/core/contracts/algorithm.py
from abc import ABC, abstractmethod
from app.core.models.timeseries import TimeSeries

class Algorithm(ABC):
    id: str                 # уникальный идентификатор плагина
    name: str               # человекочитаемое имя

    @abstractmethod
    def run(self, series: TimeSeries, **params) -> dict:
        """Возвращает словарь результатов: новые ряды, метрики, метки событий."""
```

```python
# app/core/contracts/exporter.py
from abc import ABC, abstractmethod

class Exporter(ABC):
    id: str
    name: str
    exts: tuple[str, ...]   # расширения/форматы

    @abstractmethod
    def export(self, project, path: str, **opts) -> None:
        pass
```

---

# События и обмен между слоями

* Внутри UI используйте **Qt-сигналы**; для неблокирующих операций прокидывайте результаты через `QObject`-сигналы
* В ядре предоставьте тонкий **EventBus**: класс-обёртку, где события имеют чёткие типы (`DataLoaded`, `SelectionChanged`, `AnalysisFinished`).
* UI подписывается на события, а сервисы их испускают.

---

# Многопоточность/задачи

* Главный поток: только UI и лёгкая подготовка данных.
* Вычисления: `QThreadPool` + `QRunnable` или `concurrent.futures` с обёрткой под Qt.
* Протокол задачи:

```python
@dataclass
class TaskSpec:
    name: str
    fn: Callable
    args: tuple
    kwargs: dict

# runtime/workers.py
class Worker(QRunnable):
    def __init__(self, spec: TaskSpec, on_done: Callable, on_error: Callable):
        super().__init__()
        self.spec, self.on_done, self.on_error = spec, on_done, on_error
    def run(self):
        try:
            res = self.spec.fn(*self.spec.args, **self.spec.kwargs)
            QMetaObject.invokeMethod(self.on_done, "callable", Qt.QueuedConnection, res)
        except Exception as e:
            QMetaObject.invokeMethod(self.on_error, "callable", Qt.QueuedConnection, e)
```

---

# Plotting-уровень

* **Слои**: каждый графический элемент как отдельный слой: линия, точки, маркеры событий, выделенные интервалы.
* **Оси**: X всегда время. Для разных Y используйте независимые оси/правую шкалу.
* **Интеракции**: панорамирование/зум, резиновая рамка выделения интервала, горячие клавиши.
* **Связка нескольких графиков**: синхронизация по X и курсору между несколькими `PlotItem`.

Структура:

```
plotting/
  layers/
    line_layer.py
    markers_layer.py
    interval_layer.py
  axes/
    time_axis.py
  styles/
    palettes.py
```

---

# Сервисы

* **PlotService**: управляет слоями, синхронизацией осей, стилями.
* **AnalysisService**: оркеструет запуск алгоритмов (в т.ч. пакетный запуск), хранит реестр доступных алгоритмов.
* **ProjectService**: агрегирует текущие ряды/аннотации/параметры, умеет сохранять проект и историю действий.

---

# Плагинность

* Каталог `algorithms/` и `io/exporters/` сканируются при старте.
* Каждый плагин регистрируется через декоратор:

```python
REGISTRY: dict[str, type] = {}

def register(cls):
    REGISTRY[cls.id] = cls
    return cls

@register
class PeakDetector(Algorithm):
    id = "peak_detector"
    name = "Детектор пиков"
    def run(self, series: TimeSeries, **params):
        ...
```

* Поддержите загрузку из внешних директорий: `plugins/` рядом с exe.

---

# Импорт/экспорт

* **Readers**: `xlsx`, `csv`, `json`; контракты `DataSource` + фабрика по расширению.
* **Exporters**: `docx` (отчёт с таблицами и картинками), `xlsx` (сырые и обработанные данные, логи параметров), `txt/csv` (минималка), `png/svg` (скриншоты графиков).
* Точки расширения: добавление нового экспортёра без правки UI (UI читает список экспортёров из реестра).

---

# Формат проекта

* Расширение: `.odproj` (zip).
* Внутри: `project.json` (метаданные и ссылки), `data/*.npy` или `data/*.parquet`, `fig/*.json` (состояние графиков).
* Преимущества: атомарность и переносимость.

---

# Генераторы тестовых данных

```
generators/
  sine_speed.py        # синусы/шумы/скачки
  steps_events.py      # ступеньки + метки событий
  multichannel_demo.py # несколько каналов с общей осью времени
```

Каждый генератор возвращает `Dataset` или сохраняет в `.xlsx/.csv` по интерфейсу `DataSource`.

---

# UI-структура (PyQt5)

```
MainWindow
  ├─ LeftPane (QWidget + QVBoxLayout)
  │   ├─ SelectionView (виджет выбора интервала: min/max + масштабирование по Y)
  │   ├─ ControlsView (галочки/параметры алгоритмов)
  │   └─ ExportPanel
  └─ RightPane (QSplitter)
      ├─ PlotArea (несколько вкладок/секций с PlotWidget/pyqtgraph)
      └─ DetailsPanel (таблицы результатов/метрики)
```

* Взаимодействие: LeftPane меняет параметры/интервалы, кидает события; PlotArea слушает и обновляет отображение.

---

# Конфигурация и состояние

* **QSettings** для пользовательских настроек (последние пути, геометрия окон).
* **Pydantic settings** для системных конфигов (пути плагинов, лимиты потоков).
* **История проекта**: простая команда/undo для операций над слоями.

---

# Обработка ошибок и логирование

* Единый `logger` в `app/logging.py` (стандартный `logging` либо `structlog`).
* Канал в UI: панель сообщений + всплывающие уведомления.
* Логи ошибок складывать в `%APPDATA%/ODiploma/logs/`.

---

# Тестирование

* **Юнит-тесты**: алгоритмы, конвертеры, генераторы.
* **Интеграционные**: импортер → анализ → экспорт на синтетике.
* **UI smoke** с `pytest-qt`: открытие окна, отрисовка, базовые интеракции.
* Фикстуры для генераторов данных.

---

# CI и выпуск

* `pre-commit` (flake8/black/isort/mypy по вкусу).
* GitHub Actions: прогон тестов на 3 платформах.
* Сборка артефактов PyInstaller, выкладка релизов.
* Семантическое версионирование, CHANGELOG.md.

---

# Мини-DI и точка входа

```python
# app/app.py
class App:
    def __init__(self):
        self.algorithms = load_plugins("app.algorithms")
        self.exporters = load_plugins("app.io.exporters")
        self.project = ProjectService()
        self.analysis = AnalysisService(self.algorithms)
        self.plot = PlotService()

    def start(self):
        qapp = QApplication(sys.argv)
        win = MainWindow(self)
        win.show()
        return qapp.exec_()
```

---

# Дорожная карта внедрения

1. Скелет пакетов и минимальные модели (TimeSeries/Dataset).
2. Чтение `.xlsx` и генераторы данных.
3. Plotting-слои и базовые интеракции (выделение интервала, синхронизация).
4. Пул потоков + запуск первого алгоритма.
5. Экспорт `xlsx/txt`, затем `docx/png`.
6. Формат проекта `.odproj`.
7. Плагины алгоритмов и экспортёров из внешней папки.
8. Undo/Redo и улучшение UX.

---

# Пример: добавление нового алгоритма

1. Создать файл `app/algorithms/peak_detector.py`.
2. Наследоваться от `Algorithm`, украсить `@register`.
3. Описать параметры и типы результатов.
4. UI автоматически подхватит его через реестр и сформирует панель параметров.

---

# Пример: экспорт отчёта

* Экспортёр формирует разделы: параметры, сводная таблица метрик, вставка рендеров графиков.
* Графики рендерятся в offscreen-режиме pyqtgraph → PNG → вставка в `docx`.

---

# Примечания по производительности

* Хранить `numpy` массивы, избегать частых конвертаций `pandas` ↔ `numpy` в горячем пути.
* Кэшировать производные ряды (фильтры, ревэмплинг) по ключу `(algo, params, source_hash)`.
* Для длинных сигналов использовать downsampling/LOD для предпросмотра.

---

# Итог

Эта схема даёт: чистое разделение слоёв, безопасный UI, быструю отрисовку, удобные точки расширения и предсказуемую сборку. При желании можно заменить PyQt5 на PySide6 без перелома ядра.
