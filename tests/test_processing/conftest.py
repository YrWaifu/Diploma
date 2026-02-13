# -*- coding: utf-8 -*-
"""
Общие фикстуры для тестов обработки: тестовый CSV с реалистичными данными
вибрации/тензо вертолётного оборудования и несколькими режимами работы.

Структура CSV (time; sensor_vibro; sensor_strain; mode):
    - time:          0…30 с, fs = 2000 Гц  (60 000 отсчётов)
    - sensor_vibro:  вибрационный сигнал (синусы + шум, разная интенсивность по режимам)
    - sensor_strain: тензосигнал (медленная синусоида + квазистатика, разная по режимам)
    - mode:          значение режима (0 = выключен, 1..3 = активные режимы)

Режимы:
    0.0 –  2.0 с   : mode=0 (выключен) — переходный участок
    2.0 –  8.0 с   : mode=1 (Р-1) — лёгкий режим
    8.0 – 10.0 с   : mode=0 (выключен) — переходный
   10.0 – 18.0 с   : mode=2 (Р-2) — средний режим
   18.0 – 20.0 с   : mode=0 (выключен) — переходный
   20.0 – 28.0 с   : mode=3 (Р-3) — тяжёлый режим
   28.0 – 30.0 с   : mode=0 (выключен) — переходный
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest


# ── Параметры генерации ──────────────────────────────────────────────────────
FS = 2000.0          # Гц
DURATION = 30.0      # секунд
N = int(FS * DURATION)

# Границы режимов: (t_start, t_end, mode_value)
MODE_INTERVALS: List[Tuple[float, float, int]] = [
    (0.0,  2.0,  0),
    (2.0,  8.0,  1),
    (8.0,  10.0, 0),
    (10.0, 18.0, 2),
    (18.0, 20.0, 0),
    (20.0, 28.0, 3),
    (28.0, 30.0, 0),
]

# Активные режимы (белые зоны на графике) — для удобства тестов
ACTIVE_MODES: List[Tuple[str, float, float]] = [
    ("Р-1",  2.0,  8.0),
    ("Р-2", 10.0, 18.0),
    ("Р-3", 20.0, 28.0),
]


def _generate_test_data(seed: int = 42):
    """Генерирует синтетические данные, имитирующие вертолётные датчики."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, DURATION, N, endpoint=False)

    # -- Вибросигнал ----------------------------------------------------------
    # Базовые гармоники: 16 Гц (лопастная), 32 Гц (2×), 200 Гц (редуктор)
    vibro = np.zeros(N)
    for tstart, tend, mval in MODE_INTERVALS:
        mask = (t >= tstart) & (t < tend)
        if mval == 0:
            # Переходный: слабый шум
            vibro[mask] += 0.05 * rng.randn(mask.sum())
        elif mval == 1:
            # Р-1: лёгкий
            vibro[mask] += (0.8 * np.sin(2 * np.pi * 16.0 * t[mask])
                            + 0.3 * np.sin(2 * np.pi * 32.0 * t[mask])
                            + 0.1 * np.sin(2 * np.pi * 200.0 * t[mask])
                            + 0.15 * rng.randn(mask.sum()))
        elif mval == 2:
            # Р-2: средний
            vibro[mask] += (1.5 * np.sin(2 * np.pi * 16.0 * t[mask])
                            + 0.7 * np.sin(2 * np.pi * 32.0 * t[mask])
                            + 0.4 * np.sin(2 * np.pi * 200.0 * t[mask])
                            + 0.3 * rng.randn(mask.sum()))
        elif mval == 3:
            # Р-3: тяжёлый
            vibro[mask] += (2.5 * np.sin(2 * np.pi * 16.0 * t[mask])
                            + 1.2 * np.sin(2 * np.pi * 32.0 * t[mask])
                            + 0.8 * np.sin(2 * np.pi * 200.0 * t[mask])
                            + 0.5 * rng.randn(mask.sum()))

    # -- Тензосигнал ----------------------------------------------------------
    # Квазистатическая составляющая + колебания (циклы нагружения)
    strain = np.zeros(N)
    for tstart, tend, mval in MODE_INTERVALS:
        mask = (t >= tstart) & (t < tend)
        seg_t = t[mask] - tstart  # локальное время сегмента
        if mval == 0:
            strain[mask] += 5.0 + 0.2 * rng.randn(mask.sum())
        elif mval == 1:
            # Р-1: слабые циклы ~ 2 Гц, среднее 50
            strain[mask] += 50.0 + 10.0 * np.sin(2 * np.pi * 2.0 * seg_t) + 1.0 * rng.randn(mask.sum())
        elif mval == 2:
            # Р-2: средние циклы ~ 3 Гц, среднее 120
            strain[mask] += 120.0 + 30.0 * np.sin(2 * np.pi * 3.0 * seg_t) + 2.0 * rng.randn(mask.sum())
        elif mval == 3:
            # Р-3: сильные циклы ~ 5 Гц, среднее 250
            strain[mask] += 250.0 + 60.0 * np.sin(2 * np.pi * 5.0 * seg_t) + 3.0 * rng.randn(mask.sum())

    # -- Режимный сигнал ------------------------------------------------------
    mode = np.zeros(N)
    for tstart, tend, mval in MODE_INTERVALS:
        mask = (t >= tstart) & (t < tend)
        mode[mask] = float(mval)

    return t, vibro, strain, mode


# ── Фикстуры ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_csv_path(tmp_path_factory) -> Path:
    """Создаёт тестовый CSV во временной директории (один раз за сессию)."""
    t, vibro, strain, mode_sig = _generate_test_data()
    csv_dir = tmp_path_factory.mktemp("test_data")
    csv_path = csv_dir / "helicopter_test.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["time", "sensor_vibro", "sensor_strain", "mode"])
        for i in range(len(t)):
            writer.writerow([f"{t[i]:.6f}", f"{vibro[i]:.8f}",
                             f"{strain[i]:.6f}", f"{mode_sig[i]:.0f}"])
    return csv_path


@pytest.fixture(scope="session")
def test_arrays():
    """Возвращает numpy-массивы (t, vibro, strain, mode) без записи на диск."""
    return _generate_test_data()


@pytest.fixture(scope="session")
def mode_slices(test_arrays):
    """
    Возвращает dict: mode_label → (t_slice, vibro_slice, strain_slice)
    для каждого активного режима.
    """
    t, vibro, strain, _ = test_arrays
    slices = {}
    for label, t0, t1 in ACTIVE_MODES:
        i0 = int(np.searchsorted(t, t0, side="left"))
        i1 = int(np.searchsorted(t, t1, side="right"))
        slices[label] = (t[i0:i1], vibro[i0:i1], strain[i0:i1])
    return slices
