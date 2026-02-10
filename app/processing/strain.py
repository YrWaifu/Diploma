# -*- coding: utf-8 -*-
"""
Обработка тензометрических сигналов (по разделу 2.3.1 диплома).
Вход: массив отсчётов y (и при необходимости время t). Выход: числовые характеристики.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class StrainResult:
    """Результат расчёта характеристик тензосигнала на участке."""
    y_min: float
    y_max: float
    y_mean: float
    n_cycles: int
    max_half_range: float
    min_quasi_static: float
    max_quasi_static: float
    equivalent_half_range: float


def compute_strain_characteristics(
    y: np.ndarray,
    miner_exponent: float = 5.0,
    min_prominence: float | None = None,
) -> StrainResult:
    """
    Вычисляет набор характеристик тензосигнала на участке обработки.

    Параметры:
    ----------
    y : np.ndarray
        Массив отсчётов (деформации/напряжения).
    miner_exponent : float
        Показатель степени m в гипотезе Минера для эквивалентного полуразмаха (по умолчанию 5).
    min_prominence : float | None
        Минимальная «выпуклость» экстремума, чтобы не считать шум циклом.
        Если None — 1% от размаха (y_max - y_min) по всему участку.

    Возвращает:
    -----------
    StrainResult
        Минимум, максимум, среднее, число циклов, макс. полуразмах,
        мин/макс квазистатика, эквивалентный полуразмах (Минера).
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return StrainResult(
            y_min=np.nan, y_max=np.nan, y_mean=np.nan,
            n_cycles=0, max_half_range=np.nan, min_quasi_static=np.nan,
            max_quasi_static=np.nan, equivalent_half_range=np.nan,
        )

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_mean = float(np.nanmean(y))

    extrema = _find_local_extrema(y, min_prominence)
    if len(extrema) < 2:
        return StrainResult(
            y_min=y_min, y_max=y_max, y_mean=y_mean,
            n_cycles=0, max_half_range=np.nan, min_quasi_static=np.nan,
            max_quasi_static=np.nan, equivalent_half_range=np.nan,
        )

    half_ranges, quasi_statics = _cycles_from_extrema(y, extrema)
    if not half_ranges:
        return StrainResult(
            y_min=y_min, y_max=y_max, y_mean=y_mean,
            n_cycles=0, max_half_range=np.nan, min_quasi_static=np.nan,
            max_quasi_static=np.nan, equivalent_half_range=np.nan,
        )

    max_hr = float(np.max(half_ranges))
    min_qs = float(np.min(quasi_statics))
    max_qs = float(np.max(quasi_statics))

    # Эквивалентный полуразмах по Минеру: a_eq = (sum a_k^m)^(1/m)
    a_arr = np.array(half_ranges, dtype=float)
    m = float(miner_exponent)
    if m <= 0 or not np.isfinite(m):
        m = 5.0
    a_eq = float(np.power(np.sum(np.power(a_arr, m)), 1.0 / m))

    return StrainResult(
        y_min=y_min,
        y_max=y_max,
        y_mean=y_mean,
        n_cycles=len(half_ranges),
        max_half_range=max_hr,
        min_quasi_static=min_qs,
        max_quasi_static=max_qs,
        equivalent_half_range=a_eq,
    )


def _find_local_extrema(y: np.ndarray, min_prominence: float | None) -> List[Tuple[int, str]]:
    """
    Находит локальные экстремумы: список (индекс, 'max'|'min') в порядке появления.
    """
    n = y.size
    if n < 3:
        return []

    if min_prominence is None:
        span = np.nanmax(y) - np.nanmin(y)
        min_prominence = (span * 0.01) if span > 0 else 0.0

    result: List[Tuple[int, str]] = []

    for i in range(1, n - 1):
        if not np.isfinite(y[i]):
            continue
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            # локальный максимум
            if min_prominence <= 0 or (y[i] - max(y[i - 1], y[i + 1]) >= min_prominence):
                result.append((i, "max"))
        elif y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            # локальный минимум
            if min_prominence <= 0 or (min(y[i - 1], y[i + 1]) - y[i] >= min_prominence):
                result.append((i, "min"))

    return result


def _cycles_from_extrema(
    y: np.ndarray,
    extrema: List[Tuple[int, str]],
) -> Tuple[List[float], List[float]]:
    """
    По списку экстремумов (индекс, тип) формирует циклы: полуразмах и квазистатическое значение.
    Цикл — два подряд идущих экстремума противоположного типа.
    """
    half_ranges: List[float] = []
    quasi_statics: List[float] = []

    for k in range(len(extrema) - 1):
        idx_a, kind_a = extrema[k]
        idx_b, kind_b = extrema[k + 1]
        if kind_a == kind_b:
            continue
        v_a = float(y[idx_a])
        v_b = float(y[idx_b])
        M_k = max(v_a, v_b)
        m_k = min(v_a, v_b)
        half_ranges.append((M_k - m_k) / 2.0)
        quasi_statics.append((M_k + m_k) / 2.0)

    return half_ranges, quasi_statics


# --- Самопроверка: запуск модуля как скрипта ---
if __name__ == "__main__":
    # Простой тестовый сигнал: несколько «циклов» (пики и впадины)
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    # Синус с шумом — даёт чередующиеся макс/мин
    y = 100 * np.sin(2 * np.pi * 0.5 * t) + 2 * np.random.randn(len(t))

    r = compute_strain_characteristics(y, miner_exponent=5.0)
    print("Strain characteristics (test signal):")
    print(f"  y_min = {r.y_min:.4f}")
    print(f"  y_max = {r.y_max:.4f}")
    print(f"  y_mean = {r.y_mean:.4f}")
    print(f"  n_cycles = {r.n_cycles}")
    print(f"  max_half_range = {r.max_half_range:.4f}")
    print(f"  min_quasi_static = {r.min_quasi_static:.4f}")
    print(f"  max_quasi_static = {r.max_quasi_static:.4f}")
    print(f"  equivalent_half_range (Miner) = {r.equivalent_half_range:.4f}")
    assert r.n_cycles >= 1
    assert r.y_min <= r.y_mean <= r.y_max
    print("OK: self-check passed.")
