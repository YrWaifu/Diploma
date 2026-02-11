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

    # По формуле из ВКР: a_eq = ((1/M) * Σ a_k^m)^(1/m)
    a_arr = np.array(half_ranges, dtype=float)
    m = float(miner_exponent)
    if m <= 0 or not np.isfinite(m):
        m = 5.0
    M = a_arr.size
    a_eq = float(np.power(np.sum(np.power(a_arr, m)) / M, 1.0 / m))

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

    # Значения проминентности по умолчанию — 1% от размаха по всему участку.
    if min_prominence is None:
        span = np.nanmax(y) - np.nanmin(y)
        min_prominence = (span * 0.01) if span > 0 else 0.0

    y = np.asarray(y, dtype=float)
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]

    # Маска валидных тройек (без NaN/inf), чтобы не городить сложные проверки в питоне.
    valid = np.isfinite(y0) & np.isfinite(y1) & np.isfinite(y2)

    # Локальные максимумы и минимумы векторно.
    is_max = (y1 >= y0) & (y1 >= y2)
    is_min = (y1 <= y0) & (y1 <= y2)

    # Проминентность
    max_prom = y1 - np.maximum(y0, y2)
    min_prom = np.minimum(y0, y2) - y1

    max_mask = valid & is_max & ((min_prominence <= 0) | (max_prom >= min_prominence))
    min_mask = valid & is_min & ((min_prominence <= 0) | (min_prom >= min_prominence))

    idx_base = np.arange(1, n - 1)
    max_idx = idx_base[max_mask]
    min_idx = idx_base[min_mask]

    # Объединяем и сортируем по возрастанию индекса, сохраняя тип экстремума.
    kinds: List[Tuple[int, str]] = []
    for i in max_idx:
        kinds.append((int(i), "max"))
    for i in min_idx:
        kinds.append((int(i), "min"))
    kinds.sort(key=lambda t: t[0])
    return kinds


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
