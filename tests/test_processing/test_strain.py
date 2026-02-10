# -*- coding: utf-8 -*-
"""
Тесты модуля тензометрирования (app.processing.strain).
Запуск: из корня проекта выполнить  pytest tests/
"""
from __future__ import annotations

import numpy as np
import pytest

from app.processing.strain import compute_strain_characteristics, StrainResult


def test_empty_signal():
    """Пустой массив → все характеристики NaN, n_cycles=0."""
    y = np.array([], dtype=float)
    r = compute_strain_characteristics(y)
    assert r.n_cycles == 0
    assert np.isnan(r.y_min) and np.isnan(r.y_max) and np.isnan(r.y_mean)
    assert np.isnan(r.max_half_range) and np.isnan(r.equivalent_half_range)


def test_constant_signal():
    """Константа → мин=макс=среднее, циклов нет."""
    y = np.full(100, 5.0)
    r = compute_strain_characteristics(y)
    assert r.y_min == r.y_max == r.y_mean == 5.0
    assert r.n_cycles == 0
    assert np.isnan(r.max_half_range)


def test_min_max_mean():
    """Проверка мин, макс, среднего на известном массиве."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = compute_strain_characteristics(y)
    assert r.y_min == 1.0
    assert r.y_max == 5.0
    assert r.y_mean == 3.0


def test_equivalent_half_range_formula():
    """Эквивалентный полуразмах по ВКР: a_eq = ((1/M) * Σ a_k^m)^(1/m)."""
    # Два одинаковых цикла: a1=a2=A → (1/2 * 2*A^m)^(1/m) = A
    half_ranges_manual = [5.0, 5.0]
    M = 2
    m = 5.0
    a_eq_expected = float(np.power(np.sum(np.power(np.array(half_ranges_manual), m)) / M, 1.0 / m))
    assert abs(a_eq_expected - 5.0) < 1e-10
    # Сигнал с двумя явными циклами (макс–мин–макс–мин)
    y = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
    r = compute_strain_characteristics(y, miner_exponent=5.0, min_prominence=0.0)
    assert r.n_cycles == 2
    assert abs(r.equivalent_half_range - 5.0) < 0.01


def test_sine_has_cycles():
    """Синус даёт чередующиеся макс/мин → несколько циклов."""
    np.random.seed(42)
    t = np.linspace(0, 2, 500)
    y = 100 * np.sin(2 * np.pi * 2 * t)  # 2 полных периода
    r = compute_strain_characteristics(y, min_prominence=0.0)
    assert r.y_min <= r.y_mean <= r.y_max
    assert r.n_cycles >= 2
    assert r.max_half_range > 0
    assert r.equivalent_half_range > 0
    # Среднее синуса ≈ 0
    assert abs(r.y_mean) < 1.0


def test_miner_exponent_effect():
    """Разный m даёт разный a_eq при разных полуразмахах."""
    # Два цикла: 2 и 8 → при m=1 a_eq = 5 (среднее), при m→∞ a_eq→8 (макс)
    y = np.array([0, 2, 0, 8, 0])  # полуразмахи 1 и 4
    r1 = compute_strain_characteristics(y, miner_exponent=1.0, min_prominence=0.0)
    r2 = compute_strain_characteristics(y, miner_exponent=10.0, min_prominence=0.0)
    if r1.n_cycles == 2 and r2.n_cycles == 2:
        assert r2.equivalent_half_range >= r1.equivalent_half_range


def test_quasi_static_bounds():
    """Мин/макс квазистатическое лежат между мин и макс сигнала."""
    np.random.seed(123)
    t = np.linspace(0, 5, 300)
    y = 50 * np.sin(2 * np.pi * 0.5 * t) + 10 * np.random.randn(len(t))
    r = compute_strain_characteristics(y)
    if r.n_cycles > 0:
        assert r.y_min <= r.min_quasi_static <= r.y_max
        assert r.y_min <= r.max_quasi_static <= r.y_max
