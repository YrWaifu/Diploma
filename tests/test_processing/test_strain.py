# -*- coding: utf-8 -*-
"""
Тесты app.processing.strain.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app.processing.strain import compute_strain_characteristics, StrainResult

# conftest.py предоставляет: test_csv_path, test_arrays, mode_slices
from .conftest import FS, ACTIVE_MODES


#  1. Граничные случаи
class TestEdgeCases:
    def test_empty_signal(self):
        """Пустой массив → все NaN, n_cycles = 0."""
        r = compute_strain_characteristics(np.array([], dtype=float))
        assert r.n_cycles == 0
        assert np.isnan(r.y_min) and np.isnan(r.y_max) and np.isnan(r.y_mean)
        assert np.isnan(r.max_half_range) and np.isnan(r.equivalent_half_range)

    def test_constant_signal(self):
        """Константа → мин = макс = среднее, полуразмах = 0."""
        r = compute_strain_characteristics(np.full(100, 5.0))
        assert r.y_min == r.y_max == r.y_mean == 5.0
        # Алгоритм может находить «плоские» экстремумы, но полуразмах = 0
        assert r.max_half_range == 0.0 or np.isnan(r.max_half_range)

    def test_two_points(self):
        """Два элемента — слишком мало для экстремумов."""
        r = compute_strain_characteristics(np.array([1.0, 2.0]))
        assert r.y_min == 1.0 and r.y_max == 2.0
        assert r.n_cycles == 0

    def test_monotonic_signal(self):
        """Монотонно растущий → нет циклов."""
        r = compute_strain_characteristics(np.linspace(0, 100, 500))
        assert r.n_cycles == 0
        assert r.y_min == 0.0


#  2. Базовая статистика
class TestBasicStats:
    def test_min_max_mean(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = compute_strain_characteristics(y)
        assert r.y_min == 1.0 and r.y_max == 5.0 and r.y_mean == 3.0

    def test_negative_values(self):
        y = np.array([-10.0, 0.0, 10.0, 0.0, -10.0])
        r = compute_strain_characteristics(y, min_prominence=0.0)
        assert r.y_min == -10.0 and r.y_max == 10.0 and r.y_mean == pytest.approx(-2.0)

    def test_large_offset(self):
        """Большой offset не влияет на полуразмах."""
        offset = 1e6
        y = np.array([offset, offset + 10, offset, offset + 10, offset])
        r = compute_strain_characteristics(y, min_prominence=0.0)
        assert r.y_min == offset and r.y_max == offset + 10
        if r.n_cycles > 0:
            assert abs(r.max_half_range - 5.0) < 0.01


#  3. Подсчёт циклов
class TestCycles:
    def test_explicit_cycles(self):
        """Два одинаковых цикла: макс–мин–макс–мин."""
        y = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
        r = compute_strain_characteristics(y, min_prominence=0.0)
        assert r.n_cycles == 2
        assert abs(r.max_half_range - 5.0) < 0.01

    def test_sine_has_cycles(self):
        """Синус 2 Гц × 2 с = 4 периода → несколько циклов."""
        t = np.linspace(0, 2, 500, endpoint=False)
        y = 100 * np.sin(2 * np.pi * 2 * t)
        r = compute_strain_characteristics(y, min_prominence=0.0)
        assert r.n_cycles >= 4
        assert r.max_half_range > 90

    def test_different_amplitudes(self):
        """Циклы разной амплитуды: 10 и 40."""
        y = np.array([0, 10, 0, 40, 0, 10, 0])
        r = compute_strain_characteristics(y, min_prominence=0.0)
        assert r.n_cycles >= 2
        assert abs(r.max_half_range - 20.0) < 0.01  # (40-0)/2 = 20

    def test_noise_filtered_by_prominence(self):
        """Шум с min_prominence отфильтровывает мелкие колебания."""
        np.random.seed(7)
        y = 0.1 * np.random.randn(500)  # только шум, размах ~0.6
        r = compute_strain_characteristics(y, min_prominence=1.0)
        assert r.n_cycles == 0  # шум ниже порога


#  4. Формула Минера: эквивалентный полуразмах
class TestMiner:
    def test_equal_cycles(self):
        """Все циклы одинаковые → a_eq = a_k."""
        y = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
        r = compute_strain_characteristics(y, miner_exponent=5.0, min_prominence=0.0)
        assert r.n_cycles == 2
        assert abs(r.equivalent_half_range - 5.0) < 0.01  # (10-0)/2

    def test_miner_exponent_effect(self):
        """Разный m: при m→∞, a_eq→max(a_k); при m=1, a_eq = среднее."""
        y = np.array([0, 4, 0, 16, 0])  # полуразмахи 2 и 8
        r1 = compute_strain_characteristics(y, miner_exponent=1.0, min_prominence=0.0)
        r10 = compute_strain_characteristics(y, miner_exponent=10.0, min_prominence=0.0)
        if r1.n_cycles >= 2 and r10.n_cycles >= 2:
            # При большем m, a_eq ближе к максимальному полуразмаху
            assert r10.equivalent_half_range >= r1.equivalent_half_range

    def test_formula_manual(self):
        """Проверка формулы: a_eq = ((1/M) * Σ a_k^m)^(1/m) вручную."""
        # Сигнал с двумя одинаковыми циклами → a_eq = a = 5.0
        y = np.array([0.0, 10.0, 0.0, 10.0, 0.0])
        r = compute_strain_characteristics(y, miner_exponent=5.0, min_prominence=0.0)
        assert r.n_cycles >= 2
        # Все полуразмахи равны 5, поэтому a_eq = 5
        assert abs(r.equivalent_half_range - 5.0) < 0.1

    def test_formula_different_amplitudes(self):
        """Разные амплитуды: проверяем, что a_eq > min и < max полуразмахов."""
        # 0→4→0→16→0  полуразмахи: 2 и 8
        y = np.array([0.0, 4.0, 0.0, 16.0, 0.0])
        r = compute_strain_characteristics(y, miner_exponent=5.0, min_prominence=0.0)
        assert r.n_cycles >= 2
        assert 2.0 < r.equivalent_half_range < 8.0


#  5. Квазистатические границы
class TestQuasiStatic:
    def test_bounds(self):
        """Мин/макс квазистатического лежат между мин и макс сигнала."""
        np.random.seed(123)
        t = np.linspace(0, 5, 300, endpoint=False)
        y = 50 * np.sin(2 * np.pi * 0.5 * t) + 10 * np.random.randn(len(t))
        r = compute_strain_characteristics(y)
        if r.n_cycles > 0:
            assert r.y_min <= r.min_quasi_static <= r.y_max
            assert r.y_min <= r.max_quasi_static <= r.y_max

    def test_symmetric_sine(self):
        """Симметричный синус: квазистатика ≈ 0 (среднее цикла)."""
        t = np.linspace(0, 2, 1000, endpoint=False)
        y = 10 * np.sin(2 * np.pi * 3 * t)
        r = compute_strain_characteristics(y, min_prominence=0.0)
        if r.n_cycles > 0:
            assert abs(r.min_quasi_static) < 2.0
            assert abs(r.max_quasi_static) < 2.0


#  6. Тесты на CSV-данных: три режима
class TestCsvModes:
    def test_mode_means_increase(self, mode_slices):
        """Среднее тензосигнала растёт от Р-1 к Р-3 (нагрузка увеличивается)."""
        means = {}
        for label in ("Р-1", "Р-2", "Р-3"):
            _, _, strain_slice = mode_slices[label]
            r = compute_strain_characteristics(strain_slice)
            means[label] = r.y_mean
        assert means["Р-1"] < means["Р-2"] < means["Р-3"]

    def test_mode_max_half_range_increases(self, mode_slices):
        """Макс. полуразмах растёт от лёгкого к тяжёлому режиму."""
        hr = {}
        for label in ("Р-1", "Р-2", "Р-3"):
            _, _, strain_slice = mode_slices[label]
            r = compute_strain_characteristics(strain_slice, min_prominence=0.0)
            assert r.n_cycles > 0, f"В {label} не найдены циклы!"
            hr[label] = r.max_half_range
        assert hr["Р-1"] < hr["Р-2"] < hr["Р-3"]

    def test_all_modes_have_cycles(self, mode_slices):
        """В каждом активном режиме должны быть циклы."""
        for label in ("Р-1", "Р-2", "Р-3"):
            _, _, strain_slice = mode_slices[label]
            r = compute_strain_characteristics(strain_slice, min_prominence=0.0)
            assert r.n_cycles >= 4, f"{label}: ожидалось ≥4 циклов, получено {r.n_cycles}"

    def test_equiv_half_range_increases(self, mode_slices):
        """Эквивалентный полуразмах (Минер) растёт от Р-1 к Р-3."""
        ehr = {}
        for label in ("Р-1", "Р-2", "Р-3"):
            _, _, strain_slice = mode_slices[label]
            r = compute_strain_characteristics(strain_slice, miner_exponent=5.0, min_prominence=0.0)
            assert r.n_cycles > 0
            ehr[label] = r.equivalent_half_range
        assert ehr["Р-1"] < ehr["Р-2"] < ehr["Р-3"]

    def test_quasi_static_within_bounds(self, mode_slices):
        """Квазистатика каждого режима — внутри min/max сигнала."""
        for label in ("Р-1", "Р-2", "Р-3"):
            _, _, strain_slice = mode_slices[label]
            r = compute_strain_characteristics(strain_slice, min_prominence=0.0)
            if r.n_cycles > 0:
                assert r.y_min <= r.min_quasi_static <= r.y_max
                assert r.y_min <= r.max_quasi_static <= r.y_max

    def test_different_miner_exponents(self, mode_slices):
        """На Р-3 при m=10 a_eq ≥ a_eq при m=3 (больший m → ближе к max)."""
        _, _, strain = mode_slices["Р-3"]
        r3 = compute_strain_characteristics(strain, miner_exponent=3.0, min_prominence=0.0)
        r10 = compute_strain_characteristics(strain, miner_exponent=10.0, min_prominence=0.0)
        assert r3.n_cycles > 0 and r10.n_cycles > 0
        assert r10.equivalent_half_range >= r3.equivalent_half_range


#  7. Интеграционный тест: чтение CSV и расчёт
class TestCsvIntegration:
    def test_read_csv_and_compute(self, test_csv_path):
        """Полный цикл: чтение CSV → нарезка по режиму → расчёт характеристик."""
        import csv as _csv
        times, strains = [], []
        with open(test_csv_path, encoding="utf-8") as f:
            reader = _csv.DictReader(f, delimiter=";")
            for row in reader:
                times.append(float(row["time"]))
                strains.append(float(row["sensor_strain"]))
        t = np.array(times)
        s = np.array(strains)

        results = {}
        for label, t0, t1 in ACTIVE_MODES:
            i0 = int(np.searchsorted(t, t0))
            i1 = int(np.searchsorted(t, t1))
            seg = s[i0:i1]
            r = compute_strain_characteristics(seg, miner_exponent=5.0, min_prominence=0.0)
            results[label] = r
            assert r.n_cycles > 0, f"CSV {label}: нет циклов"
            assert np.isfinite(r.equivalent_half_range)

        # Проверяем монотонный рост нагрузки
        assert results["Р-1"].y_mean < results["Р-2"].y_mean < results["Р-3"].y_mean
        assert results["Р-1"].max_half_range < results["Р-2"].max_half_range < results["Р-3"].max_half_range
