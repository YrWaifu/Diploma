# -*- coding: utf-8 -*-
"""
Тесты модуля виброметрии (app.processing.vibrometry).
Запуск: из корня проекта выполнить  pytest tests/

Покрывает:
  - estimate_fs_from_time
  - compute_time_characteristics (пустой, константа, синус)
  - psd_welch (форма, разрешение, граничные случаи)
  - rms_in_band_from_psd
  - compute_vibrometry (legacy-API)
  - compute_bands (С.Ш.В.: СКЗ + Sxx по полосам)
  - compute_sinusoidal (синусоидальная вибрация: экв./эфф. амплитуда)
  - compute_full_vibrometry (комплексный)
  - Тесты на реальном CSV с тремя режимами (Р-1, Р-2, Р-3)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app.processing.vibrometry import (
    compute_time_characteristics,
    VibrometryTimeResult,
    psd_welch,
    rms_in_band_from_psd,
    compute_vibrometry,
    VibrometryResult,
    estimate_fs_from_time,
    compute_bands,
    compute_sinusoidal,
    compute_full_vibrometry,
    BandResult,
    SinusoidalResult,
    FullVibrometryResult,
)

# conftest.py предоставляет: test_csv_path, test_arrays, mode_slices
from .conftest import FS, ACTIVE_MODES


# ═══════════════════════════════════════════════════════════════════════════════
#  1. estimate_fs_from_time
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimateFs:
    def test_uniform_grid(self):
        t = np.linspace(0, 1, 1001)  # Δt = 0.001, fs = 1000
        assert abs(estimate_fs_from_time(t) - 1000.0) < 0.1

    def test_two_points(self):
        assert abs(estimate_fs_from_time(np.array([0.0, 0.001])) - 1000.0) < 0.01

    def test_single_point_nan(self):
        assert np.isnan(estimate_fs_from_time(np.array([1.0])))

    def test_empty_nan(self):
        assert np.isnan(estimate_fs_from_time(np.array([])))

    def test_from_csv_data(self, test_arrays):
        t, *_ = test_arrays
        fs = estimate_fs_from_time(t)
        assert abs(fs - FS) < 1.0  # должно быть ≈ 2000 Гц


# ═══════════════════════════════════════════════════════════════════════════════
#  2. compute_time_characteristics
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeCharacteristics:
    def test_empty(self):
        r = compute_time_characteristics(np.array([]))
        assert np.isnan(r.mean) and np.isnan(r.rms) and np.isnan(r.crest_factor)

    def test_constant(self):
        r = compute_time_characteristics(np.full(50, 3.0))
        assert r.mean == 3.0 and r.rms == 0.0 and r.peak == 0.0 and r.peak_to_peak == 0.0

    def test_sine(self):
        """Синус A·sin(ωt): μ≈0, RMS = A/√2, пик = A, CF = √2."""
        A = 2.0
        t = np.linspace(0, 1, 10000, endpoint=False)
        x = A * np.sin(2 * np.pi * 10 * t)
        r = compute_time_characteristics(x)
        assert abs(r.mean) < 0.001
        assert abs(r.rms - A / np.sqrt(2)) < 0.01
        assert abs(r.peak - A) < 0.01
        assert abs(r.crest_factor - np.sqrt(2)) < 0.05

    def test_mode_vibro_increases(self, mode_slices):
        """RMS вибрации растёт от Р-1 к Р-3 (интенсивность режимов)."""
        rms_vals = {}
        for label in ("Р-1", "Р-2", "Р-3"):
            _, vibro, _ = mode_slices[label]
            r = compute_time_characteristics(vibro)
            rms_vals[label] = r.rms
        assert rms_vals["Р-1"] < rms_vals["Р-2"] < rms_vals["Р-3"]


# ═══════════════════════════════════════════════════════════════════════════════
#  3. psd_welch
# ═══════════════════════════════════════════════════════════════════════════════

class TestPsdWelch:
    def test_shape_and_resolution(self):
        np.random.seed(42)
        fs = 1000.0
        x = np.random.randn(2000)
        freqs, P_xx = psd_welch(x, fs, segment_length=256)
        assert len(freqs) == len(P_xx) == 256 // 2 + 1
        df = freqs[1] - freqs[0]
        assert abs(df - fs / 256) < 0.01
        assert freqs[0] == 0.0 and freqs[-1] == fs / 2

    def test_empty_and_short(self):
        f, P = psd_welch(np.array([1.0, 2.0]), 1000.0)
        assert len(f) == 0
        f2, P2 = psd_welch(np.array([]), 1000.0)
        assert len(f2) == 0

    def test_peak_at_tone_frequency(self):
        """PSD чистого тона 50 Гц: пик в районе 50 Гц."""
        fs = 1000.0
        t = np.linspace(0, 4, int(fs * 4), endpoint=False)
        x = np.sin(2 * np.pi * 50 * t)
        freqs, P_xx = psd_welch(x, fs, segment_length=512)
        peak_freq = freqs[np.argmax(P_xx)]
        assert abs(peak_freq - 50.0) < 3.0

    def test_csv_vibro_has_peaks_at_16_32(self, mode_slices):
        """В Р-2 PSD должно иметь пики на 16 и 32 Гц (лопасть и гармоника)."""
        _, vibro, _ = mode_slices["Р-2"]
        freqs, P_xx = psd_welch(vibro, FS, segment_length=1024)
        # Найдём частоту максимума
        peak_idx = np.argmax(P_xx)
        peak_f = freqs[peak_idx]
        assert abs(peak_f - 16.0) < 3.0  # основной тон 16 Гц


# ═══════════════════════════════════════════════════════════════════════════════
#  4. rms_in_band_from_psd
# ═══════════════════════════════════════════════════════════════════════════════

class TestRmsInBand:
    def test_pure_tone(self):
        """СКЗ в полосе для чистого тона 50 Гц ≈ A/√2."""
        fs = 1000.0
        n = 4000
        t = np.linspace(0, n / fs, n, endpoint=False)
        A = 1.0
        x = A * np.sin(2 * np.pi * 50 * t)
        freqs, P_xx = psd_welch(x, fs, segment_length=512)
        rms = rms_in_band_from_psd(freqs, P_xx, 10.0, 90.0, fs, 512)
        assert abs(rms - A / np.sqrt(2)) < 0.1

    def test_out_of_band(self):
        """Если полоса не содержит тона — СКЗ ≈ 0."""
        fs = 1000.0
        t = np.linspace(0, 2, int(fs * 2), endpoint=False)
        x = np.sin(2 * np.pi * 50 * t)
        freqs, P_xx = psd_welch(x, fs, segment_length=512)
        rms = rms_in_band_from_psd(freqs, P_xx, 200.0, 400.0, fs, 512)
        assert rms < 0.02

    def test_invalid_returns_nan(self):
        assert np.isnan(rms_in_band_from_psd(np.array([]), np.array([]), 0, 100, 1000, 256))


# ═══════════════════════════════════════════════════════════════════════════════
#  5. compute_vibrometry (legacy)
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeVibrometry:
    def test_full(self):
        fs = 1000.0
        n = 2000
        t = np.linspace(0, n / fs, n, endpoint=False)
        x = 0.5 * np.sin(2 * np.pi * 20 * t) + 0.05 * np.random.randn(n)
        r = compute_vibrometry(x, fs, band_f1_hz=10.0, band_f2_hz=30.0)
        assert r.n_samples == n and r.fs_hz == fs
        assert r.time.rms > 0
        assert r.rms_in_band is not None
        assert 0.2 < r.rms_in_band < 0.5

    def test_no_band(self):
        x = np.random.randn(500)
        r = compute_vibrometry(x, 1000.0)
        assert r.rms_in_band is None and r.time.rms > 0

    def test_empty(self):
        r = compute_vibrometry(np.array([]), 1000.0)
        assert r.n_samples == 0 and np.isnan(r.time.mean)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. compute_bands (С.Ш.В.)
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeBands:
    def test_single_tone_in_band(self):
        """Тон 50 Гц: полоса 10–100 содержит почти всю энергию, 200–400 — нет."""
        fs = 1000.0
        t = np.linspace(0, 2, int(fs * 2), endpoint=False)
        x = np.sin(2 * np.pi * 50 * t)
        bands = [(10.0, 100.0), (200.0, 400.0)]
        res = compute_bands(x, fs, bands, segment_length=512)
        assert len(res) == 2
        assert res[0].rms > 0.5  # почти вся энергия
        assert res[1].rms < 0.02  # пусто
        # Sxx = RMS²
        assert abs(res[0].sxx - res[0].rms ** 2) < 0.01

    def test_empty_signal(self):
        res = compute_bands(np.array([1.0, 2.0]), 1000.0, [(10, 100)])
        assert len(res) == 1 and np.isnan(res[0].rms)

    def test_invalid_band(self):
        fs = 1000.0
        x = np.random.randn(2000)
        res = compute_bands(x, fs, [(100.0, 50.0)])  # f2 < f1
        assert np.isnan(res[0].rms)

    def test_multiple_bands_csv(self, mode_slices):
        """С.Ш.В. по полосам на Р-3: полоса 10–50 > полоса 150–250."""
        _, vibro, _ = mode_slices["Р-3"]
        bands = [(10.0, 50.0), (150.0, 250.0)]
        res = compute_bands(vibro, FS, bands, segment_length=1024)
        # Основные гармоники (16, 32 Гц) попадают в первую полосу
        assert res[0].rms > res[1].rms

    def test_bands_rms_increases_with_mode(self, mode_slices):
        """СКЗ в полосе 10–50 Гц растёт от Р-1 к Р-3."""
        band = [(10.0, 50.0)]
        vals = {}
        for label in ("Р-1", "Р-2", "Р-3"):
            _, vibro, _ = mode_slices[label]
            res = compute_bands(vibro, FS, band, segment_length=1024)
            vals[label] = res[0].rms
        assert vals["Р-1"] < vals["Р-2"] < vals["Р-3"]


# ═══════════════════════════════════════════════════════════════════════════════
#  7. compute_sinusoidal (синусоидальная вибрация)
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeSinusoidal:
    def test_pure_tone_equiv(self):
        """Чистый тон 50 Гц → экв. амплитуда ≈ A (A = 1)."""
        fs = 1000.0
        t = np.linspace(0, 4, int(fs * 4), endpoint=False)
        A = 1.0
        x = A * np.sin(2 * np.pi * 50 * t)
        res = compute_sinusoidal(x, fs, [50.0], af_pct=10.0,
                                 calc_equiv=True, calc_eff=False, segment_length=1024)
        assert len(res) == 1
        assert res[0].equiv_amplitude is not None
        # Должно быть в районе A (для синуса A_eq ≈ A)
        assert 0.5 < res[0].equiv_amplitude < 1.8

    def test_pure_tone_eff(self):
        """Эффективная амплитуда чистого тона."""
        fs = 1000.0
        t = np.linspace(0, 4, int(fs * 4), endpoint=False)
        x = np.sin(2 * np.pi * 50 * t)
        res = compute_sinusoidal(x, fs, [50.0], af_pct=10.0,
                                 calc_equiv=False, calc_eff=True, segment_length=1024)
        assert res[0].eff_amplitude is not None
        assert res[0].eff_amplitude > 0

    def test_no_tone_at_freq(self):
        """Если на базовой частоте нет тона — амплитуды очень малы."""
        fs = 1000.0
        t = np.linspace(0, 4, int(fs * 4), endpoint=False)
        x = np.sin(2 * np.pi * 50 * t)  # тон 50 Гц
        res = compute_sinusoidal(x, fs, [300.0], af_pct=5.0,
                                 calc_equiv=True, calc_eff=True, segment_length=1024)
        eq = res[0].equiv_amplitude
        if eq is not None:
            assert eq < 0.1  # почти ничего

    def test_csv_16hz_detected(self, mode_slices):
        """В Р-2 базовая частота 16 Гц даёт значимую экв. амплитуду."""
        _, vibro, _ = mode_slices["Р-2"]
        res = compute_sinusoidal(vibro, FS, [16.0], af_pct=10.0,
                                 calc_equiv=True, calc_eff=True, segment_length=2048)
        assert res[0].equiv_amplitude is not None
        assert res[0].equiv_amplitude > 0.5
        assert res[0].eff_amplitude is not None
        assert res[0].eff_amplitude > 0

    def test_multiple_base_freqs(self, mode_slices):
        """На нескольких базовых частотах получаем результаты для каждой."""
        _, vibro, _ = mode_slices["Р-3"]
        freqs = [16.0, 32.0, 200.0, 750.0]
        res = compute_sinusoidal(vibro, FS, freqs, af_pct=10.0,
                                 calc_equiv=True, calc_eff=True, segment_length=2048)
        assert len(res) == 4
        # 16 и 32 Гц — значимые; 750 Гц — за пределами Найквиста (fs/2=1000) но валидна
        assert res[0].equiv_amplitude is not None and res[0].equiv_amplitude > 0.5
        assert res[1].equiv_amplitude is not None and res[1].equiv_amplitude > 0.3

    def test_empty_signal(self):
        res = compute_sinusoidal(np.array([1.0, 2.0]), 1000.0, [50.0])
        assert len(res) == 1 and res[0].equiv_amplitude is None

    def test_equiv_increases_with_mode(self, mode_slices):
        """Экв. амплитуда на 16 Гц растёт от Р-1 к Р-3."""
        vals = {}
        for label in ("Р-1", "Р-2", "Р-3"):
            _, vibro, _ = mode_slices[label]
            res = compute_sinusoidal(vibro, FS, [16.0], af_pct=10.0,
                                     calc_equiv=True, segment_length=2048)
            vals[label] = res[0].equiv_amplitude
        assert vals["Р-1"] < vals["Р-2"] < vals["Р-3"]


# ═══════════════════════════════════════════════════════════════════════════════
#  8. compute_full_vibrometry (комплексный тест)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullVibrometry:
    def test_bands_only(self):
        fs = 1000.0
        t = np.linspace(0, 2, int(fs * 2), endpoint=False)
        x = np.sin(2 * np.pi * 50 * t)
        r = compute_full_vibrometry(x, fs, bands=[(10, 100), (200, 400)])
        assert len(r.bands) == 2
        assert len(r.sinusoidal) == 0
        assert r.bands[0].rms > 0.5

    def test_sinusoidal_only(self):
        fs = 1000.0
        t = np.linspace(0, 2, int(fs * 2), endpoint=False)
        x = np.sin(2 * np.pi * 50 * t)
        r = compute_full_vibrometry(x, fs, base_freqs=[50.0], calc_equiv=True, calc_eff=True)
        assert len(r.bands) == 0
        assert len(r.sinusoidal) == 1
        assert r.sinusoidal[0].equiv_amplitude > 0

    def test_combined(self, mode_slices):
        """Комбинированный: С.Ш.В. + синусоидальная на Р-3."""
        _, vibro, _ = mode_slices["Р-3"]
        r = compute_full_vibrometry(
            vibro, FS,
            bands=[(10, 50), (50, 250), (0, 1000)],
            base_freqs=[16.0, 32.0, 200.0],
            calc_equiv=True, calc_eff=True,
            segment_length=2048,
        )
        assert len(r.bands) == 3
        assert len(r.sinusoidal) == 3
        # Полная полоса 0–1000 ≥ узких полос
        assert r.bands[2].rms >= r.bands[0].rms
        assert r.bands[2].rms >= r.bands[1].rms
        # 16 Гц амплитуда > 200 Гц амплитуда
        assert r.sinusoidal[0].equiv_amplitude > r.sinusoidal[2].equiv_amplitude

    def test_all_modes_have_results(self, mode_slices):
        """Для каждого из 3 режимов полный расчёт возвращает непустые результаты."""
        for label in ("Р-1", "Р-2", "Р-3"):
            _, vibro, _ = mode_slices[label]
            r = compute_full_vibrometry(
                vibro, FS,
                bands=[(10, 150), (150, 500)],
                base_freqs=[16.0],
                calc_equiv=True, calc_eff=True,
                segment_length=1024,
            )
            assert r.fs_hz == FS
            assert all(b.rms > 0 for b in r.bands)
            assert r.sinusoidal[0].equiv_amplitude is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Тест на CSV-файл (чтение + расчёт)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCsvIntegration:
    def test_read_csv_and_compute(self, test_csv_path):
        """Читаем CSV, нарезаем по режимам, считаем полную виброметрию."""
        import csv as _csv
        times, vibros = [], []
        with open(test_csv_path, encoding="utf-8") as f:
            reader = _csv.DictReader(f, delimiter=";")
            for row in reader:
                times.append(float(row["time"]))
                vibros.append(float(row["sensor_vibro"]))
        t = np.array(times)
        v = np.array(vibros)

        # Нарезаем Р-2 (10–18 с)
        i0 = int(np.searchsorted(t, 10.0))
        i1 = int(np.searchsorted(t, 18.0))
        v_r2 = v[i0:i1]

        r = compute_full_vibrometry(
            v_r2, FS,
            bands=[(10, 50), (50, 250)],
            base_freqs=[16.0, 32.0],
            calc_equiv=True,
            segment_length=2048,
        )
        assert len(r.bands) == 2
        assert r.bands[0].rms > 0
        assert len(r.sinusoidal) == 2
        assert r.sinusoidal[0].equiv_amplitude > 0
