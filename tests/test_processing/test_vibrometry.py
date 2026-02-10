# -*- coding: utf-8 -*-
"""
Тесты модуля виброметрии (app.processing.vibrometry).
Запуск: из корня проекта выполнить  pytest tests/
"""
from __future__ import annotations

import numpy as np
import pytest

from app.processing.vibrometry import (
    compute_time_characteristics,
    VibrometryTimeResult,
    psd_welch,
    rms_in_band_from_psd,
    compute_vibrometry,
    estimate_fs_from_time,
    VibrometryResult,
)


def test_estimate_fs_from_time():
    """Частота дискретизации: fs = 1/Δt по равномерной сетке."""
    t = np.linspace(0, 1, 1001)  # 1000 интервалов, Δt=0.001, fs=1000
    fs = estimate_fs_from_time(t)
    assert np.isfinite(fs)
    assert abs(fs - 1000.0) < 0.1
    # Один элемент — NaN
    assert np.isnan(estimate_fs_from_time(np.array([1.0])))
    # Два элемента
    fs2 = estimate_fs_from_time(np.array([0.0, 0.001]))
    assert abs(fs2 - 1000.0) < 0.01


def test_time_characteristics_empty():
    """Пустой сигнал → все NaN."""
    t = compute_time_characteristics(np.array([]))
    assert np.isnan(t.mean) and np.isnan(t.rms) and np.isnan(t.crest_factor)


def test_time_characteristics_constant():
    """Константа c → μ=c, x₀=0, RMS=0, пик=0, CF=nan или 0."""
    x = np.full(50, 3.0)
    t = compute_time_characteristics(x)
    assert t.mean == 3.0
    assert t.rms == 0.0
    assert t.peak == 0.0
    assert t.peak_to_peak == 0.0


def test_time_characteristics_sine():
    """Синус A*sin(ωt): μ≈0, RMS=A/√2, пик=A, CF=√2."""
    n = 1000
    t_arr = np.linspace(0, 1, n)
    A = 2.0
    x = A * np.sin(2 * np.pi * 10 * t_arr)
    res = compute_time_characteristics(x)
    assert abs(res.mean) < 0.001
    assert abs(res.rms - A / np.sqrt(2)) < 0.01
    assert abs(res.peak - A) < 0.01
    # Пик-фактор синуса = √2
    assert abs(res.crest_factor - np.sqrt(2)) < 0.05


def test_psd_welch_shape_and_resolution():
    """PSD Уэлча: длина частот = L/2+1, Δf = fs/L."""
    np.random.seed(42)
    fs = 1000.0
    n = 2000
    x = np.random.randn(n)
    freqs, P_xx = psd_welch(x, fs, segment_length=256)
    assert len(freqs) == len(P_xx)
    assert len(freqs) == 256 // 2 + 1  # rfft length
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0
    assert abs(df - fs / 256) < 0.01
    assert freqs[0] == 0.0
    assert freqs[-1] == fs / 2


def test_psd_welch_empty_or_short():
    """Короткий или пустой сигнал → пустые массивы."""
    f, P = psd_welch(np.array([1.0, 2.0]), 1000.0)
    assert len(f) == 0 and len(P) == 0
    f2, P2 = psd_welch(np.array([]), 1000.0)
    assert len(f2) == 0 and len(P2) == 0


def test_rms_in_band_pure_tone():
    """СКЗ в полосе для чистого тона близко к амплитуде/√2."""
    fs = 1000.0
    n = 4000
    t = np.linspace(0, n / fs, n)
    A = 1.0
    f0 = 50.0
    x = A * np.sin(2 * np.pi * f0 * t)
    freqs, P_xx = psd_welch(x, fs, segment_length=512)
    rms_band = rms_in_band_from_psd(freqs, P_xx, 10.0, 90.0, fs, 512)
    expected_rms = A / np.sqrt(2)  # RMS синуса
    assert abs(rms_band - expected_rms) < 0.1


def test_compute_vibrometry_full():
    """Полный расчёт: временные характеристики + СКЗ в полосе."""
    fs = 1000.0
    n = 2000
    t = np.linspace(0, n / fs, n)
    x = 0.5 * np.sin(2 * np.pi * 20 * t) + 0.05 * np.random.randn(n)
    r = compute_vibrometry(x, fs, band_f1_hz=10.0, band_f2_hz=30.0)
    assert r.n_samples == n
    assert r.fs_hz == fs
    assert r.time.rms > 0
    assert r.rms_in_band is not None
    assert r.band_f1_hz == 10.0 and r.band_f2_hz == 30.0
    # СКЗ в полосе с 20 Гц тоном должен быть близок к 0.5/√2
    assert 0.2 < r.rms_in_band < 0.5


def test_compute_vibrometry_no_band():
    """Без полосы → только временные характеристики, rms_in_band=None."""
    x = np.random.randn(500)
    r = compute_vibrometry(x, 1000.0, band_f1_hz=None, band_f2_hz=None)
    assert r.rms_in_band is None
    assert r.band_f1_hz is None and r.band_f2_hz is None
    assert r.time.rms > 0


def test_compute_vibrometry_empty():
    """Пустой сигнал → n_samples=0, временные поля NaN."""
    r = compute_vibrometry(np.array([]), 1000.0)
    assert r.n_samples == 0
    assert np.isnan(r.time.mean)
