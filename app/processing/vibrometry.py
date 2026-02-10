# -*- coding: utf-8 -*-
"""
Виброметрия: временные характеристики, PSD (Уэлч), СКЗ в полосе.
Формулы по документу ВКР (обозначения, спектр, PSD, связь PSD–СКЗ).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class VibrometryTimeResult:
    """Временные характеристики по документу: μ, x₀, СКЗ, пик, пик-пик, пик-фактор."""
    mean: float          # μ = (1/N) Σ x[n]
    rms: float           # СКЗ по центрированному сигналу
    peak: float          # max |x₀[n]|
    peak_to_peak: float  # max x₀ - min x₀
    crest_factor: float  # x_peak / x_rms


@dataclass
class VibrometryResult:
    """Результат анализа вибросигнала: временные характеристики + опционально СКЗ в полосе."""
    time: VibrometryTimeResult
    fs_hz: float
    n_samples: int
    # СКЗ в полосе [f1, f2] (если задана полоса)
    rms_in_band: Optional[float] = None
    band_f1_hz: Optional[float] = None
    band_f2_hz: Optional[float] = None


def compute_time_characteristics(x: np.ndarray) -> VibrometryTimeResult:
    """
    Временные характеристики по документу:
    μ = (1/N) Σ x[n], x₀[n] = x[n] - μ,
    x_rms = √[(1/N) Σ x₀²[n]], x_peak = max |x₀[n]|, x_pp = max x₀ - min x₀, CF = x_peak / x_rms.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return VibrometryTimeResult(mean=np.nan, rms=np.nan, peak=np.nan, peak_to_peak=np.nan, crest_factor=np.nan)
    n = x.size
    mu = float(np.mean(x))
    x0 = x - mu
    x_rms = float(np.sqrt(np.mean(x0 ** 2)))
    x_peak = float(np.max(np.abs(x0)))
    x_pp = float(np.max(x0) - np.min(x0))
    cf = float(x_peak / x_rms) if x_rms > 0 else np.nan
    return VibrometryTimeResult(mean=mu, rms=x_rms, peak=x_peak, peak_to_peak=x_pp, crest_factor=cf)


def _hann_window(n: int) -> np.ndarray:
    """Окно Ханна: w[n] = 0.5 * (1 - cos(2πn/(N-1))), n = 0..N-1."""
    if n <= 1:
        return np.ones(n)
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))


def psd_welch(
    x: np.ndarray,
    fs_hz: float,
    segment_length: Optional[int] = None,
    overlap_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PSD методом Уэлча по документу.
    Периодограмма сегмента: P_xx(f) = (1/(fs*U)) * |X_w(f)|², U = (1/L) Σ w²[n].
    Уэлч: P̂_xx(f) = (1/M) Σ P_xx^(m)(f).

    Возвращает (freqs, P_xx): частоты в Гц и оценка PSD в единицах²/Гц.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4 or fs_hz <= 0:
        return np.array([]), np.array([])
    # Центрируем
    x0 = x - np.mean(x)
    L = segment_length if segment_length and segment_length >= 4 else min(256, n // 4)
    L = min(L, n)
    w = _hann_window(L)
    U = float(np.mean(w ** 2))  # (1/L) Σ w²[n]
    if U <= 0:
        U = 1.0
    step = max(1, int(L * (1 - overlap_ratio)))
    periodograms = []
    start = 0
    while start + L <= n:
        seg = x0[start : start + L] * w
        X = np.fft.rfft(seg)
        # Периодограмма по документу P_xx(f) = (1/(fs*U)) * |X_w|²; односторонняя: ×2 для f∈(0,fs/2), ×1/L по Parseval (sum P*Δf = x_rms²)
        P_seg = (2.0 / (fs_hz * U * L)) * (np.abs(X) ** 2)
        P_seg[0] *= 0.5   # DC без удвоения
        if P_seg.size > 1 and L % 2 == 0:
            P_seg[-1] *= 0.5   # Nyquist без удвоения
        periodograms.append(P_seg)
        start += step
    if not periodograms:
        return np.array([]), np.array([])
    P_xx = np.mean(periodograms, axis=0)
    freqs = np.fft.rfftfreq(L, 1.0 / fs_hz)
    return freqs, P_xx.astype(float)


def rms_in_band_from_psd(
    freqs: np.ndarray,
    P_xx: np.ndarray,
    f1_hz: float,
    f2_hz: float,
    fs_hz: float,
    segment_length: int,
) -> float:
    """
    СКЗ в полосе по документу: x_rms = √(Σ P̂_xx[k] * Δf), k ∈ [f1, f2].
    Δf = fs / N для сегмента длины N (разрешение по частоте Уэлча).
    """
    if freqs.size == 0 or P_xx.size == 0 or f2_hz <= f1_hz or segment_length < 2:
        return np.nan
    df = fs_hz / segment_length
    mask = (freqs >= f1_hz) & (freqs <= f2_hz)
    if not np.any(mask):
        return 0.0
    sum_psd_df = float(np.sum(P_xx[mask]) * df)
    return float(np.sqrt(max(0.0, sum_psd_df)))


def compute_vibrometry(
    x: np.ndarray,
    fs_hz: float,
    band_f1_hz: Optional[float] = None,
    band_f2_hz: Optional[float] = None,
    segment_length: Optional[int] = None,
) -> VibrometryResult:
    """
    Полный расчёт виброметрии: временные характеристики + при заданной полосе — СКЗ в полосе (по PSD Уэлча).
    """
    x = np.asarray(x, dtype=float)
    time_res = compute_time_characteristics(x)
    n = x.size
    if n == 0:
        return VibrometryResult(time=time_res, fs_hz=fs_hz, n_samples=0)

    rms_band = None
    b1, b2 = band_f1_hz, band_f2_hz
    if b1 is not None and b2 is not None and b2 > b1 and fs_hz > 0:
        freqs, P_xx = psd_welch(x, fs_hz, segment_length=segment_length)
        if freqs.size > 0:
            L = segment_length if segment_length and segment_length >= 4 else min(256, n // 4)
            L = min(L, n)
            rms_band = rms_in_band_from_psd(freqs, P_xx, b1, b2, fs_hz, L)

    return VibrometryResult(
        time=time_res,
        fs_hz=fs_hz,
        n_samples=n,
        rms_in_band=rms_band,
        band_f1_hz=b1,
        band_f2_hz=b2,
    )


def estimate_fs_from_time(t: np.ndarray) -> float:
    """Оценка частоты дискретизации по равномерной сетке времени: fs = 1/Δt."""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return np.nan
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        return np.nan
    med_dt = float(np.median(dt))
    if med_dt <= 0:
        return np.nan
    return 1.0 / med_dt
