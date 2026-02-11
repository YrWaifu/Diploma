# -*- coding: utf-8 -*-
"""
Вычислительный слой: методы вторичной обработки сигналов.
Не зависит от GUI. Вход: массивы (t, y) и параметры. Выход: характеристики или новый ряд.
"""
from app.processing.strain import compute_strain_characteristics, StrainResult
from app.processing.vibrometry import (
    compute_vibrometry,
    compute_time_characteristics,
    VibrometryResult,
    VibrometryTimeResult,
    estimate_fs_from_time,
)
from app.processing.modes import (
    ModeSwitch,
    detect_mode_levels,
    detect_mode_switches,
)

__all__ = [
    "compute_strain_characteristics",
    "StrainResult",
    "compute_vibrometry",
    "compute_time_characteristics",
    "VibrometryResult",
    "VibrometryTimeResult",
    "estimate_fs_from_time",
    "ModeSwitch",
    "detect_mode_levels",
    "detect_mode_switches",
]
