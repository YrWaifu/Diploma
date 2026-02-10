# -*- coding: utf-8 -*-
"""
Вычислительный слой: методы вторичной обработки сигналов.
Не зависит от GUI. Вход: массивы (t, y) и параметры. Выход: характеристики или новый ряд.
"""
from app.processing.strain import compute_strain_characteristics, StrainResult

__all__ = ["compute_strain_characteristics", "StrainResult"]
