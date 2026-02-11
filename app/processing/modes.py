from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ModeSwitch:
    """Событие переключения режима: переход между двумя уровнями."""
    index: int         # индекс выборки, ближайшей к моменту переключения
    t: float           # время переключения (с интерполяцией по X)
    from_level: float  # уровень «до»
    to_level: float    # уровень «после»


def detect_mode_levels(y: np.ndarray) -> Tuple[float, float, float]:
    """
    Определяет минимальный, максимальный и пороговый уровень для сигнала режима.

    Порог берётся как середина между глобальным минимумом и максимумом.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.nan, np.nan, np.nan
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    mid = 0.5 * (y_min + y_max)
    return y_min, y_max, mid


def detect_mode_switches(
    t: np.ndarray,
    y: np.ndarray,
    smooth_window: int | None = None,
) -> Tuple[List[ModeSwitch], float, float, float]:
    """
    Находит моменты переключения режима по сигналу уровня.

    Логика:
    - берём глобальные min/max и порог mid = (min+max)/2;
    - при переходе через порог (снизу вверх или сверху вниз) считаем, что режим сменился;
    - время события — линейная интерполяция между соседними точками вокруг порога.

    Возвращает:
        (список ModeSwitch, уровень_min, уровень_max, порог_mid)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(t.size, y.size)
    if n < 2:
        return [], np.nan, np.nan, np.nan
    t = t[:n]
    y = y[:n]

    y_min, y_max, mid = detect_mode_levels(y)
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return [], y_min, y_max, mid

    # Небольшое сглаживание, чтобы уменьшить влияние шума на пороге.
    if smooth_window is None:
        smooth_window = max(3, n // 2000)  # мягкая зависимость от размера массива
    if smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        k = np.ones(smooth_window, dtype=float) / smooth_window
        y_s = np.convolve(y, k, mode="same")
    else:
        y_s = y

    switches: List[ModeSwitch] = []
    for i in range(n - 1):
        y0 = float(y_s[i])
        y1 = float(y_s[i + 1])
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        # Проверяем переход через порог (в любую сторону), избегая нулевого шага.
        if (y0 - mid) == 0 or (y1 - mid) == 0:
            # точное попадание в порог считаем переключением
            cross = True
        else:
            cross = (y0 - mid) * (y1 - mid) < 0
        if not cross or y1 == y0:
            continue

        # Линейная интерполяция времени пересечения порога.
        t0 = float(t[i])
        t1 = float(t[i + 1])
        alpha = (mid - y0) / (y1 - y0)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        t_cross = t0 + alpha * (t1 - t0)

        # Определяем уровни «до» и «после» перехода (упрощённо: по сглаженным значениям).
        from_level = y_min if y0 < mid else y_max
        to_level = y_min if y1 < mid else y_max

        switches.append(
            ModeSwitch(
                index=i,
                t=t_cross,
                from_level=from_level,
                to_level=to_level,
            )
        )

    return switches, y_min, y_max, mid

