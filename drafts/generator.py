# generate_wide_xlsx.py
# requirements: pandas, numpy, openpyxl
# run: python generate_wide_xlsx.py  (создаст demo_wide_7series.xlsx рядом)

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def generate_wide_xlsx(
    out_path: str | Path,
    n_points: int = 4000,
    n_series: int = 7,
    x_name: str = "Time",
    series_name_fmt: str = "Series_{:02d}",
    sample_rate_hz: float = 100.0,
    base_seed: int = 12345,
) -> Path:
    """
    Генерирует XLSX широкого формата:
      - первая строка: заголовки,
      - первый столбец: X (время, секунды),
      - следующие столбцы: Y-серии (числа с шумом/трендами/гармониками).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Время: равномерная сетка
    dt = 1.0 / float(sample_rate_hz)
    t = np.arange(n_points, dtype=float) * dt

    # RNG для воспроизводимости
    rng = np.random.default_rng(base_seed)

    data = {x_name: t}

    # Параметры базовых гармоник для разнообразия
    # Каждая серия: смесь синусов, небольшой тренд, шум и иногда ступенька/пик.
    for k in range(1, n_series + 1):
        local_rng = np.random.default_rng(base_seed + k)

        # Немного случайных частот и фаз
        f1 = 0.15 + 0.05 * k              # Гц
        f2 = 0.4 + 0.03 * (k % 5)         # Гц
        a1 = 1.0 + 0.2 * k
        a2 = 0.5 + 0.1 * (k % 3)
        phi1 = local_rng.uniform(0, 2*np.pi)
        phi2 = local_rng.uniform(0, 2*np.pi)

        y = (
            a1 * np.sin(2*np.pi*f1*t + phi1)
            + a2 * np.sin(2*np.pi*f2*t + phi2)
        )

        # Небольшой тренд
        trend = 0.001 * k * (t - t.mean())
        y = y + trend

        # Белый шум
        noise_sigma = 0.08 + 0.02 * (k % 4)
        y = y + local_rng.normal(0.0, noise_sigma, size=t.shape)

        # Редкие события: ступенька и одиночный пик (по желанию)
        if k % 2 == 0:
            step_idx = int(0.6 * n_points)
            y[step_idx:] += 0.3 + 0.05 * (k % 3)
        if k % 3 == 0:
            spike_idx = int(0.25 * n_points) + (k * 7) % int(0.1 * n_points)
            y[spike_idx: spike_idx + 3] += 1.2 + 0.1 * k

        data[series_name_fmt.format(k)] = y

    df = pd.DataFrame(data)
    df.to_excel(out_path, index=False, engine="openpyxl")
    return out_path


if __name__ == "__main__":
    path = generate_wide_xlsx("demo_wide_7series.xlsx")
    print(f"Сгенерировано: {path.resolve()}")
