from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

from app.core.models.dataset import Dataset


def read_multicolumn_csv(path: str | Path, sep: str = ',', decimal: str = '.') -> Dataset:
    """
    Ожидаемый формат:
    - Первая строка — заголовки: [X_name, y1_name, y2_name, ...]
    - Первый столбец — X (числовой), остальные — Y-серии
    Параметры sep/decimal позволяют читать CSV с разными разделителями и десятичными знаками.
    """
    p = Path(path)
    df = pd.read_csv(p, sep=sep, decimal=decimal)
    if df.shape[1] < 2:
        raise ValueError("CSV file must contain at least 2 columns (X and one Y)")

    columns = list(df.columns)
    x_name = str(columns[0])
    x = df.iloc[:, 0].to_numpy()

    series: Dict[str, np.ndarray] = {}
    for col in columns[1:]:
        name = str(col)
        y = df[col].to_numpy()
        series[name] = y

    return Dataset(x=x, series=series, x_name=x_name)


