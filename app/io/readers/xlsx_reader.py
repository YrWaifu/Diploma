from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

from app.core.models.dataset import Dataset


def read_two_columns_xlsx(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_excel(Path(path), header=None)
    if df.shape[1] < 2:
        raise ValueError("Excel file has less than 2 columns")

    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()

    return x, y


def read_multicolumn_xlsx(path: str | Path) -> Dataset:
    # Первая строка — заголовки: [X_name, y1_name, y2_name, ...]
    # Первый столбец — X (значения), остальные — Y-серии
    p = Path(path)
    df = pd.read_excel(p, header=0)
    if df.shape[1] < 2:
        raise ValueError("Excel file must contain at least 2 columns (X and one Y)")

    columns = list(df.columns)
    x_name = str(columns[0])
    x = df.iloc[:, 0].to_numpy()

    series: Dict[str, np.ndarray] = {}
    for col in columns[1:]:
        name = str(col)
        y = df[col].to_numpy()
        series[name] = y

    return Dataset(x=x, series=series, x_name=x_name)
