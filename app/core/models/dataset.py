from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class Dataset:
    x: np.ndarray
    series: Dict[str, np.ndarray]
    x_name: str = ""

    def is_empty(self) -> bool:
        return self.x.size == 0 or not self.series


