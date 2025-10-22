from dataclasses import dataclass
import numpy as np


@dataclass
class Timeseries:
    t: np.ndarray
    y: np.ndarray
    name: str = ""
    unit: str = ""
    meta: dict | None = None

    def is_empty(self) -> bool:
        return self.t.size == 0 or self.y.size == 0