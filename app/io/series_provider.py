from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import gc
import os
import tempfile
import logging
import traceback

import numpy as np
import pandas as pd

log = logging.getLogger("secsig")


@dataclass
class SeriesList:
    x_name: str
    y_names: List[str]


class BaseSeriesProvider:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._x_name: Optional[str] = None
        self._series_names: Optional[List[str]] = None
        self._x_memmap_path: Optional[str] = None
        self._x_memmap: Optional[np.memmap] = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None
        self._y_memmap_paths: dict[str, str] = {}
        self._y_memmaps: dict[str, np.memmap] = {}
        self._x_min_max: Optional[tuple[float, float]] = None
        self._y_min_max: dict[str, tuple[float, float]] = {}

    def list_series(self) -> SeriesList:
        if self._x_name is None or self._series_names is None:
            self._x_name, self._series_names = self._read_header()
        return SeriesList(x_name=self._x_name, y_names=list(self._series_names))

    def load_series(self, y_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if self._x_name is None or self._series_names is None:
            self.list_series()
        if y_name not in self._series_names:
            raise ValueError(f"Unknown series: {y_name}")
        x_arr = self.ensure_x_loaded()
        y_arr = self.load_y(y_name)
        return x_arr, y_arr

    def ensure_x_loaded(self, progress=None, total_hint: Optional[int] = None) -> np.ndarray:
        return self._ensure_x_memmap()

    def load_y(self, y_name: str, progress=None, total_hint: Optional[int] = None) -> np.ndarray:
        return self._read_y_column(y_name)

    def is_x_cached(self) -> bool:
        return self._x_memmap is not None

    def is_y_cached(self, y_name: str) -> bool:
        return False

    def get_x_min_max(self) -> Optional[tuple[float, float]]:
        return self._x_min_max

    def get_y_min_max(self, y_name: str) -> Optional[tuple[float, float]]:
        return self._y_min_max.get(y_name)

    def cleanup(self):
        try:
            # Drop mmap refs so Windows can delete temp files (avoid PermissionError).
            self._x_memmap = None
            self._y_memmaps.clear()
            gc.collect()

            if self._x_memmap_path and os.path.exists(self._x_memmap_path):
                try:
                    os.remove(self._x_memmap_path)
                except OSError:
                    pass
            for p in list(self._y_memmap_paths.values()):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            self._y_memmap_paths.clear()

            if self._tmpdir is not None:
                try:
                    self._tmpdir.cleanup()
                except OSError:
                    pass
        finally:
            self._x_memmap_path = None
            self._tmpdir = None

    def _read_header(self) -> Tuple[str, List[str]]:
        raise NotImplementedError

    def _read_x_column(self) -> np.ndarray:
        raise NotImplementedError

    def _read_y_column(self, y_name: str) -> np.ndarray:
        raise NotImplementedError

    def _ensure_tmpdir(self) -> str:
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="secsig_")
        return self._tmpdir.name

    def _ensure_x_memmap(self) -> np.memmap:
        if self._x_memmap is not None:
            return self._x_memmap
        x_arr = self._read_x_column()
        tmpdir = self._ensure_tmpdir()
        x_path = os.path.join(tmpdir, "x.npy")
        np.save(x_path, x_arr)
        self._x_memmap_path = x_path
        self._x_memmap = np.load(x_path, mmap_mode="r")
        return self._x_memmap


class CSVSeriesProvider(BaseSeriesProvider):
    def __init__(self, path: str | Path, sep: str = ",", decimal: str = "."):
        super().__init__(path)
        self.sep = sep
        self.decimal = decimal
        self._encoding: Optional[str] = None
        self._encodings_try: List[str] = ["utf-8", "utf-8-sig", "cp1251", "windows-1251", "koi8-r", "iso-8859-1"]
        self._cancel_requested: bool = False

    def request_cancel(self):
        self._cancel_requested = True

    def estimate_rows(self, col_name: Optional[str] = None, chunksize: int = 1_000_000) -> int:
        if self._encoding is None or self._x_name is None:
            self._resolve_read_params()
        enc = self._encoding or "utf-8"
        name = (col_name if col_name is not None else self._x_name) or 0
        log.debug("CSV estimate_rows col=%r enc=%r chunksize=%s", name, enc, chunksize)
        total = 0
        try:
            for chunk in pd.read_csv(self.path, usecols=[name], sep=None, engine="python",
                                     encoding=enc, chunksize=chunksize, dtype=str):
                if self._cancel_requested:
                    raise InterruptedError("Canceled by user")
                total += len(chunk)
        except Exception as e:
            log.error("CSV estimate_rows: %s\n%s", e, traceback.format_exc())
            raise
        log.debug("CSV estimate_rows = %s", total)
        return total

    def _stream_column_to_memmap(self, col_name: str, chunksize: int = 500_000, progress=None,
                                 total_rows: Optional[int] = None) -> tuple[str, np.memmap]:
        if self._encoding is None or self._x_name is None:
            self._resolve_read_params()
        enc = self._encoding or "utf-8"
        total = int(total_rows) if total_rows is not None else 0
        if total == 0:
            for chunk in pd.read_csv(self.path, usecols=[col_name], sep=None, engine="python",
                                     encoding=enc, chunksize=chunksize, dtype=str):
                if self._cancel_requested:
                    raise InterruptedError("Canceled by user")
                total += len(chunk)
        tmpdir = self._ensure_tmpdir()
        safe_col = str(col_name).replace(os.sep, "_").replace("/", "_").replace("\\", "_").replace(":", "_")
        out_path = os.path.join(tmpdir, f"{self.path.stem}_{safe_col}.dat")
        if os.path.exists(out_path):
            try:
                mm_r = np.memmap(out_path, dtype="float64", mode="r", shape=(total,))
                return out_path, mm_r
            except Exception:
                try:
                    os.remove(out_path)
                except Exception:
                    pass
        mm = np.memmap(out_path, dtype="float64", mode="w+", shape=(total,))
        offset = 0
        has_minmax = False
        cur_min = np.inf
        cur_max = -np.inf
        for chunk in pd.read_csv(self.path, usecols=[col_name], sep=None, engine="python",
                                 encoding=enc, chunksize=chunksize, dtype=str):
            if self._cancel_requested:
                del mm
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                raise InterruptedError("Canceled by user")
            s = chunk.iloc[:, 0].astype(str).str.replace(",", ".", regex=False)
            arr = pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()
            n = arr.shape[0]
            mm[offset:offset + n] = arr
            offset += n
            if progress is not None:
                try:
                    progress(offset, total, f"{col_name}")
                except Exception:
                    pass
            if n > 0:
                valid = np.isfinite(arr)
                if np.any(valid):
                    cmin = float(np.min(arr[valid]))
                    cmax = float(np.max(arr[valid]))
                    if not has_minmax:
                        cur_min, cur_max = cmin, cmax
                        has_minmax = True
                    else:
                        cur_min = min(cur_min, cmin)
                        cur_max = max(cur_max, cmax)
        del mm  # flush to disk
        mm_r = np.memmap(out_path, dtype="float64", mode="r", shape=(total,))
        self._cancel_requested = False
        if has_minmax:
            if str(col_name) == (self._x_name or ""):
                self._x_min_max = (cur_min, cur_max)
            else:
                self._y_min_max[str(col_name)] = (cur_min, cur_max)
        return out_path, mm_r

    def ensure_x_loaded(self, progress=None, total_hint: Optional[int] = None) -> np.ndarray:
        if self._x_memmap is not None:
            return self._x_memmap
        if self._x_name is None:
            self._resolve_read_params()
        total = int(total_hint) if total_hint else self.estimate_rows(self._x_name)
        path, mm = self._stream_column_to_memmap(self._x_name or 0, progress=progress, total_rows=total)
        self._x_memmap_path = path
        self._x_memmap = mm
        return self._x_memmap

    def load_y(self, y_name: str, progress=None, total_hint: Optional[int] = None) -> np.ndarray:
        if y_name in self._y_memmaps:
            return self._y_memmaps[y_name]
        total = int(total_hint) if total_hint else (len(self._x_memmap) if self._x_memmap is not None else 0)
        if total <= 0:
            total = self.estimate_rows(self._x_name or None)
        path, mm = self._stream_column_to_memmap(y_name, progress=progress, total_rows=total)
        self._y_memmap_paths[y_name] = path
        self._y_memmaps[y_name] = mm
        return mm

    def is_y_cached(self, y_name: str) -> bool:
        return y_name in self._y_memmaps

    def _resolve_read_params(self):
        if self._encoding is not None and self._x_name is not None and self._series_names is not None:
            return
        log.debug("CSV _resolve_read_params path=%s", self.path)
        last_err: Optional[Exception] = None
        for enc in self._encodings_try:
            try:
                log.debug("CSV пробую encoding=%r", enc)
                df = pd.read_csv(self.path, nrows=0, sep=None, engine="python", encoding=enc)
                cols = list(df.columns)
                log.debug("CSV encoding=%r cols=%s", enc, cols)
                if len(cols) < 2:
                    raise ValueError(f"CSV must contain at least 2 columns (X and one Y), got {len(cols)}: {cols}")
                self._encoding = enc
                self._x_name = str(cols[0])
                self._series_names = [str(c) for c in cols[1:]]
                log.debug("CSV ok x=%r series=%s", self._x_name, self._series_names)
                return
            except Exception as e:
                log.debug("CSV encoding=%r ошибка: %s", enc, e)
                last_err = e
        log.error("CSV: не удалось определить кодировку: %s", last_err)
        raise last_err if last_err else RuntimeError("Failed to detect CSV encoding")

    def _read_header(self) -> Tuple[str, List[str]]:
        self._resolve_read_params()
        return self._x_name or "", list(self._series_names or [])

    def _read_x_column(self) -> np.ndarray:
        if self._x_name is None or self._encoding is None:
            self._resolve_read_params()
        name = self._x_name or 0
        enc = self._encoding or "utf-8"
        df = pd.read_csv(self.path, usecols=[name], sep=None, engine="python", encoding=enc, dtype="float64")
        return df.iloc[:, 0].to_numpy()

    def _read_y_column(self, y_name: str) -> np.ndarray:
        if self._encoding is None:
            self._resolve_read_params()
        enc = self._encoding or "utf-8"
        df = pd.read_csv(self.path, usecols=[y_name], sep=None, engine="python", encoding=enc, dtype="float64")
        return df.iloc[:, 0].to_numpy()


class XLSXSeriesProvider(BaseSeriesProvider):
    def _read_header(self) -> Tuple[str, List[str]]:
        df = pd.read_excel(self.path, header=0, nrows=0)
        cols = list(df.columns)
        if len(cols) < 2:
            raise ValueError("Excel must contain at least 2 columns (X and one Y)")
        return str(cols[0]), [str(c) for c in cols[1:]]

    def _read_x_column(self) -> np.ndarray:
        if self._x_name is None:
            self._x_name, _ = self._read_header()
        df = pd.read_excel(self.path, header=0, usecols=[self._x_name])
        return df.iloc[:, 0].to_numpy()

    def _read_y_column(self, y_name: str) -> np.ndarray:
        df = pd.read_excel(self.path, header=0, usecols=[y_name])
        return df.iloc[:, 0].to_numpy()


class ParquetSeriesProvider(BaseSeriesProvider):
    def _read_header(self) -> Tuple[str, List[str]]:
        try:
            import pyarrow.parquet as pq  # type: ignore
            pf = pq.ParquetFile(self.path)
            cols = list(pf.schema.names)
        except Exception:
            df = pd.read_parquet(self.path)
            cols = list(df.columns)
            del df
        if len(cols) < 2:
            raise ValueError("Parquet must contain at least 2 columns (X and one Y)")
        return str(cols[0]), [str(c) for c in cols[1:]]

    def _read_x_column(self) -> np.ndarray:
        if self._x_name is None:
            self._x_name, _ = self._read_header()
        df = pd.read_parquet(self.path, columns=[self._x_name])
        return df.iloc[:, 0].to_numpy()

    def _read_y_column(self, y_name: str) -> np.ndarray:
        df = pd.read_parquet(self.path, columns=[y_name])
        return df.iloc[:, 0].to_numpy()


def make_series_provider(path: str | Path) -> BaseSeriesProvider:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return CSVSeriesProvider(path)
    if suffix == ".xlsx":
        return XLSXSeriesProvider(path)
    if suffix in (".parquet", ".parq"):
        return ParquetSeriesProvider(path)
    raise ValueError(f"Unsupported file extension: {suffix}")


