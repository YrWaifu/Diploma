# generate_wide_times_4096.py
# requirements:
#   core: numpy
#   xlsx: openpyxl
#   csv : (встроенный в pandas, но мы не используем pandas тут)
#   parquet (опц.): pyarrow
#
# run examples:
#   python generate_wide_times_4096.py --seconds 30 --xlsx demo_small.xlsx
#   python generate_wide_times_4096.py --seconds 11000 --csv demo_big.csv
#   python generate_wide_times_4096.py --seconds 3600 --parquet demo_hour.parquet

from __future__ import annotations
from pathlib import Path
import argparse
import math
import numpy as np

# Сигналогенератор: 7 рядов на одной оси времени
def synth_series(t: np.ndarray, k: int, base_seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(base_seed + k)
    # частоты/фазы/амплитуды немного разные для каждой серии
    f1 = 0.15 + 0.05 * k
    f2 = 0.40 + 0.03 * (k % 5)
    a1 = 1.0 + 0.2 * k
    a2 = 0.5 + 0.1 * (k % 3)
    phi1 = rng.uniform(0, 2*np.pi)
    phi2 = rng.uniform(0, 2*np.pi)
    y = a1 * np.sin(2*np.pi*f1*t + phi1) + a2 * np.sin(2*np.pi*f2*t + phi2)
    # тренд
    y += 0.001 * k * (t - t.mean())
    # шум
    y += rng.normal(0.0, 0.08 + 0.02*(k % 4), size=t.shape)
    # редкие события
    if k % 2 == 0 and t.size > 10:
        y[int(0.6*len(t)):] += 0.3 + 0.05 * (k % 3)
    if k % 3 == 0 and t.size > 10:
        ix = int(0.25*len(t)) + (k * 7) % max(5, int(0.1 * len(t)))
        y[ix: ix+3] += 1.2 + 0.1 * k
    return y

# ---------- генерация временной оси ----------
def time_axis(seconds_total: int, sample_rate: int = 4096) -> np.ndarray:
    # 0 .. seconds_total (не включая правую границу), дискрет 1/4096
    n = int(seconds_total * sample_rate)
    return np.arange(n, dtype=np.float64) / float(sample_rate)

# ---------- XLSX (много-листов, потоковая запись) ----------
def write_xlsx_streaming(
    out_path: Path,
    seconds_total: int,
    n_series: int = 7,
    sample_rate: int = 4096,
    sheet_base_name: str = "Data",
    chunk_rows: int = 100_000,
) -> Path:
    try:
        from openpyxl import Workbook
    except ImportError as e:
        raise RuntimeError("Нужен openpyxl: pip install openpyxl") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook(write_only=True)  # write_only снижает потребление памяти
    max_rows_excel = 1_048_576
    header = ["Time"] + [f"Series_{i:02d}" for i in range(1, n_series + 1)]
    t_total = seconds_total * sample_rate

    # сколько данных поместится на лист (минус строка заголовка)
    data_rows_per_sheet = max_rows_excel - 1
    n_sheets = math.ceil(t_total / data_rows_per_sheet)

    # генерация по кускам и запись
    written = 0
    for s in range(n_sheets):
        ws = wb.create_sheet(f"{sheet_base_name}_{s+1}")
        ws.append(header)
        rows_left_for_sheet = data_rows_per_sheet

        while rows_left_for_sheet > 0 and written < t_total:
            this_chunk = min(chunk_rows, rows_left_for_sheet, t_total - written)
            # построим t для этого чанка без хранения всей оси
            start_idx = written
            stop_idx = written + this_chunk
            t = (np.arange(start_idx, stop_idx, dtype=np.float64) / float(sample_rate))

            # создаём пакет из 7 серий
            series_block = [synth_series(t, k) for k in range(1, n_series + 1)]

            # построчно записываем (ws.append принимает tuple/list)
            for i in range(this_chunk):
                row = [float(t[i])] + [float(series_block[j][i]) for j in range(n_series)]
                ws.append(row)

            written += this_chunk
            rows_left_for_sheet -= this_chunk

    # Удалить дефолтный пустой лист (если он остался)
    if "Sheet" in wb.sheetnames and len(wb.sheetnames) > 1:
        ws0 = wb["Sheet"]
        wb.remove(ws0)

    wb.save(out_path)
    return out_path

# ---------- CSV (быстро и адекватно для больших объёмов) ----------
def write_csv_chunked(
    out_path: Path,
    seconds_total: int,
    n_series: int = 7,
    sample_rate: int = 4096,
    chunk_rows: int = 1_000_000,
) -> Path:
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Time"] + [f"Series_{i:02d}" for i in range(1, n_series + 1)]
    t_total = seconds_total * sample_rate
    written = 0
    mode = "w"
    while written < t_total:
        this_chunk = min(chunk_rows, t_total - written)
        start_idx = written
        stop_idx = written + this_chunk
        t = (np.arange(start_idx, stop_idx, dtype=np.float64) / float(sample_rate))
        series_block = [synth_series(t, k) for k in range(1, n_series + 1)]

        with open(out_path, mode, newline="") as f:
            w = csv.writer(f)
            if mode == "w":
                w.writerow(header)
            for i in range(this_chunk):
                w.writerow([t[i]] + [series_block[j][i] for j in range(n_series)])
        mode = "a"
        written += this_chunk
    return out_path

# ---------- Parquet (самый вменяемый для 10^7+ строк) ----------
def write_parquet(
    out_path: Path,
    seconds_total: int,
    n_series: int = 7,
    sample_rate: int = 4096,
    chunk_rows: int = 1_000_000,
) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError("Нужен pyarrow: pip install pyarrow") from e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("Time", pa.float64())] + [(f"Series_{i:02d}", pa.float64()) for i in range(1, n_series + 1)])
    writer = None
    try:
        t_total = seconds_total * sample_rate
        written = 0
        while written < t_total:
            this_chunk = min(chunk_rows, t_total - written)
            start_idx = written
            stop_idx = written + this_chunk
            t = (np.arange(start_idx, stop_idx, dtype=np.float64) / float(sample_rate))
            cols = [t] + [synth_series(t, k) for k in range(1, n_series + 1)]
            table = pa.Table.from_arrays([pa.array(c) for c in cols], schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(out_path, schema)
            writer.write_table(table)
            written += this_chunk
    finally:
        if writer is not None:
            writer.close()
    return out_path

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Генератор широких временных рядов: 7 серий, 4096 Гц.")
    p.add_argument("--seconds", type=int, default=30, help="Длительность в секундах (<= 11000).")
    p.add_argument("--xlsx", type=str, help="Путь к XLSX (создаст мульти-листовый файл).")
    p.add_argument("--csv", type=str, help="Путь к CSV (рекомендуется для больших объёмов).")
    p.add_argument("--parquet", type=str, help="Путь к Parquet (самый быстрый и компактный).")
    p.add_argument("--series", type=int, default=7, help="Сколько серий генерировать.")
    p.add_argument("--chunk", type=int, default=100_000, help="Размер чанка обработки.")
    args = p.parse_args()

    if args.seconds < 1 or args.seconds > 11000:
        raise SystemExit("seconds должен быть в диапазоне 1..11000")

    produced = []
    if args.xlsx:
        produced.append(write_xlsx_streaming(Path(args.xlsx), args.seconds, args.series, 4096, chunk_rows=args.chunk))
    if args.csv:
        produced.append(write_csv_chunked(Path(args.csv), args.seconds, args.series, 4096, chunk_rows=max(args.chunk, 1_000_000)))
    if args.parquet:
        produced.append(write_parquet(Path(args.parquet), args.seconds, args.series, 4096, chunk_rows=max(args.chunk, 1_000_000)))

    if not produced:
        # по умолчанию делаем маленький XLSX на 30 секунд, чтобы показать формат
        path = write_xlsx_streaming(Path("demo_4096_small.xlsx"), seconds_total=min(args.seconds, 30), n_series=args.series, sample_rate=4096)
        print(f"Сгенерировано (демо XLSX): {path.resolve()}")
    else:
        for pth in produced:
            print(f"Сгенерировано: {pth.resolve()}")

if __name__ == "__main__":
    main()
