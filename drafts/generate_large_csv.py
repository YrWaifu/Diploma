# -*- coding: utf-8 -*-
"""
Генератор большого тестового CSV (~20-30 ГБ), имитирующего данные
лётных испытаний вертолёта с несколькими датчиками и режимами работы.

Запуск:
    python drafts/generate_large_csv.py               # дефолт ~25 ГБ
    python drafts/generate_large_csv.py --hours 1      # ~8 ГБ   (для быстрой проверки)
    python drafts/generate_large_csv.py --hours 0.1    # ~800 МБ (быстрый тест)

Формат CSV (разделитель «;»):
    time;vib_x1;vib_y1;vib_z1;vib_x2;vib_y2;vib_z2;
    strain_1;strain_2;strain_3;strain_4;temp_bearing;temp_gearbox;
    rpm;torque;mode

Режимы полёта (реалистичный сценарий):
    0 — Земля / выключен
    1 — Земной малый газ (ЗМГ)
    2 — Взлётный режим
    3 — Крейсерский полёт
    4 — Маневрирование (виражи, развороты)
    5 — Авторотация / снижение
    6 — Посадка / пробег

Сценарий циклически повторяется, имитируя несколько вылетов за сессию.

Особенности:
    - Пишет поточно (chunk по 100 000 строк) — не ест RAM.
    - Вибрация: гармоники лопастей (≈17.5 Гц × n) + редуктор (≈200 Гц) + шум,
      амплитуды зависят от режима.
    - Тензо: квазистатика + циклы нагружения, зависят от режима.
    - Температура: медленный дрейф (инерция тепловых процессов).
    - Обороты (RPM): ступенчато, с плавным переходом.
    - Крутящий момент (torque): пропорционален режиму + пульсации.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np


# ─── Параметры ────────────────────────────────────────────────────────────────

FS = 10_000          # частота дискретизации, Гц
CHUNK = 100_000      # строк в одном чанке (≈1 с при fs=10 кГц)

# Частоты вертолётных гармоник
F_BLADE = 17.5       # основная лопастная (4-лопастной НВ, ≈260 об/мин)
F_BLADE_2 = 35.0     # 2-я гармоника
F_BLADE_3 = 52.5     # 3-я гармоника
F_GEARBOX = 198.0    # зубчатая частота редуктора
F_TAIL = 87.5        # хвостовой винт (5 лопастей × 17.5)

# Столбцы
COLUMNS = [
    "time",
    "vib_x1", "vib_y1", "vib_z1",   # вибродатчик 1 (корпус редуктора)
    "vib_x2", "vib_y2", "vib_z2",   # вибродатчик 2 (хвостовая балка)
    "strain_1", "strain_2",          # тензодатчики (лонжерон лопасти)
    "strain_3", "strain_4",          # тензодатчики (подкосы)
    "temp_bearing", "temp_gearbox",  # температура подшипника / редуктора
    "rpm", "torque",                 # обороты / крутящий момент
    "mode",
]

# ─── Профили режимов ─────────────────────────────────────────────────────────
# Каждый режим описывается набором коэффициентов:
#   vib_scale   — множитель вибрации (1 = номинал крейсера)
#   strain_mean — среднее тензо
#   strain_amp  — амплитуда циклов тензо
#   strain_freq — частота циклов тензо, Гц
#   temp_target — целевая температура (°C)
#   rpm_target  — целевые обороты (об/мин)
#   torque_mean — средний крутящий момент (Н·м)

MODE_PROFILES = {
    0: dict(vib_scale=0.02, strain_mean=5,   strain_amp=0.5,  strain_freq=0,    temp_target=40,  rpm_target=0,   torque_mean=0),
    1: dict(vib_scale=0.3,  strain_mean=30,  strain_amp=5,    strain_freq=2,    temp_target=55,  rpm_target=230, torque_mean=800),
    2: dict(vib_scale=1.2,  strain_mean=180, strain_amp=40,   strain_freq=5,    temp_target=85,  rpm_target=265, torque_mean=3500),
    3: dict(vib_scale=1.0,  strain_mean=120, strain_amp=25,   strain_freq=3.5,  temp_target=78,  rpm_target=258, torque_mean=2800),
    4: dict(vib_scale=1.5,  strain_mean=200, strain_amp=55,   strain_freq=6,    temp_target=90,  rpm_target=262, torque_mean=3800),
    5: dict(vib_scale=0.5,  strain_mean=60,  strain_amp=15,   strain_freq=2.5,  temp_target=70,  rpm_target=240, torque_mean=500),
    6: dict(vib_scale=0.4,  strain_mean=40,  strain_amp=8,    strain_freq=1.5,  temp_target=60,  rpm_target=235, torque_mean=600),
}

# Сценарий одного «вылета» (режим, длительность в секундах)
SINGLE_SORTIE = [
    (0,  60),    # земля, двигатели выключены
    (1, 120),    # ЗМГ — прогрев
    (2,  45),    # взлёт
    (3, 600),    # крейсер
    (4,  90),    # маневрирование
    (3, 480),    # крейсер
    (4, 120),    # маневрирование
    (3, 300),    # крейсер
    (5,  60),    # авторотация / снижение
    (6,  45),    # посадка
    (1,  90),    # ЗМГ — охлаждение
    (0,  90),    # земля, выключение
]

SORTIE_DURATION = sum(d for _, d in SINGLE_SORTIE)  # ~2100 с ≈ 35 мин


# ─── Генерация ────────────────────────────────────────────────────────────────

def build_mode_timeline(total_seconds: float) -> list[tuple[int, float]]:
    """Создаёт полную ленту режимов, повторяя вылет нужное кол-во раз."""
    timeline: list[tuple[int, float]] = []
    elapsed = 0.0
    while elapsed < total_seconds:
        for mode_val, dur in SINGLE_SORTIE:
            actual = min(dur, total_seconds - elapsed)
            if actual <= 0:
                break
            timeline.append((mode_val, actual))
            elapsed += actual
    return timeline


def generate_chunk(
    t_start: float,
    n_samples: int,
    mode_val: int,
    rng: np.random.RandomState,
    temp_bearing: float,
    temp_gearbox: float,
    rpm_current: float,
) -> tuple[np.ndarray, float, float, float]:
    """
    Генерирует один чанк данных (n_samples строк).
    Возвращает (data_2d, temp_bearing, temp_gearbox, rpm_current) — обновлённые.
    data_2d: shape (n_samples, len(COLUMNS))
    """
    prof = MODE_PROFILES[mode_val]
    dt = 1.0 / FS
    t = np.arange(n_samples) * dt + t_start

    vs = prof["vib_scale"]

    # ── Вибрация: датчик 1 (корпус редуктора) ──
    base = (
        vs * 2.0 * np.sin(2 * np.pi * F_BLADE * t + rng.uniform(0, 2 * np.pi))
        + vs * 0.8 * np.sin(2 * np.pi * F_BLADE_2 * t + rng.uniform(0, 2 * np.pi))
        + vs * 0.3 * np.sin(2 * np.pi * F_BLADE_3 * t + rng.uniform(0, 2 * np.pi))
        + vs * 0.5 * np.sin(2 * np.pi * F_GEARBOX * t + rng.uniform(0, 2 * np.pi))
    )
    noise_level = max(0.01, vs * 0.15)
    vib_x1 = base + noise_level * rng.randn(n_samples)
    vib_y1 = base * 0.85 + noise_level * rng.randn(n_samples) + vs * 0.2 * np.sin(2 * np.pi * F_TAIL * t)
    vib_z1 = base * 0.6 + noise_level * 1.2 * rng.randn(n_samples)

    # ── Вибрация: датчик 2 (хвостовая балка) ──
    tail_component = vs * 1.5 * np.sin(2 * np.pi * F_TAIL * t + rng.uniform(0, 2 * np.pi))
    vib_x2 = tail_component * 0.5 + noise_level * 0.8 * rng.randn(n_samples)
    vib_y2 = tail_component + noise_level * rng.randn(n_samples)
    vib_z2 = tail_component * 0.7 + vs * 0.3 * np.sin(2 * np.pi * F_BLADE * t) + noise_level * rng.randn(n_samples)

    # ── Тензо ──
    s_mean = prof["strain_mean"]
    s_amp = prof["strain_amp"]
    s_freq = prof["strain_freq"]
    local_t = t - t_start
    cyclic = s_amp * np.sin(2 * np.pi * s_freq * local_t) if s_freq > 0 else np.zeros(n_samples)
    strain_1 = s_mean + cyclic + 0.8 * rng.randn(n_samples)
    strain_2 = s_mean * 0.9 + cyclic * 1.1 + 0.9 * rng.randn(n_samples)
    strain_3 = s_mean * 0.5 + cyclic * 0.6 + s_amp * 0.3 * np.sin(2 * np.pi * s_freq * 1.7 * local_t) + 0.5 * rng.randn(n_samples)
    strain_4 = s_mean * 0.4 + cyclic * 0.5 + 0.4 * rng.randn(n_samples)

    # ── Температура: экспоненциальный дрейф к целевой ──
    tau = 200.0  # постоянная времени, с
    alpha = dt / tau
    temp_b = np.empty(n_samples)
    temp_g = np.empty(n_samples)
    tb, tg = temp_bearing, temp_gearbox
    target_b = prof["temp_target"]
    target_g = prof["temp_target"] + 8.0  # редуктор чуть горячее
    for i in range(n_samples):
        tb += alpha * (target_b - tb) + 0.005 * rng.randn()
        tg += alpha * (target_g - tg) + 0.005 * rng.randn()
        temp_b[i] = tb
        temp_g[i] = tg

    # ── RPM: плавный переход ──
    rpm_target = prof["rpm_target"]
    rpm_arr = np.empty(n_samples)
    r = rpm_current
    rpm_tau = 5.0  # с
    rpm_alpha = dt / rpm_tau
    for i in range(n_samples):
        r += rpm_alpha * (rpm_target - r) + 0.05 * rng.randn()
        rpm_arr[i] = r

    # ── Крутящий момент ──
    torque_mean = prof["torque_mean"]
    torque = torque_mean + torque_mean * 0.03 * np.sin(2 * np.pi * F_BLADE * t) + 5.0 * rng.randn(n_samples)

    # ── Mode ──
    mode_arr = np.full(n_samples, float(mode_val))

    # Сборка
    data = np.column_stack([
        t, vib_x1, vib_y1, vib_z1, vib_x2, vib_y2, vib_z2,
        strain_1, strain_2, strain_3, strain_4,
        temp_b, temp_g, rpm_arr, torque, mode_arr,
    ])
    return data, float(tb), float(tg), float(r)


def write_chunk_to_file(f: io.TextIOBase, data: np.ndarray):
    """Пишет чанк в CSV максимально быстро через numpy → bytes → file."""
    buf = io.BytesIO()
    np.savetxt(buf, data, delimiter=";", fmt="%.6f")
    f.write(buf.getvalue().decode("ascii"))


def main():
    parser = argparse.ArgumentParser(description="Генератор большого тестового CSV")
    parser.add_argument("--hours", type=float, default=6.0,
                        help="Длительность записи в часах (по умолчанию 6 — ~35 ГБ)")
    parser.add_argument("--output", type=str, default=None,
                        help="Путь к выходному файлу (по умолчанию drafts/test_flight_data.csv)")
    parser.add_argument("--seed", type=int, default=42, help="Seed генератора (по умолчанию 42)")
    args = parser.parse_args()

    total_seconds = args.hours * 3600.0
    total_samples = int(total_seconds * FS)
    out_path = Path(args.output) if args.output else Path(__file__).parent / "test_flight_data.csv"

    # Оценка размера: ~25 байт на значение × 16 столбцов ≈ 220 байт/строка
    est_gb = total_samples * 220 / 1e9
    print(f"Generation params:")
    print(f"  Duration:    {args.hours:.1f} h ({total_seconds:.0f} s)")
    print(f"  Sample rate: {FS} Hz")
    print(f"  Samples:     {total_samples:,}")
    print(f"  Columns:     {len(COLUMNS)}")
    print(f"  Est. size:   ~{est_gb:.1f} GB")
    print(f"  Output:      {out_path}")
    print()

    timeline = build_mode_timeline(total_seconds)
    rng = np.random.RandomState(args.seed)

    # Начальные состояния
    temp_bearing = 25.0
    temp_gearbox = 25.0
    rpm_current = 0.0

    t_global = 0.0
    written_samples = 0
    start_wall = time.time()

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        # Заголовок
        f.write(";".join(COLUMNS) + "\n")

        for mode_val, duration in timeline:
            seg_samples = int(duration * FS)
            seg_written = 0

            while seg_written < seg_samples:
                chunk_n = min(CHUNK, seg_samples - seg_written)
                data, temp_bearing, temp_gearbox, rpm_current = generate_chunk(
                    t_global, chunk_n, mode_val, rng,
                    temp_bearing, temp_gearbox, rpm_current,
                )
                write_chunk_to_file(f, data)
                t_global += chunk_n / FS
                seg_written += chunk_n
                written_samples += chunk_n

                # Прогресс
                pct = written_samples / total_samples * 100
                elapsed = time.time() - start_wall
                rate = written_samples / elapsed if elapsed > 0 else 0
                eta = (total_samples - written_samples) / rate if rate > 0 else 0
                file_size_gb = os.path.getsize(out_path) / 1e9
                sys.stdout.write(
                    f"\r  [{pct:5.1f}%]  {written_samples:>12,} / {total_samples:,} "
                    f"| {file_size_gb:.2f} GB "
                    f"| {rate/1e6:.1f} M samp/s "
                    f"| ETA {int(eta//60)}m{int(eta%60):02d}s   "
                )
                sys.stdout.flush()

    final_size = os.path.getsize(out_path) / 1e9
    wall = time.time() - start_wall
    print(f"\n\nDone!")
    print(f"  File:    {out_path}")
    print(f"  Size:    {final_size:.2f} GB")
    print(f"  Time:    {int(wall//60)} min {int(wall%60)} s")
    print(f"  Rows:    {written_samples:,}")


if __name__ == "__main__":
    main()
