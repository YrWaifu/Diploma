# -*- coding: utf-8 -*-
"""Дополнительные численные проверки модулей обработки (из корня: python tests/run_diploma_tests.py)."""
from __future__ import annotations
import sys, math
import numpy as np

sys.path.insert(0, ".")
from app.processing.strain import compute_strain_characteristics
from app.processing.vibrometry import (
    compute_time_characteristics,
    psd_welch,
    rms_in_band_from_psd,
    compute_bands,
    compute_sinusoidal,
    estimate_fs_from_time,
)

PASS = "ПРОЙДЕН"
FAIL = "ПРОВАЛЕН"

results_log = []  # (section, name, status, details)

def check(name, condition, details=""):
    status = PASS if condition else FAIL
    results_log.append((name, status, details))
    mark = "✓" if condition else "✗"
    print(f"  {mark} {name}: {status}  {details}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# Вспомогательная генерация тестовых данных
FS = 2000.0   # Гц — частота дискретизации тестовых сигналов

def make_sine(freq_hz, amplitude, duration_s, fs=FS, offset=0.0, noise_std=0.0):
    """Синусоидальный сигнал с опциональным смещением и шумом."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    x = offset + amplitude * np.sin(2 * np.pi * freq_hz * t)
    if noise_std > 0:
        x += noise_std * rng.randn(len(x))
    return t, x

def make_multi_sine(components, duration_s, fs=FS, offset=0.0, noise_std=0.0):
    """Сумма синусоид. components = [(freq, amp), ...]"""
    rng = np.random.RandomState(7)
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    x = np.full(len(t), offset, dtype=float)
    for freq, amp in components:
        x += amp * np.sin(2 * np.pi * freq * t)
    if noise_std > 0:
        x += noise_std * rng.randn(len(x))
    return t, x


# Блок 1. Верификация базовой статистики (тензо)
section("БЛОК 1. Базовая статистика тензосигнала")

print("\n  Тест 1.1 — Экстремумы и среднее для синусоиды с постоянной составляющей")
print("  Сигнал: A=100, f=2 Гц, смещение=200, длительность=10 с, fs=2000 Гц")
A, F, OFFSET, DUR = 100.0, 2.0, 200.0, 10.0
_, y = make_sine(F, A, DUR, offset=OFFSET)
r = compute_strain_characteristics(y, min_prominence=0.0)

expected_min = OFFSET - A
expected_max = OFFSET + A
expected_mean = OFFSET
tol = 1.0  # допуск 1 единица

print(f"  {'Характеристика':<30} {'Ожидаемое':>14} {'Получено':>14} {'Отклонение':>12}")
print(f"  {'-'*72}")
for name, expected, got in [
    ("Минимум",  expected_min,  r.y_min),
    ("Максимум", expected_max,  r.y_max),
    ("Среднее",  expected_mean, r.y_mean),
]:
    diff = abs(got - expected)
    print(f"  {name:<30} {expected:>14.4f} {got:>14.4f} {diff:>12.4f}")
    check(f"1.1 {name}", diff < tol, f"(допуск ±{tol})")

print("\n  Тест 1.2 — Граничные случаи")
r_empty = compute_strain_characteristics(np.array([]))
check("1.2 Пустой массив → n_cycles=0", r_empty.n_cycles == 0)
check("1.2 Пустой массив → y_min=NaN",  np.isnan(r_empty.y_min))

r_const = compute_strain_characteristics(np.full(1000, 42.0))
check("1.2 Константа → y_min=y_max=mean=42", 
      r_const.y_min == 42.0 and r_const.y_max == 42.0 and r_const.y_mean == 42.0)

r_mono = compute_strain_characteristics(np.linspace(0, 100, 500))
check("1.2 Монотонный рост → n_cycles=0", r_mono.n_cycles == 0)


# Блок 2. Верификация выявления циклов и полуразмахов
section("БЛОК 2. Выявление циклов и полуразмахи")

print("\n  Тест 2.1 — Количество циклов и максимальный полуразмах")
print("  Сигналы с известным числом периодов:\n")
print(f"  {'Сигнал':<40} {'Ожид. цикл':>12} {'Получ. цикл':>13} {'Ожид. a_max':>12} {'Получ. a_max':>13}")
print(f"  {'-'*95}")

test_cases = [
    # (freq, amp, duration, expected_min_cycles, expected_amp_halfrange)
    (2.0,  100.0, 5.0,   8,  100.0),   # 2 Гц × 5 с = 10 периодов → ≥8 циклов, полуразмах=100
    (5.0,   50.0, 4.0,  18,   50.0),   # 5 Гц × 4 с = 20 периодов → ≥18 циклов
    (10.0,  25.0, 2.0,  18,   25.0),   # 10 Гц × 2 с = 20 периодов
]

for freq, amp, dur, exp_cycles, exp_hr in test_cases:
    _, y = make_sine(freq, amp, dur)
    r = compute_strain_characteristics(y, min_prominence=0.0)
    label = f"f={freq} Гц, A={amp}, t={dur} с"
    print(f"  {label:<40} {exp_cycles:>12} {r.n_cycles:>13} {exp_hr:>12.1f} {r.max_half_range:>13.4f}")
    check(f"2.1 Циклы [{label}]",       r.n_cycles >= exp_cycles)
    check(f"2.1 Полуразмах [{label}]",  abs(r.max_half_range - exp_hr) < 5.0,
          f"(допуск ±5, получено {r.max_half_range:.4f})")

print("\n  Тест 2.2 — Цикл из явного набора точек (аналитическая проверка)")
y_explicit = np.array([0.0, 40.0, 0.0, 40.0, 0.0, 40.0, 0.0])
r_ex = compute_strain_characteristics(y_explicit, min_prominence=0.0)
print(f"  Сигнал: [0, 40, 0, 40, 0, 40, 0]")
print(f"  Ожидаемый полуразмах: 20.0,  получено: {r_ex.max_half_range:.4f}")
print(f"  Ожидаемое n_cycles ≥ 3,     получено: {r_ex.n_cycles}")
check("2.2 Явные циклы — полуразмах=20", abs(r_ex.max_half_range - 20.0) < 0.1)
check("2.2 Явные циклы — n_cycles≥3",    r_ex.n_cycles >= 3)


# Блок 3. Верификация эквивалентного полуразмаха (Майнер)
section("БЛОК 3. Эквивалентный полуразмах (гипотеза Майнера)")

print("\n  Тест 3.1 — Одинаковые циклы: a_eq = a_k при любом m")
y_eq = np.array([0.0, 20.0, 0.0, 20.0, 0.0, 20.0, 0.0])
print(f"  Сигнал: 3 цикла с полуразмахом 10.0")
print(f"  {'m':>6} {'Ожид. a_eq':>12} {'Получ. a_eq':>13} {'Отклонение':>13}")
print(f"  {'-'*50}")
for m in [1.0, 3.0, 5.0, 10.0]:
    r = compute_strain_characteristics(y_eq, miner_exponent=m, min_prominence=0.0)
    diff = abs(r.equivalent_half_range - 10.0)
    print(f"  {m:>6.1f} {10.0:>12.4f} {r.equivalent_half_range:>13.4f} {diff:>13.6f}")
    check(f"3.1 a_eq при m={m}", diff < 0.1)

print("\n  Тест 3.2 — Разные амплитуды: a_eq зависит от m")
print("  Сигнал: 2 цикла с полуразмахами 5 и 20 (среднее=12.5, макс=20)")
y_diff = np.array([0.0, 10.0, 0.0, 40.0, 0.0])   # a_k = 5 и 20
print(f"  Формула: a_eq = ((a1^m + a2^m) / 2)^(1/m)")
print(f"\n  {'m':>6} {'Ожид. a_eq':>12} {'Получ. a_eq':>13} {'Статус':>10}")
print(f"  {'-'*45}")
for m in [1.0, 3.0, 5.0, 10.0]:
    a1, a2 = 5.0, 20.0
    expected = ((a1**m + a2**m) / 2) ** (1.0/m)
    r = compute_strain_characteristics(y_diff, miner_exponent=m, min_prominence=0.0)
    ok = r.n_cycles >= 2 and abs(r.equivalent_half_range - expected) < 0.5
    print(f"  {m:>6.1f} {expected:>12.4f} {r.equivalent_half_range:>13.4f} {'OK' if ok else 'ERR':>10}")
    check(f"3.2 Майнер m={m:.0f}", ok, f"ожид={expected:.4f}, получ={r.equivalent_half_range:.4f}")

print("\n  Тест 3.3 — При увеличении m, a_eq стремится к max(a_k)")
r_vals = {}
y_miner = np.array([0.0, 10.0, 0.0, 40.0, 0.0])
for m in [1, 2, 5, 10, 20]:
    rv = compute_strain_characteristics(y_miner, miner_exponent=float(m), min_prominence=0.0)
    r_vals[m] = rv.equivalent_half_range
print(f"  m:     {' '.join(f'{m:>8}' for m in r_vals)}")
print(f"  a_eq:  {' '.join(f'{v:>8.4f}' for v in r_vals.values())}")
vals = list(r_vals.values())
monotone = all(vals[i] <= vals[i+1] + 0.01 for i in range(len(vals)-1))
check("3.3 a_eq монотонно не убывает с ростом m", monotone)


# Блок 4. Верификация квазистатических значений
section("БЛОК 4. Квазистатические значения")

print("\n  Тест 4.1 — Симметричный синус: квазистатика ≈ 0")
_, y = make_sine(3.0, 50.0, 5.0)
r = compute_strain_characteristics(y, min_prominence=0.0)
print(f"  Сигнал: A=50, f=3 Гц, смещение=0")
print(f"  min_квазистатика = {r.min_quasi_static:.4f}  (ожид. ≈ 0)")
print(f"  max_квазистатика = {r.max_quasi_static:.4f}  (ожид. ≈ 0)")
check("4.1 min_qs ≈ 0", abs(r.min_quasi_static) < 5.0)
check("4.1 max_qs ≈ 0", abs(r.max_quasi_static) < 5.0)

print("\n  Тест 4.2 — Синус со смещением +100: квазистатика ≈ 100")
_, y = make_sine(3.0, 30.0, 5.0, offset=100.0)
r = compute_strain_characteristics(y, min_prominence=0.0)
print(f"  Сигнал: A=30, f=3 Гц, смещение=100")
print(f"  min_квазистатика = {r.min_quasi_static:.4f}  (ожид. ≈ 100)")
print(f"  max_квазистатика = {r.max_quasi_static:.4f}  (ожид. ≈ 100)")
check("4.2 min_qs ≈ 100", abs(r.min_quasi_static - 100.0) < 5.0)
check("4.2 max_qs ≈ 100", abs(r.max_quasi_static - 100.0) < 5.0)


# Блок 5. Верификация временных характеристик вибросигнала
section("БЛОК 5. Временные характеристики вибросигнала")

print("\n  Тест 5.1 — Синусоида: μ, RMS, пик, пик-фактор")
print("  Теория: μ=0, RMS=A/√2, пик=A, CF=√2 ≈ 1.4142\n")
print(f"  {'Амплитуда A':>12} {'Ожид. RMS':>12} {'Получ. RMS':>12} "
      f"{'Ожид. пик':>12} {'Получ. пик':>12} {'CF':>8}")
print(f"  {'-'*75}")

for A in [1.0, 5.0, 10.0]:
    _, x = make_sine(50.0, A, 4.0)
    r = compute_time_characteristics(x)
    exp_rms = A / math.sqrt(2)
    diff_rms = abs(r.rms - exp_rms)
    diff_peak = abs(r.peak - A)
    print(f"  {A:>12.1f} {exp_rms:>12.4f} {r.rms:>12.4f} "
          f"{A:>12.4f} {r.peak:>12.4f} {r.crest_factor:>8.4f}")
    check(f"5.1 RMS при A={A}", diff_rms < 0.01, f"(отклон. {diff_rms:.5f})")
    check(f"5.1 Пик при A={A}", diff_peak < 0.01, f"(отклон. {diff_peak:.5f})")

CF_expected = math.sqrt(2)
_, x_cf = make_sine(50.0, 3.0, 4.0)
r_cf = compute_time_characteristics(x_cf)
check("5.1 Пик-фактор CF=√2", abs(r_cf.crest_factor - CF_expected) < 0.05,
      f"ожид={CF_expected:.4f}, получ={r_cf.crest_factor:.4f}")


# Блок 6. Верификация PSD методом Уэлча
section("БЛОК 6. Спектральная плотность мощности (метод Уэлча)")

print("\n  Тест 6.1 — Разрешение PSD и диапазон частот")
L = 1024
_, x = make_sine(50.0, 1.0, 8.0)
freqs, P_xx = psd_welch(x, FS, segment_length=L)
df_actual = freqs[1] - freqs[0]
df_expected = FS / L
print(f"  Длина сегмента L={L}, fs={FS} Гц")
print(f"  Ожид. Δf = {df_expected:.4f} Гц,  получено Δf = {df_actual:.4f} Гц")
print(f"  Ожид. f_max = {FS/2:.1f} Гц,  получено f_max = {freqs[-1]:.1f} Гц")
print(f"  Число бинов: ожид. {L//2+1}, получено {len(freqs)}")
check("6.1 Частотное разрешение", abs(df_actual - df_expected) < 0.01)
check("6.1 Верхняя частота = fs/2", abs(freqs[-1] - FS/2) < 1.0)
check("6.1 Число бинов = L/2+1", len(freqs) == L//2 + 1)

print("\n  Тест 6.2 — Пик PSD на частоте тона")
print(f"  {'Частота тона, Гц':>20} {'Пик PSD на, Гц':>18} {'Отклонение, Гц':>18}")
print(f"  {'-'*60}")
for f_tone in [16.0, 32.0, 50.0, 100.0, 200.0]:
    _, x = make_sine(f_tone, 1.0, 8.0)
    freqs, P_xx = psd_welch(x, FS, segment_length=1024)
    peak_f = float(freqs[np.argmax(P_xx)])
    diff = abs(peak_f - f_tone)
    print(f"  {f_tone:>20.1f} {peak_f:>18.2f} {diff:>18.2f}")
    check(f"6.2 Пик PSD на {f_tone} Гц", diff < 3.0, f"(допуск ±3 Гц)")

print("\n  Тест 6.3 — Связь PSD и СКЗ: ∫ P_xx(f)df ≈ x_rms²")
print(f"  {'Амплитуда':>12} {'x_rms (врем.)':>16} {'√∫PSD·df':>14} {'Отклонение':>12}")
print(f"  {'-'*60}")
for A in [1.0, 2.0, 5.0]:
    _, x = make_sine(50.0, A, 8.0)
    r_t = compute_time_characteristics(x)
    freqs, P_xx = psd_welch(x, FS, segment_length=1024)
    df = freqs[1] - freqs[0]
    rms_from_psd = math.sqrt(float(np.sum(P_xx)) * df)
    diff = abs(rms_from_psd - r_t.rms)
    print(f"  {A:>12.1f} {r_t.rms:>16.4f} {rms_from_psd:>14.4f} {diff:>12.4f}")
    check(f"6.3 ПСВ=RMS² при A={A}", diff < 0.05, f"(допуск 0.05)")


# Блок 7. Верификация СКЗ в полосе частот
section("БЛОК 7. СКЗ в полосе частот")

print("\n  Тест 7.1 — Тон в полосе: СКЗ ≈ A/√2")
print(f"  {'Амплитуда':>12} {'Полоса, Гц':>18} {'Ожид. СКЗ':>12} {'Получ. СКЗ':>12} {'Откл.':>10}")
print(f"  {'-'*70}")
for A, f_tone, band in [(1.0, 50.0, (10, 100)), (2.0, 100.0, (80, 120)), (0.5, 200.0, (150, 250))]:
    _, x = make_sine(f_tone, A, 8.0)
    freqs, P_xx = psd_welch(x, FS, segment_length=1024)
    rms = rms_in_band_from_psd(freqs, P_xx, band[0], band[1], FS, 1024)
    exp = A / math.sqrt(2)
    diff = abs(rms - exp)
    print(f"  {A:>12.1f} {str(band):>18} {exp:>12.4f} {rms:>12.4f} {diff:>10.4f}")
    check(f"7.1 СКЗ в полосе {band}", diff < 0.1, f"(допуск 0.1)")

print("\n  Тест 7.2 — Тон вне полосы: СКЗ ≈ 0")
_, x = make_sine(50.0, 1.0, 8.0)
freqs, P_xx = psd_welch(x, FS, segment_length=1024)
rms_out = rms_in_band_from_psd(freqs, P_xx, 200.0, 400.0, FS, 1024)
print(f"  Тон 50 Гц, полоса 200–400 Гц → СКЗ = {rms_out:.6f}  (ожид. ≈ 0)")
check("7.2 Тон вне полосы → СКЗ≈0", rms_out < 0.02)


# Блок 8. Расчёт по полосам (С.Ш.В.) — таблица
section("БЛОК 8. Расчёт СКЗ по полосам (С.Ш.В.)")

print("\n  Тест 8.1 — Многотональный сигнал, расчёт по нескольким полосам")
# Сигнал: три гармоники с известными амплитудами
comps = [(16.0, 1.5), (32.0, 0.8), (200.0, 0.3)]
_, x = make_multi_sine(comps, duration_s=8.0)

bands = [(5.0, 25.0), (25.0, 50.0), (150.0, 250.0), (5.0, 250.0)]
band_res = compute_bands(x, FS, bands, segment_length=2048)

print(f"  Компоненты сигнала: {comps}")
print(f"\n  {'Полоса, Гц':>18} {'СКЗ (Sxx^0.5)':>16} {'Sxx':>14}")
print(f"  {'-'*52}")
for (f1, f2), br in zip(bands, band_res):
    print(f"  {f'{f1:.0f}–{f2:.0f}':>18} {br.rms:>16.4f} {br.sxx:>14.6f}")

# Проверки
check("8.1 Полоса 5–25 Гц содержит тон 16 Гц (RMS>0.9)", band_res[0].rms > 0.9)
check("8.1 Полоса 25–50 Гц содержит тон 32 Гц (RMS>0.4)", band_res[1].rms > 0.4)
check("8.1 Полоса 150–250 Гц содержит тон 200 Гц (RMS>0.15)", band_res[2].rms > 0.15)
check("8.1 Полная полоса ≥ суммы частичных",
      band_res[3].rms >= max(band_res[0].rms, band_res[1].rms, band_res[2].rms))
check("8.1 Sxx = RMS²",
      all(abs(br.sxx - br.rms**2) < 0.001 for br in band_res if br.rms > 0))


# Блок 9. Синусоидальная вибрация: эквивалентная амплитуда
section("БЛОК 9. Синусоидальная вибрация: эквивалентная амплитуда")

print("\n  Тест 9.1 — Эквивалентная амплитуда чистого тона")
print(f"  {'Амплитуда A':>12} {'f0, Гц':>10} {'Ожид. A_eq ≈ A':>16} {'Получ. A_eq':>14} {'Откл.':>10}")
print(f"  {'-'*68}")
for A, f0 in [(1.0, 50.0), (2.0, 100.0), (0.8, 16.0)]:
    _, x = make_sine(f0, A, 8.0)
    res = compute_sinusoidal(x, FS, [f0], af_pct=10.0,
                             calc_equiv=True, calc_eff=False, segment_length=2048)
    got = res[0].equiv_amplitude
    diff = abs(got - A)
    print(f"  {A:>12.2f} {f0:>10.1f} {A:>16.4f} {got:>14.4f} {diff:>10.4f}")
    check(f"9.1 A_eq при f0={f0} Гц, A={A}", diff < 0.6,
          f"(допуск 0.6, отклон. {diff:.4f})")

print("\n  Тест 9.2 — Тон вне полосы поиска → A_eq малая")
_, x = make_sine(50.0, 1.0, 8.0)
res = compute_sinusoidal(x, FS, [300.0], af_pct=5.0, calc_equiv=True, segment_length=2048)
got = res[0].equiv_amplitude or 0.0
print(f"  Тон 50 Гц, базовая частота 300 Гц → A_eq = {got:.6f}  (ожид. < 0.05)")
check("9.2 Тон вне полосы → A_eq≈0", got < 0.05)

print("\n  Тест 9.3 — Многогармонический сигнал: A_eq на нескольких базовых частотах")
comps_sin = [(16.0, 1.5), (32.0, 0.8), (200.0, 0.3)]
_, x = make_multi_sine(comps_sin, duration_s=8.0)
base_freqs = [16.0, 32.0, 200.0]
res = compute_sinusoidal(x, FS, base_freqs, af_pct=10.0,
                         calc_equiv=True, calc_eff=True, segment_length=4096)
print(f"  {'f0, Гц':>10} {'A сигн.':>10} {'A_eq':>10} {'A_eff':>10}")
print(f"  {'-'*45}")
amp_map = {f: a for f, a in comps_sin}
for r, f0 in zip(res, base_freqs):
    a_sig = amp_map.get(f0, 0)
    aeq = r.equiv_amplitude or float('nan')
    aef = r.eff_amplitude or float('nan')
    print(f"  {f0:>10.1f} {a_sig:>10.2f} {aeq:>10.4f} {aef:>10.4f}")
check("9.3 A_eq(16 Гц) > A_eq(32 Гц)", res[0].equiv_amplitude > res[1].equiv_amplitude)
check("9.3 A_eq(32 Гц) > A_eq(200 Гц)", res[1].equiv_amplitude > res[2].equiv_amplitude)


# Блок 10. Сводная таблица по трём режимам (имитация лётных данных)
section("БЛОК 10. Сводные результаты по режимам Р-1 / Р-2 / Р-3")

# Генерируем реалистичный многорежимный сигнал
rng = np.random.RandomState(42)
MODE_CFG = {
    "Р-1": dict(vibro_amps=[(16.0, 0.8), (32.0, 0.3), (200.0, 0.1)],
                vibro_noise=0.15, strain_mean=50.0, strain_amp=10.0, strain_freq=2.0, strain_noise=1.0),
    "Р-2": dict(vibro_amps=[(16.0, 1.5), (32.0, 0.7), (200.0, 0.4)],
                vibro_noise=0.30, strain_mean=120.0, strain_amp=30.0, strain_freq=3.0, strain_noise=2.0),
    "Р-3": dict(vibro_amps=[(16.0, 2.5), (32.0, 1.2), (200.0, 0.8)],
                vibro_noise=0.50, strain_mean=250.0, strain_amp=60.0, strain_freq=5.0, strain_noise=3.0),
}
DUR_MODE = 8.0

print("\n  10.1 — Характеристики тензосигнала по режимам")
print(f"  {'Режим':>8} {'Мин':>10} {'Макс':>10} {'Среднее':>10} "
      f"{'Цикл':>8} {'a_max':>10} {'a_eq(m=5)':>12}")
print(f"  {'-'*74}")
strain_results = {}
for label, cfg in MODE_CFG.items():
    t = np.linspace(0, DUR_MODE, int(FS * DUR_MODE), endpoint=False)
    y_s = (cfg["strain_mean"]
           + cfg["strain_amp"] * np.sin(2 * np.pi * cfg["strain_freq"] * t)
           + cfg["strain_noise"] * rng.randn(len(t)))
    r = compute_strain_characteristics(y_s, miner_exponent=5.0, min_prominence=0.0)
    strain_results[label] = r
    print(f"  {label:>8} {r.y_min:>10.2f} {r.y_max:>10.2f} {r.y_mean:>10.2f} "
          f"{r.n_cycles:>8} {r.max_half_range:>10.4f} {r.equivalent_half_range:>12.4f}")

check("10.1 Среднее Р-1 < Р-2 < Р-3",
      strain_results["Р-1"].y_mean < strain_results["Р-2"].y_mean < strain_results["Р-3"].y_mean)
check("10.1 a_max Р-1 < Р-2 < Р-3",
      strain_results["Р-1"].max_half_range < strain_results["Р-2"].max_half_range < strain_results["Р-3"].max_half_range)
check("10.1 a_eq Р-1 < Р-2 < Р-3",
      strain_results["Р-1"].equivalent_half_range < strain_results["Р-2"].equivalent_half_range < strain_results["Р-3"].equivalent_half_range)

print("\n  10.2 — Вибрационные характеристики по режимам")
print(f"  {'Режим':>8} {'μ':>8} {'RMS':>8} {'Пик':>8} {'CF':>8} "
      f"{'СКЗ 5–25':>12} {'СКЗ 25–50':>12} {'СКЗ 150–250':>13}")
print(f"  {'-'*85}")
vibro_results = {}
for label, cfg in MODE_CFG.items():
    t = np.linspace(0, DUR_MODE, int(FS * DUR_MODE), endpoint=False)
    y_v = sum(a * np.sin(2 * np.pi * f * t) for f, a in cfg["vibro_amps"])
    y_v = y_v + cfg["vibro_noise"] * rng.randn(len(t))
    rt = compute_time_characteristics(y_v)
    bands_res = compute_bands(y_v, FS, [(5, 25), (25, 50), (150, 250)], segment_length=2048)
    vibro_results[label] = (rt, bands_res)
    print(f"  {label:>8} {rt.mean:>8.4f} {rt.rms:>8.4f} {rt.peak:>8.4f} {rt.crest_factor:>8.4f} "
          f"{bands_res[0].rms:>12.4f} {bands_res[1].rms:>12.4f} {bands_res[2].rms:>13.4f}")

check("10.2 RMS Р-1 < Р-2 < Р-3",
      vibro_results["Р-1"][0].rms < vibro_results["Р-2"][0].rms < vibro_results["Р-3"][0].rms)

print("\n  10.3 — Эквивалентная амплитуда синусоидальной вибрации по режимам")
print(f"  {'Режим':>8} {'A_eq 16 Гц':>14} {'A_eq 32 Гц':>14} {'A_eq 200 Гц':>14}")
print(f"  {'-'*55}")
for label, cfg in MODE_CFG.items():
    t = np.linspace(0, DUR_MODE, int(FS * DUR_MODE), endpoint=False)
    y_v = sum(a * np.sin(2 * np.pi * f * t) for f, a in cfg["vibro_amps"])
    y_v = y_v + cfg["vibro_noise"] * rng.randn(len(t))
    sr = compute_sinusoidal(y_v, FS, [16.0, 32.0, 200.0],
                            af_pct=10.0, calc_equiv=True, segment_length=4096)
    print(f"  {label:>8} {sr[0].equiv_amplitude:>14.4f} {sr[1].equiv_amplitude:>14.4f} {sr[2].equiv_amplitude:>14.4f}")

# ИТОГ
section("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
total = len(results_log)
passed = sum(1 for _, s, _ in results_log if s == PASS)
failed = total - passed

print(f"\n  Всего тестов:   {total}")
print(f"  Пройдено:       {passed}")
print(f"  Провалено:      {failed}")

if failed:
    print(f"\n  Провалено тестов ({failed}):")
    for name, status, details in results_log:
        if status == FAIL:
            print(f"    ✗ {name}  {details}")

print(f"\n  {'='*40}")
print(f"  Результат: {'ВСЕ ТЕСТЫ ПРОЙДЕНЫ ✓' if failed == 0 else f'ИМЕЮТСЯ ОШИБКИ ✗ ({failed})'}")
print(f"  {'='*40}\n")
