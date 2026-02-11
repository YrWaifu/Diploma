import random
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

def build_mode_segments(total_seconds: float, min_s: int = 7, max_s: int = 10, start_value: int = 0):
    segments = []
    t = 0.0
    v = start_value
    while t < total_seconds - 1e-12:
        dur = random.randint(min_s, max_s)
        end_t = min(t + dur, total_seconds)
        segments.append((t, end_t, v))
        t = end_t
        v = 1 - v
    return segments

def mode_cmd_at_time(segments, t):
    for a, b, v in segments:
        if a <= t < b or (abs(t - b) < 1e-12 and b == segments[-1][1]):
            return v
    return segments[-1][2]

class SmoothSignal:
    def __init__(self, value: float, target: float, pull: float, target_jitter: float, noise: float):
        self.value = value
        self.target = target
        self.pull = pull
        self.target_jitter = target_jitter
        self.noise = noise

    def step(self):
        self.target += random.gauss(0, self.target_jitter)
        self.value += (self.target - self.value) * self.pull
        return self.value + random.gauss(0, self.noise)

class ModeLinearRamp:
    """
    Линейные переходы как в примере:
    0, 0.2, 0.4, 0.6, 0.8, 1, 1, ... потом 0.8, 0.6, ...
    Ramp задаётся либо временем (ramp_seconds), либо числом шагов (ramp_steps).
    """
    def __init__(self, initial: float = 0.0):
        self.value = initial
        self.target = initial
        self.step_delta = 0.0
        self.steps_left = 0

    def start_ramp(self, new_target: float, ramp_steps: int):
        self.target = new_target
        ramp_steps = max(1, int(ramp_steps))
        self.steps_left = ramp_steps
        self.step_delta = (self.target - self.value) / ramp_steps

    def step(self):
        if self.steps_left > 0:
            self.value += self.step_delta
            self.steps_left -= 1
            if self.steps_left == 0:
                self.value = self.target
        return self.value

def generate_xlsx(
    filename: str = "helicopter_like.xlsx",
    rows: int = 4000,
    total_seconds: float = 40.0,
    seed: int | None = 42,
    ramp_seconds_range=(0.6, 1.2),   # длительность линейного перехода
    quantize_step=0.2                # сделать значения как 0.2,0.4,... (поставь None если не надо)
):
    if seed is not None:
        random.seed(seed)

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"

    headers = ["Time", "Temp_C", "Rotor_RPM", "Vib_g", "Altitude_m", "Pressure_kPa", "Current_A", "Режим"]
    ws.append(headers)

    dt = total_seconds / (rows - 1)
    segments = build_mode_segments(total_seconds, 7, 10, start_value=0)

    temp = SmoothSignal(35.0, 36.0, pull=0.01, target_jitter=0.002, noise=0.05)
    rpm  = SmoothSignal(410.0, 412.0, pull=0.02, target_jitter=0.01,  noise=0.2)
    alt  = SmoothSignal(120.0, 120.0, pull=0.005, target_jitter=0.02, noise=0.1)
    vib_base = SmoothSignal(0.15, 0.15, pull=0.03, target_jitter=0.001, noise=0.01)
    current = SmoothSignal(30.0, 30.0, pull=0.03, target_jitter=0.02, noise=0.1)

    vib_pulse = 0.0

    def pressure_from_alt(a_m):
        return 101.3 - 0.012 * a_m

    # Режим с линейным ramp
    mode_lin = ModeLinearRamp(initial=float(mode_cmd_at_time(segments, 0.0)))
    last_cmd = int(round(mode_lin.value))

    for i in range(rows):
        t = i * dt
        cmd = mode_cmd_at_time(segments, t)

        # если команда изменилась -> запускаем ramp
        if cmd != last_cmd:
            ramp_s = random.uniform(*ramp_seconds_range)
            ramp_steps = max(1, int(round(ramp_s / dt)))
            mode_lin.start_ramp(float(cmd), ramp_steps)
            last_cmd = cmd

        mode_val = mode_lin.step()

        # опционально: квантуем в шаг 0.2, чтобы было как в примере
        if quantize_step is not None:
            mode_val = round(mode_val / quantize_step) * quantize_step
            # ограничим
            mode_val = max(0.0, min(1.0, mode_val))

        # влияние режима на другие каналы
        temp.target = 36.0 + (2.5 * mode_val)
        rpm.target  = 412.0 + (8.0 * mode_val)
        current.target = 30.0 + (18.0 * mode_val)

        # редкие события вибрации
        if random.random() < 0.002:
            vib_pulse += random.uniform(0.3, 0.9)
        vib_pulse *= 0.97

        temp_val = temp.step()
        rpm_val = rpm.step()
        alt_val = alt.step()
        vib_val = vib_base.step() + vib_pulse + abs(random.gauss(0, 0.005))
        pressure_val = pressure_from_alt(alt_val) + random.gauss(0, 0.05)
        current_val = current.step() + (0.3 * vib_pulse)

        ws.append([
            round(t, 6),
            round(temp_val, 3),
            round(rpm_val, 2),
            round(vib_val, 4),
            round(alt_val, 2),
            round(pressure_val, 3),
            round(current_val, 2),
            round(mode_val, 4),
        ])

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"

    widths = [10, 10, 12, 10, 12, 13, 12, 10]
    for idx, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = w

    wb.save(filename)
    return filename, segments

if __name__ == "__main__":
    fn, segs = generate_xlsx(
        filename="helicopter_like.xlsx",
        rows=4000,
        total_seconds=40.0,
        seed=42,
        ramp_seconds_range=(0.6, 1.2),  # делай больше для более длинного “склона”
        quantize_step=0.2               # как в примере; поставь None для плавных дробных
    )
    print("Saved:", fn)
    print("Mode segments:", segs)
